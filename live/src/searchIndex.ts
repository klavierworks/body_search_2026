/**
 * Nearest-pose search over the index the preprocessor built.
 *
 * Two stages, the same ones `bodypose query` runs:
 *
 *   1. Recall — cosine similarity against every row. Both sides are unit
 *      vectors so this is a dot product, and 43k rows x 34 dims is about 1.5M
 *      multiply-adds. That is well under a millisecond, which is why there is
 *      no VP-tree here; Move Mirror needed one in 2018, a modern laptop
 *      does not.
 *   2. Re-rank — confidence-weighted L2 over the shortlist. This is what
 *      stops an ankle the detector was guessing at from dragging the match
 *      somewhere silly. It is too expensive to run over everything and too
 *      valuable to skip, hence the two stages.
 *
 * Results are then collapsed per image, because one photograph can contribute
 * several rows (two figures, plus their mirrors) and a filmstrip of the same
 * picture four times is not a result set.
 */

import { DIMS, NUM_JOINTS, type EncodedPose, type EncodingParams } from './encoding'

export interface IndexManifest {
  schema_version: number
  built_at: string
  encoding: EncodingParams & { keypoints: string[]; dims: number; y_axis: string }
  detector: { backend: string; compute_device: string; max_side: number | null }
  filters: Record<string, unknown>
  images_root: string
  thumbs_root: string | null
  images_scanned: number | null
  images_with_figures: number
  count: number
  mirrored_included: boolean
  rejected: Record<string, number>
  files: { vectors: string; weights: string; meta: string }
}

interface IndexMeta {
  count: number
  paths: string[]
  path_id: number[]
  bbox: [number, number, number, number][]
  score: number[]
  mirrored?: number[]
}

export interface Match {
  row: number
  pathId: number
  path: string
  /** Cosine similarity from the recall stage, 1 = identical configuration. */
  cosine: number
  /** Confidence-weighted distance from the re-rank stage, 0 = identical. */
  distance: number
  /** The indexed figure's box within the image, normalised. */
  bbox: [number, number, number, number]
  /** True when this row is the left-right reflection of the stored image. */
  mirrored: boolean
}

export class SearchIndex {
  readonly manifest: IndexManifest
  readonly params: EncodingParams
  private readonly vectors: Float32Array
  private readonly weights: Float32Array
  private readonly meta: IndexMeta
  private readonly scores: Float32Array

  constructor(manifest: IndexManifest, vectors: Float32Array, weights: Float32Array, meta: IndexMeta) {
    const n = manifest.count
    if (vectors.length !== n * DIMS) {
      throw new Error(`vectors.bin has ${vectors.length} floats, expected ${n * DIMS}`)
    }
    if (weights.length !== n * NUM_JOINTS) {
      throw new Error(`weights.bin has ${weights.length} floats, expected ${n * NUM_JOINTS}`)
    }
    this.manifest = manifest
    this.params = manifest.encoding
    this.vectors = vectors
    this.weights = weights
    this.meta = meta
    this.scores = new Float32Array(n)
  }

  get size(): number {
    return this.manifest.count
  }

  get imageCount(): number {
    return this.meta.paths.length
  }

  static async load(base = '/api'): Promise<SearchIndex> {
    const [manifest, meta, vectorsBuf, weightsBuf] = await Promise.all([
      fetchJson<IndexManifest>(`${base}/index.json`),
      fetchJson<IndexMeta>(`${base}/meta.json`),
      fetchBuffer(`${base}/vectors.bin`),
      fetchBuffer(`${base}/weights.bin`),
    ])
    // The .bin files are little-endian float32 and every platform this runs on
    // is little-endian, so the typed-array view is a free reinterpret.
    return new SearchIndex(manifest, new Float32Array(vectorsBuf), new Float32Array(weightsBuf), meta)
  }

  /**
   * Find the closest indexed poses to a query.
   *
   * @param topK      how many distinct images to return
   * @param recall    shortlist depth handed to the re-rank stage
   */
  search(query: EncodedPose, topK = 24, recall = 400): Match[] {
    const n = this.manifest.count
    const q = query.vector
    const scores = this.scores

    for (let row = 0; row < n; row++) {
      const off = row * DIMS
      let dot = 0
      for (let i = 0; i < DIMS; i++) dot += this.vectors[off + i] * q[i]
      scores[row] = dot
    }

    const shortlist = topIndices(scores, Math.min(recall, n))

    const qw = query.weights
    const ranked: Match[] = []
    for (const row of shortlist) {
      const voff = row * DIMS
      const woff = row * NUM_JOINTS
      let acc = 0
      let total = 0
      for (let j = 0; j < NUM_JOINTS; j++) {
        const omega = this.weights[woff + j] * qw[j]
        if (omega <= 0) continue
        const dx = this.vectors[voff + j * 2] - q[j * 2]
        const dy = this.vectors[voff + j * 2 + 1] - q[j * 2 + 1]
        acc += omega * (dx * dx + dy * dy)
        total += omega
      }
      // No joint the detector was confident about on both sides: nothing to
      // compare, so this row cannot be trusted regardless of its cosine.
      if (total <= 0) continue
      const pathId = this.meta.path_id[row]
      ranked.push({
        row,
        pathId,
        path: this.meta.paths[pathId],
        cosine: scores[row],
        distance: Math.sqrt(acc / total),
        bbox: this.meta.bbox[row],
        mirrored: this.meta.mirrored ? this.meta.mirrored[row] === 1 : false,
      })
    }

    ranked.sort((a, b) => a.distance - b.distance)

    const seen = new Set<number>()
    const out: Match[] = []
    for (const match of ranked) {
      if (seen.has(match.pathId)) continue
      seen.add(match.pathId)
      out.push(match)
      if (out.length >= topK) break
    }
    return out
  }

  /** URL for an indexed image, served by the dev server's middleware. */
  imageUrl(match: Match): string {
    return `/img/${match.path.split('/').map(encodeURIComponent).join('/')}`
  }
}

/**
 * Indices of the `k` largest values.
 *
 * A bounded min-heap rather than a sort: sorting 43k scores to look at 400 of
 * them is most of a frame's budget thrown away, and this runs every frame.
 */
function topIndices(scores: Float32Array, k: number): Int32Array {
  const heapIdx = new Int32Array(k)
  const heapVal = new Float32Array(k)
  let size = 0

  for (let i = 0; i < scores.length; i++) {
    const v = scores[i]
    if (size < k) {
      let c = size++
      heapIdx[c] = i
      heapVal[c] = v
      while (c > 0) {
        const p = (c - 1) >> 1
        if (heapVal[p] <= heapVal[c]) break
        swap(heapIdx, heapVal, p, c)
        c = p
      }
    } else if (v > heapVal[0]) {
      heapIdx[0] = i
      heapVal[0] = v
      let p = 0
      for (;;) {
        const l = p * 2 + 1
        const r = l + 1
        let m = p
        if (l < size && heapVal[l] < heapVal[m]) m = l
        if (r < size && heapVal[r] < heapVal[m]) m = r
        if (m === p) break
        swap(heapIdx, heapVal, p, m)
        p = m
      }
    }
  }
  return size === k ? heapIdx : heapIdx.slice(0, size)
}

function swap(idx: Int32Array, val: Float32Array, a: number, b: number): void {
  const ti = idx[a]
  idx[a] = idx[b]
  idx[b] = ti
  const tv = val[a]
  val[a] = val[b]
  val[b] = tv
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} -> ${res.status} ${res.statusText}`)
  return (await res.json()) as T
}

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} -> ${res.status} ${res.statusText}`)
  return await res.arrayBuffer()
}
