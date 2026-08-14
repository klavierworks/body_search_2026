/**
 * The browser half of the pose vector encoding.
 *
 * This is a port of `preprocessing/bodypose/encoding.py` and has to stay
 * identical to it — a query vector encoded differently from the index is not a
 * worse match, it is a meaningless one. The parameters that could drift
 * (origin, scale, confidence floor) are not duplicated here as constants; they
 * arrive from the index's own `index.json` and are applied at runtime, and
 * `assertCompatible` refuses an index this code was not written against.
 */

export const ENCODING_VERSION = 'coco17-v1'

export const COCO17 = [
  'nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
] as const

export const NUM_JOINTS = 17
export const DIMS = NUM_JOINTS * 2
export const MIN_EXTENT = 1e-4

/** Index permutation that swaps left and right. */
export const MIRROR_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

/** BlazePose's 33 landmarks contain COCO's 17; this is the gather. */
export const BLAZEPOSE_TO_COCO17 = [
  0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
]

/** Bones, as index pairs into COCO17 — for the skeleton overlay. */
export const SKELETON: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6],
]

export interface EncodingParams {
  version: string
  origin: 'center' | 'corner'
  scale: 'uniform' | 'independent'
  conf_floor: number
}

export interface EncodedPose {
  /** 34 floats, unit L2 norm. */
  vector: Float32Array
  /** 17 confidences in [0, 1]; zero where the joint was below the floor. */
  weights: Float32Array
  bbox: [number, number, number, number]
  visible: number
}

/**
 * Keypoints are a flat `[x, y, confidence] * 17` array in image-normalised,
 * top-left-origin coordinates — the same convention the indexer uses.
 */
export type Keypoints = Float32Array

export function assertCompatible(params: EncodingParams): void {
  if (params.version !== ENCODING_VERSION) {
    throw new Error(
      `Index was built with encoding "${params.version}" but this build speaks ` +
        `"${ENCODING_VERSION}". Rebuild the index, or check out the matching commit.`,
    )
  }
  if (params.origin !== 'center' && params.origin !== 'corner') {
    throw new Error(`Unknown origin mode "${params.origin}" in index.json`)
  }
  if (params.scale !== 'uniform' && params.scale !== 'independent') {
    throw new Error(`Unknown scale mode "${params.scale}" in index.json`)
  }
}

/**
 * Translate, scale and normalise a pose into the comparable unit vector.
 * Returns null when there is nothing to encode — no confident joints, or a
 * bounding box with no extent in either axis.
 */
export function encode(kp: Keypoints, params: EncodingParams): EncodedPose | null {
  let visible = 0
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity

  for (let j = 0; j < NUM_JOINTS; j++) {
    const c = kp[j * 3 + 2]
    if (c < params.conf_floor) continue
    visible++
    const x = kp[j * 3]
    const y = kp[j * 3 + 1]
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
  }
  if (visible === 0) return null

  const w = x1 - x0
  const h = y1 - y0
  if (w < MIN_EXTENT && h < MIN_EXTENT) return null

  const ox = params.origin === 'center' ? (x0 + x1) * 0.5 : x0
  const oy = params.origin === 'center' ? (y0 + y1) * 0.5 : y0

  let sx: number
  let sy: number
  if (params.scale === 'uniform') {
    sx = sy = Math.max(w, h, MIN_EXTENT)
  } else {
    sx = Math.max(w, MIN_EXTENT)
    sy = Math.max(h, MIN_EXTENT)
  }

  const vector = new Float32Array(DIMS)
  const weights = new Float32Array(NUM_JOINTS)
  for (let j = 0; j < NUM_JOINTS; j++) {
    const c = kp[j * 3 + 2]
    if (c < params.conf_floor) continue
    // Joints below the floor stay at the origin, which is neutral once weighted.
    vector[j * 2] = (kp[j * 3] - ox) / sx
    vector[j * 2 + 1] = (kp[j * 3 + 1] - oy) / sy
    weights[j] = Math.min(1, Math.max(0, c))
  }

  let norm = 0
  for (let i = 0; i < DIMS; i++) norm += vector[i] * vector[i]
  norm = Math.sqrt(norm)
  if (norm < MIN_EXTENT) return null
  const inv = 1 / norm
  for (let i = 0; i < DIMS; i++) vector[i] *= inv

  return { vector, weights, bbox: [x0, y0, x1, y1], visible }
}

/** Reflect a pose about the vertical axis, swapping left/right identities. */
export function mirrorKeypoints(kp: Keypoints): Keypoints {
  const out = new Float32Array(NUM_JOINTS * 3)
  for (let j = 0; j < NUM_JOINTS; j++) {
    const src = MIRROR_PERM[j]
    out[j * 3] = 1 - kp[src * 3]
    out[j * 3 + 1] = kp[src * 3 + 1]
    out[j * 3 + 2] = kp[src * 3 + 2]
  }
  return out
}

/** How many of the four shoulder/hip joints cleared the floor. */
export function torsoCount(kp: Keypoints, confFloor: number): number {
  let n = 0
  for (const j of [5, 6, 11, 12]) if (kp[j * 3 + 2] >= confFloor) n++
  return n
}
