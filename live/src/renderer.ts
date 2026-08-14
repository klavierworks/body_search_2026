/**
 * The compositor.
 *
 * The skeleton is the persistent thing on screen, not the photograph. It tracks
 * where you actually are in the camera frame, so it moves around as you move,
 * and the matched image is placed *onto* it — scaled and positioned so the
 * indexed figure's bounding box registers with the live figure's bounding box.
 * Then the skeleton is drawn over the top.
 *
 * That registration is the whole point, and it is only possible because the
 * index stores each figure's bounding box alongside its vector. Matching finds
 * a picture of a body in the same configuration; the box is what lets that body
 * be laid over yours at the right size and place rather than just displayed.
 *
 * Previous matches fade out behind the current one, so a moving body leaves a
 * short trail of the images it passed through.
 *
 * All of that is per person. Each tracked body owns its own stack of layers and
 * its own trail, so two people standing side by side get two independent
 * pictures registered onto them, and one person leaving the frame fades out
 * only their own images.
 */

import { NUM_JOINTS, SKELETON, type Keypoints } from './encoding'
import type { Match } from './searchIndex'
import type { Person } from './tracking'

/** A figure's extent on screen, in device pixels. */
interface Box {
  x: number
  y: number
  width: number
  height: number
}

interface Layer {
  /** The person this image was matched for. */
  trackId: number
  match: Match
  image: HTMLImageElement
  /** Where the live skeleton was when this match arrived — the image stays put. */
  box: Box
  addedAt: number
  /** When the image stopped being current, or null while it still is. */
  fadeFrom: number | null
}

export interface RendererOptions {
  /** How long a superseded image takes to fade out, in ms. */
  fadeMs?: number
  /** How many superseded images stay on screen, per person. */
  trail?: number
  /** Confidence below which a joint is not drawn. */
  confFloor?: number
}

/**
 * A colour per person, so two bodies crossing stay legible as two bodies.
 *
 * Golden-angle hue steps give well-separated colours for any number of people
 * without a fixed palette to run out of. Kept pale and barely saturated: enough
 * to tell two skeletons apart, not so much that the overlay stops reading as
 * the white line it was.
 */
export function personColor(trackId: number, alpha = 0.92): string {
  const hue = (trackId * 137.508) % 360
  return `hsl(${hue.toFixed(1)} 45% 88% / ${alpha})`
}

export class Renderer {
  private readonly canvas: HTMLCanvasElement
  private readonly ctx: CanvasRenderingContext2D
  private readonly images = new Map<string, HTMLImageElement>()
  private layers: Layer[] = []
  private readonly fadeMs: number
  private readonly trail: number
  private readonly confFloor: number

  /** Camera aspect ratio, for mapping normalised pose coords to the canvas. */
  private aspect = 16 / 9
  mirror = true
  showSkeleton = true

  constructor(canvas: HTMLCanvasElement, options: RendererOptions = {}) {
    this.canvas = canvas
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('could not get a 2d context')
    this.ctx = ctx
    this.fadeMs = options.fadeMs ?? 900
    this.trail = options.trail ?? 3
    this.confFloor = options.confFloor ?? 0.1
  }

  setCameraAspect(width: number, height: number): void {
    if (width > 0 && height > 0) this.aspect = width / height
  }

  resize(): void {
    // Cap the backing store at 2x. On a Retina display a full-screen canvas at
    // native density is a lot of fill for something redrawn every frame, and
    // the difference is invisible on photographs.
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const width = Math.round(this.canvas.clientWidth * dpr)
    const height = Math.round(this.canvas.clientHeight * dpr)
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width
      this.canvas.height = height
    }
  }

  /**
   * Show a new match for one person, pinned to wherever that figure is now.
   *
   * The image keeps that position as it fades, rather than following the body —
   * a trail that tracked you would smear instead of layering.
   */
  push(trackId: number, match: Match, url: string, kp: Keypoints): void {
    const image = this.imageFor(match.path, url)
    const box = this.liveBox(kp)
    if (!box) return

    const now = performance.now()
    // Whatever this person was showing starts fading from now, not from
    // whenever it arrived — an image that held for ten seconds should still
    // take the full fade to go.
    const previous = this.layers.find((layer) => layer.trackId === trackId)
    if (previous && previous.fadeFrom === null) previous.fadeFrom = now

    this.layers.unshift({ trackId, match, image, box, addedAt: now, fadeFrom: null })
    this.trimTrail(trackId)
  }

  /**
   * @param people everyone currently tracked; anyone absent from this list has
   *               left, and their images begin to fade.
   */
  draw(people: readonly Person[], now: number): void {
    const { ctx, canvas } = this
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const present = new Set(people.map((person) => person.id))
    // Oldest first, so each current match lands on top of its own trail.
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i]
      if (layer.fadeFrom === null && !present.has(layer.trackId)) layer.fadeFrom = now
      const alpha = layer.fadeFrom === null ? 1 : Math.max(0, 1 - (now - layer.fadeFrom) / this.fadeMs)
      if (alpha <= 0) continue
      this.drawLayer(layer, alpha)
    }
    this.layers = this.layers.filter(
      (layer) => layer.fadeFrom === null || now - layer.fadeFrom < this.fadeMs,
    )

    if (this.showSkeleton) {
      for (const person of people) this.drawSkeleton(person.kp, personColor(person.id))
    }
  }

  /** Keep at most `trail + 1` images per person; the rest are already invisible. */
  private trimTrail(trackId: number): void {
    let kept = 0
    this.layers = this.layers.filter((layer) => {
      if (layer.trackId !== trackId) return true
      return ++kept <= this.trail + 1
    })
  }

  private drawLayer(layer: Layer, alpha: number): void {
    const { image, match, box } = layer
    if (!image.complete || image.naturalWidth === 0) return

    const { naturalWidth: iw, naturalHeight: ih } = image
    const [x0, y0, x1, y1] = match.bbox

    // Scale so the indexed figure's height fills the live figure's height, then
    // place it so the two boxes share a centre. Uniform scale — matching the
    // width independently would stretch the photograph.
    const figureH = Math.max(y1 - y0, 1e-3) * ih
    const scale = box.height / figureH
    const drawW = iw * scale
    const drawH = ih * scale
    const dx = box.x + box.width / 2 - ((x0 + x1) / 2) * drawW
    const dy = box.y + box.height / 2 - ((y0 + y1) / 2) * drawH

    const { ctx } = this
    ctx.save()
    ctx.globalAlpha = alpha
    if (match.mirrored) {
      // The reflection of this image is what matched, so show the reflection.
      ctx.translate(dx + drawW / 2, 0)
      ctx.scale(-1, 1)
      ctx.translate(-(dx + drawW / 2), 0)
    }
    ctx.drawImage(image, dx, dy, drawW, drawH)
    ctx.restore()
  }

  private drawSkeleton(kp: Keypoints, color: string): void {
    const { ctx, canvas } = this
    const scale = canvas.width / 1280

    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = color
    ctx.lineWidth = Math.max(1.5, 3 * scale)
    // A dark halo keeps the white readable over a pale engraving as well as
    // over black.
    ctx.shadowColor = 'rgba(0, 0, 0, 0.55)'
    ctx.shadowBlur = 8 * scale

    for (const [a, b] of SKELETON) {
      if (kp[a * 3 + 2] < this.confFloor || kp[b * 3 + 2] < this.confFloor) continue
      const pa = this.toScreen(kp[a * 3], kp[a * 3 + 1])
      const pb = this.toScreen(kp[b * 3], kp[b * 3 + 1])
      ctx.beginPath()
      ctx.moveTo(pa.x, pa.y)
      ctx.lineTo(pb.x, pb.y)
      ctx.stroke()
    }
    ctx.restore()
  }

  /** The live figure's bounding box on screen, or null if there isn't one. */
  private liveBox(kp: Keypoints): Box | null {
    let x0 = Infinity
    let y0 = Infinity
    let x1 = -Infinity
    let y1 = -Infinity
    for (let j = 0; j < NUM_JOINTS; j++) {
      if (kp[j * 3 + 2] < this.confFloor) continue
      const p = this.toScreen(kp[j * 3], kp[j * 3 + 1])
      if (p.x < x0) x0 = p.x
      if (p.x > x1) x1 = p.x
      if (p.y < y0) y0 = p.y
      if (p.y > y1) y1 = p.y
    }
    if (!Number.isFinite(x0) || y1 - y0 < 1) return null
    return { x: x0, y: y0, width: Math.max(x1 - x0, 1), height: y1 - y0 }
  }

  /**
   * Normalised camera coordinates to canvas pixels.
   *
   * The camera frame is fitted to the canvas the way `object-fit: cover` would,
   * so the skeleton's position on screen corresponds to where the body actually
   * is in frame — walk left, the skeleton goes left.
   */
  private toScreen(nx: number, ny: number): { x: number; y: number } {
    const { width, height } = this.canvas
    const canvasAspect = width / height
    let drawW = width
    let drawH = height
    if (canvasAspect > this.aspect) {
      drawH = width / this.aspect
    } else {
      drawW = height * this.aspect
    }
    const offsetX = (width - drawW) / 2
    const offsetY = (height - drawH) / 2
    const x = this.mirror ? 1 - nx : nx
    return { x: offsetX + x * drawW, y: offsetY + ny * drawH }
  }

  private imageFor(path: string, url: string): HTMLImageElement {
    let image = this.images.get(path)
    if (!image) {
      image = new Image()
      image.decoding = 'async'
      image.src = url
      this.images.set(path, image)
      // Unbounded caching of full-size photographs is a memory leak over a
      // long session; oldest-first eviction is enough here.
      if (this.images.size > 240) {
        const oldest = this.images.keys().next().value
        if (oldest !== undefined) this.images.delete(oldest)
      }
    }
    return image
  }

  clear(): void {
    this.layers = []
  }
}
