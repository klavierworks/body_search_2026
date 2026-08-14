/**
 * Who is who, frame to frame.
 *
 * MediaPipe returns an unordered array of poses per frame with no identity
 * attached. The person at index 0 this frame may be at index 1 the next, and
 * anyone the detector misses for a frame comes back as a fresh entry. That
 * matters because everything downstream is stateful *per person*: the smoother
 * averages a query vector over time and the hold keeps an image on screen for a
 * minimum dwell. Feed one person's pose into another's smoother and their
 * pictures swap over.
 *
 * So each detection is associated with a track from the previous frame by
 * proximity — nearest centroid, measured in units of the figure's own size so
 * that someone close to the camera is allowed to move further per frame than
 * someone at the back of the room. A track that gets no detection coasts on its
 * last pose for a short grace period rather than being destroyed, which is what
 * stops a single dropped frame from resetting that person's state.
 */

import { NUM_JOINTS, type Keypoints } from './encoding'

export interface Person {
  /** Stable for as long as this person is tracked. Never reused. */
  id: number
  /** The latest detection, as it came out of the model. */
  kp: Keypoints
  /**
   * The same pose, smoothed over time — what gets drawn, and what the image is
   * registered onto. BlazePose's landmarks wander by a pixel or two frame to
   * frame even on a body holding still, which reads as a shimmer on the
   * skeleton and, through the bounding box, as a twitch on the photograph.
   * Kept separate from `kp` so the search still queries with what was actually
   * detected; that path has its own smoothing, over the encoded vector.
   */
  render: Keypoints
  /** Centroid of the confident joints, in normalised camera coordinates. */
  cx: number
  cy: number
  /** Bounding-box diagonal — the figure's apparent size, same units. */
  size: number
  /** Consecutive frames this track has been matched to a detection. */
  seen: number
  /** True when this frame produced a detection; false when coasting. */
  visible: boolean
  lastSeenMs: number
}

export interface TrackerOptions {
  /** Confidence below which a joint does not count towards the centroid. */
  confFloor?: number
  /** Association radius, as a multiple of the figure's size. */
  maxDistance?: number
  /** How long a track survives with no detection before it is retired. */
  graceMs?: number
  /**
   * Time constant for the drawn pose, in ms — roughly how long it takes to
   * close two thirds of the gap to the latest detection. 0 disables it.
   */
  smoothMs?: number
}

export interface TrackerUpdate {
  /** Everyone currently tracked, coasting tracks included. */
  people: Person[]
  /** Ids retired by this update — their per-person state can be discarded. */
  retired: number[]
}

/** Where a figure is and how big it is, or null if too little of it is visible. */
export function figureExtent(
  kp: Keypoints,
  confFloor: number,
): { cx: number; cy: number; size: number } | null {
  let n = 0
  let sx = 0
  let sy = 0
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (let j = 0; j < NUM_JOINTS; j++) {
    if (kp[j * 3 + 2] < confFloor) continue
    const x = kp[j * 3]
    const y = kp[j * 3 + 1]
    n++
    sx += x
    sy += y
    if (x < x0) x0 = x
    if (x > x1) x1 = x
    if (y < y0) y0 = y
    if (y > y1) y1 = y
  }
  if (n === 0) return null
  const w = x1 - x0
  const h = y1 - y0
  return { cx: sx / n, cy: sy / n, size: Math.max(Math.hypot(w, h), 1e-3) }
}

/**
 * Move `current` a fraction `k` of the way towards `target`, in place.
 *
 * Confidences are eased along with the coordinates, so a joint the detector is
 * unsure about ramps across the drawing threshold instead of flickering over it
 * several times a second.
 */
function ease(current: Keypoints, target: Keypoints, k: number): void {
  if (k >= 1) {
    current.set(target)
    return
  }
  for (let i = 0; i < current.length; i++) {
    current[i] += (target[i] - current[i]) * k
  }
}

export class PersonTracker {
  private tracks: Person[] = []
  private nextId = 1
  private readonly confFloor: number
  private readonly maxDistance: number
  private readonly graceMs: number
  private readonly smoothMs: number

  constructor(options: TrackerOptions = {}) {
    this.confFloor = options.confFloor ?? 0.1
    this.maxDistance = options.maxDistance ?? 0.6
    this.graceMs = options.graceMs ?? 350
    this.smoothMs = options.smoothMs ?? 60
  }

  /**
   * Fold this frame's detections into the running set of tracks.
   *
   * Association is greedy over the cheapest pairs first rather than optimal
   * (Hungarian). With at most a handful of people on screen the two agree
   * except in situations — bodies crossing at the same depth — where the
   * optimal answer is not obviously the right one either.
   */
  update(poses: readonly Keypoints[], nowMs: number): TrackerUpdate {
    const detections = poses
      .map((kp) => {
        const extent = figureExtent(kp, this.confFloor)
        return extent ? { kp, ...extent } : null
      })
      .filter((d): d is NonNullable<typeof d> => d !== null)

    const pairs: { track: number; det: number; cost: number }[] = []
    for (let t = 0; t < this.tracks.length; t++) {
      const track = this.tracks[t]
      for (let d = 0; d < detections.length; d++) {
        const det = detections[d]
        const scale = Math.max(track.size, det.size)
        const cost = Math.hypot(det.cx - track.cx, det.cy - track.cy) / scale
        if (cost <= this.maxDistance) pairs.push({ track: t, det: d, cost })
      }
    }
    pairs.sort((a, b) => a.cost - b.cost)

    const takenTrack = new Set<number>()
    const takenDet = new Set<number>()
    for (const pair of pairs) {
      if (takenTrack.has(pair.track) || takenDet.has(pair.det)) continue
      takenTrack.add(pair.track)
      takenDet.add(pair.det)
      const track = this.tracks[pair.track]
      const det = detections[pair.det]
      // Framerate-independent: the fraction of the gap closed comes from the
      // time since this track was last seen, so the pose settles at the same
      // rate whether the camera delivers 15fps or 60.
      const dt = nowMs - track.lastSeenMs
      ease(track.render, det.kp, this.smoothMs > 0 ? 1 - Math.exp(-dt / this.smoothMs) : 1)
      track.kp = det.kp
      track.cx = det.cx
      track.cy = det.cy
      track.size = det.size
      track.seen++
      track.visible = true
      track.lastSeenMs = nowMs
    }

    const retired: number[] = []
    this.tracks = this.tracks.filter((track, t) => {
      if (takenTrack.has(t)) return true
      track.visible = false
      if (nowMs - track.lastSeenMs < this.graceMs) return true
      retired.push(track.id)
      return false
    })

    for (let d = 0; d < detections.length; d++) {
      if (takenDet.has(d)) continue
      const det = detections[d]
      this.tracks.push({
        id: this.nextId++,
        kp: det.kp,
        render: det.kp.slice(),
        cx: det.cx,
        cy: det.cy,
        size: det.size,
        seen: 1,
        visible: true,
        lastSeenMs: nowMs,
      })
    }

    return { people: this.tracks, retired }
  }

  reset(): number[] {
    const retired = this.tracks.map((track) => track.id)
    this.tracks = []
    return retired
  }
}
