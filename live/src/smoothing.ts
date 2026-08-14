/**
 * Temporal smoothing.
 *
 * Nearest-neighbour lookup per frame, shown raw, strobes unwatchably: keypoints
 * jitter by a pixel or two between frames, that moves the query vector slightly,
 * and near the boundary between two equally good matches the display flickers
 * between them many times a second. Two separate mechanisms fix two separate
 * problems.
 */

import { DIMS, NUM_JOINTS, type EncodedPose } from './encoding'

/**
 * Exponential moving average over the query vector, renormalised to stay on the
 * unit sphere. Damps detector jitter without adding the latency a longer window
 * would.
 *
 * `alpha` is the weight of the newest frame: 1 is no smoothing, 0.2 is heavy.
 */
export class PoseSmoother {
  private vector = new Float32Array(DIMS)
  private weights = new Float32Array(NUM_JOINTS)
  private primed = false

  constructor(public alpha = 0.35) {}

  push(pose: EncodedPose): EncodedPose {
    if (!this.primed) {
      this.vector.set(pose.vector)
      this.weights.set(pose.weights)
      this.primed = true
      return { ...pose, vector: this.vector.slice(), weights: this.weights.slice() }
    }

    const a = this.alpha
    const b = 1 - a
    let norm = 0
    for (let i = 0; i < DIMS; i++) {
      const v = this.vector[i] * b + pose.vector[i] * a
      this.vector[i] = v
      norm += v * v
    }
    norm = Math.sqrt(norm)
    if (norm > 1e-6) {
      const inv = 1 / norm
      for (let i = 0; i < DIMS; i++) this.vector[i] *= inv
    }
    for (let j = 0; j < NUM_JOINTS; j++) {
      this.weights[j] = this.weights[j] * b + pose.weights[j] * a
    }

    return { ...pose, vector: this.vector.slice(), weights: this.weights.slice() }
  }

  reset(): void {
    this.primed = false
    this.vector.fill(0)
    this.weights.fill(0)
  }
}

export interface HoldOptions {
  /** Shortest time a match stays on screen, in ms. */
  minDwellMs?: number
  /** How much better a challenger must be before it displaces the incumbent. */
  margin?: number
}

/**
 * Hysteresis on the displayed result.
 *
 * Smoothing the query stops the vector jittering; it does not stop two images
 * with near-identical distances trading places. A challenger has to be
 * meaningfully better *and* the incumbent has to have had its moment.
 */
export class MatchHold<T extends { path: string; distance: number }> {
  private current: T | null = null
  private since = 0
  private readonly minDwellMs: number
  private readonly margin: number

  constructor(options: HoldOptions = {}) {
    this.minDwellMs = options.minDwellMs ?? 450
    this.margin = options.margin ?? 0.012
  }

  /**
   * Returns the match that should be on screen, and whether it just changed.
   *
   * Takes the whole ranked result set, not just the winner, because the
   * incumbent has to be re-priced against the current query every frame. Held
   * at the distance it scored on the frame it won, it would only ever be
   * compared against its own best moment, and since each change then has to
   * beat the last by `margin`, the displayed distance could only ratchet
   * downwards until nothing in the corpus could beat it and the screen locked.
   */
  update(matches: readonly T[], nowMs: number): { match: T | null; changed: boolean } {
    const candidate = matches.length ? matches[0] : null
    if (!candidate) return { match: this.current, changed: false }
    if (!this.current) {
      this.current = candidate
      this.since = nowMs
      return { match: candidate, changed: true }
    }
    if (candidate.path === this.current.path) {
      // Same image, refreshed distance — keep the dwell clock running.
      this.current = candidate
      return { match: candidate, changed: false }
    }
    // Today's price for the incumbent. Absent from the results altogether means
    // it is not among the closest poses any more, at any margin.
    const incumbent = matches.find((m) => m.path === this.current!.path)
    if (incumbent) this.current = incumbent // refreshes distance, not the clock
    const settled = nowMs - this.since >= this.minDwellMs
    const better = !incumbent || candidate.distance < incumbent.distance - this.margin
    if (settled && better) {
      this.current = candidate
      this.since = nowMs
      return { match: candidate, changed: true }
    }
    return { match: this.current, changed: false }
  }

  reset(): void {
    this.current = null
    this.since = 0
  }
}
