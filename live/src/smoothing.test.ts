/**
 * The hold is the thing that decides what stays on screen, and its failure mode
 * is not a wrong picture but a frozen one: judge a challenger against the
 * incumbent's historical best rather than its current fit and the accepted
 * distances can only ratchet downwards, until nothing in the corpus can clear
 * the margin and the display locks for the rest of the session.
 */

import { describe, expect, it } from 'vitest'

import { MatchHold } from './smoothing'

interface Row {
  path: string
  distance: number
}

const at = (path: string, distance: number): Row => ({ path, distance })

describe('MatchHold', () => {
  it('shows the first match immediately', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    expect(hold.update([at('a.jpg', 0.2)], 0)).toEqual({ match: at('a.jpg', 0.2), changed: true })
  })

  it('holds through the dwell time even when a challenger is clearly better', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    hold.update([at('a.jpg', 0.2)], 0)
    const result = hold.update([at('b.jpg', 0.01), at('a.jpg', 0.2)], 100)
    expect(result).toEqual({ match: at('a.jpg', 0.2), changed: false })
  })

  it('ignores a challenger inside the margin', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    hold.update([at('a.jpg', 0.2)], 0)
    const result = hold.update([at('b.jpg', 0.195), at('a.jpg', 0.2)], 1000)
    expect(result.match?.path).toBe('a.jpg')
    expect(result.changed).toBe(false)
  })

  it('switches for a settled challenger outside the margin', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    hold.update([at('a.jpg', 0.2)], 0)
    const result = hold.update([at('b.jpg', 0.18), at('a.jpg', 0.2)], 1000)
    expect(result).toEqual({ match: at('b.jpg', 0.18), changed: true })
  })

  it('re-prices the incumbent each frame rather than holding its best score', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    // 'a' wins at a very good distance — the body is momentarily in that pose.
    hold.update([at('a.jpg', 0.02)], 0)
    // The body moves on. 'a' now fits badly, 'b' fits well. Judged against 'a'
    // as it stands now, 'b' takes over; judged against 'a' at its best, nothing
    // ever could.
    const result = hold.update([at('b.jpg', 0.05), at('a.jpg', 0.4)], 1000)
    expect(result).toEqual({ match: at('b.jpg', 0.05), changed: true })
  })

  it('yields to the challenger when the incumbent drops out of the results', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    hold.update([at('a.jpg', 0.02)], 0)
    const result = hold.update([at('b.jpg', 0.3)], 1000)
    expect(result).toEqual({ match: at('b.jpg', 0.3), changed: true })
  })

  it('does not ratchet the displayed distance downwards over many changes', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    let now = 0
    let shown: Row | null = null
    // A body moving through a sequence of poses: each frame some image fits
    // well, and the previous one no longer does.
    for (let i = 0; i < 12; i++) {
      now += 1000
      const winner = at(`pose-${i}.jpg`, 0.04)
      const stale = shown ? [at(shown.path, 0.5)] : []
      const result = hold.update([winner, ...stale], now)
      expect(result.match).toEqual(winner)
      shown = result.match
    }
  })

  it('keeps the dwell clock running across a re-price', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    hold.update([at('a.jpg', 0.2)], 0)
    // Re-pricing must not restart the clock, or a stream of challengers would
    // keep pushing the dwell deadline out and nothing could ever take over.
    hold.update([at('b.jpg', 0.05), at('a.jpg', 0.3)], 200)
    hold.update([at('b.jpg', 0.05), at('a.jpg', 0.3)], 400)
    const result = hold.update([at('b.jpg', 0.05), at('a.jpg', 0.3)], 460)
    expect(result).toEqual({ match: at('b.jpg', 0.05), changed: true })
  })

  it('drops the incumbent on reset', () => {
    const hold = new MatchHold<Row>({ minDwellMs: 450, margin: 0.012 })
    hold.update([at('a.jpg', 0.02)], 0)
    hold.reset()
    expect(hold.update([at('b.jpg', 0.9)], 10)).toEqual({ match: at('b.jpg', 0.9), changed: true })
  })
})
