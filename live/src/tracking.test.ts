/**
 * The tracker's job is identity, and its failure mode is not a lost person but
 * a swapped one: if two people trade ids, they trade smoothers and holds, and
 * each ends up wearing the other's photograph. So the cases that matter are the
 * ones where MediaPipe's output order changes under us, and where a detection
 * goes missing for a frame.
 */

import { describe, expect, it } from 'vitest'

import { NUM_JOINTS, type Keypoints } from './encoding'
import { PersonTracker } from './tracking'

/** A figure standing at (x, y), roughly 0.3 of the frame tall. */
function figure(x: number, y: number): Keypoints {
  const kp = new Float32Array(NUM_JOINTS * 3)
  for (let j = 0; j < NUM_JOINTS; j++) {
    kp[j * 3] = x + (j % 3) * 0.02
    kp[j * 3 + 1] = y + (j / NUM_JOINTS) * 0.3
    kp[j * 3 + 2] = 0.9
  }
  return kp
}

const ids = (people: readonly { id: number }[]) => people.map((p) => p.id)

describe('PersonTracker', () => {
  it('gives each person in the first frame an id', () => {
    const tracker = new PersonTracker()
    const { people } = tracker.update([figure(0.2, 0.3), figure(0.7, 0.3)], 0)
    expect(ids(people)).toEqual([1, 2])
  })

  it('keeps ids when the detection order flips', () => {
    const tracker = new PersonTracker()
    tracker.update([figure(0.2, 0.3), figure(0.7, 0.3)], 0)
    const { people } = tracker.update([figure(0.71, 0.3), figure(0.21, 0.3)], 33)
    const byPosition = [...people].sort((a, b) => a.cx - b.cx)
    expect(ids(byPosition)).toEqual([1, 2])
  })

  it('follows a person across the frame in small steps', () => {
    const tracker = new PersonTracker()
    tracker.update([figure(0.1, 0.3)], 0)
    let now = 0
    for (let x = 0.15; x < 0.8; x += 0.05) {
      now += 33
      const { people } = tracker.update([figure(x, 0.3)], now)
      expect(ids(people)).toEqual([1])
    }
  })

  it('coasts through a dropped frame rather than reissuing an id', () => {
    const tracker = new PersonTracker()
    tracker.update([figure(0.4, 0.3)], 0)
    const missed = tracker.update([], 33)
    expect(ids(missed.people)).toEqual([1])
    expect(missed.people[0].visible).toBe(false)
    expect(missed.retired).toEqual([])

    const back = tracker.update([figure(0.4, 0.3)], 66)
    expect(ids(back.people)).toEqual([1])
    expect(back.people[0].visible).toBe(true)
  })

  it('retires a person who stays gone past the grace period', () => {
    const tracker = new PersonTracker({ graceMs: 350 })
    tracker.update([figure(0.4, 0.3)], 0)
    tracker.update([], 100)
    const { people, retired } = tracker.update([], 500)
    expect(people).toEqual([])
    expect(retired).toEqual([1])
  })

  it('does not reuse an id after someone leaves', () => {
    const tracker = new PersonTracker({ graceMs: 350 })
    tracker.update([figure(0.4, 0.3)], 0)
    tracker.update([], 500)
    const { people } = tracker.update([figure(0.4, 0.3)], 600)
    expect(ids(people)).toEqual([2])
  })

  it('treats a jump much larger than the figure as a new person', () => {
    const tracker = new PersonTracker()
    tracker.update([figure(0.05, 0.3)], 0)
    // The old track is not matched to it — it coasts out its grace period while
    // the far-away figure starts its own identity.
    const { people } = tracker.update([figure(0.9, 0.3)], 33)
    expect(people.filter((p) => p.visible).map((p) => p.id)).toEqual([2])
    expect(people.find((p) => p.id === 1)?.visible).toBe(false)
  })

  it('ignores a pose with no confident joints', () => {
    const tracker = new PersonTracker()
    const blank = new Float32Array(NUM_JOINTS * 3)
    const { people } = tracker.update([blank], 0)
    expect(people).toEqual([])
  })
})
