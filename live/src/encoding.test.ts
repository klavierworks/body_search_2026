/**
 * The encoding is implemented twice — once in Python for the index, once in
 * TypeScript for the query — and the two must agree exactly. A silent drift
 * between them does not degrade matching, it destroys it, and it would look
 * like "the model is bad" rather than like a bug.
 *
 * So the important test here is the golden fixture: vectors produced by
 * `preprocessing/bodypose/encoding.py` that this implementation has to
 * reproduce. Regenerate it whenever the encoding changes, with the snippet in
 * `preprocessing/README.md`.
 */

import { describe, expect, it } from 'vitest'

import golden from './__fixtures__/golden.json'
import {
  DIMS,
  NUM_JOINTS,
  encode,
  mirrorKeypoints,
  torsoCount,
  type EncodingParams,
  type Keypoints,
} from './encoding'

const flatten = (triples: number[][]): Keypoints => Float32Array.from(triples.flat())

const PARAMS: EncodingParams = {
  version: 'coco17-v1',
  origin: 'center',
  scale: 'uniform',
  conf_floor: 0.1,
}

function pose(mutate: (kp: number[][]) => void = () => {}): Keypoints {
  const kp: number[][] = []
  for (let j = 0; j < NUM_JOINTS; j++) {
    kp.push([0.3 + 0.02 * j, 0.2 + 0.03 * j, 0.9])
  }
  mutate(kp)
  return flatten(kp)
}

describe('parity with the Python indexer', () => {
  it.each(golden.map((c, i) => [i, c] as const))(
    'case %i reproduces the reference vector',
    (_i, testCase) => {
      const result = encode(flatten(testCase.keypoints), testCase.params as EncodingParams)
      expect(result).not.toBeNull()
      for (let d = 0; d < DIMS; d++) {
        expect(result!.vector[d]).toBeCloseTo(testCase.vector[d], 5)
      }
      for (let j = 0; j < NUM_JOINTS; j++) {
        expect(result!.weights[j]).toBeCloseTo(testCase.weights[j], 5)
      }
      expect(result!.visible).toBe(testCase.visible)
      for (let b = 0; b < 4; b++) {
        expect(result!.bbox[b]).toBeCloseTo(testCase.bbox[b], 5)
      }
    },
  )

  it.each(golden.map((c, i) => [i, c] as const))(
    'case %i mirrors the same way Python does',
    (_i, testCase) => {
      const result = encode(mirrorKeypoints(flatten(testCase.keypoints)), testCase.params as EncodingParams)
      expect(result).not.toBeNull()
      for (let d = 0; d < DIMS; d++) {
        expect(result!.vector[d]).toBeCloseTo(testCase.mirrored_vector[d], 5)
      }
    },
  )
})

describe('encode invariants', () => {
  it('returns a unit vector', () => {
    const result = encode(pose(), PARAMS)!
    let norm = 0
    for (const v of result.vector) norm += v * v
    expect(Math.sqrt(norm)).toBeCloseTo(1, 6)
  })

  it('ignores where the figure sits in frame', () => {
    const a = encode(pose(), PARAMS)!
    const b = encode(
      pose((kp) => kp.forEach((p) => { p[0] += 0.25; p[1] -= 0.1 })),
      PARAMS,
    )!
    for (let d = 0; d < DIMS; d++) expect(b.vector[d]).toBeCloseTo(a.vector[d], 5)
  })

  it('ignores how large the figure is', () => {
    const a = encode(pose(), PARAMS)!
    const b = encode(
      pose((kp) => kp.forEach((p) => { p[0] *= 0.4; p[1] *= 0.4 })),
      PARAMS,
    )!
    for (let d = 0; d < DIMS; d++) expect(b.vector[d]).toBeCloseTo(a.vector[d], 5)
  })

  it('drops joints below the confidence floor and zeroes their weight', () => {
    const result = encode(pose((kp) => { kp[15][2] = 0.02; kp[16][2] = 0.02 }), PARAMS)!
    expect(result.visible).toBe(NUM_JOINTS - 2)
    expect(result.weights[15]).toBe(0)
    expect(result.weights[16]).toBe(0)
  })

  it('returns null when nothing is confident enough', () => {
    expect(encode(pose((kp) => kp.forEach((p) => { p[2] = 0 })), PARAMS)).toBeNull()
  })

  it('returns null for a collapsed bounding box', () => {
    expect(encode(pose((kp) => kp.forEach((p) => { p[0] = 0.5; p[1] = 0.5 })), PARAMS)).toBeNull()
  })

  it('preserves aspect ratio under uniform scale but not independent', () => {
    // A figure stretched horizontally is a different pose under uniform scaling
    // and the same pose under independent scaling. That difference is the whole
    // reason the mode is configurable.
    const base = pose()
    const stretched = pose((kp) => kp.forEach((p) => { p[0] = 0.5 + (p[0] - 0.5) * 2 }))

    const uniformA = encode(base, PARAMS)!
    const uniformB = encode(stretched, PARAMS)!
    let uniformDot = 0
    for (let d = 0; d < DIMS; d++) uniformDot += uniformA.vector[d] * uniformB.vector[d]

    const indep: EncodingParams = { ...PARAMS, scale: 'independent' }
    const indepA = encode(base, indep)!
    const indepB = encode(stretched, indep)!
    let indepDot = 0
    for (let d = 0; d < DIMS; d++) indepDot += indepA.vector[d] * indepB.vector[d]

    expect(indepDot).toBeCloseTo(1, 5)
    expect(uniformDot).toBeLessThan(0.999)
  })
})

describe('mirrorKeypoints', () => {
  it('is its own inverse', () => {
    const original = pose()
    const twice = mirrorKeypoints(mirrorKeypoints(original))
    for (let i = 0; i < original.length; i++) expect(twice[i]).toBeCloseTo(original[i], 6)
  })

  it('swaps left and right joint identities', () => {
    const kp = pose((p) => { p[5][2] = 0.9; p[6][2] = 0.2 })
    const flipped = mirrorKeypoints(kp)
    expect(flipped[5 * 3 + 2]).toBeCloseTo(0.2, 6)
    expect(flipped[6 * 3 + 2]).toBeCloseTo(0.9, 6)
  })
})

describe('torsoCount', () => {
  it('counts the shoulder and hip joints above the floor', () => {
    expect(torsoCount(pose(), 0.1)).toBe(4)
    expect(torsoCount(pose((kp) => { kp[11][2] = 0; kp[12][2] = 0 }), 0.1)).toBe(2)
  })
})
