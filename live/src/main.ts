/**
 * Live pose search: webcam -> MediaPipe -> the same 34-float vector the index
 * is built on -> nearest neighbour -> the matching photograph laid onto your
 * own skeleton.
 *
 * The whole design rests on one thing: the query vector and the indexed vectors
 * live in the same space. Nothing here compares images to images.
 *
 * Several people can be in frame at once. Each is tracked, searched and held
 * independently — see `tracking.ts` for why identity has to be maintained here
 * rather than trusted from MediaPipe's output order.
 */

import './styles.css'

import { encode, mirrorKeypoints, torsoCount, assertCompatible } from './encoding'
import { PoseSource, toKeypointsAll } from './pose'
import { Renderer } from './renderer'
import { SearchIndex, type Match } from './searchIndex'
import { MatchHold, PoseSmoother } from './smoothing'
import { PersonTracker, type Person } from './tracking'

const TOP_K = 12
/** Below this many torso joints the pose is not worth querying with. */
const MIN_TORSO = 3
/** Figures detected per frame. Detection cost is roughly linear in this. */
const MAX_PEOPLE = 4

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

const stage = el<HTMLDivElement>('stage')
const scene = el<HTMLCanvasElement>('scene')
const video = el<HTMLVideoElement>('video')
const boot = el<HTMLDivElement>('boot')
const bootStatus = el<HTMLParagraphElement>('boot-status')
const bootStart = el<HTMLButtonElement>('boot-start')

const hud = {
  corpus: el<HTMLElement>('hud-corpus'),
  people: el<HTMLElement>('hud-people'),
  cosine: el<HTMLElement>('hud-cosine'),
  distance: el<HTMLElement>('hud-distance'),
  search: el<HTMLElement>('hud-search'),
  fps: el<HTMLElement>('hud-fps'),
  path: el<HTMLElement>('hud-path'),
}

const view = {
  /** Selfie view — your raised right hand should appear on the right. */
  mirror: true,
  showHud: true,
  showSkeleton: true,
  showCamera: false,
}

const tracker = new PersonTracker()
const renderer = new Renderer(scene)

/**
 * Everything that has to persist between frames for one person: the smoother
 * carries their query vector, the hold carries the image on their skeleton.
 * Keyed by track id and discarded when the tracker retires them.
 */
interface PersonState {
  smoother: PoseSmoother
  hold: MatchHold<Match>
  shown: Match | null
}

const people = new Map<number, PersonState>()

function stateFor(id: number): PersonState {
  let state = people.get(id)
  if (!state) {
    state = {
      smoother: new PoseSmoother(0.35),
      hold: new MatchHold<Match>({ minDwellMs: 450, margin: 0.012 }),
      shown: null,
    }
    people.set(id, state)
  }
  return state
}

async function main() {
  let index: SearchIndex
  try {
    bootStatus.textContent = 'loading index…'
    index = await SearchIndex.load()
    assertCompatible(index.params)
  } catch (err) {
    return fail(
      'Could not load the pose index.\n\n' +
        `${err instanceof Error ? err.message : String(err)}\n\n` +
        'Build one first:\n' +
        '  ln -s /path/to/images INPUT/images\n' +
        '  preprocessing/.venv/bin/bodypose run --thumbs\n\n' +
        'The dev server picks it up from OUTPUT/ with no further configuration.',
    )
  }

  hud.corpus.textContent = `${index.imageCount.toLocaleString()} images · ${index.size.toLocaleString()} poses`
  bootStatus.textContent =
    `${index.size.toLocaleString()} poses from ${index.imageCount.toLocaleString()} images\n` +
    `indexed with ${index.manifest.detector.backend} on ${index.manifest.detector.compute_device}`
  bootStart.hidden = false

  bootStart.addEventListener('click', async () => {
    bootStart.hidden = true
    bootStatus.textContent = 'starting camera…'
    const source = new PoseSource(video)
    try {
      await source.start({ model: 'lite', delegate: 'GPU', maxPoses: MAX_PEOPLE })
    } catch (err) {
      return fail(
        'Could not start the camera.\n\n' +
          `${err instanceof Error ? err.message : String(err)}\n\n` +
          'getUserMedia needs a secure context: localhost is fine, a bare LAN\n' +
          'IP is not. Check Screen & Camera permissions in System Settings too.',
      )
    }
    boot.hidden = true
    renderer.setCameraAspect(video.videoWidth, video.videoHeight)
    loop(index, source)
  })
}

function loop(index: SearchIndex, source: PoseSource) {
  let frames = 0
  let fpsAt = performance.now()
  let tracked: readonly Person[] = []

  const tick = () => {
    requestAnimationFrame(tick)
    const now = performance.now()
    renderer.resize()

    const result = source.detect(now)
    if (result) {
      const update = tracker.update(toKeypointsAll(result), now)
      for (const id of update.retired) people.delete(id)
      tracked = update.people
      query(index, tracked, now)
    }

    renderer.draw(tracked, now)

    frames++
    if (now - fpsAt > 500) {
      hud.fps.textContent = (frames / ((now - fpsAt) / 1000)).toFixed(0)
      frames = 0
      fpsAt = now
    }
  }
  requestAnimationFrame(tick)
}

/**
 * Search for everyone in frame, then hand out the results.
 *
 * The search itself is per person and independent. The allocation afterwards is
 * not: two people in the same pose would otherwise be shown the same
 * photograph, twice, which reads as a bug rather than as the coincidence it is.
 * So each image goes to at most one person per frame — whoever is already
 * showing it keeps it, and among the rest the closest fit wins, everyone else
 * falling through to their next-best match.
 */
function query(index: SearchIndex, tracked: readonly Person[], now: number) {
  const confFloor = index.params.conf_floor
  const searching: { person: Person; state: PersonState; matches: Match[] }[] = []

  const t0 = performance.now()
  for (const person of tracked) {
    // Coasting on a stale pose: the query vector cannot have changed, so
    // re-running the search would only cost time.
    if (!person.visible) continue
    if (torsoCount(person.kp, confFloor) < MIN_TORSO) continue
    const state = stateFor(person.id)
    // The keypoints do both jobs: drawn as-is in camera space, and mirrored for
    // the query only in selfie view. Mirroring the drawing instead would put
    // the skeleton on the wrong side of the frame.
    const raw = encode(view.mirror ? mirrorKeypoints(person.kp) : person.kp, index.params)
    if (!raw) continue
    const matches = index.search(state.smoother.push(raw), TOP_K)
    if (matches.length) searching.push({ person, state, matches })
  }
  hud.search.textContent = `${(performance.now() - t0).toFixed(1)} ms`

  // Incumbents claim first, so an image already on someone stays on them. That
  // includes people not searched this frame — they are still wearing theirs.
  const claimed = new Set<number>()
  for (const person of tracked) {
    const shown = people.get(person.id)?.shown
    if (shown) claimed.add(shown.pathId)
  }
  searching.sort((a, b) => a.matches[0].distance - b.matches[0].distance)

  for (const { person, state, matches } of searching) {
    const mine = state.shown?.pathId
    const available = matches.filter((m) => m.pathId === mine || !claimed.has(m.pathId))
    if (!available.length) continue

    const { match, changed } = state.hold.update(available, now)
    if (!match) continue
    claimed.add(match.pathId)
    state.shown = match
    if (changed) renderer.push(person.id, match, index.imageUrl(match), person.kp)
  }

  report(tracked)
}

/**
 * The HUD has room for one match, so it reports the largest figure in frame —
 * usually whoever is closest to the camera — and counts the rest.
 */
function report(tracked: readonly Person[]) {
  const visible = tracked.filter((person) => person.visible)
  hud.people.textContent = String(visible.length)

  let primary: Person | null = null
  for (const person of visible) {
    if (!primary || person.size > primary.size) primary = person
  }
  const match = primary ? people.get(primary.id)?.shown : null
  hud.cosine.textContent = match ? match.cosine.toFixed(3) : '—'
  hud.distance.textContent = match ? match.distance.toFixed(4) : '—'
  hud.path.textContent = match ? match.path : '—'
}

function fail(message: string) {
  boot.hidden = false
  bootStart.hidden = true
  bootStatus.classList.add('error')
  bootStatus.textContent = message
}

document.addEventListener('keydown', (event) => {
  switch (event.key.toLowerCase()) {
    case 'm':
      view.mirror = !view.mirror
      // Every query vector in flight was encoded in the old handedness, and
      // every image on screen was chosen for it.
      people.clear()
      renderer.clear()
      break
    case 's':
      view.showSkeleton = !view.showSkeleton
      break
    case 'v':
      view.showCamera = !view.showCamera
      break
    case 'h':
      view.showHud = !view.showHud
      break
    default:
      return
  }
  renderer.mirror = view.mirror
  renderer.showSkeleton = view.showSkeleton
  stage.classList.toggle('mirror', view.mirror)
  stage.classList.toggle('show-camera', view.showCamera)
  stage.classList.toggle('no-hud', !view.showHud)
})

renderer.mirror = view.mirror
stage.classList.toggle('mirror', view.mirror)
window.addEventListener('resize', () => renderer.resize())
main()
