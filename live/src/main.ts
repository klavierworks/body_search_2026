/**
 * Live pose search: webcam -> MediaPipe -> the same 34-float vector the index
 * is built on -> nearest neighbour -> the matching photograph laid onto your
 * own skeleton.
 *
 * The whole design rests on one thing: the query vector and the indexed vectors
 * live in the same space. Nothing here compares images to images.
 */

import './styles.css'

import { encode, mirrorKeypoints, torsoCount, assertCompatible, type Keypoints } from './encoding'
import { PoseSource, toKeypoints } from './pose'
import { Renderer } from './renderer'
import { SearchIndex, type Match } from './searchIndex'
import { MatchHold, PoseSmoother } from './smoothing'

const TOP_K = 12
/** Below this many torso joints the pose is not worth querying with. */
const MIN_TORSO = 3

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

const stage = el<HTMLDivElement>('stage')
const scene = el<HTMLCanvasElement>('scene')
const video = el<HTMLVideoElement>('video')
const boot = el<HTMLDivElement>('boot')
const bootStatus = el<HTMLParagraphElement>('boot-status')
const bootStart = el<HTMLButtonElement>('boot-start')

const hud = {
  corpus: el<HTMLElement>('hud-corpus'),
  cosine: el<HTMLElement>('hud-cosine'),
  distance: el<HTMLElement>('hud-distance'),
  search: el<HTMLElement>('hud-search'),
  fps: el<HTMLElement>('hud-fps'),
  path: el<HTMLElement>('hud-path'),
}

const state = {
  /** Selfie view — your raised right hand should appear on the right. */
  mirror: true,
  showHud: true,
  showSkeleton: true,
  showCamera: false,
}

const smoother = new PoseSmoother(0.35)
const hold = new MatchHold<Match>({ minDwellMs: 450, margin: 0.012 })
const renderer = new Renderer(scene)

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
        '  preprocessing/.venv/bin/bodypose run /path/to/images --thumbs\n\n' +
        'then point the dev server at it:\n' +
        '  BODY_ARTIFACTS=../preprocessing/out npm run dev',
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
      await source.start({ model: 'lite', delegate: 'GPU' })
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
  const confFloor = index.params.conf_floor
  let frames = 0
  let fpsAt = performance.now()
  let live: Keypoints | null = null

  const tick = () => {
    requestAnimationFrame(tick)
    const now = performance.now()
    renderer.resize()

    const result = source.detect(now)
    if (result) {
      const kp = toKeypoints(result)
      // One set of keypoints does both jobs: drawn as-is in camera space, and
      // mirrored for the query only if we are in selfie view. Mirroring the
      // drawing instead would put the skeleton on the wrong side of the frame.
      live = kp
      if (kp && torsoCount(kp, confFloor) >= MIN_TORSO) {
        query(index, state.mirror ? mirrorKeypoints(kp) : kp, kp, now)
      }
    }

    renderer.draw(live, now)

    frames++
    if (now - fpsAt > 500) {
      hud.fps.textContent = (frames / ((now - fpsAt) / 1000)).toFixed(0)
      frames = 0
      fpsAt = now
    }
  }
  requestAnimationFrame(tick)
}

function query(index: SearchIndex, queryKp: Keypoints, drawKp: Keypoints, now: number) {
  const raw = encode(queryKp, index.params)
  if (!raw) return
  const smoothed = smoother.push(raw)

  const t0 = performance.now()
  const matches = index.search(smoothed, TOP_K)
  hud.search.textContent = `${(performance.now() - t0).toFixed(1)} ms`
  if (!matches.length) return

  const { match, changed } = hold.update(matches, now)
  if (!match) return

  if (changed) {
    renderer.push(match, index.imageUrl(match), drawKp)
    hud.path.textContent = match.path
  }
  hud.cosine.textContent = match.cosine.toFixed(3)
  hud.distance.textContent = match.distance.toFixed(4)
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
      state.mirror = !state.mirror
      smoother.reset()
      hold.reset()
      renderer.clear()
      break
    case 's':
      state.showSkeleton = !state.showSkeleton
      break
    case 'v':
      state.showCamera = !state.showCamera
      break
    case 'h':
      state.showHud = !state.showHud
      break
    default:
      return
  }
  renderer.mirror = state.mirror
  renderer.showSkeleton = state.showSkeleton
  stage.classList.toggle('mirror', state.mirror)
  stage.classList.toggle('show-camera', state.showCamera)
  stage.classList.toggle('no-hud', !state.showHud)
})

renderer.mirror = state.mirror
stage.classList.toggle('mirror', state.mirror)
window.addEventListener('resize', () => renderer.resize())
main()
