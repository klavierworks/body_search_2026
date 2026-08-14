/**
 * Webcam capture and MediaPipe pose detection.
 *
 * BlazePose via `@mediapipe/tasks-vision` replaces the PoseNet the original
 * project would have used — it is both more accurate and cheaper, and its 33
 * landmarks contain COCO's 17 as a subset, which is the set the index is built
 * on. See `BLAZEPOSE_TO_COCO17` in encoding.ts.
 *
 * The wasm runtime and the .task model are served from `public/`, not a CDN,
 * so an installation with no network still runs. `npm run setup` puts them
 * there.
 */

import { FilesetResolver, PoseLandmarker, type PoseLandmarkerResult } from '@mediapipe/tasks-vision'

import { BLAZEPOSE_TO_COCO17, NUM_JOINTS, type Keypoints } from './encoding'

export type ModelSize = 'lite' | 'full' | 'heavy'

export interface PoseSourceOptions {
  model?: ModelSize
  /** GPU is roughly 3x faster here; CPU is the fallback for odd drivers. */
  delegate?: 'GPU' | 'CPU'
  width?: number
  height?: number
}

export class PoseSource {
  readonly video: HTMLVideoElement
  private landmarker: PoseLandmarker | null = null
  private stream: MediaStream | null = null
  private lastTimestamp = -1

  constructor(video: HTMLVideoElement) {
    this.video = video
  }

  async start(options: PoseSourceOptions = {}): Promise<void> {
    const { model = 'lite', delegate = 'GPU', width = 1280, height = 720 } = options

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: width }, height: { ideal: height }, facingMode: 'user' },
      audio: false,
    })
    this.video.srcObject = this.stream
    await this.video.play()
    await new Promise<void>((resolve) => {
      if (this.video.videoWidth > 0) return resolve()
      this.video.onloadedmetadata = () => resolve()
    })

    const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
    this.landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: `/mediapipe/pose_landmarker_${model}.task`,
        delegate,
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })
  }

  /**
   * Detect on the current video frame.
   *
   * Returns null when the frame has not advanced — MediaPipe's VIDEO mode
   * rejects a timestamp it has already seen, and at 120Hz display refresh the
   * render loop outruns a 30fps camera regularly.
   */
  detect(timestampMs: number): PoseLandmarkerResult | null {
    if (!this.landmarker) return null
    if (this.video.readyState < 2) return null
    if (timestampMs <= this.lastTimestamp) return null
    this.lastTimestamp = timestampMs
    return this.landmarker.detectForVideo(this.video, timestampMs)
  }

  stop(): void {
    this.stream?.getTracks().forEach((track) => track.stop())
    this.landmarker?.close()
    this.landmarker = null
    this.stream = null
  }
}

/**
 * Gather BlazePose's 33 landmarks down to the 17 the index speaks, as a flat
 * `[x, y, confidence] * 17` array.
 *
 * MediaPipe's `visibility` is the closest analogue to the per-joint confidence
 * Vision reports, and it is what the weighted re-rank consumes.
 */
export function toKeypoints(result: PoseLandmarkerResult, poseIndex = 0): Keypoints | null {
  const landmarks = result.landmarks?.[poseIndex]
  if (!landmarks || landmarks.length < 29) return null

  const kp = new Float32Array(NUM_JOINTS * 3)
  for (let j = 0; j < NUM_JOINTS; j++) {
    const lm = landmarks[BLAZEPOSE_TO_COCO17[j]]
    kp[j * 3] = lm.x
    kp[j * 3 + 1] = lm.y
    kp[j * 3 + 2] = lm.visibility ?? 1
  }
  return kp
}
