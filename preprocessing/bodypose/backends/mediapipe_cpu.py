"""MediaPipe BlazePose backend — the exact-parity option.

The browser runs MediaPipe. Running it here too removes any question about two
detectors disagreeing on where a shoulder is. It is a good deal slower than
Vision (no ANE path from Python; this is TFLite on CPU cores), so it exists for
`doctor` comparisons and for anyone who would rather trade wall-clock for a
guaranteed-consistent vector space.

BlazePose emits 33 landmarks. The 17 COCO joints are a subset, so the mapping
is a straight index gather.
"""

from __future__ import annotations

import os

from .base import Backend, DetectedPose, DetectionResult
from ..encoding import NUM_JOINTS

# BlazePose landmark index for each COCO17 joint, in order.
BLAZEPOSE_TO_COCO17 = (
    0,   # nose
    2,   # left_eye        (BlazePose has inner/centre/outer; centre is 2)
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
)
assert len(BLAZEPOSE_TO_COCO17) == NUM_JOINTS

DEFAULT_MODEL_ENV = "BODYPOSE_MP_MODEL"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _resolve_model(path: str | None) -> str:
    candidate = path or os.environ.get(DEFAULT_MODEL_ENV)
    if not candidate:
        raise RuntimeError(
            "No MediaPipe model. Pass --mp-model /path/to/pose_landmarker_lite.task "
            f"or set {DEFAULT_MODEL_ENV}. Download it from:\n  {MODEL_URL}\n"
            "(the same file live/scripts/setup.mjs fetches for the browser)"
        )
    if not os.path.isfile(candidate):
        raise RuntimeError(f"MediaPipe model not found at {candidate}")
    return candidate


class MediaPipeBackend(Backend):
    name = "mediapipe"

    def __init__(self, max_side: int = 1024, max_poses: int = 3, model_path: str | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The mediapipe backend needs the `mediapipe` package:\n"
                "  pip install mediapipe\n"
                "It has no wheel for every Python version — 3.11 or 3.12 is the "
                "safe choice on Apple silicon."
            ) from exc

        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The mediapipe backend needs numpy and Pillow.") from exc

        self._mp = mp
        self._np = np
        self._Image = Image
        self.max_side = max_side
        self.device = "CPU (TFLite/XNNPACK)"

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_resolve_model(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=max_poses,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, path: str) -> DetectionResult:
        try:
            with self._Image.open(path) as im:
                im.load()
                width, height = im.size
                rgb = im.convert("RGB")
                longest = max(rgb.size)
                if longest > self.max_side:
                    ratio = self.max_side / longest
                    rgb = rgb.resize(
                        (max(1, round(rgb.width * ratio)), max(1, round(rgb.height * ratio))),
                        self._Image.Resampling.BILINEAR,
                    )
                array = self._np.asarray(rgb, dtype=self._np.uint8)
        except Exception as exc:  # noqa: BLE001 - a corrupt file is a data fact
            return DetectionResult(error=f"decode failed: {exc}")

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=array)
        try:
            result = self.landmarker.detect(image)
        except Exception as exc:  # noqa: BLE001
            return DetectionResult(width=width, height=height, error=f"mediapipe: {exc}")

        poses: list[DetectedPose] = []
        for landmarks in result.pose_landmarks or []:
            keypoints = []
            for idx in BLAZEPOSE_TO_COCO17:
                lm = landmarks[idx]
                visibility = getattr(lm, "visibility", None)
                conf = 1.0 if visibility is None else float(visibility)
                keypoints.append([round(float(lm.x), 5), round(float(lm.y), 5), round(conf, 5)])
            mean_conf = sum(k[2] for k in keypoints) / NUM_JOINTS
            poses.append(DetectedPose(keypoints=keypoints, score=round(mean_conf, 5)))

        return DetectionResult(width=width, height=height, poses=poses)

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:  # noqa: BLE001
            pass
