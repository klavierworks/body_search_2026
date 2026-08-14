"""MediaPipe BlazePose — the detector both stages of this project run.

The browser runs MediaPipe, so running it here too means there is no
cross-detector question to answer: an offline vector and a live vector come out
of the same model with the same joint definitions and the same left/right
convention. That is the whole point of the `universal` layout, and it is also
what makes the pipeline run anywhere rather than only on Apple silicon.

BlazePose emits 33 landmarks. The 17 COCO joints the index speaks are a subset,
so the mapping is a straight index gather — see `BLAZEPOSE_TO_COCO17` here and
the identical table in `live/src/encoding.ts`.

Compute
-------
TFLite runs the model through a *delegate*: a plug-in that decides which
hardware executes the graph. Three matter here:

    XNNPACK   the optimised CPU path, always available
    GPU       OpenGL ES compute shaders on Linux, Metal on macOS
    (none)    the reference CPU kernels, much slower

`delegate="auto"` asks for GPU, proves it by running one frame through it, and
falls back to CPU if either step fails — a machine with no usable GL driver
gets a working slow run and a line saying so, rather than a crash in a worker
thread an hour in.

Whether GPU actually beats CPU here is hardware-dependent and worth measuring
rather than assuming: `bodypose bench` times both on the machine in
front of you. The model is small, so on a many-core CPU with a modest GPU,
XNNPACK across several threads can win.
"""

from __future__ import annotations

from .base import Backend, DetectedPose, DetectionResult
from .gpuprobe import gpu_available
from ..encoding import NUM_JOINTS
from ..imaging import open_downscaled
from ..models import DEFAULT_MODEL_SIZE, resolve_model

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

DELEGATES = ("auto", "gpu", "cpu")

#: Kept in step with `live/src/pose.ts`. The live side runs at 0.5; indexing
#: uses a lower bar because `bodypose build` re-filters on score afterwards and
#: a figure discarded during detection cannot be recovered without a rerun.
MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3


def _import_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:  # pragma: no cover - install-time failure
        raise SystemExit(
            "the mediapipe package is missing:\n"
            "  pip install -e preprocessing\n"
            "MediaPipe does not publish a wheel for every Python version; "
            "3.11 to 3.13 are known to work."
        ) from exc
    return mp, mp_python, mp_vision


class MediaPipeBackend(Backend):
    name = "mediapipe"

    def __init__(
        self,
        max_side: int = 1024,
        max_poses: int = 3,
        model_path: str | None = None,
        model_size: str = DEFAULT_MODEL_SIZE,
        delegate: str = "auto",
    ):
        mp, mp_python, mp_vision = _import_mediapipe()
        self._mp = mp
        self.max_side = max_side
        self.model = resolve_model(model_size, model_path)
        self.model_size = model_size

        if delegate not in DELEGATES:
            raise SystemExit(f"unknown delegate {delegate!r}, expected one of {', '.join(DELEGATES)}")

        self.fallback_from = None
        choice = delegate
        if delegate == "auto":
            works, why = gpu_available(self.model)
            choice = "gpu" if works else "cpu"
            if not works:
                self.fallback_from = why
        elif delegate == "gpu":
            works, why = gpu_available(self.model)
            if not works:
                raise SystemExit(
                    f"--delegate gpu was asked for but the GPU delegate does not work here:\n"
                    f"  {why}\n"
                    "  Use --delegate cpu, or auto to pick whichever runs."
                )

        try:
            self.landmarker = self._build(mp_python, mp_vision, choice, max_poses)
        except Exception as exc:  # noqa: BLE001 - nothing left to fall back to
            raise SystemExit(
                f"could not create a MediaPipe pose landmarker on {choice}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self.delegate = choice
        self.device = "GPU (TFLite delegate)" if choice == "gpu" else "CPU (TFLite/XNNPACK)"
        # The GPU path takes frames as 4-channel RGBA; handing it 3-channel RGB
        # aborts the process inside MediaPipe's Objective-C buffer conversion
        # rather than raising. The CPU path takes RGB directly, and converting
        # for it would only add a copy per image.
        self._mode = "RGBA" if choice == "gpu" else "RGB"
        self._format = (
            self._mp.ImageFormat.SRGBA if choice == "gpu" else self._mp.ImageFormat.SRGB
        )

    def _build(self, mp_python, mp_vision, choice: str, max_poses: int):
        base = mp_python.BaseOptions(
            model_asset_path=self.model,
            delegate=(
                mp_python.BaseOptions.Delegate.GPU
                if choice == "gpu"
                else mp_python.BaseOptions.Delegate.CPU
            ),
        )
        return mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=max_poses,
                min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            )
        )

    def detect(self, path: str) -> DetectionResult:
        try:
            image, width, height = open_downscaled(path, self.max_side, mode=self._mode)
        except Exception as exc:  # noqa: BLE001 - a corrupt file is a data fact
            return DetectionResult(error=f"decode failed: {type(exc).__name__}: {exc}")

        import numpy as np

        array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        try:
            result = self.landmarker.detect(
                self._mp.Image(image_format=self._format, data=array)
            )
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
