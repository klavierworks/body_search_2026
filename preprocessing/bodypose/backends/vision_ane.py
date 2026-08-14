"""Apple Vision body-pose detection, pinned to the Neural Engine.

``VNDetectHumanBodyPoseRequest`` is the fast path on Apple silicon: the model
ships with the OS, and on macOS 14+ Vision will hand us the list of compute
devices it can run each stage on, so we can explicitly ask for the ANE instead
of hoping the scheduler picks it.

Vision reports 19 joints (COCO's 17 plus a neck and a root); we keep the 17 the
browser side can also produce. Its coordinates are normalised with the origin
at the *bottom* left, so y gets flipped on the way out.

Decoding is the other half of the speed story, and close to half the cost:
detection runs at ~134 img/s per thread while a single-threaded ImageIO decode
of the same images runs at ~142 img/s. Museum scans reach tens of megapixels
and Vision only needs a modest input, so we ask ImageIO for a downscaled
thumbnail straight from the file — it never materialises the full-resolution
bitmap.
"""

from __future__ import annotations

import threading

import Quartz
import Vision
from Foundation import NSURL

from .base import Backend, DetectedPose, DetectionResult
from ..encoding import COCO17, NUM_JOINTS

# Vision's joint-name constants, in COCO17 order.
_VISION_JOINT_NAMES = (
    "VNHumanBodyPoseObservationJointNameNose",
    "VNHumanBodyPoseObservationJointNameLeftEye",
    "VNHumanBodyPoseObservationJointNameRightEye",
    "VNHumanBodyPoseObservationJointNameLeftEar",
    "VNHumanBodyPoseObservationJointNameRightEar",
    "VNHumanBodyPoseObservationJointNameLeftShoulder",
    "VNHumanBodyPoseObservationJointNameRightShoulder",
    "VNHumanBodyPoseObservationJointNameLeftElbow",
    "VNHumanBodyPoseObservationJointNameRightElbow",
    "VNHumanBodyPoseObservationJointNameLeftWrist",
    "VNHumanBodyPoseObservationJointNameRightWrist",
    "VNHumanBodyPoseObservationJointNameLeftHip",
    "VNHumanBodyPoseObservationJointNameRightHip",
    "VNHumanBodyPoseObservationJointNameLeftKnee",
    "VNHumanBodyPoseObservationJointNameRightKnee",
    "VNHumanBodyPoseObservationJointNameLeftAnkle",
    "VNHumanBodyPoseObservationJointNameRightAnkle",
)
assert len(_VISION_JOINT_NAMES) == NUM_JOINTS == len(COCO17)


def _joint_keys() -> tuple[str, ...]:
    """Resolve the constants to the strings Vision actually keys its dict with."""
    keys = []
    for const in _VISION_JOINT_NAMES:
        value = getattr(Vision, const, None)
        if value is None:
            raise RuntimeError(
                f"Vision is missing {const}. This needs macOS 11 or newer with "
                "pyobjc-framework-Vision installed."
            )
        keys.append(str(value))
    return tuple(keys)


_JOINT_KEYS: tuple[str, ...] | None = None
_JOINT_KEYS_LOCK = threading.Lock()


def _joint_keys_cached() -> tuple[str, ...]:
    global _JOINT_KEYS
    with _JOINT_KEYS_LOCK:
        if _JOINT_KEYS is None:
            _JOINT_KEYS = _joint_keys()
        return _JOINT_KEYS


def _autorelease_pool():
    """Per-image drain pool; without one, CGImages pile up until the run ends."""
    import objc

    return objc.autorelease_pool()


#: Vision hands back Core ML device objects, so the accelerator is identified by
#: class rather than by any VNComputeDeviceType constant — those are not in the
#: pyobjc bindings. Measured on an M4 Pro over 60 Rijksmuseum scans at 1024px:
#:
#:   pinned to MLNeuralEngineComputeDevice   7.46 ms/img   134 img/s
#:   pinned to MLGPUComputeDevice            7.46 ms/img   134 img/s
#:   left unpinned (Vision schedules)        8.67 ms/img   115 img/s
#:   pinned to MLCPUComputeDevice           53.58 ms/img    19 img/s
#:
#: So the ANE is 7.2x the CPU path and pinning beats letting Vision choose. GPU
#: ties with the ANE on this model, but the ANE is the better pick anyway: it
#: leaves the GPU free and draws far less power over a run of this length.
_ANE_DEVICE_CLASS = "MLNeuralEngineComputeDevice"


def _pin_to_neural_engine(request) -> str:
    """Ask Vision to run this request's stages on the Neural Engine.

    Returns a label describing what we actually got, which `detect` logs — so a
    machine where the ANE is unavailable says so rather than quietly running
    seven times slower. The API is macOS 14+; older systems get Vision's own
    scheduling, which still reaches the ANE, we just cannot confirm it.
    """
    query = getattr(request, "supportedComputeStageDevicesAndReturnError_", None)
    if query is None:
        return "auto (macOS < 14: no compute-device API)"

    try:
        stage_devices, err = query(None)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return f"auto (device query failed: {exc})"
    if err is not None or not stage_devices:
        return "auto (device query returned nothing)"

    pinned: list[str] = []
    unavailable: list[str] = []
    for stage, devices in stage_devices.items():
        stage_name = str(stage).replace("VNComputeStage", "")
        device = next((d for d in devices if d.__class__.__name__ == _ANE_DEVICE_CLASS), None)
        if device is None:
            unavailable.append(stage_name)
            continue
        try:
            request.setComputeDevice_forComputeStage_(device, stage)
            pinned.append(stage_name)
        except Exception:  # noqa: BLE001 - fall back to Vision's own scheduling
            unavailable.append(stage_name)

    if not pinned:
        return "auto (no Neural Engine device offered)"
    label = "NeuralEngine[" + ",".join(sorted(pinned)) + "]"
    if unavailable:
        label += " + auto[" + ",".join(sorted(unavailable)) + "]"
    return label


class VisionBackend(Backend):
    name = "vision"

    def __init__(self, max_side: int = 1024, max_poses: int = 3):
        self.max_side = max_side
        self.max_poses = max_poses
        self.keys = _joint_keys_cached()
        self.request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        self.device = _pin_to_neural_engine(self.request)
        self._thumb_options = {
            Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
            Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
            Quartz.kCGImageSourceThumbnailMaxPixelSize: int(max_side),
            Quartz.kCGImageSourceShouldCacheImmediately: True,
        }

    def detect(self, path: str) -> DetectionResult:
        with _autorelease_pool():
            return self._detect(path)

    def _detect(self, path: str) -> DetectionResult:
        url = NSURL.fileURLWithPath_(path)
        source = Quartz.CGImageSourceCreateWithURL(url, None)
        if source is None:
            return DetectionResult(error="unreadable: not a decodable image")

        props = Quartz.CGImageSourceCopyPropertiesAtIndex(source, 0, None) or {}
        width = int(props.get(Quartz.kCGImagePropertyPixelWidth, 0) or 0)
        height = int(props.get(Quartz.kCGImagePropertyPixelHeight, 0) or 0)

        image = Quartz.CGImageSourceCreateThumbnailAtIndex(source, 0, self._thumb_options)
        if image is None:
            return DetectionResult(width=width, height=height, error="decode failed")

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
        ok, err = handler.performRequests_error_([self.request], None)
        if not ok:
            return DetectionResult(width=width, height=height, error=f"vision: {err}")

        observations = self.request.results() or []
        poses: list[DetectedPose] = []
        for obs in observations[: self.max_poses]:
            pose = self._read_observation(obs)
            if pose is not None:
                poses.append(pose)

        return DetectionResult(width=width, height=height, poses=poses)

    def _read_observation(self, obs) -> DetectedPose | None:
        group = Vision.VNHumanBodyPoseObservationJointsGroupNameAll
        points, err = obs.recognizedPointsForJointsGroupName_error_(group, None)
        if err is not None or not points:
            return None

        keypoints: list[list[float]] = []
        for key in self.keys:
            point = points.get(key)
            if point is None:
                keypoints.append([0.0, 0.0, 0.0])
                continue
            location = point.location()
            # Vision's origin is bottom-left; everything downstream is top-left.
            keypoints.append(
                [round(float(location.x), 5), round(1.0 - float(location.y), 5),
                 round(float(point.confidence()), 5)]
            )
        return DetectedPose(keypoints=keypoints, score=round(float(obs.confidence()), 5))
