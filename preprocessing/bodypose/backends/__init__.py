"""Backend registry."""

from __future__ import annotations

from .base import Backend, DetectedPose, DetectionResult

BACKENDS = ("vision", "mediapipe")


def make_backend(name: str, *, max_side: int, max_poses: int, mp_model: str | None = None) -> Backend:
    if name == "vision":
        from .vision_ane import VisionBackend

        return VisionBackend(max_side=max_side, max_poses=max_poses)
    if name == "mediapipe":
        from .mediapipe_cpu import MediaPipeBackend

        return MediaPipeBackend(max_side=max_side, max_poses=max_poses, model_path=mp_model)
    raise ValueError(f"unknown backend {name!r}, expected one of {BACKENDS}")


__all__ = ["Backend", "DetectedPose", "DetectionResult", "BACKENDS", "make_backend"]
