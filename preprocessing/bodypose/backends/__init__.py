"""Backend registry.

MediaPipe is the default and the portable one: it is the same detector the
browser runs, and it works on Linux, macOS and Windows. Apple Vision remains
available on macOS, where it is faster still, but it puts a second detector in
the loop — see `bodypose doctor` for whether that costs you anything.
"""

from __future__ import annotations

import sys

from .base import Backend, DetectedPose, DetectionResult

BACKENDS = ("mediapipe", "vision")
DEFAULT_BACKEND = "mediapipe"


def available_backends() -> tuple[str, ...]:
    """Backends this machine can actually run."""
    if sys.platform == "darwin":
        return BACKENDS
    return ("mediapipe",)


def make_backend(
    name: str,
    *,
    max_side: int,
    max_poses: int,
    mp_model: str | None = None,
    model_size: str = "lite",
    delegate: str = "auto",
) -> Backend:
    if name == "mediapipe":
        from .mediapipe_pose import MediaPipeBackend

        return MediaPipeBackend(
            max_side=max_side,
            max_poses=max_poses,
            model_path=mp_model,
            model_size=model_size,
            delegate=delegate,
        )
    if name == "vision":
        if sys.platform != "darwin":
            raise SystemExit(
                "the vision backend is Apple Vision and needs macOS — this is "
                f"{sys.platform}. Use --backend mediapipe (the default)."
            )
        try:
            from .vision_ane import VisionBackend
        except ImportError as exc:
            raise SystemExit(
                "the vision backend needs the pyobjc bindings:\n"
                "  pip install -e 'preprocessing[vision]'"
            ) from exc

        return VisionBackend(max_side=max_side, max_poses=max_poses)
    raise SystemExit(f"unknown backend {name!r}, expected one of {', '.join(BACKENDS)}")


__all__ = [
    "Backend",
    "DetectedPose",
    "DetectionResult",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "available_backends",
    "make_backend",
]
