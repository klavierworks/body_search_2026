"""Backend contract.

A backend turns a file path into zero or more poses, each of which is 17
``(x, y, confidence)`` triples in image-normalised, top-left-origin
coordinates — whatever joint topology the underlying model uses natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectedPose:
    #: 17 ``[x, y, confidence]`` triples in COCO order (see encoding.COCO17).
    keypoints: list[list[float]]
    #: Detector's overall confidence in this person, if it reports one.
    score: float = 1.0


@dataclass
class DetectionResult:
    width: int = 0
    height: int = 0
    poses: list[DetectedPose] = field(default_factory=list)
    error: str | None = None


class Backend:
    """One instance per worker thread — none of these are thread-safe."""

    name: str = "base"
    #: Filled in once the backend knows what hardware it landed on.
    device: str = "unknown"

    def detect(self, path: str) -> DetectionResult:
        raise NotImplementedError

    def close(self) -> None:
        pass
