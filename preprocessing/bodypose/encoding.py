"""The pose vector encoding. This module is the specification.

Everything downstream — the on-disk index, the browser matcher in
``live/src/encoding.ts`` — has to agree with what happens here, so keep the two
implementations in lockstep. ``EncodingParams`` is written into ``index.json``
and read back by the browser, so a mismatch is loud rather than silent.

The pipeline, per detected person:

  1. Take 17 COCO keypoints as ``(x, y, confidence)`` in image-normalised,
     top-left-origin coordinates. Every backend converts into this convention.
  2. Mark a joint visible when ``confidence >= conf_floor``.
  3. Bounding box over the visible joints only.
  4. Translate (bbox centre or corner) and scale (uniformly by the longer bbox
     side, or independently per axis) so the vector encodes limb configuration
     and nothing about where the figure sat in frame.
  5. Invisible joints collapse to the origin, which is neutral once weighted.
  6. Flatten to 34 floats and L2-normalise.

Cosine similarity between two of these is the recall metric; the per-joint
confidences ride along as weights for the re-rank stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

ENCODING_VERSION = "coco17-v1"

#: Canonical joint order. Both backends map into this; the browser mirrors it.
COCO17 = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

NUM_JOINTS = len(COCO17)
DIMS = NUM_JOINTS * 2

JOINT_INDEX = {name: i for i, name in enumerate(COCO17)}

#: Index permutation that swaps left and right. Used for mirroring a pose.
MIRROR_PERM = tuple(
    JOINT_INDEX[
        name.replace("left_", "right_") if name.startswith("left_")
        else name.replace("right_", "left_") if name.startswith("right_")
        else name
    ]
    for name in COCO17
)

TORSO = tuple(JOINT_INDEX[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))

# A degenerate bounding box (a figure seen edge-on, or a detection that
# collapsed) would otherwise divide by ~zero and produce a vector of garbage
# that happens to have unit norm.
MIN_EXTENT = 1e-4


@dataclass(frozen=True)
class EncodingParams:
    """Knobs that must match between the indexer and the browser."""

    version: str = ENCODING_VERSION
    #: "center" puts the bbox centre at the origin; "corner" uses the top-left,
    #: which is what Move Mirror did.
    origin: str = "center"
    #: "uniform" divides both axes by the longer bbox side, preserving aspect
    #: ratio. "independent" divides each axis by its own extent, the Move Mirror
    #: behaviour — it makes every pose fill a unit square, which throws away the
    #: difference between a wide stance and a narrow one.
    scale: str = "uniform"
    #: Below this confidence a joint is treated as absent.
    conf_floor: float = 0.1

    def validate(self) -> None:
        if self.origin not in ("center", "corner"):
            raise ValueError(f"origin must be 'center' or 'corner', got {self.origin!r}")
        if self.scale not in ("uniform", "independent"):
            raise ValueError(f"scale must be 'uniform' or 'independent', got {self.scale!r}")
        if not 0.0 <= self.conf_floor < 1.0:
            raise ValueError(f"conf_floor must be in [0, 1), got {self.conf_floor}")

    def to_json(self) -> dict:
        d = asdict(self)
        d["keypoints"] = list(COCO17)
        d["dims"] = DIMS
        d["y_axis"] = "top-left"
        d["min_extent"] = MIN_EXTENT
        return d


@dataclass
class EncodedPose:
    vector: list[float]        # 34 floats, unit L2 norm
    weights: list[float]       # 17 confidences, clamped to [0, 1]
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in image coords
    visible: int               # number of joints above conf_floor


def encode(
    keypoints: Sequence[Sequence[float]],
    params: EncodingParams = EncodingParams(),
) -> EncodedPose | None:
    """Turn 17 ``(x, y, confidence)`` triples into a comparable unit vector.

    Returns ``None`` when the pose is too degenerate to encode — no visible
    joints, or a bounding box with no extent in either axis.
    """
    if len(keypoints) != NUM_JOINTS:
        raise ValueError(f"expected {NUM_JOINTS} keypoints, got {len(keypoints)}")

    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    for x, y, c in keypoints:
        xs.append(float(x))
        ys.append(float(y))
        cs.append(min(1.0, max(0.0, float(c))))

    vis = [c >= params.conf_floor for c in cs]
    n_vis = sum(vis)
    if n_vis == 0:
        return None

    vx = [xs[i] for i in range(NUM_JOINTS) if vis[i]]
    vy = [ys[i] for i in range(NUM_JOINTS) if vis[i]]
    x0, x1 = min(vx), max(vx)
    y0, y1 = min(vy), max(vy)
    w = x1 - x0
    h = y1 - y0
    if w < MIN_EXTENT and h < MIN_EXTENT:
        return None

    if params.origin == "center":
        ox, oy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    else:
        ox, oy = x0, y0

    if params.scale == "uniform":
        sx = sy = max(w, h, MIN_EXTENT)
    else:
        sx, sy = max(w, MIN_EXTENT), max(h, MIN_EXTENT)

    vec = [0.0] * DIMS
    for i in range(NUM_JOINTS):
        if vis[i]:
            vec[i * 2] = (xs[i] - ox) / sx
            vec[i * 2 + 1] = (ys[i] - oy) / sy
        # else: leave at the origin, neutral once the weight is applied

    norm = math.sqrt(sum(v * v for v in vec))
    if norm < MIN_EXTENT:
        return None
    inv = 1.0 / norm
    vec = [v * inv for v in vec]

    weights = [cs[i] if vis[i] else 0.0 for i in range(NUM_JOINTS)]
    return EncodedPose(vector=vec, weights=weights, bbox=(x0, y0, x1, y1), visible=n_vis)


def mirror_keypoints(keypoints: Sequence[Sequence[float]]) -> list[list[float]]:
    """Reflect a pose about the vertical axis of the image.

    Flips x within the unit image box and swaps the left/right joint identities,
    so a figure facing one way becomes the same figure facing the other.
    """
    return [
        [1.0 - float(keypoints[src][0]), float(keypoints[src][1]), float(keypoints[src][2])]
        for src in MIRROR_PERM
    ]


def torso_visible(keypoints: Sequence[Sequence[float]], conf_floor: float, need: int = 3) -> bool:
    """Whether enough of the shoulder/hip quad survived to trust the bbox."""
    return sum(1 for i in TORSO if keypoints[i][2] >= conf_floor) >= need


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def weighted_distance(
    va: Sequence[float], wa: Sequence[float],
    vb: Sequence[float], wb: Sequence[float],
) -> float:
    """Confidence-weighted L2 between two encoded poses.

    Each joint contributes in proportion to the product of both detectors'
    confidence in it, so an ankle one side was guessing at cannot drag the
    match around. Normalised by total weight, so it stays comparable across
    poses with different numbers of visible joints.
    """
    total = 0.0
    acc = 0.0
    for j in range(NUM_JOINTS):
        omega = wa[j] * wb[j]
        if omega <= 0.0:
            continue
        dx = va[j * 2] - vb[j * 2]
        dy = va[j * 2 + 1] - vb[j * 2 + 1]
        acc += omega * (dx * dx + dy * dy)
        total += omega
    if total <= 0.0:
        return float("inf")
    return math.sqrt(acc / total)
