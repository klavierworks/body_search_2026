"""Check that the two detectors agree before you trust a cross-detector match.

The whole design rests on one assumption: a vector produced offline by Apple
Vision lands in the same place as a vector produced in the browser by MediaPipe
for the same body configuration. That assumption is testable — run both over
the same sample and measure.

Two things this catches:

  * A systematic left/right convention mismatch. Both frameworks document
    anatomical (subject's own) left and right, but a swap here would be
    invisible in any single-detector test and would quietly ruin every match.
    So we also score the mirrored comparison; if mirroring *improves*
    agreement, the conventions disagree.
  * Ordinary drift — different joint definitions for ears and eyes, different
    behaviour on occluded limbs — which sets your expectation for how tight a
    cosine threshold can usefully be.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .backends import available_backends, make_backend
from .discovery import walk_images
from .encoding import (
    COCO17,
    MIRROR_PERM,
    NUM_JOINTS,
    EncodingParams,
    cosine,
    encode,
    mirror_keypoints,
)
from .util import Progress, note


@dataclass
class DoctorConfig:
    root: str
    sample: int = 60
    seed: int = 0
    max_side: int = 1024
    mp_model: str | None = None
    model_size: str = "lite"
    delegate: str = "auto"
    params: EncodingParams = EncodingParams()


def run_doctor(cfg: DoctorConfig) -> dict:
    if "vision" not in available_backends():
        raise SystemExit(
            "doctor compares Apple Vision against MediaPipe, and Vision needs macOS.\n"
            "  There is nothing to compare here: with --backend mediapipe (the\n"
            "  default) the index and the browser already run the same detector,\n"
            "  which is the agreement this command exists to verify.\n"
            "  To check throughput instead, run: bodypose bench"
        )

    note(f"sampling up to {cfg.sample} images from {cfg.root} ...")
    all_paths = list(walk_images(cfg.root))
    if not all_paths:
        raise SystemExit(f"no images under {cfg.root}")
    rng = random.Random(cfg.seed)
    sample = rng.sample(all_paths, min(cfg.sample, len(all_paths)))

    vision = make_backend("vision", max_side=cfg.max_side, max_poses=1)
    note(f"vision compute device: {vision.device}")
    mediapipe = make_backend(
        "mediapipe", max_side=cfg.max_side, max_poses=1, mp_model=cfg.mp_model,
        model_size=cfg.model_size, delegate=cfg.delegate,
    )
    note(f"mediapipe compute device: {mediapipe.device}")

    cos_direct: list[float] = []
    cos_mirrored: list[float] = []
    joint_error = [0.0] * NUM_JOINTS
    joint_count = [0] * NUM_JOINTS
    both_found = 0
    only_vision = 0
    only_mediapipe = 0
    neither = 0

    progress = Progress(len(sample), label="doctor")
    for path in sample:
        progress.advance()
        v_result = vision.detect(path)
        m_result = mediapipe.detect(path)
        v_pose = v_result.poses[0] if v_result.poses else None
        m_pose = m_result.poses[0] if m_result.poses else None

        if v_pose is None and m_pose is None:
            neither += 1
            continue
        if v_pose is None:
            only_mediapipe += 1
            continue
        if m_pose is None:
            only_vision += 1
            continue
        both_found += 1

        v_enc = encode(v_pose.keypoints, cfg.params)
        m_enc = encode(m_pose.keypoints, cfg.params)
        if v_enc is None or m_enc is None:
            continue

        cos_direct.append(cosine(v_enc.vector, m_enc.vector))
        m_flipped = encode(mirror_keypoints(m_pose.keypoints), cfg.params)
        if m_flipped is not None:
            cos_mirrored.append(cosine(v_enc.vector, m_flipped.vector))

        for j in range(NUM_JOINTS):
            if v_enc.weights[j] > 0 and m_enc.weights[j] > 0:
                dx = v_enc.vector[j * 2] - m_enc.vector[j * 2]
                dy = v_enc.vector[j * 2 + 1] - m_enc.vector[j * 2 + 1]
                joint_error[j] += (dx * dx + dy * dy) ** 0.5
                joint_count[j] += 1
    progress.done()
    vision.close()
    mediapipe.close()

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    report = {
        "sampled": len(sample),
        "both_found": both_found,
        "only_vision": only_vision,
        "only_mediapipe": only_mediapipe,
        "neither": neither,
        "mean_cosine": mean(cos_direct),
        "mean_cosine_mirrored": mean(cos_mirrored),
        "per_joint_error": {
            COCO17[j]: (joint_error[j] / joint_count[j] if joint_count[j] else None)
            for j in range(NUM_JOINTS)
        },
    }

    note("")
    note("  detection agreement")
    note(f"    both found a figure   {both_found}/{len(sample)}")
    note(f"    vision only           {only_vision}")
    note(f"    mediapipe only        {only_mediapipe}")
    note(f"    neither               {neither}")
    note("")
    note("  vector agreement (cosine, 1.0 = identical)")
    note(f"    as-is                 {report['mean_cosine']:.4f}")
    note(f"    mediapipe mirrored    {report['mean_cosine_mirrored']:.4f}")

    if cos_direct and cos_mirrored and mean(cos_mirrored) > mean(cos_direct) + 0.05:
        note("")
        note("  !! Mirroring MediaPipe improves agreement. The two detectors")
        note("     disagree on left/right convention. Matches will be reflected.")
        note(f"     Joint order to correct with: {list(MIRROR_PERM)}")
    elif cos_direct and mean(cos_direct) < 0.85:
        note("")
        note("  !  Agreement is lower than expected for one shared vector space.")
        note("     Stay on the default `--backend mediapipe` so both sides use")
        note("     the same detector; --backend vision is the faster option on")
        note("     Apple silicon but only worth it if this number is high.")
    else:
        note("")
        note("  Conventions match. Vision-indexed vectors are safe to query")
        note("  with MediaPipe from the browser, so --backend vision is a")
        note("  usable speed-up on this machine.")

    note("")
    note("  worst-agreeing joints (normalised distance):")
    ranked = sorted(
        ((v, k) for k, v in report["per_joint_error"].items() if v is not None), reverse=True
    )
    for value, name in ranked[:5]:
        note(f"    {name:<16} {value:.4f}")

    return report
