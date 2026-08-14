"""The cheap half: turn raw detections into a searchable index.

Filtering happens here, not during detection, so you can re-cut the corpus at
different thresholds in seconds instead of re-reading the drive. This is the
step that decides what `-filtered-` means.

Output is three files the browser fetches directly:

  vectors.bin   float32 little-endian, N x 34, unit L2 norm per row
  weights.bin   float32 little-endian, N x 17, per-joint confidence
  meta.json     columnar: paths, bounding boxes, scores, mirror flags

plus index.json, which records the encoding parameters so the browser can
refuse to run against an index it would misinterpret.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from .detect import DETECT_META_FILE, iter_detections
from .paths import dataset_name_for
from .encoding import (
    DIMS,
    ENCODING_VERSION,
    NUM_JOINTS,
    EncodingParams,
    encode,
    mirror_keypoints,
    torso_visible,
)
from .util import Progress, human_count, note

INDEX_FILE = "index.json"
VECTORS_FILE = "vectors.bin"
WEIGHTS_FILE = "weights.bin"
META_FILE = "meta.json"


@dataclass
class Filters:
    #: How many of the 17 joints must clear ``conf_floor``.
    min_keypoints: int = 8
    #: Minimum detector confidence for the person as a whole.
    min_score: float = 0.3
    #: Reject figures smaller than this fraction of the frame's shorter side.
    #: A crowd scene's background extras produce noisy, near-degenerate poses.
    min_bbox_frac: float = 0.05
    #: Require at least 3 of the 4 shoulder/hip joints. Without a torso the
    #: bounding box is not a body box and the normalisation is meaningless.
    require_torso: bool = True
    #: Keep at most this many figures per image, best-scoring first.
    max_poses_per_image: int = 2


@dataclass
class BuildConfig:
    out_dir: str
    filters: Filters
    params: EncodingParams
    include_mirror: bool = False
    images_root: str | None = None
    thumbs_root: str | None = None


def _load_detect_meta(out_dir: str) -> dict:
    path = os.path.join(out_dir, DETECT_META_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_build(cfg: BuildConfig) -> dict:
    cfg.params.validate()
    detect_meta = _load_detect_meta(cfg.out_dir)
    images_root = cfg.images_root or detect_meta.get("images_root")
    if not images_root:
        raise SystemExit("cannot determine images root — pass --images-root")

    f = cfg.filters
    vectors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    paths: list[str] = []
    path_ids: list[int] = []
    bboxes: list[list[float]] = []
    scores: list[float] = []
    mirrored: list[int] = []

    path_id_of: dict[str, int] = {}
    rejected = {
        "no_detection": 0,
        "unreadable": 0,
        "low_score": 0,
        "few_keypoints": 0,
        "no_torso": 0,
        "too_small": 0,
        "degenerate": 0,
        "extra_poses": 0,
    }
    seen: dict[str, dict] = {}

    note("reading detections ...")
    for record in iter_detections(cfg.out_dir):
        # A resumed run can re-detect an image whose line was torn; last wins.
        seen[record["p"]] = record

    progress = Progress(len(seen), label="build")
    for rel_path, record in seen.items():
        progress.advance()
        if "e" in record:
            rejected["unreadable"] += 1
            continue
        poses = record.get("ps") or []
        if not poses:
            rejected["no_detection"] += 1
            continue

        width = float(record.get("w") or 0)
        height = float(record.get("h") or 0)
        # bbox extents are in normalised image coordinates; convert the
        # threshold into that space so it means the same thing on a tall
        # portrait scan and a wide stereograph.
        aspect = (width / height) if (width > 0 and height > 0) else 1.0

        kept_for_image = 0
        for pose in sorted(poses, key=lambda p: -float(p.get("s", 0.0))):
            if kept_for_image >= f.max_poses_per_image:
                rejected["extra_poses"] += 1
                continue
            score = float(pose.get("s", 0.0))
            if score < f.min_score:
                rejected["low_score"] += 1
                continue
            keypoints = pose["k"]
            if len(keypoints) != NUM_JOINTS:
                rejected["degenerate"] += 1
                continue

            visible = sum(1 for k in keypoints if k[2] >= cfg.params.conf_floor)
            if visible < f.min_keypoints:
                rejected["few_keypoints"] += 1
                continue
            if f.require_torso and not torso_visible(keypoints, cfg.params.conf_floor):
                rejected["no_torso"] += 1
                continue

            encoded = encode(keypoints, cfg.params)
            if encoded is None:
                rejected["degenerate"] += 1
                continue

            x0, y0, x1, y1 = encoded.bbox
            # Normalised coords stretch the short axis; undo that before
            # measuring how much of the frame the figure occupies.
            box_w = (x1 - x0) * (aspect if aspect > 1 else 1.0)
            box_h = (y1 - y0) * (1.0 / aspect if aspect < 1 else 1.0)
            if max(box_w, box_h) < f.min_bbox_frac:
                rejected["too_small"] += 1
                continue

            pid = path_id_of.get(rel_path)
            if pid is None:
                pid = len(paths)
                path_id_of[rel_path] = pid
                paths.append(rel_path)

            vectors.append(np.asarray(encoded.vector, dtype=np.float32))
            weights.append(np.asarray(encoded.weights, dtype=np.float32))
            path_ids.append(pid)
            bboxes.append([round(v, 4) for v in encoded.bbox])
            scores.append(round(score, 4))
            mirrored.append(0)
            kept_for_image += 1

            if cfg.include_mirror:
                flipped = encode(mirror_keypoints(keypoints), cfg.params)
                if flipped is not None:
                    vectors.append(np.asarray(flipped.vector, dtype=np.float32))
                    weights.append(np.asarray(flipped.weights, dtype=np.float32))
                    path_ids.append(pid)
                    bboxes.append([round(v, 4) for v in flipped.bbox])
                    scores.append(round(score, 4))
                    mirrored.append(1)
    progress.done()

    if not vectors:
        raise SystemExit(
            "no poses survived the filters. Loosen --min-keypoints / --min-score "
            "/ --min-bbox-frac, or check that detection actually found figures."
        )

    matrix = np.stack(vectors).astype(np.float32, copy=False)
    weight_matrix = np.stack(weights).astype(np.float32, copy=False)
    assert matrix.shape[1] == DIMS and weight_matrix.shape[1] == NUM_JOINTS

    os.makedirs(cfg.out_dir, exist_ok=True)
    _write_binary(os.path.join(cfg.out_dir, VECTORS_FILE), matrix)
    _write_binary(os.path.join(cfg.out_dir, WEIGHTS_FILE), weight_matrix)

    meta = {
        "count": len(path_ids),
        "paths": paths,
        "path_id": path_ids,
        "bbox": bboxes,
        "score": scores,
    }
    if cfg.include_mirror:
        meta["mirrored"] = mirrored
    with open(os.path.join(cfg.out_dir, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, separators=(",", ":"))

    index = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "encoding": cfg.params.to_json(),
        "detector": {
            "backend": detect_meta.get("backend", "unknown"),
            "compute_device": detect_meta.get("compute_device", "unknown"),
            "max_side": detect_meta.get("max_side"),
        },
        "filters": asdict(f),
        # The dataset name is what makes an index portable: it resolves through
        # INPUT/ wherever the repository is checked out, whereas images_root is
        # only true on the machine that built it.
        "dataset": dataset_name_for(images_root),
        "images_root": images_root,
        "thumbs_root": cfg.thumbs_root,
        "images_scanned": detect_meta.get("scanned"),
        "images_with_figures": len(paths),
        "count": len(path_ids),
        "mirrored_included": bool(cfg.include_mirror),
        "rejected": rejected,
        "files": {"vectors": VECTORS_FILE, "weights": WEIGHTS_FILE, "meta": META_FILE},
        "dtype": "float32-le",
    }
    with open(os.path.join(cfg.out_dir, INDEX_FILE), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    total_seen = len(seen)
    note("")
    note(f"  images scanned      {total_seen:,}")
    note(f"  images with figures {len(paths):,}  ({100.0 * len(paths) / max(1, total_seen):.1f}%)")
    note(f"  index rows          {len(path_ids):,}" + ("  (incl. mirrors)" if cfg.include_mirror else ""))
    note(f"  vectors.bin         {human_count(matrix.nbytes)}B")
    note("  rejected: " + ", ".join(f"{k}={v:,}" for k, v in rejected.items() if v))
    note(f"\nwrote {cfg.out_dir}/{{index.json,vectors.bin,weights.bin,meta.json}}")
    return index


def _write_binary(path: str, array: np.ndarray) -> None:
    # Explicit little-endian: the browser reads these with a plain Float32Array,
    # which is host-endian, and every platform running this is little-endian —
    # but being explicit means a big-endian host fails loudly at build time.
    array.astype("<f4", copy=False).tofile(path)


def load_index(out_dir: str) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Read a built index back in, for stats/queries from Python."""
    with open(os.path.join(out_dir, INDEX_FILE), "r", encoding="utf-8") as fh:
        index = json.load(fh)
    count = index["count"]
    vectors = np.fromfile(os.path.join(out_dir, VECTORS_FILE), dtype="<f4").reshape(count, DIMS)
    weights = np.fromfile(os.path.join(out_dir, WEIGHTS_FILE), dtype="<f4").reshape(count, NUM_JOINTS)
    with open(os.path.join(out_dir, META_FILE), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return index, vectors, weights, meta
