"""bodypose — build a pose-searchable index from a folder of images."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import DEFAULT_BACKEND, available_backends
from .encoding import EncodingParams
from .models import DEFAULT_MODEL_SIZE, MODEL_SIZES
from .paths import (
    has_index,
    input_root,
    list_input_entries,
    output_root,
    require_input_root,
)


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    """Flags that decide what runs the model and on what hardware."""
    g = p.add_argument_group("detector")
    g.add_argument(
        "--backend", choices=available_backends(), default=DEFAULT_BACKEND,
        help="mediapipe = the same detector the browser runs, works everywhere "
             "(default); vision = Apple Vision on the Neural Engine, macOS only "
             "and faster, but puts a second detector in the loop",
    )
    g.add_argument(
        "--delegate", choices=("auto", "gpu", "cpu"), default="auto",
        help="which hardware TFLite runs the model on: gpu (OpenGL/Metal), cpu "
             "(XNNPACK), or auto to try the GPU and fall back (default: auto). "
             "Run `bodypose bench` to see which is faster here",
    )
    g.add_argument(
        "--model-size", choices=MODEL_SIZES, default=DEFAULT_MODEL_SIZE,
        help="BlazePose model: lite is fastest and is what the live app loads, "
             "heavy is most accurate and several times slower (default: lite). "
             "Downloaded on first use",
    )
    g.add_argument(
        "--mp-model", default=None,
        help="explicit path to a pose_landmarker_*.task, instead of the "
             "downloaded or live/public/mediapipe copy",
    )


def _add_encoding_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("encoding (must match the live app; recorded in index.json)")
    g.add_argument(
        "--origin", choices=("center", "corner"), default="center",
        help="where the bounding box maps to the origin (default: center)",
    )
    g.add_argument(
        "--scale", choices=("uniform", "independent"), default="uniform",
        help="uniform preserves aspect ratio; independent reproduces Move Mirror (default: uniform)",
    )
    g.add_argument(
        "--conf-floor", type=float, default=0.1,
        help="below this confidence a joint counts as absent (default: 0.1)",
    )


def _add_exclude_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exclude", action="append", default=[], metavar="PATTERN",
        help="skip folders whose name matches PATTERN while walking INPUT/; "
             "repeatable, globs allowed (e.g. --exclude google-ls22500)",
    )


def _params(args) -> EncodingParams:
    params = EncodingParams(origin=args.origin, scale=args.scale, conf_floor=args.conf_floor)
    params.validate()
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bodypose",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
layout — two directories, both relative to the repository:

  INPUT/     the images. Everything under it, at any depth, is the corpus
  OUTPUT/    the index built from them

INPUT/ is the dataset. Subfolders are just organisation, symlinks are followed,
and no command takes a dataset name.

typical run:

  ln -s /path/to/images INPUT/images
  bodypose bench                 # which delegate is faster here?
  bodypose run --thumbs
  cd live && npm run dev

detection is the expensive step and resumes if interrupted; build is seconds
and can be rerun with different filters without re-reading the images.

Detection runs MediaPipe BlazePose by default — the same model the browser
runs — so the offline and live vectors come out of one detector, on Linux,
macOS or Windows alike.
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- detect --------------------------------------------------------
    d = sub.add_parser("detect", help="run pose detection over INPUT/ (the slow step)")
    d.add_argument("-o", "--out", default=None, help="override the output directory")
    d.add_argument("-j", "--workers", type=int, default=0, help="worker threads (default: from cores and delegate)")
    d.add_argument(
        "--max-side", type=int, default=1024,
        help="decode images down to this longest side before detection (default: 1024)",
    )
    d.add_argument("--max-poses", type=int, default=3, help="figures to detect per image (default: 3)")
    d.add_argument("--limit", type=int, default=0, help="stop after N new images — for a quick trial run")
    d.add_argument("--restart", action="store_true", help="discard existing detections and start over")
    _add_detector_args(d)
    _add_exclude_arg(d)

    # ---- build ---------------------------------------------------------
    b = sub.add_parser("build", help="turn detections into a searchable index (the fast step)")
    b.add_argument("-o", "--out", default=None, help="override the output directory")
    b.add_argument("--min-keypoints", type=int, default=8, help="joints that must be visible (default: 8)")
    b.add_argument("--min-score", type=float, default=0.3, help="minimum detector confidence (default: 0.3)")
    b.add_argument(
        "--min-bbox-frac", type=float, default=0.05,
        help="reject figures smaller than this fraction of the frame (default: 0.05)",
    )
    b.add_argument("--no-torso", action="store_true", help="keep figures without a visible shoulder/hip quad")
    b.add_argument("--max-poses-per-image", type=int, default=2, help="index at most N figures per image (default: 2)")
    b.add_argument(
        "--include-mirror", action="store_true",
        help="also index the left-right reflection of every pose — doubles recall, doubles index size",
    )
    b.add_argument("--images-root", default=None, help="override where the images are read from")
    _add_encoding_args(b)

    # ---- run -----------------------------------------------------------
    r = sub.add_parser("run", help="detect then build in one go")
    r.add_argument("-o", "--out", default=None, help="override the output directory")
    r.add_argument("-j", "--workers", type=int, default=0)
    r.add_argument("--max-side", type=int, default=1024)
    r.add_argument("--max-poses", type=int, default=3)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--restart", action="store_true")
    _add_detector_args(r)
    _add_exclude_arg(r)
    r.add_argument("--min-keypoints", type=int, default=8)
    r.add_argument("--min-score", type=float, default=0.3)
    r.add_argument("--min-bbox-frac", type=float, default=0.05)
    r.add_argument("--no-torso", action="store_true")
    r.add_argument("--max-poses-per-image", type=int, default=2)
    r.add_argument("--include-mirror", action="store_true")
    r.add_argument("--images-root", default=None)
    r.add_argument("--thumbs", action="store_true", help="also generate web-sized thumbnails")
    r.add_argument("--thumb-size", type=int, default=1000)
    _add_encoding_args(r)

    # ---- thumbs --------------------------------------------------------
    t = sub.add_parser("thumbs", help="generate web-sized thumbnails for indexed images")
    t.add_argument("-o", "--out", default=None)
    t.add_argument("--size", type=int, default=1000, help="longest side in pixels (default: 1000)")
    t.add_argument("--quality", type=float, default=0.82, help="JPEG quality 0-1 (default: 0.82)")
    t.add_argument("-j", "--workers", type=int, default=8)

    # ---- doctor --------------------------------------------------------
    doc = sub.add_parser(
        "doctor",
        help="compare Vision against MediaPipe on a sample — checks the two detectors agree",
    )
    doc.add_argument("-n", "--sample", type=int, default=60, help="images to test (default: 60)")
    doc.add_argument("--seed", type=int, default=0)
    doc.add_argument("--max-side", type=int, default=1024)
    doc.add_argument("--delegate", choices=("auto", "gpu", "cpu"), default="auto")
    doc.add_argument("--model-size", choices=MODEL_SIZES, default=DEFAULT_MODEL_SIZE)
    doc.add_argument("--mp-model", default=None)
    _add_encoding_args(doc)

    # ---- bench ---------------------------------------------------------
    bn = sub.add_parser(
        "bench",
        help="time GPU against CPU on this machine before starting a long run",
    )
    bn.add_argument("-n", "--sample", type=int, default=120, help="images to time (default: 120)")
    bn.add_argument("--seed", type=int, default=0)
    bn.add_argument("--max-side", type=int, default=1024)
    bn.add_argument("--max-poses", type=int, default=3)
    bn.add_argument("--model-size", choices=MODEL_SIZES, default=DEFAULT_MODEL_SIZE)
    bn.add_argument("--mp-model", default=None)
    bn.add_argument(
        "-j", "--threads", type=int, action="append", default=[],
        help="thread count to try; repeatable (default: a spread up to your core count)",
    )
    bn.add_argument(
        "--delegate", action="append", choices=("gpu", "cpu"), default=[],
        help="delegate to try; repeatable (default: both)",
    )

    # ---- status / stats / query -----------------------------------------
    sub.add_parser("status", help="what is in INPUT/, and whether OUTPUT/ has an index")

    s = sub.add_parser("stats", help="summarise a built index")
    s.add_argument("-o", "--out", default=None)

    q = sub.add_parser("query", help="find the nearest indexed poses to a query image")
    q.add_argument("image", help="path to a query image")
    q.add_argument("-o", "--out", default=None)
    q.add_argument("-k", "--top", type=int, default=10)
    q.add_argument("--max-side", type=int, default=1024)

    return parser


def _output_dir(args, *, must_exist: bool) -> str:
    """Where the index lives: OUTPUT/, or wherever ``--out`` says."""
    out_dir = os.path.abspath(args.out) if getattr(args, "out", None) else output_root()
    if must_exist and not has_index(out_dir):
        raise SystemExit(
            f"no index in {out_dir}\n"
            "  run `bodypose run` first"
        )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "status":
        return _status()

    if args.command == "detect":
        from .detect import run_detect

        run_detect(_detect_config(args))
        return 0

    if args.command == "build":
        from .build import run_build

        run_build(_build_config(args))
        return 0

    if args.command == "run":
        from .detect import run_detect
        from .build import run_build

        run_detect(_detect_config(args))
        run_build(_build_config(args, must_exist=False))
        if args.thumbs:
            from .thumbs import run_thumbs

            run_thumbs(_output_dir(args, must_exist=False), max_side=args.thumb_size)
        return 0

    if args.command == "thumbs":
        from .thumbs import run_thumbs

        run_thumbs(
            _output_dir(args, must_exist=True),
            max_side=args.size,
            quality=args.quality,
            workers=args.workers,
        )
        return 0

    if args.command == "doctor":
        from .doctor import DoctorConfig, run_doctor

        run_doctor(
            DoctorConfig(
                root=require_input_root(),
                sample=args.sample,
                seed=args.seed,
                max_side=args.max_side,
                mp_model=args.mp_model,
                model_size=args.model_size,
                delegate=args.delegate,
                params=_params(args),
            )
        )
        return 0

    if args.command == "bench":
        from .bench import BenchConfig, run_bench

        run_bench(
            BenchConfig(
                root=require_input_root(),
                sample=args.sample,
                seed=args.seed,
                max_side=args.max_side,
                max_poses=args.max_poses,
                model_size=args.model_size,
                mp_model=args.mp_model,
                threads=tuple(args.threads),
                delegates=tuple(args.delegate) or ("gpu", "cpu"),
            )
        )
        return 0

    if args.command == "stats":
        return _stats(_output_dir(args, must_exist=True))

    if args.command == "query":
        return _query(args)

    return 1


def _detect_config(args):
    from .detect import DetectConfig

    return DetectConfig(
        root=require_input_root(),
        out_dir=_output_dir(args, must_exist=False),
        backend=args.backend,
        workers=args.workers,
        max_side=args.max_side,
        max_poses=args.max_poses,
        mp_model=args.mp_model,
        model_size=args.model_size,
        delegate=args.delegate,
        limit=args.limit,
        restart=args.restart,
        exclude=tuple(args.exclude),
    )


def _build_config(args, must_exist: bool = True):
    from .build import BuildConfig, Filters

    return BuildConfig(
        out_dir=_output_dir(args, must_exist=must_exist and args.command != "run"),
        filters=Filters(
            min_keypoints=args.min_keypoints,
            min_score=args.min_score,
            min_bbox_frac=args.min_bbox_frac,
            require_torso=not args.no_torso,
            max_poses_per_image=args.max_poses_per_image,
        ),
        params=_params(args),
        include_mirror=args.include_mirror,
        images_root=getattr(args, "images_root", None),
    )


def _status() -> int:
    entries = list_input_entries()

    print(f"INPUT   {input_root()}")
    if not entries:
        print("  (empty — put the images in, or link them:")
        print(f"   ln -s /path/to/images {os.path.join(input_root(), 'images')})")
    for name in entries:
        target = os.path.join(input_root(), name)
        where = os.path.realpath(target)
        link = f"  -> {where}" if where != os.path.abspath(target) else ""
        kind = "dir" if os.path.isdir(target) else "file"
        print(f"  {name:<28} {kind}{link}")

    out_dir = output_root()
    print(f"\nOUTPUT  {out_dir}")
    if not has_index(out_dir):
        print("  (nothing built yet — run `bodypose run`)")
        return 0
    try:
        with open(os.path.join(out_dir, "index.json"), "r", encoding="utf-8") as fh:
            index = json.load(fh)
        print(f"  {index['count']:,} poses from {index['images_with_figures']:,} images")
        print(f"  built {index.get('built_at', '?')} with {index['detector'].get('backend')}")
    except (OSError, KeyError, json.JSONDecodeError):
        print("  (unreadable index.json)")
    return 0


def _stats(out_dir: str) -> int:
    from .build import load_index

    index, vectors, weights, meta = load_index(out_dir)
    print(json.dumps(
        {
            "count": index["count"],
            "unique_images": len(meta["paths"]),
            "images_scanned": index.get("images_scanned"),
            "encoding": index["encoding"]["version"],
            "origin": index["encoding"]["origin"],
            "scale": index["encoding"]["scale"],
            "detector": index["detector"],
            "filters": index["filters"],
            "mirrored_included": index["mirrored_included"],
            "rejected": index.get("rejected", {}),
            "vectors_bytes": int(vectors.nbytes),
            "mean_visible_joints": float((weights > 0).sum(axis=1).mean()),
        },
        indent=2,
    ))
    return 0


def _query(args) -> int:
    import numpy as np

    from .backends import make_backend
    from .build import load_index
    from .encoding import EncodingParams, encode

    index, vectors, weights, meta = load_index(_output_dir(args, must_exist=True))
    params = EncodingParams(
        origin=index["encoding"]["origin"],
        scale=index["encoding"]["scale"],
        conf_floor=index["encoding"]["conf_floor"],
    )

    # Query with whatever built the index. Detecting the query image with a
    # different model than the corpus would measure the gap between two
    # detectors as much as the gap between two poses.
    detector = index.get("detector") or {}
    backend = make_backend(
        detector.get("backend") or DEFAULT_BACKEND,
        max_side=args.max_side,
        max_poses=1,
        model_size=detector.get("model_size") or DEFAULT_MODEL_SIZE,
    )
    result = backend.detect(args.image)
    backend.close()
    if result.error:
        print(f"could not read query image: {result.error}", file=sys.stderr)
        return 1
    if not result.poses:
        print("no figure detected in the query image", file=sys.stderr)
        return 1

    encoded = encode(result.poses[0].keypoints, params)
    if encoded is None:
        print("query pose too degenerate to encode", file=sys.stderr)
        return 1

    q = np.asarray(encoded.vector, dtype=np.float32)
    qw = np.asarray(encoded.weights, dtype=np.float32)

    # Cosine recall over everything, then a confidence-weighted re-rank of the
    # shortlist — the same two stages the browser runs.
    similarity = vectors @ q
    depth = min(300, len(similarity))
    shortlist = np.argpartition(-similarity, depth - 1)[:depth]

    diff = (vectors[shortlist].reshape(-1, 17, 2) - q.reshape(17, 2)) ** 2
    omega = weights[shortlist] * qw
    numer = (omega * diff.sum(axis=2)).sum(axis=1)
    denom = np.maximum(omega.sum(axis=1), 1e-6)
    distance = np.sqrt(numer / denom)

    ranking = np.argsort(distance)[: args.top]
    for rank, slot in enumerate(ranking, 1):
        row = int(shortlist[slot])
        rel = meta["paths"][meta["path_id"][row]]
        print(f"{rank:3d}  cos={similarity[row]:.4f}  wdist={distance[slot]:.4f}  {rel}")
    return 0
