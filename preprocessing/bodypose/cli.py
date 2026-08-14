"""bodypose — build a pose-searchable index from a folder of images."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .encoding import EncodingParams

DEFAULT_OUT = os.environ.get("BODYPOSE_OUT", "out")


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
typical run:

  bodypose run /Volumes/SS2_OSX/output-images --thumbs
  cd ../live && BODY_ARTIFACTS=../preprocessing/out npm run dev

detection is the expensive step and resumes if interrupted; build is seconds
and can be rerun with different filters without touching the drive again.
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- detect --------------------------------------------------------
    d = sub.add_parser("detect", help="run pose detection over a folder tree (the slow step)")
    d.add_argument("root", help="top-level folder to walk, recursively")
    d.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    d.add_argument(
        "--backend", choices=("vision", "mediapipe"), default="vision",
        help="vision = Apple Neural Engine (fast, default); mediapipe = same detector as the browser (slow)",
    )
    d.add_argument("-j", "--workers", type=int, default=0, help="worker threads (default: cores - 2, capped at 10)")
    d.add_argument(
        "--max-side", type=int, default=1024,
        help="decode images down to this longest side before detection (default: 1024)",
    )
    d.add_argument("--max-poses", type=int, default=3, help="figures to detect per image (default: 3)")
    d.add_argument("--limit", type=int, default=0, help="stop after N new images — for a quick trial run")
    d.add_argument("--restart", action="store_true", help="discard existing detections and start over")
    d.add_argument("--mp-model", default=None, help="path to pose_landmarker_*.task (mediapipe backend only)")
    d.add_argument(
        "--exclude", action="append", default=[], metavar="PATTERN",
        help="skip folders whose name or path matches PATTERN; repeatable, globs allowed "
             "(e.g. --exclude google-ls22500 --exclude 'rm-1?')",
    )


    # ---- build ---------------------------------------------------------
    b = sub.add_parser("build", help="turn detections into a searchable index (the fast step)")
    b.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
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
    b.add_argument("--images-root", default=None, help="override the root recorded at detect time")
    _add_encoding_args(b)

    # ---- run -----------------------------------------------------------
    r = sub.add_parser("run", help="detect then build in one go")
    r.add_argument("root", help="top-level folder to walk, recursively")
    r.add_argument("-o", "--out", default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    r.add_argument("--backend", choices=("vision", "mediapipe"), default="vision")
    r.add_argument("-j", "--workers", type=int, default=0)
    r.add_argument("--max-side", type=int, default=1024)
    r.add_argument("--max-poses", type=int, default=3)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--restart", action="store_true")
    r.add_argument("--mp-model", default=None)
    r.add_argument(
        "--exclude", action="append", default=[], metavar="PATTERN",
        help="skip folders whose name or path matches PATTERN; repeatable, globs allowed "
             "(e.g. --exclude google-ls22500 --exclude 'rm-1?')",
    )
    r.add_argument("--min-keypoints", type=int, default=8)
    r.add_argument("--min-score", type=float, default=0.3)
    r.add_argument("--min-bbox-frac", type=float, default=0.05)
    r.add_argument("--no-torso", action="store_true")
    r.add_argument("--max-poses-per-image", type=int, default=2)
    r.add_argument("--include-mirror", action="store_true")
    r.add_argument("--thumbs", action="store_true", help="also generate web-sized thumbnails")
    r.add_argument("--thumb-size", type=int, default=1000)
    _add_encoding_args(r)

    # ---- thumbs --------------------------------------------------------
    t = sub.add_parser("thumbs", help="generate web-sized thumbnails for indexed images")
    t.add_argument("-o", "--out", default=DEFAULT_OUT)
    t.add_argument("--size", type=int, default=1000, help="longest side in pixels (default: 1000)")
    t.add_argument("--quality", type=float, default=0.82, help="JPEG quality 0-1 (default: 0.82)")
    t.add_argument("-j", "--workers", type=int, default=8)

    # ---- doctor --------------------------------------------------------
    doc = sub.add_parser(
        "doctor",
        help="compare Vision against MediaPipe on a sample — checks the two detectors agree",
    )
    doc.add_argument("root", help="folder to sample images from")
    doc.add_argument("-n", "--sample", type=int, default=60, help="images to test (default: 60)")
    doc.add_argument("--seed", type=int, default=0)
    doc.add_argument("--max-side", type=int, default=1024)
    doc.add_argument("--mp-model", default=None)
    _add_encoding_args(doc)

    # ---- stats / query --------------------------------------------------
    s = sub.add_parser("stats", help="summarise a built index")
    s.add_argument("-o", "--out", default=DEFAULT_OUT)

    q = sub.add_parser("query", help="find the nearest indexed poses to a query image")
    q.add_argument("image", help="path to a query image")
    q.add_argument("-o", "--out", default=DEFAULT_OUT)
    q.add_argument("-k", "--top", type=int, default=10)
    q.add_argument("--max-side", type=int, default=1024)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
        run_build(_build_config(args))
        if args.thumbs:
            from .thumbs import run_thumbs

            run_thumbs(args.out, max_side=args.thumb_size)
        return 0

    if args.command == "thumbs":
        from .thumbs import run_thumbs

        run_thumbs(args.out, max_side=args.size, quality=args.quality, workers=args.workers)
        return 0

    if args.command == "doctor":
        from .doctor import DoctorConfig, run_doctor

        run_doctor(
            DoctorConfig(
                root=args.root,
                sample=args.sample,
                seed=args.seed,
                max_side=args.max_side,
                mp_model=args.mp_model,
                params=_params(args),
            )
        )
        return 0

    if args.command == "stats":
        return _stats(args.out)

    if args.command == "query":
        return _query(args)

    return 1


def _detect_config(args):
    from .detect import DetectConfig

    return DetectConfig(
        root=args.root,
        out_dir=args.out,
        backend=args.backend,
        workers=args.workers,
        max_side=args.max_side,
        max_poses=args.max_poses,
        mp_model=args.mp_model,
        limit=args.limit,
        restart=args.restart,
        exclude=tuple(args.exclude),
    )


def _build_config(args):
    from .build import BuildConfig, Filters

    return BuildConfig(
        out_dir=args.out,
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

    index, vectors, weights, meta = load_index(args.out)
    params = EncodingParams(
        origin=index["encoding"]["origin"],
        scale=index["encoding"]["scale"],
        conf_floor=index["encoding"]["conf_floor"],
    )

    backend = make_backend("vision", max_side=args.max_side, max_poses=1)
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
