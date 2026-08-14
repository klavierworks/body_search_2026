"""Generate web-sized thumbnails for the images that made it into the index.

Only images with a surviving pose get one, so this is tens of thousands of
files rather than half a million. Encoding goes through ImageIO the same way
decoding does — no Pillow, and the JPEG path is hardware-assisted.

Worth doing before a live session: the raw Rijksmuseum scans are large enough
that fetching one over a dev server visibly stalls the match display.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import Quartz
from Foundation import NSURL

from .build import META_FILE, INDEX_FILE
from .util import Progress, note

THUMBS_DIR = "thumbs"


def _thumb_options(max_side: int) -> dict:
    return {
        Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
        Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
        Quartz.kCGImageSourceThumbnailMaxPixelSize: int(max_side),
        Quartz.kCGImageSourceShouldCacheImmediately: True,
    }


def _make_thumb(src: str, dst: str, options: dict, quality: float) -> bool:
    import objc

    pool = getattr(objc, "autorelease_pool", None)
    ctx = pool() if pool is not None else None
    try:
        if ctx is not None:
            ctx.__enter__()
        source = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(src), None)
        if source is None:
            return False
        image = Quartz.CGImageSourceCreateThumbnailAtIndex(source, 0, options)
        if image is None:
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        dest = Quartz.CGImageDestinationCreateWithURL(
            NSURL.fileURLWithPath_(dst), "public.jpeg", 1, None
        )
        if dest is None:
            return False
        Quartz.CGImageDestinationAddImage(
            dest, image, {Quartz.kCGImageDestinationLossyCompressionQuality: quality}
        )
        return bool(Quartz.CGImageDestinationFinalize(dest))
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)


def run_thumbs(out_dir: str, max_side: int = 1000, quality: float = 0.82, workers: int = 8) -> str:
    index_path = os.path.join(out_dir, INDEX_FILE)
    meta_path = os.path.join(out_dir, META_FILE)
    if not os.path.exists(meta_path):
        raise SystemExit(f"no index at {out_dir} — run `bodypose build` first")

    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    images_root = index["images_root"]
    thumbs_root = os.path.join(out_dir, THUMBS_DIR)
    paths = meta["paths"]
    note(f"thumbnailing {len(paths):,} images to {thumbs_root} at {max_side}px")

    options = _thumb_options(max_side)
    progress = Progress(len(paths), label="thumbs")
    counters = {"made": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    def work(rel: str) -> None:
        dst = os.path.join(thumbs_root, rel + ".jpg")
        try:
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                outcome = "skipped"
            else:
                ok = _make_thumb(os.path.join(images_root, rel), dst, options, quality)
                outcome = "made" if ok else "failed"
        except Exception:  # noqa: BLE001 - a bad file is not a run-ending event
            outcome = "failed"
        with lock:
            counters[outcome] += 1
            progress.advance(suffix=f"made {counters['made']:,} · failed {counters['failed']:,}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(work, paths):
            pass
    progress.done()

    index["thumbs_root"] = thumbs_root
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    note(
        f"thumbs: {counters['made']:,} written, {counters['skipped']:,} already present, "
        f"{counters['failed']:,} failed"
    )
    note(f"index.json now points the live app at {thumbs_root}")
    return thumbs_root
