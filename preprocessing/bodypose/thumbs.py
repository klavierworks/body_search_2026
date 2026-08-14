"""Generate web-sized thumbnails for the images that made it into the index.

Only images with a surviving pose get one, so this is tens of thousands of
files rather than half a million.

Worth doing before a live session: the raw Rijksmuseum scans are large enough
that fetching one over a dev server visibly stalls the match display.

Decoding goes through the same `draft()` fast path the detector uses, so a
40-megapixel scan is never fully materialised just to write a 1000px JPEG.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .build import META_FILE, INDEX_FILE
from .imaging import open_downscaled
from .paths import resolve_images_root
from .util import Progress, note

THUMBS_DIR = "thumbs"


def _make_thumb(src: str, dst: str, max_side: int, quality: int) -> bool:
    image, _, _ = open_downscaled(src, max_side)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Write to a temp name first: a killed run must not leave a half-written
    # JPEG behind, because the skip-if-exists check would then accept it
    # forever.
    temporary = dst + ".partial"
    image.save(temporary, "JPEG", quality=quality, optimize=True, progressive=True)
    os.replace(temporary, dst)
    return True


def run_thumbs(
    out_dir: str,
    max_side: int = 1000,
    quality: float = 0.82,
    workers: int = 8,
) -> str:
    index_path = os.path.join(out_dir, INDEX_FILE)
    meta_path = os.path.join(out_dir, META_FILE)
    if not os.path.exists(meta_path):
        raise SystemExit(f"no index at {out_dir} — run `bodypose build` first")

    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    images_root = resolve_images_root(index)
    thumbs_root = os.path.join(out_dir, THUMBS_DIR)
    paths = meta["paths"]
    # Pillow takes JPEG quality as 1-100; the flag is 0-1 to match the
    # ImageIO-era interface, and both spellings are worth accepting rather than
    # silently writing quality-1 garbage for someone who passed 82.
    q = int(round(quality * 100)) if quality <= 1.0 else int(round(quality))
    q = max(1, min(100, q))
    note(f"thumbnailing {len(paths):,} images to {thumbs_root} at {max_side}px, quality {q}")

    progress = Progress(len(paths), label="thumbs")
    counters = {"made": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    def work(rel: str) -> None:
        dst = os.path.join(thumbs_root, rel + ".jpg")
        try:
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                outcome = "skipped"
            else:
                ok = _make_thumb(os.path.join(images_root, rel), dst, max_side, q)
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
