"""The expensive half: run a pose detector over every image under a root.

Detection is separated from index building on purpose. Detection is the part
that costs real time and you want to do it exactly once; building an index from
the results takes seconds and you will want to redo it every time you change a
filter threshold.

Results stream to an append-only JSONL as they land, so an interrupted run —
closed lid, Ctrl-C, a disconnected volume — resumes from where it stopped.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .backends import make_backend
from .discovery import excluded_dirs, walk_images
from .paths import dataset_name_for
from .util import Progress, note

DETECTIONS_FILE = "detections.jsonl"
DETECT_META_FILE = "detect-meta.json"


@dataclass
class DetectConfig:
    root: str
    out_dir: str
    backend: str = "mediapipe"
    workers: int = 0            # 0 = pick from core count and delegate
    max_side: int = 1024
    max_poses: int = 3
    mp_model: str | None = None
    model_size: str = "lite"
    #: auto | gpu | cpu — which TFLite delegate the mediapipe backend asks for.
    delegate: str = "auto"
    limit: int = 0              # 0 = no limit, otherwise stop after N new images
    restart: bool = False
    #: Folder names or glob patterns to skip while walking.
    exclude: tuple[str, ...] = ()


def _default_workers(backend: str, delegate: str) -> int:
    """Threads to run detection on.

    Both backends release the GIL inside the native call, so these are
    genuinely concurrent and the Python side is not the bottleneck. How far it
    scales depends on what is doing the arithmetic:

      * A GPU delegate is one shared unit, like the Neural Engine. Extra
        threads do not multiply it; they only overlap decoding with inference,
        and past a handful they add contention and VRAM for nothing. Decode is
        roughly half the total cost here, so a few threads is still clearly
        better than one.
      * The CPU/XNNPACK delegate is the opposite: it scales with cores, and
        already runs its own internal thread pool. Oversubscribing hurts, so
        leaving a couple of cores free is the safe default.

    Measured numbers for the ANE path are in `vision_ane.py`; it plateaued at 4
    threads and stayed flat to 10. Nothing equivalent is measured for GPU
    delegates because it depends entirely on the card — run `bodypose bench` on
    the machine you actually care about and pass `-j` if the default is wrong.
    """
    cpu = os.cpu_count() or 8
    if backend == "mediapipe" and delegate == "gpu":
        return max(2, min(4, cpu - 2))
    return max(2, min(8, cpu - 2))


def _already_done(path: str) -> set[str]:
    """Relative paths present in an existing detections file."""
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["p"])
            except (json.JSONDecodeError, KeyError):
                # A torn last line from a hard kill. Everything before it is
                # still good; that one image gets redone.
                continue
    return done


class _Writer:
    """Serialises JSONL appends from the worker threads."""

    def __init__(self, path: str):
        self.fh = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()
        self._since_flush = 0

    def write(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self.lock:
            self.fh.write(line)
            self._since_flush += 1
            if self._since_flush >= 200:
                self.fh.flush()
                self._since_flush = 0

    def close(self) -> None:
        with self.lock:
            self.fh.flush()
            os.fsync(self.fh.fileno())
            self.fh.close()


def run_detect(cfg: DetectConfig) -> dict:
    root = os.path.abspath(cfg.root)
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    detections_path = os.path.join(cfg.out_dir, DETECTIONS_FILE)
    if cfg.restart and os.path.exists(detections_path):
        os.remove(detections_path)

    done = _already_done(detections_path)
    if done:
        note(f"resuming: {len(done):,} images already detected")

    if cfg.exclude:
        skipped = excluded_dirs(root, cfg.exclude)
        if skipped:
            note(f"excluding {len(skipped)} folder(s): {', '.join(skipped)}")
        else:
            # Almost always a typo. Saying so beats silently indexing the very
            # folder the flag was meant to leave out.
            note(f"warning: --exclude {' '.join(cfg.exclude)} matched no folders under {root}")

    # Counted as we go rather than collected silently: listing a directory with
    # tens of thousands of entries takes long enough that a quiet scan is
    # indistinguishable from a hang, and the obvious response to that is Ctrl-C.
    scan = Progress(None, label="scanning")
    all_paths: list[str] = []
    for path in walk_images(root, cfg.exclude):
        all_paths.append(path)
        scan.advance()
    scan.done()
    note(f"found {len(all_paths):,} images")

    pending = [p for p in all_paths if os.path.relpath(p, root) not in done]
    if cfg.limit:
        pending = pending[: cfg.limit]
    if not pending:
        note("nothing to do — every image already has a detection")
        return _write_detect_meta(cfg, root, len(all_paths), cfg.backend, "n/a")

    # Resolve the delegate before sizing the pool: `--delegate auto` may land on
    # either GPU or CPU, and the two want very different thread counts. Building
    # one backend up front also means a broken model path or an unusable GPU
    # driver fails here, on the main thread, with a readable message — rather
    # than N times over inside the pool.
    def new_backend():
        return make_backend(
            cfg.backend,
            max_side=cfg.max_side,
            max_poses=cfg.max_poses,
            mp_model=cfg.mp_model,
            model_size=cfg.model_size,
            delegate=cfg.delegate,
        )

    probe = new_backend()
    device = probe.device
    resolved_delegate = getattr(probe, "delegate", "n/a")
    if getattr(probe, "fallback_from", None):
        note(f"GPU delegate unavailable, using CPU ({probe.fallback_from})")
    note(f"compute device: {device}")

    workers = cfg.workers or _default_workers(cfg.backend, resolved_delegate)
    note(f"detecting {len(pending):,} images with {workers} workers, backend={cfg.backend}")

    local = threading.local()
    backends: list = []
    backends_lock = threading.Lock()
    # The probe is a perfectly good worker backend; hand it to whichever thread
    # asks first instead of paying for the model load twice.
    spare: list = [probe]

    def backend_for_thread():
        backend = getattr(local, "backend", None)
        if backend is None:
            with backends_lock:
                backend = spare.pop() if spare else None
            if backend is None:
                backend = new_backend()
            local.backend = backend
            with backends_lock:
                backends.append(backend)
        return backend

    writer = _Writer(detections_path)
    progress = Progress(len(pending), label="detect")
    counters = {"ok": 0, "empty": 0, "error": 0, "poses": 0}
    counters_lock = threading.Lock()

    def work(path: str) -> None:
        backend = backend_for_thread()
        started = time.monotonic()
        try:
            result = backend.detect(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            result = None
            record = {"p": os.path.relpath(path, root), "e": f"{type(exc).__name__}: {exc}"}
        else:
            record = {
                "p": os.path.relpath(path, root),
                "w": result.width,
                "h": result.height,
                "ms": round((time.monotonic() - started) * 1000, 1),
            }
            if result.error:
                record["e"] = result.error
            else:
                record["ps"] = [{"s": p.score, "k": p.keypoints} for p in result.poses]
        writer.write(record)

        with counters_lock:
            if "e" in record:
                counters["error"] += 1
            elif record.get("ps"):
                counters["ok"] += 1
                counters["poses"] += len(record["ps"])
            else:
                counters["empty"] += 1
            progress.advance(
                suffix=f"figures {counters['ok']:,} · empty {counters['empty']:,} · err {counters['error']:,}"
            )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map() rather than a list of futures: it keeps memory flat across
            # half a million items and propagates Ctrl-C promptly.
            for _ in pool.map(work, pending):
                pass
    except KeyboardInterrupt:
        note("\ninterrupted — progress is saved, rerun the same command to resume")
    finally:
        progress.done()
        writer.close()
        for backend in backends + spare:
            backend.close()

    note(
        f"detected: {counters['ok']:,} images with figures "
        f"({counters['poses']:,} poses), {counters['empty']:,} with none, "
        f"{counters['error']:,} unreadable"
    )
    return _write_detect_meta(cfg, root, len(all_paths), cfg.backend, device)


def _write_detect_meta(cfg: DetectConfig, root: str, scanned: int, backend: str, device: str) -> dict:
    meta = {
        "dataset": dataset_name_for(root),
        "images_root": root,
        "scanned": scanned,
        "backend": backend,
        "compute_device": device,
        "max_side": cfg.max_side,
        "max_poses_per_image": cfg.max_poses,
        "excluded": list(cfg.exclude),
        "detections_file": DETECTIONS_FILE,
    }
    if backend == "mediapipe":
        # Recorded so `build` can stamp it into index.json: an index built with
        # `heavy` and queried with `lite` is a subtly different vector space,
        # and that is worth being able to see after the fact.
        meta["model_size"] = cfg.model_size
    path = os.path.join(cfg.out_dir, DETECT_META_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def iter_detections(out_dir: str):
    """Stream records back out of the JSONL, skipping torn lines."""
    path = os.path.join(out_dir, DETECTIONS_FILE)
    if not os.path.exists(path):
        raise SystemExit(
            f"no detections at {path} — run `bodypose detect <images-root>` first"
        )
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
