"""Measure detection throughput on this machine, before committing to a run.

Whether the GPU delegate beats the CPU one is not knowable in advance. The pose
model is small, so a many-core CPU running XNNPACK across eight threads is a
real competitor to a mid-range GPU, and on a machine with no usable GL driver
the GPU path does not run at all. A 600k-image corpus is several hours either
way, which is long enough that it is worth spending two minutes finding out.

The numbers this prints are end-to-end — decode plus inference, which is what
the actual run costs — over a random sample of INPUT/, so image sizes
are representative rather than a synthetic best case.
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .backends import make_backend
from .discovery import walk_images
from .util import note


@dataclass
class BenchConfig:
    root: str
    sample: int = 120
    seed: int = 0
    max_side: int = 1024
    max_poses: int = 3
    model_size: str = "lite"
    mp_model: str | None = None
    backend: str = "mediapipe"
    #: Thread counts to try. Empty means a sensible spread for this machine.
    threads: tuple[int, ...] = ()
    delegates: tuple[str, ...] = ("gpu", "cpu")


def _time_one(
    cfg: BenchConfig, delegate: str, workers: int, paths: list[str]
) -> tuple[float, int, str]:
    """Return (images per second, images that produced a figure, device label)."""
    local = threading.local()
    made: list = []
    lock = threading.Lock()

    def backend_for_thread():
        backend = getattr(local, "backend", None)
        if backend is None:
            backend = make_backend(
                cfg.backend,
                max_side=cfg.max_side,
                max_poses=cfg.max_poses,
                mp_model=cfg.mp_model,
                model_size=cfg.model_size,
                delegate=delegate,
            )
            local.backend = backend
            with lock:
                made.append(backend)
        return backend

    found = [0]

    def work(path: str) -> None:
        result = backend_for_thread().detect(path)
        if result.poses:
            with lock:
                found[0] += 1

    # Warm up outside the timed region: the first call through a delegate pays
    # for shader compilation or XNNPACK setup, which is a one-off cost the real
    # run amortises over hundreds of thousands of images.
    warm = make_backend(
        cfg.backend, max_side=cfg.max_side, max_poses=cfg.max_poses,
        mp_model=cfg.mp_model, model_size=cfg.model_size, delegate=delegate,
    )
    device = warm.device
    for path in paths[: min(5, len(paths))]:
        warm.detect(path)
    warm.close()

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(work, paths):
            pass
    elapsed = time.monotonic() - started

    for backend in made:
        backend.close()
    return (len(paths) / elapsed if elapsed > 0 else 0.0), found[0], device


def run_bench(cfg: BenchConfig) -> dict:
    note(f"sampling up to {cfg.sample} images from {cfg.root} ...")
    all_paths = list(walk_images(cfg.root))
    if not all_paths:
        raise SystemExit(f"no images under {cfg.root}")
    rng = random.Random(cfg.seed)
    paths = rng.sample(all_paths, min(cfg.sample, len(all_paths)))

    cores = os.cpu_count() or 8
    thread_counts = cfg.threads or tuple(
        sorted({1, 2, 4, min(8, cores), max(2, cores - 2)})
    )

    note(f"{len(paths)} images, model={cfg.model_size}, max-side={cfg.max_side}, {cores} cores")
    note("")
    note(f"  {'delegate':<10} {'threads':>7}  {'img/s':>8}  {'est. 600k':>10}  device")

    results: dict[str, dict[int, float]] = {}
    for delegate in cfg.delegates:
        for workers in thread_counts:
            try:
                rate, _found, device = _time_one(cfg, delegate, workers, paths)
            except SystemExit as exc:
                note(f"  {delegate:<10} {'—':>7}  unavailable: {str(exc).splitlines()[0][:60]}")
                break
            except Exception as exc:  # noqa: BLE001 - a dead delegate is a result
                note(f"  {delegate:<10} {'—':>7}  failed: {type(exc).__name__}: {exc}")
                break
            hours = 600_000 / rate / 3600 if rate > 0 else float("inf")
            note(f"  {delegate:<10} {workers:>7}  {rate:>8.1f}  {hours:>9.1f}h  {device}")
            results.setdefault(delegate, {})[workers] = rate

    best = None
    for delegate, by_threads in results.items():
        for workers, rate in by_threads.items():
            if best is None or rate > best[2]:
                best = (delegate, workers, rate)

    if best:
        note("")
        note(f"  fastest: --delegate {best[0]} -j {best[1]}  ({best[2]:.1f} img/s)")
        note("")
        note("  Detection figures include decoding, which is roughly half the cost")
        note("  and scales with threads regardless of delegate. If GPU and CPU land")
        note("  close together, prefer CPU — it leaves the machine usable and has")
        note("  no driver surprises over a multi-hour run.")
    return {"results": results, "best": best}
