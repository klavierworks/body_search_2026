"""Find out whether the GPU delegate works here, survivably.

MediaPipe is C++ underneath, and when it does not like something it calls
`CHECK`, which aborts the process. An abort is not a Python exception: no
`except` catches it, no `finally` runs, the interpreter is simply gone. A
half-working GL driver can therefore take down an eight-hour indexing run at
image 400,000 with no way to fall back in-process.

So the probe runs in a subprocess. If the GPU delegate aborts, a child dies and
the parent reads a non-zero exit code and quietly uses the CPU instead. It
costs one interpreter start and one model load, once per run.

Run directly to see the raw outcome:

    python -m bodypose.backends.gpuprobe /path/to/pose_landmarker_lite.task
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Cached per process — the answer cannot change while we run, and the probe
#: costs an interpreter start.
_RESULT: tuple[bool, str] | None = None

PROBE_TIMEOUT_SECONDS = 180


def _probe_body(model: str) -> None:
    """The part that runs in the child. Raises or aborts if the GPU is unusable."""
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=model, delegate=mp_python.BaseOptions.Delegate.GPU
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    # Four channels, matching what the real backend sends down the GPU path.
    # A 3-channel frame is exactly what aborts on the Metal backend, so probing
    # with one would answer a question nobody asked.
    frame = np.zeros((256, 256, 4), dtype=np.uint8)
    frame[..., 3] = 255
    landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame))
    landmarker.close()


def gpu_available(model: str) -> tuple[bool, str]:
    """``(works, explanation)`` for the GPU delegate with this model."""
    global _RESULT
    if _RESULT is not None:
        return _RESULT

    env = dict(os.environ, BODYPOSE_GPUPROBE_MODEL=model)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "bodypose.backends.gpuprobe"],
            env=env,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _RESULT = (False, f"GPU probe timed out after {PROBE_TIMEOUT_SECONDS}s")
        return _RESULT
    except Exception as exc:  # noqa: BLE001 - probing must never be fatal
        _RESULT = (False, f"GPU probe could not run: {type(exc).__name__}: {exc}")
        return _RESULT

    if completed.returncode == 0:
        _RESULT = (True, "GPU delegate probe passed")
        return _RESULT

    detail = _last_meaningful_line(completed.stderr) or f"exit code {completed.returncode}"
    _RESULT = (False, detail)
    return _RESULT


def _last_meaningful_line(stderr: str) -> str:
    """The most useful line of a MediaPipe failure.

    Its stack traces are hundreds of `@ 0x...` frames wrapping one line that
    says what actually went wrong, and that is the line worth printing.
    """
    interesting = [
        line.strip()
        for line in (stderr or "").splitlines()
        if line.strip()
        and not line.lstrip().startswith("@")
        and not line.startswith(("I0", "W0"))
    ]
    # GL and EGL failures rarely use the word "error" — they say things like
    # "Unable to initialize EGL" or "cannot open shared object file" — so match
    # the vocabulary these actually fail in.
    markers = (
        "check failed", "error", "fail", "unable", "cannot", "not supported",
        "no such", "undefined symbol", "egl", "gl_context",
    )
    for line in reversed(interesting):
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            return line[:300]
    return interesting[-1][:300] if interesting else ""


def main() -> int:
    model = os.environ.get("BODYPOSE_GPUPROBE_MODEL") or (
        sys.argv[1] if len(sys.argv) > 1 else ""
    )
    if not model:
        print("usage: python -m bodypose.backends.gpuprobe <model.task>", file=sys.stderr)
        return 2
    try:
        _probe_body(model)
    except Exception as exc:  # noqa: BLE001 - the exit code is the answer
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
