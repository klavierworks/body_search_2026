"""Finding — and if necessary fetching — the MediaPipe pose model.

Both stages of this project run the same detector, so they should run the same
*file*. The browser's copy is put under `live/public/mediapipe/` by
`npm run setup`; if it is there, preprocessing uses that one rather than
downloading a second copy. Failing that we keep one in `.models/` at the
repository root.

Three sizes exist. They emit the same 33 BlazePose landmarks and therefore the
same 17-joint vector, so the choice is accuracy against speed, not a change of
vector space:

    lite    ~5 MB    fastest, what the live app defaults to
    full    ~9 MB    noticeably better on partial and small figures
    heavy   ~29 MB   best, several times slower

Indexing with one size and querying with another does work, but the small
systematic differences between them widen the gap between an offline vector and
a live one for no benefit. Match them unless you have measured otherwise.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Iterator

from .paths import repo_root
from .util import human_count, note

MODEL_SIZES = ("lite", "full", "heavy")
DEFAULT_MODEL_SIZE = "lite"

MODEL_ENV = "BODYPOSE_MP_MODEL"
MODELS_DIRNAME = ".models"

#: Google's published location for each size, float16 variants.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{size}/float16/latest/pose_landmarker_{size}.task"
)


def model_filename(size: str) -> str:
    return f"pose_landmarker_{size}.task"


def models_dir() -> str:
    return os.path.join(repo_root(), MODELS_DIRNAME)


def live_model_dir() -> str:
    """Where `live/scripts/setup.mjs` puts the browser's copy."""
    return os.path.join(repo_root(), "live", "public", "mediapipe")


def _candidates(size: str) -> Iterator[str]:
    name = model_filename(size)
    yield os.path.join(live_model_dir(), name)
    yield os.path.join(models_dir(), name)


def resolve_model(size: str = DEFAULT_MODEL_SIZE, explicit: str | None = None,
                  download: bool = True) -> str:
    """Absolute path to a `.task` model, downloading it once if needed.

    Order: an explicit ``--mp-model`` path, then ``$BODYPOSE_MP_MODEL``, then
    the browser's copy under `live/public/mediapipe/`, then `.models/`.
    """
    if size not in MODEL_SIZES:
        raise SystemExit(f"unknown model size {size!r}, expected one of {', '.join(MODEL_SIZES)}")

    named = explicit or os.environ.get(MODEL_ENV)
    if named:
        if not os.path.isfile(named):
            raise SystemExit(f"MediaPipe model not found at {named}")
        return os.path.abspath(named)

    for candidate in _candidates(size):
        if os.path.isfile(candidate):
            return candidate

    target = os.path.join(models_dir(), model_filename(size))
    if not download:
        raise SystemExit(
            f"no {size} model on disk. Expected one of:\n"
            + "\n".join(f"  {c}" for c in _candidates(size))
        )
    return fetch_model(size, target)


def fetch_model(size: str, target: str) -> str:
    """Download one model to ``target``. Writes via a temp file so an
    interrupted download cannot leave a truncated .task that loads and then
    fails somewhere much less obvious."""
    url = MODEL_URL.format(size=size)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    note(f"downloading {size} pose model from {url}")
    temporary = target + ".partial"
    try:
        with urllib.request.urlopen(url) as response, open(temporary, "wb") as fh:
            fetched = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                fetched += len(chunk)
        os.replace(temporary, target)
    except Exception as exc:  # noqa: BLE001 - network failure needs a clear exit
        if os.path.exists(temporary):
            os.remove(temporary)
        raise SystemExit(
            f"could not download the {size} model: {exc}\n"
            f"  fetch it manually to {target}\n  from {url}"
        ) from exc
    note(f"model saved to {target} ({human_count(fetched)}B)")
    return target
