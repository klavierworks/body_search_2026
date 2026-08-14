"""Where things live.

The repository is the frame of reference, and there are exactly two directories:

    INPUT/     the images — everything under here, however it is arranged
    OUTPUT/    the index built from them

INPUT/ *is* the dataset. Not a folder of datasets: whatever is beneath it, at
any depth, is one corpus, and one index is built from all of it. Subfolders are
just organisation. Nothing needs naming and no command takes a dataset
argument.

The images themselves do not have to be there physically. Symlinks are the
normal case, and the walk follows them:

    ln -s /media/you/SS2_OSX/DATASETS_BACKUP INPUT/datasets-backup
    ln -s /media/you/SS2_OSX/output-images   INPUT/output-images

Both of those are then part of the same corpus. Image paths are recorded
relative to INPUT/, so an index survives the repository moving, or a symlink
being repointed at a local copy of the same images.
"""

from __future__ import annotations

import os

INPUT_DIRNAME = "INPUT"
OUTPUT_DIRNAME = "OUTPUT"


def repo_root() -> str:
    """The directory holding INPUT/ and OUTPUT/.

    Found by walking up from the working directory, then from this file, so the
    CLI behaves the same whether it is run from the repository root, from inside
    `preprocessing/`, or from an editable install elsewhere. `BODY_ROOT`
    overrides the search.
    """
    override = os.environ.get("BODY_ROOT")
    if override:
        return os.path.abspath(override)

    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        current = start
        while True:
            if os.path.isdir(os.path.join(current, INPUT_DIRNAME)):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    # Nothing found: fall back to the repository this file lives in, two levels
    # up from bodypose/. Creating INPUT/ there is the right thing on first run.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def input_root() -> str:
    return os.environ.get("BODY_INPUT") or os.path.join(repo_root(), INPUT_DIRNAME)


def output_root() -> str:
    return os.environ.get("BODY_OUTPUT") or os.path.join(repo_root(), OUTPUT_DIRNAME)


def require_input_root() -> str:
    """INPUT/, with something in it, or an actionable exit."""
    root = input_root()
    if not os.path.isdir(root):
        raise SystemExit(
            f"no INPUT directory at {root}\n"
            "  create it and put the images in, or link them:\n"
            f"    mkdir -p {root}\n"
            f"    ln -s /path/to/images {os.path.join(root, 'images')}"
        )
    if not _has_entries(root):
        raise SystemExit(
            f"INPUT/ is empty ({root})\n"
            "  link some images in, for example:\n"
            f"    ln -s /path/to/images {os.path.join(root, 'images')}"
        )
    return root


def _has_entries(root: str) -> bool:
    """Anything at all besides the README that documents the layout."""
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.startswith(".") or entry.name == "README.md":
                continue
            return True
    return False


def resolve_images_root(index: dict) -> str:
    """Where an index's images are now.

    Always INPUT/ — paths in the index are relative to it, which is what makes
    an index portable between machines and survive a repointed symlink. The
    absolute path recorded at build time is only a fallback for an index whose
    INPUT/ has since gone away.
    """
    root = input_root()
    if os.path.isdir(root):
        return root
    return index.get("images_root") or ""


def list_input_entries() -> list[str]:
    """Top-level names under INPUT/, for reporting what is in the corpus."""
    root = input_root()
    if not os.path.isdir(root):
        return []
    with os.scandir(root) as entries:
        return sorted(
            e.name for e in entries
            if not e.name.startswith(".") and e.name != "README.md"
        )


def has_index(out_dir: str | None = None) -> bool:
    return os.path.isfile(os.path.join(out_dir or output_root(), "index.json"))
