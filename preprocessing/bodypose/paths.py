"""Where things live.

The repository is the frame of reference:

    INPUT/<dataset>/     source images, one folder per dataset
    OUTPUT/<dataset>/    the index built from it
    preprocessing/
    live/

So a dataset is named, not pathed. `bodypose run output-images` reads
`INPUT/output-images` and writes `OUTPUT/output-images`, and the live app finds
both without being told where anything is.

Datasets are usually symlinks — the images themselves can sit on an external
volume, or be copied to the internal disk, without any of that leaking into the
index or the commands:

    ln -s /Volumes/SS2_OSX/output-images INPUT/output-images

Because a dataset is identified by name rather than by absolute path, an index
built on one machine still resolves on another. Absolute paths still work
everywhere a dataset name does, for anything that does not fit the layout.
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


def resolve_dataset(name: str) -> tuple[str, str | None]:
    """Turn a dataset name or path into ``(absolute_path, dataset_name)``.

    ``dataset_name`` is the path relative to INPUT/ when the images are inside
    it, and None otherwise — that is the flag for whether the index is portable
    or pinned to this machine's filesystem.
    """
    if os.path.isabs(name):
        absolute = os.path.abspath(name)
    else:
        candidate = os.path.join(input_root(), name)
        # A relative path that exists in the working directory but not under
        # INPUT/ is honoured as-is, so `bodypose detect ./scratch` still works.
        absolute = os.path.abspath(candidate if os.path.exists(candidate) else name)

    return absolute, dataset_name_for(absolute)


def dataset_name_for(absolute: str) -> str | None:
    """The name a path would have as a dataset, or None if it is outside INPUT."""
    root = input_root()
    try:
        # realpath both sides: INPUT/<dataset> is typically a symlink to another
        # volume, and a textual comparison would call every dataset external.
        real_root = os.path.realpath(root)
        real_path = os.path.realpath(absolute)
    except OSError:
        return None

    for base, target in ((root, absolute), (real_root, real_path)):
        try:
            relative = os.path.relpath(target, base)
        except ValueError:  # different drives on some platforms
            continue
        if not relative.startswith(os.pardir) and relative != os.curdir:
            return relative
    return None


def resolve_images_root(index: dict) -> str:
    """Where an index's images are now.

    Prefers the dataset name over the absolute path recorded at build time, so
    moving the repository, or rebuilding the symlink to point at a local copy
    instead of the external drive, does not invalidate an index.
    """
    dataset = index.get("dataset")
    if dataset:
        candidate = os.path.join(input_root(), dataset)
        if os.path.isdir(candidate):
            return candidate
    return index.get("images_root") or ""


def default_out_dir(dataset: str | None, absolute: str) -> str:
    """Where an index for this dataset belongs."""
    name = dataset or os.path.basename(os.path.normpath(absolute))
    return os.path.join(output_root(), name)


def list_datasets() -> list[str]:
    """Immediate subfolders of INPUT/, symlinks included."""
    root = input_root()
    if not os.path.isdir(root):
        return []
    with os.scandir(root) as entries:
        return sorted(
            e.name for e in entries if e.is_dir() and not e.name.startswith(".")
        )


def list_indexes() -> list[str]:
    """Datasets under OUTPUT/ that have a built index."""
    root = output_root()
    if not os.path.isdir(root):
        return []
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "index.json" in filenames:
            found.append(os.path.relpath(dirpath, root))
            dirnames[:] = []  # an index directory has no nested indexes
    return sorted(found)
