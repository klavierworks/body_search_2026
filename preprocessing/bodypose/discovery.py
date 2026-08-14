"""Walking an HFS+ scratch disk without tripping over its housekeeping."""

from __future__ import annotations

import os
from fnmatch import fnmatch
from typing import Iterator, Sequence

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp", ".webp", ".heic", ".heif"}
)

# Directories macOS scatters across an external volume. None hold source images.
SKIP_DIRS = frozenset(
    {
        ".Spotlight-V100",
        ".fseventsd",
        ".Trashes",
        ".TemporaryItems",
        ".DocumentRevisions-V100",
        ".DS_Store",
        "__MACOSX",
        ".git",
        "node_modules",
    }
)


def is_image(name: str) -> bool:
    # `._foo.jpg` files are AppleDouble resource forks — a few hundred bytes of
    # metadata carrying the extension of the file they shadow. On this drive
    # they are also where the download URLs in catalogue-filtered-ls78 live.
    if name.startswith("._") or name.startswith("."):
        return False
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def matches_any(name: str, rel_path: str, patterns: Sequence[str]) -> bool:
    """Whether a directory matches an exclusion pattern.

    Patterns are matched against both the folder's own name and its path
    relative to the scan root, so ``google-ls22500`` and ``rm-*`` and
    ``sub/dir`` all work.
    """
    for pattern in patterns:
        if fnmatch(name, pattern) or fnmatch(rel_path, pattern):
            return True
    return False


def walk_images(
    root: str,
    exclude: Sequence[str] = (),
    follow_symlinks: bool = True,
) -> Iterator[str]:
    """Yield absolute paths to images beneath ``root``, in a stable order.

    ``exclude`` skips whole directories before descending into them, so an
    excluded folder costs nothing — nothing under it is read or even stat'd.

    Symlinks are followed by default because that is the normal way images get
    into INPUT/: the corpus is usually one or more links out to another volume,
    and not following them would find nothing at all. Following links means
    cycles are possible, so directories are tracked by real path and visited
    once — a link back to an ancestor ends that branch instead of the walk.
    """
    root = os.path.abspath(root)
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        if follow_symlinks:
            try:
                real = os.path.realpath(dirpath)
            except OSError:
                real = dirpath
            if real in seen:
                dirnames[:] = []
                continue
            seen.add(real)

        keep = []
        for d in sorted(dirnames):
            if d in SKIP_DIRS or d.startswith("._"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, d), root)
            if exclude and matches_any(d, rel, exclude):
                continue
            keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            if is_image(name):
                yield os.path.join(dirpath, name)


def excluded_dirs(root: str, exclude: Sequence[str], depth: int = 2) -> list[str]:
    """Folders near the top of ``root`` that an exclusion list will skip.

    Reported before the scan so a typo shows up as "matched nothing" rather
    than silently indexing the folder you meant to leave out.

    Deliberately shallow: walking the whole tree to enumerate exclusions would
    cost exactly the scan the flag exists to avoid. Two levels is enough for
    INPUT/, where the first level is usually a symlink to a volume and the
    folders worth excluding sit just inside it.
    """
    if not exclude:
        return []
    root = os.path.abspath(root)
    hits: list[str] = []

    def scan(directory: str, prefix: str, remaining: int) -> None:
        if remaining <= 0:
            return
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda e: e.name)
        except OSError:
            return
        for entry in children:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            rel = os.path.join(prefix, entry.name) if prefix else entry.name
            if matches_any(entry.name, rel, exclude):
                hits.append(rel)
                continue  # no need to look inside something already excluded
            scan(entry.path, rel, remaining - 1)

    scan(root, "", depth)
    return hits


def count_images(root: str, exclude: Sequence[str] = ()) -> int:
    return sum(1 for _ in walk_images(root, exclude))
