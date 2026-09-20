"""Uncategorized project utilities"""

import logging
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

# HEIC needs the extra `pillow-heif` plugin, and tif is left out until something actually needs
# it - scanners are the only real source of it
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "gif", "bmp")


def get_logger(name: str) -> logging.Logger:
    """Return a logger named without the package prefix - level & format are set in `main.py`"""
    return logging.getLogger(name.removeprefix("local_img_organizer."))


def find_images(folder: Path, extensions: Iterable[str] = IMG_EXTENSIONS) -> list[Path]:
    """Return the image files directly in `folder`, sorted, matching `extensions` case-insensitively

    :param folder: Dir to look in
    :param extensions: Extensions to keep, with or without a leading "." - ex. "jpg" / ".JPG"
    """
    wanted = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in wanted)


def import_cls(module: str, name: str, *, kind: str) -> Any:  # noqa: ANN401
    """Import `module` and return its `name.capitalize()` class; raise ValueError if missing"""
    try:
        mod = import_module(module)
        return getattr(mod, name.capitalize())
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Unknown {kind}: {name!r}") from None


def move_file(src: Path, dest: Path) -> None:
    """Move `src` to `dest`, verifying the filesystem still matches what was planned

    `Path.rename` replaces an existing `dest` silently, so the checks here are what keep a stale
    plan from destroying a file - every operation that moves bytes around goes through this.
    """
    if not src.is_file():
        raise ValueError(f"{src} is not a file")
    if dest.exists():
        raise ValueError(f"Dest {dest} already exists?")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)
    if not dest.is_file():
        raise ValueError(f"{src} was not moved to {dest}?")
