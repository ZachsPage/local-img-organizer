"""Uncategorized project utilities"""

import logging
from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

# HEIC needs the extra `pillow-heif` plugin, and tif is left out until something actually needs
# it - scanners are the only real source of it
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "gif", "bmp")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger"""
    pkg_name = "local_img_organizer."
    log = logging.getLogger(name.removeprefix(pkg_name))
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
        )
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


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


def defer_exceptions(actions: list[Callable[[], None]]) -> None:
    """Run every action, deferring any exceptions until all have been attempted, then raise
    one combined error (message text only) covering all failures
    """
    errors = []
    for action in actions:
        try:
            action()
        except Exception as ex:  # noqa: BLE001
            errors.append(str(ex))
    if errors:
        raise RuntimeError("\n".join(errors))
