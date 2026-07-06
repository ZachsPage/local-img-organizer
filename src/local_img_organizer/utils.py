"""Uncategorized project utilities"""

import logging
from collections.abc import Callable
from importlib import import_module
from typing import Any


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
