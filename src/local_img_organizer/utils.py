"""Uncategorized project utilities"""

import logging


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
