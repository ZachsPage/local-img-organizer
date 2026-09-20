"""Rename operation - renames an image after when it was taken, so files sort chronologically"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import override

from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Journal, Operation, OpOut
from local_img_organizer.utils import get_logger

_log = get_logger(__name__)

type _Data = Operation.Data

_MIN_YEAR, _MAX_YEAR = 1900, 2099

_DATE_PATTERNS = (
    (re.compile(r"(?=(\d{4})[-_.]?(\d{2})[-_.]?(\d{2}))"), "%Y%m%d"),
    (re.compile(r"(?<!\d)(\d{2})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)"), "%y%m%d"),
)


def _has_date(name: str) -> bool:
    """Return whether `name` already holds a valid year-first date - ex. `20231015`, `2023-10-15`,
    or a standalone `170624` / `23_07_31`
    """
    for pattern, fmt in _DATE_PATTERNS:
        for match in pattern.finditer(name):
            try:
                found = datetime.strptime("".join(match.groups()), fmt)  # noqa: DTZ007
            except ValueError:
                continue
            if _MIN_YEAR <= found.year <= _MAX_YEAR:
                return True
    return False


@dataclass
class Rename(Operation):
    """Renames an image to `IMG{YYYY}{MM}{DD}{HH}{MM}{SS}{ms}` from its metadata

    Names that already hold a date, or hold no digits at all (a human-readable name), are skipped
    """

    class Cfg(Operation.Cfg):
        """Rename operation configuration"""

    cfg: Cfg

    @override
    def plan(self, data: _Data) -> OpOut:
        path = data.src.path
        stem = path.stem
        if _has_date(stem):
            _log.debug(f"{path.name}: skipping, name already holds a date")
            return {}
        if not any(c.isdigit() for c in stem):
            _log.debug(f"{path.name}: skipping, name has no digits")
            return {}

        if date_taken := data.ext.get("date_taken"):
            source, dt = "date_taken", datetime.fromisoformat(date_taken)
        elif file_modified := data.ext.get("file_modified"):
            source, dt = (
                "file_modified",
                datetime.fromisoformat(file_modified).replace(microsecond=0),
            )
        else:
            raise ValueError(f"{path}: no date_taken / file_modified to rename from")

        base = f"IMG{dt:%Y%m%d%H%M%S}"
        if ms := dt.microsecond // 1000:
            base += f"{ms:03d}"
        suffix = path.suffix.lower()
        dest = path.with_name(f"{base}{suffix}")
        index = 0
        while data.src.is_taken(dest):
            index += 1
            dest = path.with_name(f"{base}_{index}{suffix}")
        return data.src.plan_path_change(dest) | {"date_source": source}

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        data.src.apply_path_change(Path(planned["dest"]))

    @classmethod
    @override
    def plan_undo(cls, entry: Journal.Entry, img: ImgFile) -> OpOut:
        return img.plan_move_back(entry)

    @classmethod
    @override
    def undo(cls, img: ImgFile, planned: OpOut) -> None:
        img.apply_path_change(Path(planned["dest"]))
