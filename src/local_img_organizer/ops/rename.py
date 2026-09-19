"""Rename operation - renames an image after when it was taken, so files sort chronologically"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import override

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
    _planned: set[Path] = field(default_factory=set, init=False, repr=False)

    @override
    def plan(self, data: _Data) -> OpOut:
        if not data.src.is_file():
            raise ValueError(f"{data.src} is not a file")
        stem = data.src.stem
        if _has_date(stem):
            _log.debug(f"{data.src.name}: skipping, name already holds a date")
            return {}
        if not any(c.isdigit() for c in stem):
            _log.debug(f"{data.src.name}: skipping, name has no digits")
            return {}

        if date_taken := data.ext.get("date_taken"):
            source, dt = "date_taken", datetime.fromisoformat(date_taken)
        elif file_modified := data.ext.get("file_modified"):
            source, dt = (
                "file_modified",
                datetime.fromisoformat(file_modified).replace(microsecond=0),
            )
        else:
            raise ValueError(f"{data.src}: no date_taken / file_modified to rename from")

        base = f"IMG{dt:%Y%m%d%H%M%S}"
        if ms := dt.microsecond // 1000:
            base += f"{ms:03d}"
        suffix = data.src.suffix.lower()
        dest = data.src.with_name(f"{base}{suffix}")
        index = 0
        while dest.exists() or dest in self._planned:
            index += 1
            dest = data.src.with_name(f"{base}_{index}{suffix}")
        self._planned.add(dest)
        return {"dest": str(dest), "date_source": source}

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        if not planned:
            return
        data.src.rename(planned["dest"])

    @classmethod
    @override
    def can_undo(cls, entry: Journal.Entry) -> None:
        dest = entry.op_out.get("dest")
        if not dest:
            return
        if Path(entry.src).exists():
            raise ValueError(f"{entry.src}: already exists, undo would overwrite it")
        if not Path(dest).exists():
            raise ValueError(f"{entry.src}: dest {dest} is missing, cannot undo rename")

    @classmethod
    @override
    def undo(cls, og_data: _Data, og_planned: OpOut) -> OpOut:
        dest = og_planned.get("dest")
        if not dest:
            return {}
        Path(dest).rename(og_data.src)
        return {"dest": str(og_data.src)}
