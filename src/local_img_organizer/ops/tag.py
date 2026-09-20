"""Tag operation - writes EXIF / XMP metadata in place with exiftool"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, override

from pydantic import model_validator

from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Journal, Operation, OpOut
from local_img_organizer.utils import get_logger

_log = get_logger(__name__)

type _Data = Operation.Data

# The formats with somewhere to put metadata - gif & bmp have neither EXIF nor XMP
_TAGGABLE = {".jpg", ".jpeg", ".png", ".webp"}

# EXIF stores its timestamps as local wall-clock text in this format, with no timezone
_EXIF_DT_FMT = "%Y:%m:%d %H:%M:%S"

# `DateTimeOriginal` is what viewers show as "date taken"; `CreateDate` is the one some read
# instead, so both get written. `DateTime` is deliberately left out - it means "last edited"
_DATE_TAGS = ("DateTimeOriginal", "CreateDate")

# EXIF 2.31's offset tags - only written when the source time actually carried a zone, so a
# naive wall-clock is never dressed up as a real instant
_OFFSET_TAGS = ("OffsetTimeOriginal", "OffsetTimeDigitized")

# Keywords have no single home - XMP is the modern standard & IPTC is what older tools read
_KEYWORD_TAGS = ("XMP-dc:Subject", "IPTC:Keywords")


@dataclass
class Tag(Operation):
    """Writes metadata into an image in place, without re-encoding it or bumping its mtime

    Does either or both of, per configured op:
      - `date: true` - backfills the "date taken" tags from the file's modified time, only for
        images with no capture time of their own
      - `name` / `value` - adds a `name:value` keyword to the fields photo managers read
    """

    class Cfg(Operation.Cfg):
        """Tag operation configuration"""

        op: str = "tag"
        date: bool = False
        name: str | None = None
        value: str | None = None

        @model_validator(mode="after")
        def _check_there_is_something_to_write(self) -> "Tag.Cfg":
            """Reject a tag op that would never write anything"""
            if (self.name is None) != (self.value is None):
                raise ValueError("tag: `name` & `value` have to be set together")
            if not self.date and self.name is None:
                raise ValueError("tag: set `date: true` and/or a `name` & `value` to write")
            return self

    cfg: Cfg

    @override
    def plan(self, data: _Data) -> OpOut:
        # Read from where the file really is - a dry run leaves earlier ops in the chain unapplied
        path = data.src.disk_path
        if path.suffix.lower() not in _TAGGABLE:
            _log.debug(f"{path.name}: skipping, {path.suffix} cannot hold metadata")
            return {}
        _exiftool()  # surface a missing binary while planning, not part way through a run
        planned: OpOut = {}
        if self.cfg.date:
            planned |= self._plan_date(data, path)
        if self.cfg.name and (keywords := self._plan_keywords(path)):
            planned["add"] = keywords
        return planned

    def _plan_date(self, data: _Data, path: Path) -> OpOut:
        """Return the capture time tags to write, or `{}` if the image already has one

        The fallback is the file's modified time, which is a guess at when the photo was taken -
        so it is only ever used to fill a gap, never to overwrite what the camera recorded.
        """
        if data.ext.get("date_taken"):
            _log.debug(f"{path.name}: skipping date, it already holds a capture time")
            return {}
        if not (file_modified := data.ext.get("file_modified")):
            _log.debug(f"{path.name}: skipping date, no file_modified to fall back on")
            return {}
        dt = datetime.fromisoformat(file_modified)
        tags = dict.fromkeys(_DATE_TAGS, dt.strftime(_EXIF_DT_FMT))
        if offset := dt.strftime("%:z"):
            tags |= dict.fromkeys(_OFFSET_TAGS, offset)
        return {"set": tags, "date_source": "file_modified"}

    def _plan_keywords(self, path: Path) -> dict[str, str]:
        """Return the keyword tags still missing `name:value`

        Reading first is what keeps undo safe - only the tags this run actually added get removed,
        so a keyword the user already had survives.
        """
        keyword = f"{self.cfg.name}:{self.cfg.value}"
        existing = _read_keywords(path)
        missing = {tag: keyword for tag in _KEYWORD_TAGS if keyword not in existing[tag]}
        if not missing:
            _log.debug(f"{path.name}: skipping keyword, {keyword} is already set")
        return missing

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        _write(data.src.disk_path, _to_args(planned))

    @classmethod
    @override
    def plan_undo(cls, entry: Journal.Entry, img: ImgFile) -> OpOut:
        if not _to_args(entry.op_out):
            return {}  # nothing was written - ex. the entry recording a failed run
        if img.path != entry.src:
            raise ValueError(f"{entry.src}: expected the file there, found {img.path}")
        return entry.op_out

    @classmethod
    @override
    def undo(cls, img: ImgFile, planned: OpOut) -> None:
        _write(img.disk_path, _undo_args(planned))


def _to_args(planned: OpOut) -> list[str]:
    """Return the exiftool arguments that apply `planned`"""
    return [f"-{tag}={value}" for tag, value in planned.get("set", {}).items()] + [
        f"-{tag}+={value}" for tag, value in planned.get("add", {}).items()
    ]


def _undo_args(planned: OpOut) -> list[str]:
    """Return the exiftool arguments that reverse `planned`

    A `set` tag was only written because the image had none, so its undo is a delete. An `add`
    drops just the one value, leaving any keyword that was already there alone.
    """
    return [f"-{tag}=" for tag in planned.get("set", {})] + [
        f"-{tag}-={value}" for tag, value in planned.get("add", {}).items()
    ]


def _write(path: Path, args: list[str]) -> None:
    """Apply `args` to `path` in place, keeping its mtime & leaving no backup file behind

    `-P` is what preserves the mtime, and it matters more than it looks - without it, tagging an
    image bumps its modified time to now, so the next run reads the write time as `file_modified`
    and the only evidence the date was ever guessed from is gone.
    """
    if not args:
        return
    _run([_exiftool(), "-P", "-overwrite_original", *args, str(path)])


def _run(cmd: list[str]) -> str:
    """Run `cmd`, returning its stdout & raising with exiftool's own message on failure"""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"exiftool failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def _exiftool() -> str:
    """Return the exiftool binary, or raise with how to install it

    Pillow is not an option here - it re-encodes the pixels & drops the tags it does not know.
    """
    if found := shutil.which("exiftool"):
        return found
    raise ValueError("exiftool is not installed - ex. `apt install libimage-exiftool-perl`")


def _read_keywords(path: Path) -> dict[str, list[str]]:
    """Return the values `path` already holds for each of `_KEYWORD_TAGS`"""
    raw = _run([_exiftool(), "-json", "-G1", *(f"-{tag}" for tag in _KEYWORD_TAGS), str(path)])
    found: dict[str, Any] = json.loads(raw)[0] if raw.strip() else {}
    result: dict[str, list[str]] = {}
    for tag in _KEYWORD_TAGS:
        # exiftool's JSON keys carry whichever group prefix it feels like, so match on the tag
        # name itself - ex. `XMP-dc:Subject` can come back as `Subject` or `XMP:Subject`
        wanted = tag.rsplit(":", 1)[-1]
        values: Any = next((v for k, v in found.items() if k.rsplit(":", 1)[-1] == wanted), [])
        result[tag] = [str(v) for v in (values if isinstance(values, list) else [values])]
    return result
