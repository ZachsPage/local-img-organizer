"""Photo metadata extraction - ex. when a photo was taken & where"""

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, NamedTuple, override

from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational
from pydantic import Field, model_validator

from local_img_organizer.config import parse_operations
from local_img_organizer.interfaces import ExtOut, Extractor, Journal, Operation
from local_img_organizer.ops.noop import Noop
from local_img_organizer.utils import find_images, get_logger

_log = get_logger(__name__)

# EXIF stores its timestamps as local wall-clock text in this format, with no timezone
_EXIF_DT_FMT = "%Y:%m:%d %H:%M:%S"

# GPS coordinates are stored as (degrees, minutes, seconds)
_DMS_PARTS = 3

# EXIF is stored in the layout TIFF uses, where tags live in tables called IFDs ("Image File
# Directory") - each just maps a numeric tag to its value. Every file has a top level ("0th") IFD
# describing the image itself (ex. Make / Model / Orientation / DateTime), which then points at
# nested sub-IFDs holding everything else - `Exif` for the camera's own tags (DateTimeOriginal
# and the sub-second / UTC offset tags) and `GPSInfo` for coordinates.
#
# This extractor has to know about them because Pillow only follows those pointers when asked
# (`Image.Exif.get_ifd`), so the table it hands back by default is missing most of what we want
type _Tags = dict[int, Any]


class _DateSource(NamedTuple):
    """One EXIF timestamp - split across three tags, since the original 20-byte ASCII date had
    room for neither fractional seconds nor a timezone, and both were added in later revisions

    :param in_sub_ifd: Whether the date tag is in the `Exif` sub-IFD or the 0th one. Only the date
        moves - the sub-second & offset tags are always in the `Exif` sub-IFD
    """

    date: int
    subsec: int
    offset: int
    in_sub_ifd: bool = True


# Ordered date sources - the first one that parses wins. `DateTimeOriginal` is when the shutter
# fired, `DateTimeDigitized` when it became a file (they differ for scanned film), and `DateTime`
# only when software last changed it - so it is the weakest of the three
_DATE_SOURCES = (
    _DateSource(
        ExifTags.Base.DateTimeOriginal,
        ExifTags.Base.SubsecTimeOriginal,
        ExifTags.Base.OffsetTimeOriginal,
    ),
    _DateSource(
        ExifTags.Base.DateTimeDigitized,
        ExifTags.Base.SubsecTimeDigitized,
        ExifTags.Base.OffsetTimeDigitized,
    ),
    _DateSource(
        ExifTags.Base.DateTime,
        ExifTags.Base.SubsecTime,
        ExifTags.Base.OffsetTime,
        in_sub_ifd=False,
    ),
)


@dataclass
class Metadata(Extractor):
    """Extracts photo metadata (ex. capture time / GPS location) to be used in Operations

    With no `operations` configured every image runs `noop`, so the metadata still shows up in
    the journal - handy to see what is available before deciding what to do with it.
    """

    class Cfg(Extractor.Cfg):
        """Metadata extractor configuration"""

        operations: list[dict[str, Any]] = Field(default_factory=list)
        # Only run the operations for images whose metadata has all of these keys - ex. ["gps"]
        require: list[str] = Field(default_factory=list)
        # Turn any coordinates found into a city / state / country under "location"
        lookup_location: bool = True

        @model_validator(mode="after")
        def _check_location_is_reachable(self) -> "Metadata.Cfg":
            """Reject requiring a key that this config can never produce"""
            if "location" in self.require and not self.lookup_location:
                msg = "require: [location] never matches while lookup_location is false"
                raise ValueError(msg)
            return self

    cfg: Cfg
    ops: list[Operation]

    @classmethod
    def from_cfg(cls, data: dict[str, Any]) -> "Metadata":
        """Build a Metadata extractor from raw YAML config data"""
        cfg = cls.Cfg.model_validate(data)
        return cls(cfg=cfg, ops=parse_operations(cfg.operations) or [Noop()])

    @override
    def run(self, img_dir: Path, *, is_dry: bool) -> Generator[Callable[[], Journal.Entry]]:
        cfg = self.cfg
        paths = find_images(img_dir)
        _log.info(f"Extracting metadata from {len(paths)} images in {img_dir}...")
        if cfg.require:
            _log.info(f"- Only running operations for images with: {cfg.require}")
        counts = {"date_taken": 0, "gps": 0, "error": 0}
        for path in paths:
            meta = _extract(path, lookup_location=cfg.lookup_location)
            for key in counts:
                counts[key] += key in meta
            ops = self.ops if self._should_run_ops(meta) else [Noop()]
            for op in ops:
                yield op.prepare(Operation.Data(src=path, is_dry=is_dry), ext_data=meta)
        _log.info(
            f"Found a capture time for {counts['date_taken']}/{len(paths)} images, "
            f"coordinates for {counts['gps']}, and failed to read {counts['error']}"
        )

    def _should_run_ops(self, meta: ExtOut) -> bool:
        """Return whether this image's metadata satisfies the configured `require` keys"""
        if "error" in meta:
            return False
        return all(key in meta for key in self.cfg.require)


def _extract(path: Path, *, lookup_location: bool) -> ExtOut:
    """Return the metadata found for a single image - keys are omitted when unavailable

    Keys that may be present:
        date_taken: ISO-8601 capture time read from EXIF - carries a UTC offset only when one was
            recorded, otherwise it is the camera's local wall-clock with no way to place it on a
            timeline. Absent when the image has no embedded date
        file_modified: ISO-8601 file modification time, present whenever the image was readable.
            Kept separate from `date_taken` so a real capture time is never confused with a guess
            at one - it is up to an operation to decide whether falling back to it makes sense
        gps: {lat, lon} in signed decimal degrees
        location: the {city, county, state, country, country_code} the gps resolves to
        error: why the image could not be read - the other keys will be missing
    """
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            base: _Tags = dict(exif)
            # EXIF nests most of the interesting tags in sub-directories (IFDs) that are only
            # read on demand - pull them while the file is still open
            sub: _Tags = dict(exif.get_ifd(ExifTags.IFD.Exif))
            gps: _Tags = dict(exif.get_ifd(ExifTags.IFD.GPSInfo))
            png_text = {k: v for k, v in img.info.items() if isinstance(v, str)}
    except Exception as ex:  # noqa: BLE001
        _log.error(f"Unable to read {path}: {ex}")
        return {"error": str(ex)}

    out: ExtOut = {}
    if date_taken := _find_date(base, sub, png_text):
        out["date_taken"] = date_taken
    # Unlike EXIF, a file time is a real instant - so its offset is known rather than guessed
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).astimezone()
    out["file_modified"] = mtime.isoformat()

    if coords := _find_gps(gps):
        out["gps"] = coords
        if lookup_location and (place := _find_location(coords)):
            out["location"] = place
    return out


def _find_date(base: _Tags, sub: _Tags, png_text: dict[str, str]) -> str | None:
    """Return an ISO-8601 timestamp from the first date source that parses"""
    for src in _DATE_SOURCES:
        date = (sub if src.in_sub_ifd else base).get(src.date)
        if parsed := _parse_dt(date, sub.get(src.subsec), sub.get(src.offset)):
            return parsed
    # PNGs carry no EXIF unless a tool wrote one, but often have a "Creation Time" text chunk
    for key, value in png_text.items():
        if "creat" in key.lower() and (parsed := _parse_dt(value, None, None)):
            return parsed
    return None


def _parse_dt(raw: Any, subsec: Any, offset: Any) -> str | None:  # noqa: ANN401
    """Return an ISO-8601 timestamp for an EXIF "YYYY:MM:DD HH:MM:SS" value, or None if unparsable

    :param subsec: Matching sub-second tag, holding the fractional digits as text - ex. "250" = .25s
    :param offset: Matching UTC offset tag, holding the timezone as text - ex. "-07:00"
    """
    text = _clean(raw)
    if not text:
        return None
    dt = _try_formats(text)
    if dt is None:
        return None
    if (digits := _clean(subsec)) and digits.isdigit():
        dt = dt.replace(microsecond=int(digits.ljust(6, "0")[:6]))
    if dt.tzinfo is None and (tz := _parse_offset(offset)):
        dt = dt.replace(tzinfo=tz)
    return dt.isoformat()


def _try_formats(text: str) -> datetime | None:
    """Return `text` parsed by the first date format that accepts it, or None if none do

    EXIF's own format comes first. The other two are for PNG's "Creation Time" text chunk, which
    is not EXIF and has no enforced format - the spec asks for RFC 1123 ("Sat, 03 Feb 2019
    04:05:06 GMT"), but plenty of tools write ISO-8601 into it instead
    """
    try:
        # Naive for now - EXIF records the camera's local wall-clock, and the timezone (if the
        # camera recorded one at all) lives in a separate tag
        return datetime.strptime(text, _EXIF_DT_FMT)  # noqa: DTZ007
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        # Unlike the other two this can carry its own zone, which `_parse_dt` then leaves alone
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def _parse_offset(raw: Any) -> tzinfo | None:  # noqa: ANN401
    """Return the timezone for an EXIF "+HH:MM" UTC offset value, or None if unset / unparsable

    These tags were only added in EXIF 2.31, so plenty of photos have a capture time with no
    offset to pair it with - those stay naive rather than being guessed at
    """
    text = _clean(raw)
    if not text:
        return None
    try:
        # An offset on its own is a complete `%z`, so the rest of the parsed value is unused
        return datetime.strptime(text, "%z").tzinfo
    except ValueError:
        return None


def _find_gps(gps: _Tags) -> dict[str, float] | None:
    """Return the {lat, lon} decimal degrees recorded for the photo"""
    lat = _to_degrees(gps.get(ExifTags.GPS.GPSLatitude), gps.get(ExifTags.GPS.GPSLatitudeRef))
    lon = _to_degrees(gps.get(ExifTags.GPS.GPSLongitude), gps.get(ExifTags.GPS.GPSLongitudeRef))
    if lat is None or lon is None:
        return None
    return {"lat": round(lat, 7), "lon": round(lon, 7)}


def _find_location(coords: dict[str, float]) -> dict[str, str] | None:
    """Return the place the photo's coordinates fall in - ex. {"city": "Denver", "state": ...}

    This is an offline lookup against a bundled GeoNames dataset, which holds a point per
    populated place rather than real borders - so the answer is the *nearest* known place, and
    can be wrong for a photo taken far from one (out at sea, in a park, just across a border)
    """
    # Imported here so the dataset & its k-d tree are only built once a photo actually has
    # coordinates - it costs ~0.3s, and most folders of photos have none
    import reverse_geocode  # type: ignore[import-untyped] # noqa: PLC0415

    try:
        found = reverse_geocode.get((coords["lat"], coords["lon"]))
    except Exception as ex:  # noqa: BLE001
        _log.error(f"Unable to look up {coords}: {ex}")
        return None
    keys = ("city", "county", "state", "country", "country_code")
    return {key: text for key in keys if (text := _clean(found.get(key)))} or None


def _to_degrees(dms: Any, ref: Any) -> float | None:  # noqa: ANN401
    """Convert an EXIF (degrees, minutes, seconds) triple + N/S/E/W ref to signed decimal degrees"""
    if not isinstance(dms, Sequence) or len(dms) != _DMS_PARTS:
        return None
    parts = [_to_float(v) for v in dms]
    if any(p is None for p in parts):
        return None
    degrees, minutes, seconds = (p for p in parts if p is not None)
    value = degrees + minutes / 60 + seconds / 3600
    return -value if (_clean(ref) or "").upper() in {"S", "W"} else value


def _to_float(value: Any) -> float | None:  # noqa: ANN401
    """Return `value` as a float - EXIF numbers are often rationals, which can be 0-denominated"""
    if isinstance(value, IFDRational | int | float):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except ZeroDivisionError:
            pass
    return None


def _clean(value: Any) -> str | None:  # noqa: ANN401
    """Return `value` as trimmed text - EXIF strings are frequently bytes and/or null padded"""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    return value.replace("\x00", "").strip() or None
