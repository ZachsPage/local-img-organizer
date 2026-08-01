import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import ExifTags, Image, PngImagePlugin
from PIL.TiffImagePlugin import IFDRational

from local_img_organizer.extractors.metadata import Metadata
from local_img_organizer.interfaces import Journal

_DATE = "2023:07:14 10:22:31"
_ISO = "2023-07-14T10:22:31"


def _exif(
    *,
    date: str | None = _DATE,
    subsec: str | None = None,
    offset: str | None = None,
    gps: bool = False,
) -> Image.Exif:
    exif = Image.Exif()
    # Pillow drops the whole EXIF block when the top level IFD is empty, so always write something
    exif[ExifTags.Base.Orientation] = 1
    if date:
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        sub[ExifTags.Base.DateTimeOriginal] = date
        if subsec:
            sub[ExifTags.Base.SubsecTimeOriginal] = subsec
        if offset:
            sub[ExifTags.Base.OffsetTimeOriginal] = offset
    if gps:
        # 37deg 46' 29.88" N, 122deg 24' 31.32" W - San Francisco
        ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        ifd[ExifTags.GPS.GPSLatitudeRef] = "N"
        ifd[ExifTags.GPS.GPSLatitude] = (IFDRational(37), IFDRational(46), IFDRational(2988, 100))
        ifd[ExifTags.GPS.GPSLongitudeRef] = "W"
        ifd[ExifTags.GPS.GPSLongitude] = (IFDRational(122), IFDRational(24), IFDRational(3132, 100))
    return exif


def _write(path: Path, exif: Image.Exif | None = None, **kwargs: Any) -> Path:
    img = Image.new("RGB", (8, 6), "red")
    if exif is not None:
        kwargs["exif"] = exif
    img.save(path, **kwargs)
    return path


def _run(img_dir: Path, **cfg: object) -> list[Journal.Entry]:
    extractor = Metadata.from_cfg(dict(cfg))
    return [prepared() for prepared in extractor.run(img_dir, is_dry=False)]


def test_extracts_date_and_gps(tmp_path: Path) -> None:
    """Verify the headline fields are pulled out of a JPEG's EXIF"""
    _write(tmp_path / "a.jpg", _exif(gps=True, subsec="250"))
    entries = _run(tmp_path)

    assert len(entries) == 1
    found = entries[0].ext_out
    assert found["date_taken"] == f"{_ISO}.250000"
    assert found["gps"] == {"lat": 37.7749667, "lon": -122.4087}


def test_looks_up_location(tmp_path: Path) -> None:
    """Verify coordinates are turned into a place, and that it can be turned off"""
    _write(tmp_path / "a.jpg", _exif(gps=True))

    found = _run(tmp_path)[0].ext_out["location"]
    assert found["city"] == "San Francisco"
    assert found["state"] == "California"
    assert found["country"] == "United States"
    assert found["country_code"] == "US"

    assert "location" not in _run(tmp_path, lookup_location=False)[0].ext_out


def test_journals_noop_with_no_operations(tmp_path: Path) -> None:
    """Verify metadata is journaled under a noop when no operations are configured"""
    _write(tmp_path / "a.jpg", _exif())
    entries = _run(tmp_path)

    assert [e.op for e in entries] == ["noop"]
    assert entries[0].op_out == {}
    assert entries[0].src == tmp_path / "a.jpg"
    assert entries[0].ext_out["date_taken"] == _ISO


def test_feeds_operations(tmp_path: Path) -> None:
    """Verify configured operations run per image and receive the metadata as ext_out"""
    src = _write(tmp_path / "a.jpg", _exif())
    entries = _run(tmp_path, operations=[{"op": "move", "subdir_name": "dated"}])

    assert [e.op for e in entries] == ["move"]
    assert entries[0].ext_out["date_taken"] == _ISO
    assert entries[0].op_out == {"dest": str(tmp_path / "dated" / "a.jpg")}
    assert not src.exists()


def test_require_gates_operations(tmp_path: Path) -> None:
    """Verify `require` keeps operations off images missing those keys, but still journals them"""
    _write(tmp_path / "located.jpg", _exif(gps=True))
    _write(tmp_path / "plain.jpg", _exif())
    entries = _run(tmp_path, require=["gps"], operations=[{"op": "move", "subdir_name": "located"}])

    by_name = {e.src.name: e for e in entries}
    assert by_name["located.jpg"].op == "move"
    assert by_name["plain.jpg"].op == "noop"
    assert "gps" not in by_name["plain.jpg"].ext_out
    assert (tmp_path / "located" / "located.jpg").exists()
    assert (tmp_path / "plain.jpg").exists()


def test_file_modified_is_reported_separately(tmp_path: Path) -> None:
    """Verify the file time is always reported, and never stands in for a missing capture time"""
    dated = _write(tmp_path / "dated.jpg", _exif())
    plain = _write(tmp_path / "plain.jpg", _exif(date=None))
    stamp = datetime(2019, 2, 3, 4, 5, 6, tzinfo=UTC).timestamp()
    for path in (dated, plain):
        os.utime(path, (stamp, stamp))

    expected = datetime.fromtimestamp(stamp, tz=UTC).astimezone().isoformat()
    by_name = {e.src.name: e.ext_out for e in _run(tmp_path)}
    assert by_name["dated.jpg"]["date_taken"] == _ISO
    assert by_name["dated.jpg"]["file_modified"] == expected
    assert "date_taken" not in by_name["plain.jpg"]
    assert by_name["plain.jpg"]["file_modified"] == expected


def test_utc_offset(tmp_path: Path) -> None:
    """Verify a recorded UTC offset is attached to the capture time, and never invented"""
    _write(tmp_path / "zoned.jpg", _exif(offset="-07:00", subsec="250"))
    _write(tmp_path / "naive.jpg", _exif())
    # EXIF says to blank these tags out when the camera does not know its own timezone
    _write(tmp_path / "blank.jpg", _exif(offset="   :  "))

    by_name = {e.src.name: e.ext_out for e in _run(tmp_path)}
    assert by_name["zoned.jpg"]["date_taken"] == f"{_ISO}.250000-07:00"
    # The wall-clock is unchanged - only the timezone it belongs to was added
    assert by_name["naive.jpg"]["date_taken"] == _ISO
    assert by_name["blank.jpg"]["date_taken"] == _ISO


def test_png_creation_time_and_exif(tmp_path: Path) -> None:
    """Verify PNGs work via both an embedded EXIF chunk and a "Creation Time" text chunk"""
    _write(tmp_path / "a.png", _exif())
    text = PngImagePlugin.PngInfo()
    text.add_text("Creation Time", "2019:02:03 04:05:06")
    _write(tmp_path / "b.png", pnginfo=text)

    by_name = {e.src.name: e.ext_out for e in _run(tmp_path)}
    assert by_name["a.png"]["date_taken"] == _ISO
    assert by_name["b.png"]["date_taken"] == "2019-02-03T04:05:06"


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("2019:02:03 04:05:06", "2019-02-03T04:05:06"),
        ("2019-02-03T04:05:06", "2019-02-03T04:05:06"),
        ("Sun, 03 Feb 2019 04:05:06 GMT", "2019-02-03T04:05:06+00:00"),
        ("3 Feb 2019 04:05:06 -0700", "2019-02-03T04:05:06-07:00"),
        ("sometime last summer", None),
    ],
)
def test_png_creation_time_formats(tmp_path: Path, written: str, expected: str | None) -> None:
    """Verify the formats tools actually write into "Creation Time", which has none enforced"""
    text = PngImagePlugin.PngInfo()
    text.add_text("Creation Time", written)
    _write(tmp_path / "a.png", pnginfo=text)

    assert _run(tmp_path)[0].ext_out.get("date_taken") == expected


def test_bad_and_ignored_files(tmp_path: Path) -> None:
    """Verify unreadable images are journaled as errors & non-images are skipped entirely"""
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "subdir").mkdir()
    entries = _run(tmp_path, operations=[{"op": "move", "subdir_name": "x"}])

    assert len(entries) == 1
    assert entries[0].src.name == "broken.jpg"
    # Nothing is known about the file, so its ops are skipped rather than run blind
    assert entries[0].op == "noop"
    assert "error" in entries[0].ext_out


def test_reads_supported_extensions_only(tmp_path: Path) -> None:
    """Verify the supported extensions are read case-insensitively & anything else is skipped"""
    _write(tmp_path / "a.JPG", _exif(), format="JPEG")
    _write(tmp_path / "b.png", _exif())
    _write(tmp_path / "c.ppm", _exif(), format="PPM")

    assert [e.src.name for e in _run(tmp_path)] == ["a.JPG", "b.png"]


def test_rejects_unknown_cfg_keys() -> None:
    """Verify a typo'd config key is caught rather than silently ignored"""
    with pytest.raises(ValueError, match="typo"):
        Metadata.from_cfg({"typo": True})


def test_rejects_requiring_location_without_the_lookup() -> None:
    """Verify a `require` that can never be satisfied is caught instead of silently running noop"""
    with pytest.raises(ValueError, match="never matches"):
        Metadata.from_cfg({"require": ["location"], "lookup_location": False})

    Metadata.from_cfg({"require": ["gps"], "lookup_location": False})
