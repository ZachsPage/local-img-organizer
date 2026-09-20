from pathlib import Path

import pytest
from PIL import ExifTags

from local_img_organizer.extractors.metadata import _DATE_SOURCES
from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Journal
from local_img_organizer.ops import tag as tag_mod
from local_img_organizer.ops.tag import Tag

_MODIFIED = "2023-07-14T10:22:31.250000-06:00"
_EXIF_DATE = "2023:07:14 10:22:31"


def _tag(**cfg: object) -> Tag:
    """Build a Tag op from raw config data"""
    return Tag.from_cfg({"op": "tag", **cfg})


def _data(tmp_path: Path, name: str = "test.jpg", **ext: object) -> Tag.Data:
    """Build op data for an image that exists on disk"""
    src = tmp_path / name
    src.touch()
    return Tag.Data(src=ImgFile(path=src), is_dry=False, ext=ext)


@pytest.fixture(autouse=True)
def fake_exiftool(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Pretend exiftool is installed with no keywords set, recording every command it would run"""
    calls: list[list[str]] = []
    monkeypatch.setattr(tag_mod, "_exiftool", lambda: "exiftool")
    monkeypatch.setattr(tag_mod, "_run", lambda cmd: calls.append(cmd) or "")
    return calls


def test_plan_date_from_file_modified(tmp_path: Path) -> None:
    """With no capture time, the file's modified time is written to every date tag"""
    plan = _tag(date=True).plan(_data(tmp_path, file_modified=_MODIFIED))
    assert plan == {
        "set": {
            "DateTimeOriginal": _EXIF_DATE,
            "CreateDate": _EXIF_DATE,
            "OffsetTimeOriginal": "-06:00",
            "OffsetTimeDigitized": "-06:00",
        },
        "date_source": "file_modified",
    }


def test_plan_date_writes_no_offset_it_does_not_have(tmp_path: Path) -> None:
    """A modified time with no zone writes no offset tags rather than guessing one"""
    plan = _tag(date=True).plan(_data(tmp_path, file_modified="2023-07-14T10:22:31"))
    assert set(plan["set"]) == {"DateTimeOriginal", "CreateDate"}


def test_plan_date_skips_existing_capture_time(tmp_path: Path) -> None:
    """A real capture time is never overwritten by the modified time guess"""
    data = _data(tmp_path, date_taken="2020-01-02T03:04:05", file_modified=_MODIFIED)
    assert _tag(date=True).plan(data) == {}


def test_plan_date_without_any_date(tmp_path: Path) -> None:
    """Nothing to fall back on plans nothing"""
    assert _tag(date=True).plan(_data(tmp_path)) == {}


def test_plan_skips_formats_without_metadata(tmp_path: Path) -> None:
    """A gif has nowhere to hold tags, so it is skipped rather than failed"""
    data = _data(tmp_path, name="test.gif", file_modified=_MODIFIED)
    assert _tag(date=True).plan(data) == {}


def test_plan_keyword(tmp_path: Path) -> None:
    """A name/value pair is planned as a `name:value` keyword for both keyword tags"""
    plan = _tag(name="type", value="outdoor").plan(_data(tmp_path))
    assert plan == {"add": {"XMP-dc:Subject": "type:outdoor", "IPTC:Keywords": "type:outdoor"}}


def test_plan_keyword_already_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyword the image already carries is not added again"""
    monkeypatch.setattr(
        tag_mod,
        "_run",
        lambda _: '[{"XMP-dc:Subject": ["type:outdoor"], "IPTC:Keywords": ["type:outdoor"]}]',
    )
    assert _tag(name="type", value="outdoor").plan(_data(tmp_path)) == {}


def test_plan_keyword_under_any_group_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exiftool picks its own group prefixes, and a single keyword comes back unwrapped"""
    found = '[{"SourceFile": "x.jpg", "XMP:Subject": ["type:outdoor"], "IPTC:Keywords": "a"}]'
    monkeypatch.setattr(tag_mod, "_run", lambda _: found)
    plan = _tag(name="type", value="outdoor").plan(_data(tmp_path))
    assert plan == {"add": {"IPTC:Keywords": "type:outdoor"}}


def test_plan_date_and_keyword(tmp_path: Path) -> None:
    """One op can write both a date & a keyword"""
    op = _tag(date=True, name="type", value="outdoor")
    plan = op.plan(_data(tmp_path, file_modified=_MODIFIED))
    assert set(plan) == {"set", "date_source", "add"}


def test_run_preserves_mtime_and_leaves_no_backup(
    tmp_path: Path, fake_exiftool: list[list[str]]
) -> None:
    """The write keeps the original mtime & overwrites in place rather than leaving a copy"""
    op = _tag(date=True)
    data = _data(tmp_path, file_modified=_MODIFIED)
    op.run(data, op.plan(data))

    cmd = fake_exiftool[-1]
    assert "-P" in cmd
    assert "-overwrite_original" in cmd
    assert f"-DateTimeOriginal={_EXIF_DATE}" in cmd
    assert cmd[-1] == str(data.src.disk_path)


def test_undo_args(tmp_path: Path, fake_exiftool: list[list[str]]) -> None:
    """Undo deletes the tags it set & removes only the keyword value it added"""
    planned = {"set": {"DateTimeOriginal": _EXIF_DATE}, "add": {"IPTC:Keywords": "a:b"}}
    Tag.undo(ImgFile(path=tmp_path / "test.jpg"), planned)

    cmd = fake_exiftool[-1]
    assert "-DateTimeOriginal=" in cmd
    assert "-IPTC:Keywords-=a:b" in cmd


def test_undo_of_nothing(tmp_path: Path, fake_exiftool: list[list[str]]) -> None:
    """An empty plan runs no exiftool at all"""
    Tag.undo(ImgFile(path=tmp_path / "test.jpg"), {})
    assert fake_exiftool == []


def _entry(src: Path, op_out: dict[str, object]) -> Journal.Entry:
    """Build a journal entry for a tag op"""
    return Journal.Entry(op="tag", img_uid="uid", src=src, ext_out={}, op_out=op_out, is_dry=False)


def test_plan_undo(tmp_path: Path) -> None:
    """The undo is planned against where the file sits once later ops have been unwound"""
    src = tmp_path / "test.jpg"
    entry = _entry(src, {"set": {"DateTimeOriginal": _EXIF_DATE}})
    assert Tag.plan_undo(entry, ImgFile(path=src)) == entry.op_out
    with pytest.raises(ValueError, match="expected the file there"):
        Tag.plan_undo(entry, ImgFile(path=tmp_path / "elsewhere.jpg"))


def test_plan_undo_of_a_failed_run(tmp_path: Path) -> None:
    """The entry journaled for a failed run wrote nothing, so there is nothing to reverse"""
    entry = _entry(tmp_path / "test.jpg", {"error": "boom"})
    assert Tag.plan_undo(entry, ImgFile(path=tmp_path / "elsewhere.jpg")) == {}


def test_missing_exiftool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing binary is reported while planning, not part way through a run"""
    monkeypatch.undo()
    monkeypatch.setattr(tag_mod.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="exiftool is not installed"):
        _tag(date=True).plan(_data(tmp_path, file_modified=_MODIFIED))


def test_exiftool_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed exiftool call surfaces its own stderr"""
    monkeypatch.undo()
    monkeypatch.setattr(tag_mod.shutil, "which", lambda _: "exiftool")

    class _Result:
        returncode = 1
        stderr = "Error: Not a valid JPG"
        stdout = ""

    monkeypatch.setattr(tag_mod.subprocess, "run", lambda *_a, **_kw: _Result())
    with pytest.raises(ValueError, match="Error: Not a valid JPG"):
        Tag.undo(ImgFile(path=tmp_path / "x.jpg"), {"set": {"DateTimeOriginal": _EXIF_DATE}})


def test_from_cfg_needs_something_to_write() -> None:
    """A tag op that would never write anything is rejected at config time"""
    with pytest.raises(ValueError, match="date"):
        _tag()
    with pytest.raises(ValueError, match="together"):
        _tag(name="type")


def test_date_tags_feed_the_metadata_extractor(tmp_path: Path) -> None:
    """What this op writes is what the metadata extractor reads back as `date_taken`"""
    plan = _tag(date=True).plan(_data(tmp_path, file_modified=_MODIFIED))
    read_back = {ExifTags.Base(src.date).name for src in _DATE_SOURCES}
    assert "DateTimeOriginal" in plan["set"]
    assert "DateTimeOriginal" in read_back
