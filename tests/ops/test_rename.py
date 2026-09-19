from pathlib import Path

import pytest

from local_img_organizer.interfaces import Journal, Operation
from local_img_organizer.ops.rename import Rename, _has_date

type Data = Operation.Data

_MTIME = "2021-09-15T20:17:56.123456-04:00"


def _rename() -> Rename:
    return Rename(cfg=Rename.Cfg(op="rename"))


def _data(tmp_path: Path, name: str, **ext: str) -> Data:
    src = tmp_path / name
    src.touch()
    return Rename.Data(src=src, is_dry=False, ext={"file_modified": _MTIME, **ext})


@pytest.mark.parametrize(
    "name",
    [
        "IMG_20231015_175522",
        "IMG20251117103846",
        "IMG20251117103846123",
        "IMG20251117103846_1",
        "IMG19850601120000",
        "IMG19700101000000",
        "20190704_120000",
        "PXL_20231015_175522385",
        "photo-2023-10-15",
        "170624-J-E-final-selects-546",
        "zach_miller_rx_23_07_31",
    ],
)
def test_has_date(name: str) -> None:
    """Verify dated names are detected, including this op's own output"""
    assert _has_date(name)


@pytest.mark.parametrize(
    "name",
    [
        "13641144_1228325863845354_612491603760188281_o",
        "007adc99-a23b-4e3f-9d86-e3c807773059",
        "IMG_3052",
        "12051130",
        "Rachel-Michael-Wedding-186_websize",
    ],
)
def test_has_no_date(name: str) -> None:
    """Verify digits that only look like a date are rejected"""
    assert not _has_date(name)


def test_skips_dated_name(tmp_path: Path) -> None:
    """Verify an already dated name is left alone, even with a date_taken"""
    data = _data(tmp_path, "IMG20251117103846.jpg", date_taken="2024-01-01T12:00:00")
    assert _rename().plan(data) == {}


def test_skips_name_without_digits(tmp_path: Path) -> None:
    """Verify a human-readable name is never renamed, even with a date_taken"""
    data = _data(tmp_path, "DylansArbys.PNG", date_taken="2024-01-01T12:00:00")
    assert _rename().plan(data) == {}


def test_plan_from_date_taken(tmp_path: Path) -> None:
    """Verify date_taken wins over file_modified, and its sub-seconds are kept as ms"""
    data = _data(tmp_path, "DSC_2057.JPG", date_taken="2024-01-01T12:00:00.123000")
    assert _rename().plan(data) == {
        "dest": str(tmp_path / "IMG20240101120000123.jpg"),
        "date_source": "date_taken",
    }


def test_plan_from_file_modified(tmp_path: Path) -> None:
    """Verify file_modified is the fallback, truncated to whole seconds"""
    data = _data(tmp_path, "IMG_3052.jpg")
    assert _rename().plan(data) == {
        "dest": str(tmp_path / "IMG20210915201756.jpg"),
        "date_source": "file_modified",
    }


def test_plan_no_date(tmp_path: Path) -> None:
    """Verify planning without any date to use raises"""
    src = tmp_path / "IMG_3052.jpg"
    src.touch()
    with pytest.raises(ValueError, match="no date_taken"):
        _rename().plan(Rename.Data(src=src, is_dry=False))


def test_plan_not_a_file(tmp_path: Path) -> None:
    """Verify planning for a src that does not exist raises"""
    with pytest.raises(ValueError, match="is not a file"):
        _rename().plan(Rename.Data(src=tmp_path / "IMG_1.jpg", is_dry=False))


def test_collisions(tmp_path: Path) -> None:
    """Verify the counter covers files on disk & names planned earlier in the same (dry) run"""
    (tmp_path / "IMG20210915201756.jpg").touch()
    op = _rename()
    dests = [
        Path(op.plan(_data(tmp_path, n))["dest"]).name for n in ("IMG_3052.jpg", "IMG_3067.jpg")
    ]
    assert dests == ["IMG20210915201756_1.jpg", "IMG20210915201756_2.jpg"]


def test_roundtrip(tmp_path: Path) -> None:
    """Verifies the plan -> run -> undo round trip"""
    data = _data(tmp_path, "IMG_3052.jpg")
    op = _rename()
    plan = op.plan(data)
    dest = Path(plan["dest"])
    op.run(data, plan)
    assert dest.exists()
    assert not data.src.exists()
    assert op.undo(data, plan) == {"dest": str(data.src)}
    assert data.src.exists()
    assert not dest.exists()


def test_can_undo(tmp_path: Path) -> None:
    """Verify undo is rejected when it would overwrite src or dest is missing"""
    src = tmp_path / "IMG_3052.jpg"
    dest = tmp_path / "IMG20210915201756.jpg"
    entry = Journal.Entry(
        op="rename", src=src, ext_out={}, op_out={"dest": str(dest)}, is_dry=False
    )
    with pytest.raises(ValueError, match="is missing"):
        Rename.can_undo(entry)
    src.touch()
    dest.touch()
    with pytest.raises(ValueError, match="already exists"):
        Rename.can_undo(entry)
    src.unlink()
    Rename.can_undo(entry)
