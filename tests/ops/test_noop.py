from pathlib import Path

from local_img_organizer.img_file import ImgFile
from local_img_organizer.ops.noop import Noop


def test_does_nothing(tmp_path: Path) -> None:
    """Verify the file is untouched and nothing is planned"""
    src = tmp_path / "test.png"
    src.touch()
    data = Noop.Data(src=ImgFile(path=src), is_dry=False)
    op = Noop()

    plan = op.plan(data)
    assert plan == {"noop": True}
    op.run(data, plan)
    assert src.exists()
    assert list(tmp_path.iterdir()) == [src]


def test_undo(tmp_path: Path) -> None:
    """Verify undoing nothing also does nothing"""
    src = tmp_path / "test.png"

    assert Noop.undo(ImgFile(path=src), {}) is None
    assert not src.exists()


def test_journal_entry(tmp_path: Path) -> None:
    """Verify the extractor's findings are journaled under the noop op, despite planning nothing"""
    src = tmp_path / "test.png"
    src.touch()
    found = {"date_taken": "2023-07-14T10:22:31"}

    entry = Noop().plan_entry(Noop.Data(src=ImgFile(path=src), is_dry=False, ext=found))

    assert entry is not None
    assert entry.op == "noop"
    assert entry.src == src
    assert entry.ext_out == found
    assert entry.op_out == {"noop": True}


def test_from_cfg() -> None:
    """Verify `op: noop` in a config builds the op"""
    assert isinstance(Noop.from_cfg({"op": "noop"}), Noop)
