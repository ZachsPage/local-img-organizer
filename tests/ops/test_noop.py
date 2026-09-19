from pathlib import Path

from local_img_organizer.interfaces import Journal
from local_img_organizer.ops.noop import Noop


def test_does_nothing(tmp_path: Path) -> None:
    """Verify the file is untouched and nothing is planned"""
    src = tmp_path / "test.png"
    src.touch()
    data = Noop.Data(src=src, is_dry=False)
    op = Noop()

    plan = op.plan(data)
    assert plan == {}
    op.run(data, plan)
    assert src.exists()
    assert list(tmp_path.iterdir()) == [src]


def test_undo(tmp_path: Path) -> None:
    """Verify undoing nothing is always valid and also does nothing"""
    src = tmp_path / "test.png"
    entry = Journal.Entry(op="noop", src=src, ext_out={"a": 1}, op_out={}, is_dry=False)

    Noop.can_undo(entry)
    assert Noop.undo(Noop.Data(src=src, is_dry=False), entry.op_out) == {}
    assert not src.exists()


def test_journal_entry(tmp_path: Path) -> None:
    """Verify the extractor's findings are journaled under the noop op"""
    src = tmp_path / "test.png"
    src.touch()
    found = {"date_taken": "2023-07-14T10:22:31"}

    entry = Noop().prepare(Noop.Data(src=src, is_dry=False), ext_data=found)()

    assert entry.op == "noop"
    assert entry.src == src
    assert entry.ext_out == found
    assert entry.op_out == {}


def test_from_cfg() -> None:
    """Verify `op: noop` in a config builds the op"""
    assert isinstance(Noop.from_cfg({"op": "noop"}), Noop)
