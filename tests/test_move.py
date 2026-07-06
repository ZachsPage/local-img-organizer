from pathlib import Path

import pytest

from local_img_organizer.interfaces import Journal, Operation
from local_img_organizer.ops.move import Move


def _entry(src: Path, op_out: dict) -> Journal.Entry:
    return Journal.Entry(op="move", src=src, ext_out={}, op_out=op_out)


def test_plan_not_a_file(tmp_path):
    """Test planning a move for a src that does not exist raises"""
    op = Move(cfg=Move.Cfg(op="move", subdir_name="cats"))
    with pytest.raises(ValueError, match="is not a file"):
        op.plan(Operation.Data(src=tmp_path / "missing.png", is_dry=False))


def test_plan_already_in_subdir(tmp_path):
    """Test planning a move for a file already in the target subdir is a no-op"""
    src = tmp_path / "cats" / "a.png"
    src.parent.mkdir()
    src.touch()
    op = Move(cfg=Move.Cfg(op="move", subdir_name="cats"))
    assert op.plan(Operation.Data(src=src, is_dry=False)) == {}


def test_plan_dest_exists(tmp_path):
    """Test planning a move raises if the destination is already occupied"""
    src = tmp_path / "a.png"
    src.touch()
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    op = Move(cfg=Move.Cfg(op="move", subdir_name="cats"))
    with pytest.raises(ValueError, match="already exists"):
        op.plan(Operation.Data(src=src, is_dry=False))


def test_plan_returns_dest(tmp_path):
    """Test planning a move returns the intended destination"""
    src = tmp_path / "a.png"
    src.touch()
    op = Move(cfg=Move.Cfg(op="move", subdir_name="cats"))
    planned = op.plan(Operation.Data(src=src, is_dry=False))
    assert planned == {"dest": str(tmp_path / "cats" / "a.png")}


def test_run_moves_file(tmp_path):
    """Test running a move relocates the file to the planned destination"""
    src = tmp_path / "a.png"
    src.touch()
    op = Move(cfg=Move.Cfg(op="move", subdir_name="cats"))
    planned = op.plan(Operation.Data(src=src, is_dry=False))
    op.run(Operation.Data(src=src, is_dry=False), planned)
    assert not src.exists()
    assert (tmp_path / "cats" / "a.png").exists()


def test_run_noop_for_empty_planned(tmp_path):
    """Test running a move with an empty planned dict does nothing"""
    src = tmp_path / "a.png"
    src.touch()
    op = Move(cfg=Move.Cfg(op="move", subdir_name="cats"))
    op.run(Operation.Data(src=src, is_dry=False), {})
    assert src.exists()


def test_can_undo_noop_entry(tmp_path):
    """Test an entry with no op_out (already-in-subdir case) is always undoable"""
    Move.can_undo(_entry(tmp_path / "a.png", {}))


def test_can_undo_valid(tmp_path):
    """Test a valid undo: src absent, dest present"""
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    Move.can_undo(_entry(tmp_path / "a.png", {"dest": str(dest)}))


def test_can_undo_src_exists(tmp_path):
    """Test undo is rejected if the original src already exists (would overwrite it)"""
    src = tmp_path / "a.png"
    src.touch()
    dest = tmp_path / "cats" / "a.png"
    with pytest.raises(ValueError, match="already exists"):
        Move.can_undo(_entry(src, {"dest": str(dest)}))


def test_can_undo_dest_missing(tmp_path):
    """Test undo is rejected if the moved file's dest no longer exists"""
    dest = tmp_path / "cats" / "a.png"
    with pytest.raises(ValueError, match="missing"):
        Move.can_undo(_entry(tmp_path / "a.png", {"dest": str(dest)}))


def test_can_undo_failed_run_entry(tmp_path):
    """Test an entry from a failed run (op_out holds an error, no dest) is a no-op to undo"""
    Move.can_undo(_entry(tmp_path / "a.png", {"error": "something went wrong"}))


def test_undo_moves_file_back(tmp_path):
    """Test undoing a move relocates the file back to its original src"""
    src = tmp_path / "a.png"
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    Move.undo(Operation.Data(src=src, is_dry=False), {"dest": str(dest)})
    assert src.exists()
    assert not dest.exists()


def test_undo_noop_for_empty_planned(tmp_path):
    """Test undoing an empty planned dict does nothing"""
    src = tmp_path / "a.png"
    Move.undo(Operation.Data(src=src, is_dry=False), {})
    assert not src.exists()


def test_undo_noop_for_failed_run(tmp_path):
    """Test undoing a failed-run entry (op_out holds an error, no dest) does nothing"""
    src = tmp_path / "a.png"
    Move.undo(Operation.Data(src=src, is_dry=False), {"error": "something went wrong"})
    assert not src.exists()
