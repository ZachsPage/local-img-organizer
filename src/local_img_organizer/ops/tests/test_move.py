from pathlib import Path

import pytest

from local_img_organizer.interfaces import Operation
from local_img_organizer.ops.move import Move

type Data = Operation.Data


def _move(subdir_name: str) -> Move:
    return Move(cfg=Move.Cfg(op="move", subdir_name=subdir_name))


def test_roundtrip(tmp_path: Path) -> None:
    """Verifies the plan -> run -> undo round trip"""
    src = tmp_path / "test.png"
    src.touch()
    subdir_name = "subdir"
    op = _move(subdir_name)
    data_in = Move.Data(src=src, is_dry=False)
    # Verify expected path but is not moved yet
    plan = op.plan(data_in)
    exp_dest = Path(plan["dest"])
    assert subdir_name in [str(p.name) for p in exp_dest.parents]
    assert not exp_dest.exists()
    # Verify moved
    op.run(data_in, plan)
    assert exp_dest.exists()
    assert not src.exists()
    # Verify move is undone, and undo reports the file's real post-undo location
    undo_out = op.undo(data_in, plan)
    assert not exp_dest.exists()
    assert src.exists()
    assert undo_out == {"dest": str(src)}


def test_undo_removes_emptied_subdir(tmp_path: Path) -> None:
    """Verify undo deletes the subdir it moved into if undo leaves it empty"""
    src = tmp_path / "test.png"
    src.touch()
    op = _move("subdir")
    data_in = Move.Data(src=src, is_dry=False)
    plan = op.plan(data_in)
    subdir = Path(plan["dest"]).parent
    op.run(data_in, plan)
    op.undo(data_in, plan)
    assert not subdir.exists()


def test_undo_keeps_nonempty_subdir(tmp_path: Path) -> None:
    """Verify undo leaves the subdir if other files remain in it"""
    src = tmp_path / "test.png"
    src.touch()
    op = _move("subdir")
    data_in = Move.Data(src=src, is_dry=False)
    plan = op.plan(data_in)
    subdir = Path(plan["dest"]).parent
    op.run(data_in, plan)
    (subdir / "other.png").touch()
    op.undo(data_in, plan)
    assert subdir.exists()


def test_noop(tmp_path: Path) -> None:
    """Verify move was already done"""
    subdir_name = "subdir"
    src = tmp_path / subdir_name / "test.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    op = _move(subdir_name)
    data_in = Move.Data(src=src, is_dry=False)
    # Verify nothing to do - already in subdir - rest of calls also do nothing
    plan = op.plan(data_in)
    assert not plan
    op.run(data_in, plan)
    assert op.undo(data_in, plan) == {}


def test_failures(tmp_path: Path) -> None:
    """Verify any failure paths"""
    # Verify input is not a file
    with pytest.raises(ValueError, match="is not a file"):
        _move("test").plan(Move.Data(src=tmp_path, is_dry=False))
