from pathlib import Path

import pytest

from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Journal, Operation
from local_img_organizer.ops.move import Move


def _move(subdir_name: str = "cats") -> Move:
    return Move(cfg=Move.Cfg(op="move", subdir_name=subdir_name))


def _data(src: Path) -> Operation.Data:
    return Operation.Data(src=ImgFile(path=src), is_dry=False)


def test_plan_already_in_subdir(tmp_path):
    """Test planning a move for a file already in the target subdir is a no-op"""
    src = tmp_path / "cats" / "a.png"
    src.parent.mkdir()
    src.touch()
    assert _move().plan(_data(src)) == {}


def test_plan_dest_exists(tmp_path):
    """Test planning a move raises if the destination is already occupied"""
    src = tmp_path / "a.png"
    src.touch()
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    with pytest.raises(ValueError, match="already exists"):
        _move().plan(_data(src))


def test_plan_returns_dest_and_advances_path(tmp_path):
    """Test planning returns the intended destination & moves the image's planned path there"""
    src = tmp_path / "a.png"
    src.touch()
    data = _data(src)
    assert _move().plan(data) == {"dest": str(tmp_path / "cats" / "a.png")}
    # Planned path advances for the next op in a chain, the real file has not moved yet
    assert data.src.path == tmp_path / "cats" / "a.png"
    assert data.src.disk_path == src


def test_run_moves_file(tmp_path):
    """Test running a move relocates the file to the planned destination"""
    src = tmp_path / "a.png"
    src.touch()
    op, data = _move(), _data(src)
    op.run(data, op.plan(data))
    assert not src.exists()
    assert (tmp_path / "cats" / "a.png").exists()
    assert data.src.disk_path == tmp_path / "cats" / "a.png"


def test_run_raises_if_dest_appeared(tmp_path):
    """Test a dest created between plan & run stops the move instead of overwriting it"""
    src = tmp_path / "a.png"
    src.touch()
    op, data = _move(), _data(src)
    planned = op.plan(data)
    dest = Path(planned["dest"])
    dest.parent.mkdir()
    dest.write_text("do not clobber me")
    with pytest.raises(ValueError, match="already exists"):
        op.run(data, planned)
    assert dest.read_text() == "do not clobber me"


def test_run_raises_if_src_is_gone(tmp_path):
    """Test a src that disappeared between plan & run stops the move"""
    src = tmp_path / "a.png"
    src.touch()
    op, data = _move(), _data(src)
    planned = op.plan(data)
    src.unlink()
    with pytest.raises(ValueError, match="is not a file"):
        op.run(data, planned)


def _undone(src: Path, dest: Path) -> ImgFile:
    """Return the image sitting at `dest`, with its move back to `src` planned"""
    entry = Journal.Entry(
        op="move", img_uid="img", src=src, ext_out={}, op_out={"dest": str(dest)}, is_dry=False
    )
    img = ImgFile.from_journal_for_undo(entry)
    Move.undo(img, Move.plan_undo(entry, img))
    return img


def test_roundtrip(tmp_path):
    """Test the plan -> run -> undo round trip, with undo putting the file back where it began"""
    src = tmp_path / "a.png"
    src.touch()
    op, data = _move(), _data(src)
    planned = op.plan(data)
    op.run(data, planned)
    img = _undone(src, Path(planned["dest"]))
    assert img.disk_path == src
    assert src.exists()
    assert not Path(planned["dest"]).exists()


def test_undo_removes_emptied_subdir(tmp_path):
    """Test undo deletes the subdir it moved into if undo leaves it empty"""
    src = tmp_path / "a.png"
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    _undone(src, dest)
    assert not dest.parent.exists()


def test_undo_keeps_nonempty_subdir(tmp_path):
    """Test undo leaves the subdir if other files remain in it"""
    src = tmp_path / "a.png"
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    (dest.parent / "other.png").touch()
    _undone(src, dest)
    assert dest.parent.exists()


def test_plan_undo_rejects_overwriting_the_original(tmp_path):
    """Test an undo that would clobber a file now sitting at the original name is refused"""
    src = tmp_path / "a.png"
    src.touch()
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    entry = Journal.Entry(
        op="move", img_uid="img", src=src, ext_out={}, op_out={"dest": str(dest)}, is_dry=False
    )
    with pytest.raises(ValueError, match="cannot tell whether the op ran"):
        ImgFile.from_journal_for_undo(entry)
