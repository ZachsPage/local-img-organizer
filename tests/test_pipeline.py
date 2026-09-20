"""Covers a real multi-op chain - the ops only see the shared ImgFile, never each other"""

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pytest
from PIL import Image

from local_img_organizer.extractors.metadata import Metadata
from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Extractor, Operation, run_ops, run_undos
from local_img_organizer.ops.move import Move
from local_img_organizer.ops.rename import Rename
from tests.stubs import StubJournal

_MTIME = "2021-09-15T20:17:56-04:00"


@dataclass
class _Extractor(Extractor):
    """Feeds a fixed file_modified to every image so `rename` has a date to work from"""

    class Cfg(Extractor.Cfg):
        pass

    ops: list[Operation] = field(default_factory=list)

    @override
    def run(
        self, images: list[ImgFile], *, is_dry: bool
    ) -> Generator[tuple[Operation, Operation.Data]]:
        for img in images:
            data = Operation.Data(src=img, is_dry=is_dry, ext={"file_modified": _MTIME})
            for op in self.ops:
                yield op, data


def _chain() -> list[Operation]:
    return [Rename(cfg=Rename.Cfg(op="rename")), Move(cfg=Move.Cfg(op="move", subdir_name="dated"))]


def _run(tmp_path: Path, *, is_dry: bool = False) -> StubJournal:
    journal = StubJournal()
    run_ops(tmp_path, journal, [_Extractor(ops=_chain())], is_dry=is_dry)
    return journal


def test_rename_then_move(tmp_path):
    """Test the move plans against the renamed file, not the name the run started with"""
    (tmp_path / "IMG_3052.jpg").touch()

    journal = _run(tmp_path)
    renamed, moved = journal.entries

    assert renamed.src == tmp_path / "IMG_3052.jpg"
    assert renamed.op_out["dest"] == str(tmp_path / "IMG20210915201756.jpg")
    # The move's src is where rename left the file, and both entries share one img_uid
    assert moved.src == tmp_path / "IMG20210915201756.jpg"
    assert moved.op_out == {"dest": str(tmp_path / "dated" / "IMG20210915201756.jpg")}
    assert renamed.img_uid == moved.img_uid
    assert (tmp_path / "dated" / "IMG20210915201756.jpg").exists()
    assert list(tmp_path.glob("*.jpg")) == []


def test_dry_run_plans_the_chain_without_touching_anything(tmp_path):
    """Test a dry run reports both steps, which needs the planned path without the file moving"""
    src = tmp_path / "IMG_3052.jpg"
    src.touch()

    journal = _run(tmp_path, is_dry=True)

    assert [e.op_out["dest"] for e in journal.entries] == [
        str(tmp_path / "IMG20210915201756.jpg"),
        str(tmp_path / "dated" / "IMG20210915201756.jpg"),
    ]
    assert src.exists()
    assert not (tmp_path / "dated").exists()


def test_undo_unwinds_the_chain(tmp_path):
    """Test undoing a chain restores the original name & location"""
    src = tmp_path / "IMG_3052.jpg"
    src.touch()
    journal = _run(tmp_path)

    run_undos(journal, source=Path("unused"))

    assert src.exists()
    assert not (tmp_path / "dated").exists()
    assert [e.op for e in journal.entries[2:]] == ["move", "rename"]


def test_second_image_reuses_a_name_the_chain_freed(tmp_path):
    """Test a name is only claimed once per run, even after the chain moves that file away

    Both images share one mtime, so they want the same name - the 2nd takes the collision suffix
    rather than the name the 1st vacated, which keeps the plan honest during a dry run.
    """
    for name in ("IMG_3052.jpg", "IMG_3067.jpg"):
        (tmp_path / name).touch()

    journal = _run(tmp_path)

    assert [Path(e.op_out["dest"]).name for e in journal.entries] == [
        "IMG20210915201756.jpg",
        "IMG20210915201756.jpg",
        "IMG20210915201756_1.jpg",
        "IMG20210915201756_1.jpg",
    ]


def test_failed_run_mid_chain_leaves_an_undoable_journal(tmp_path):
    """Test a move that fails after a rename stops the run, with the rename still undoable"""
    src = tmp_path / "IMG_3052.jpg"
    src.touch()
    # Occupy the move's destination after planning is impossible to time, so block the subdir
    # itself - mkdir then fails the way a permission problem would
    (tmp_path / "dated").write_text("not a directory")

    journal = StubJournal()
    with pytest.raises(FileExistsError):
        run_ops(tmp_path, journal, [_Extractor(ops=_chain())])

    assert [e.op for e in journal.entries] == ["rename", "move", "move"]
    assert "error" in journal.entries[-1].op_out

    run_undos(journal, source=Path("unused"))
    assert src.exists()


def test_dry_run_later_extractor_reads_where_the_file_still_is(tmp_path):
    """Test an extractor after a planned move reads the real file, not the path ops plan to use

    A dry run moves nothing, so `ImgFile.path` (where the chain plans to leave the file) and
    `disk_path` (where its bytes are) diverge for every extractor after the first.
    """
    src = tmp_path / "IMG_3052.jpg"
    Image.new("RGB", (1, 1)).save(src)

    journal = StubJournal()
    mover = _Extractor(ops=[Move(cfg=Move.Cfg(op="move", subdir_name="dated"))])
    metadata = Metadata.from_cfg({"lookup_location": False})
    run_ops(tmp_path, journal, [mover, metadata], is_dry=True)

    moved, found = journal.entries
    assert moved.op_out == {"dest": str(tmp_path / "dated" / "IMG_3052.jpg")}
    assert "error" not in found.ext_out
    assert "file_modified" in found.ext_out
