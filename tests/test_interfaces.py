import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, override

import pytest

from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Extractor, Journal, Operation, OpOut, run_ops, run_undos
from local_img_organizer.ops.noop import Noop
from tests.stubs import StubJournal

_log = logging.getLogger(__name__)


class StubOperation(Operation):
    class Cfg(Operation.Cfg):
        pass

    @override
    def plan(self, data: Operation.Data) -> OpOut:
        return data.src.plan_path_change(Path(f"{data.src.path}_renamed"))

    @override
    def run(self, data: Operation.Data, planned: OpOut) -> None:
        _log.info("would rename %s -> %s", data.src.path, planned["dest"])

    @classmethod
    @override
    def plan_undo(cls, entry: Journal.Entry, img: ImgFile) -> OpOut:
        return img.plan_move_back(entry)

    @classmethod
    @override
    def undo(cls, img: ImgFile, planned: OpOut) -> None:
        _log.info("would undo %s -> %s", img.path, planned["dest"])


@dataclass
class StubExtractor(Extractor):
    class Cfg(Extractor.Cfg):
        pass

    ops: list[Operation] = field(default_factory=list)
    label: ClassVar[str] = "test_ext_label"

    @override
    def run(
        self, images: list[ImgFile], *, is_dry: bool
    ) -> Generator[tuple[Operation, Operation.Data]]:
        for img in images:
            data = Operation.Data(src=img, is_dry=is_dry, ext={"label": self.label})
            for op in self.ops:
                yield op, data


def _entry(src: Path, op_out: dict, *, op: str = "move", img_uid: str = "img") -> Journal.Entry:
    return Journal.Entry(op=op, img_uid=img_uid, src=src, ext_out={}, op_out=op_out, is_dry=False)


def test_run_ops(tmp_path):
    """Test set up & running run_ops"""
    files = [tmp_path / f"test_file_{i}.png" for i in range(5)]
    for f in files:
        f.touch()

    journal = StubJournal()
    op = StubOperation()
    run_ops(tmp_path, journal, [StubExtractor(ops=[op])])
    entries = list(journal.read())

    # Verify all files were processed
    assert len(entries) == len(files)

    # Verify entry data
    for entry in entries:
        assert entry.op == "stuboperation"
        assert entry.src in files
        assert entry.ext_out == {"label": StubExtractor.label}
        assert entry.op_out == {"dest": f"{entry.src}_renamed"}
        assert entry.is_dry is False

    # Verify each image got its own id for undo to group a chain by
    assert len({e.img_uid for e in entries}) == len(files)


def test_run_ops_chains_ops_on_one_image(tmp_path):
    """Test a 2nd op plans against where the 1st op puts the file, and both share an img_uid"""
    (tmp_path / "a.png").touch()

    journal = StubJournal()
    run_ops(tmp_path, journal, [StubExtractor(ops=[StubOperation(), StubOperation()])])
    first, second = journal.entries

    assert first.src == tmp_path / "a.png"
    assert second.src == Path(first.op_out["dest"])
    assert first.img_uid == second.img_uid


def test_run_ops_journals_before_running(tmp_path):
    """Test the entry is journaled before the operation executes, so a crash cannot lose it"""
    (tmp_path / "a.png").touch()

    class LogChecker(StubJournal):
        @override
        def log(self, entry: Journal.Entry) -> None:
            super().log(entry)
            assert not ran, "entry must be journaled before the op runs"

    ran = False

    class SlowOp(StubOperation):
        class Cfg(Operation.Cfg):
            pass

        @override
        def run(self, data: Operation.Data, planned: OpOut) -> None:
            nonlocal ran
            ran = True

    run_ops(tmp_path, LogChecker(), [StubExtractor(ops=[SlowOp()])])
    assert ran


def test_run_ops_skips_entries_without_action(tmp_path):
    """Test run_ops only journals entries whose op did something, plus noop entries"""
    (tmp_path / "test_file.png").touch()

    class SkipOp(StubOperation):
        class Cfg(Operation.Cfg):
            pass

        @override
        def plan(self, data: Operation.Data) -> OpOut:
            return {}

        @override
        def run(self, data: Operation.Data, planned: OpOut) -> None:
            pass

    journal = StubJournal()
    run_ops(tmp_path, journal, [StubExtractor(ops=[SkipOp(), Noop()])])
    assert [e.op for e in journal.read()] == ["noop"]


def test_run_ops_dry_run_tags_entries(tmp_path):
    """Test entries produced during a dry run are tagged is_dry so they can't be undone later"""
    (tmp_path / "test_file.png").touch()

    journal = StubJournal()
    run_ops(tmp_path, journal, [StubExtractor(ops=[StubOperation()])], is_dry=True)
    entries = list(journal.read())

    assert len(entries) == 1
    assert entries[0].is_dry is True


def test_run_ops_resolves_relative_img_dir(tmp_path, monkeypatch):
    """Test run_ops resolves a relative img_dir so journaled entries carry absolute paths"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "test_file.png").touch()
    monkeypatch.chdir(tmp_path)

    journal = StubJournal()
    run_ops(Path("sub"), journal, [StubExtractor(ops=[StubOperation()])])
    entries = list(journal.read())

    assert len(entries) == 1
    assert entries[0].src.is_absolute()
    assert entries[0].src == tmp_path / "sub" / "test_file.png"


def test_bad_op_no_cfg():
    """Test catching if a new Operation does not define a Cfg inner class"""
    with pytest.raises(TypeError, match="must define a Cfg"):

        class BadOperation(Operation):
            pass


def test_failed_run_journals_error_and_stops(tmp_path):
    """Test a failed run journals the error, then stops the run rather than carrying on"""
    for name in ("a.png", "b.png"):
        (tmp_path / name).touch()

    class FailingOp(StubOperation):
        class Cfg(Operation.Cfg):
            pass

        @override
        def run(self, data: Operation.Data, planned: OpOut) -> None:
            msg = "something went wrong"
            raise RuntimeError(msg)

    journal = StubJournal()
    with pytest.raises(RuntimeError, match="something went wrong"):
        run_ops(tmp_path, journal, [StubExtractor(ops=[FailingOp()])])

    # The planned entry, then the error - and nothing for the 2nd image
    assert [e.op_out for e in journal.entries] == [
        {"dest": f"{tmp_path / 'a.png'}_renamed"},
        {"error": "something went wrong"},
    ]


# run_undos resolves op classes dynamically via `local_img_organizer.ops.<entry.op>`, so these
# tests use the real "move" op rather than a test-local stub.


def test_run_undos_collects_all_invalid_entries(tmp_path):
    """Test run_undos reports every entry it cannot undo, rather than stopping at the first"""
    blocked_src = tmp_path / "already_here.png"
    blocked_src.touch()
    blocked_dest = tmp_path / "cats" / "already_here.png"
    blocked_dest.parent.mkdir()
    blocked_dest.touch()  # src & dest both exist -> cannot tell whether the move ran
    entries = [
        _entry(blocked_src, {"dest": str(blocked_dest)}),
        _entry(tmp_path / "other.png", {"dest": str(tmp_path / "x.png")}, op="bogus_op"),
    ]
    journal = StubJournal(entries=list(entries))

    with pytest.raises(RuntimeError) as exc_info:
        run_undos(journal, source=Path("unused"))

    message = str(exc_info.value)
    assert "cannot tell whether the op ran" in message
    assert "Unknown op: 'bogus_op'" in message
    # Neither entry was valid, so nothing should have been undone or logged
    assert journal.entries == entries


def test_run_undos_rejects_dry_run_entry(tmp_path):
    """Test run_undos refuses to undo an entry that was only planned, never executed"""
    entry = Journal.Entry(
        op="move",
        img_uid="img",
        src=tmp_path / "a.png",
        ext_out={},
        op_out={"dest": str(tmp_path / "cats" / "a.png")},
        is_dry=True,
    )
    journal = StubJournal(entries=[entry])

    with pytest.raises(RuntimeError, match="dry run"):
        run_undos(journal, source=Path("unused"))

    # Nothing should have been undone or logged
    assert journal.entries == [entry]


def test_run_undos_undoes_valid_entries(tmp_path):
    """Test run_undos executes the undo for each valid entry and logs the result"""
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    src = tmp_path / "a.png"

    entry = _entry(src, {"dest": str(dest)})
    journal = StubJournal(entries=[entry])

    run_undos(journal, source=Path("unused"))

    assert src.exists()
    assert not dest.exists()
    undo_entry = journal.entries[-1]
    assert undo_entry.op == "move"
    # op_out should reflect where the file actually ended up (src), not echo the original entry
    assert undo_entry.op_out == {"dest": str(src)}


def test_run_undos_unwinds_a_chain_in_reverse(tmp_path):
    """Test a chain undoes newest first, since its earlier entries name paths that only exist
    again once the later ones are undone
    """
    final = tmp_path / "cats" / "IMG20240101120000.jpg"
    final.parent.mkdir()
    final.touch()
    original = tmp_path / "a.jpg"
    renamed = tmp_path / "IMG20240101120000.jpg"

    journal = StubJournal(
        entries=[
            _entry(original, {"dest": str(renamed)}, op="rename"),
            _entry(renamed, {"dest": str(final)}),
        ]
    )
    run_undos(journal, source=Path("unused"))

    assert original.exists()
    assert not final.exists()
    assert not final.parent.exists()
    assert [e.op for e in journal.entries[2:]] == ["move", "rename"]


def test_run_undos_warns_for_an_entry_that_never_ran(tmp_path, caplog):
    """Test an entry whose op never ran (a failed run, or a crash) warns & undoes from where the
    file really is, rather than refusing to undo the journal
    """
    original = tmp_path / "a.jpg"
    renamed = tmp_path / "IMG20240101120000.jpg"
    renamed.touch()
    journal = StubJournal(
        entries=[
            _entry(original, {"dest": str(renamed)}, op="rename"),
            # This move was journaled, then failed - the file is still where rename left it
            _entry(renamed, {"dest": str(tmp_path / "dated" / renamed.name)}),
        ]
    )

    run_undos(journal, source=Path("unused"))

    assert "never moved to" in caplog.text
    assert original.exists()  # the rename before it still unwound
    assert [e.op for e in journal.entries[2:]] == ["rename"]


def test_run_undos_dry_run_does_not_execute(tmp_path):
    """Test run_undos in dry mode validates & logs but does not actually move any files"""
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    src = tmp_path / "a.png"

    entry = _entry(src, {"dest": str(dest)})
    journal = StubJournal(entries=[entry])

    run_undos(journal, source=Path("unused"), is_dry=True)

    assert not src.exists()
    assert dest.exists()
    # The undo is journaled as planned - putting the file back at the entry's src - but not run
    assert journal.entries[-1].op_out == {"dest": str(src)}
    assert journal.entries[-1].is_dry is True


def test_run_undos_no_journal_files():
    """Test run_undos raises if no source is given and no journal files are available"""
    journal = StubJournal()
    with pytest.raises(RuntimeError, match="No journal files available"):
        run_undos(journal, source=None)
