import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, override

import pytest

from local_img_organizer.interfaces import Extractor, Journal, Operation, OpOut, run_ops, run_undos

_log = logging.getLogger(__name__)


@dataclass
class StubJournal(Journal):
    entries: list[Journal.Entry] = field(default_factory=list)

    @override
    def log(self, entry: Journal.Entry) -> None:
        _log.info(entry)
        self.entries.append(entry)

    @override
    def read(self, source: Path | None = None) -> Generator[Journal.Entry]:
        yield from self.entries


class StubOperation(Operation):
    class Cfg(Operation.Cfg):
        pass

    @override
    def plan(self, data: Operation.Data) -> OpOut:
        return {"from": str(data.src), "to": f"{data.src}_renamed"}

    @override
    def run(self, data: Operation.Data, planned: OpOut) -> None:
        _log.info("would rename %s -> %s", planned["from"], planned["to"])

    @classmethod
    @override
    def can_undo(cls, entry: Journal.Entry) -> None:
        pass

    @classmethod
    @override
    def undo(cls, og_data: Operation.Data, og_out: OpOut) -> OpOut:
        _log.info("would undo %s -> %s", og_out["to"], og_out["from"])
        return {"from": og_out["to"], "to": og_out["from"]}


@dataclass
class StubExtractor(Extractor):
    class Cfg(Extractor.Cfg):
        pass

    ops: list[Operation] = field(default_factory=list)
    label: ClassVar[str] = "test_ext_label"

    @override
    def run(self, img_dir: Path, *, is_dry: bool) -> Generator[Callable[[], Journal.Entry]]:
        for file in img_dir.iterdir():
            data = Operation.Data(src=file, is_dry=is_dry)
            for op in self.ops:
                yield op.prepare(data, ext_data={"label": self.label})


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
        assert entry.op_out == {"from": str(entry.src), "to": f"{entry.src}_renamed"}
        assert entry.is_dry is False

    # Verify undo - op_out should reflect the reversal, not just echo the original entry
    for entry in entries:
        undo_entry = op.prepare_undo(entry)()
        assert undo_entry.op == "stuboperation"
        assert undo_entry.op_out == {"from": entry.op_out["to"], "to": entry.op_out["from"]}


def test_run_ops_dry_run_tags_entries(tmp_path):
    """Test entries produced during a dry run are tagged is_dry so they can't be undone later"""
    (tmp_path / "test_file.png").touch()

    journal = StubJournal()
    op = StubOperation()
    run_ops(tmp_path, journal, [StubExtractor(ops=[op])], is_dry=True)
    entries = list(journal.read())

    assert len(entries) == 1
    assert entries[0].is_dry is True


def test_run_ops_resolves_relative_img_dir(tmp_path, monkeypatch):
    """Test run_ops resolves a relative img_dir so journaled entries carry absolute paths"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "test_file.png").touch()
    monkeypatch.chdir(tmp_path)

    journal = StubJournal()
    op = StubOperation()
    run_ops(Path("sub"), journal, [StubExtractor(ops=[op])])
    entries = list(journal.read())

    assert len(entries) == 1
    assert entries[0].src.is_absolute()
    assert entries[0].src == tmp_path / "sub" / "test_file.png"


def test_bad_op_no_cfg():
    """Test catching if a new Operation does not define a Cfg inner class"""
    with pytest.raises(TypeError, match="must define a Cfg"):

        class BadOperation(Operation):
            pass


def test_bad_op_run():
    """Test avoiding bubbling up operation exceptions, but ensure they are journaled"""

    class FailingOp(Operation):
        class Cfg(Operation.Cfg):
            pass

        @override
        def plan(self, data: Operation.Data) -> OpOut:
            return {"planned": "something"}

        @override
        def run(self, data: Operation.Data, planned: OpOut) -> None:
            msg = "something went wrong"
            raise RuntimeError(msg)

        @classmethod
        @override
        def can_undo(cls, entry: Journal.Entry) -> None:
            pass

        @classmethod
        @override
        def undo(cls, og_data: Operation.Data, og_out: OpOut) -> OpOut:
            return {}

    data = Operation.Data(src=Path("fake.png"), is_dry=False)
    entry = FailingOp().prepare(data, ext_data={"label": "x"})()

    # Verify error was captured, not raised
    assert entry.op_out == {"error": "something went wrong"}


# run_undos resolves op classes dynamically via `local_img_organizer.ops.<entry.op>`, so these
# tests use the real "move" op rather than a test-local stub.


def test_run_undos_collects_all_invalid_entries(tmp_path):
    """Test run_undos validates every entry before raising, collecting all failure reasons
    together (rather than stopping at the first bad entry) - this is what defer_exceptions buys us
    """
    blocked_src = tmp_path / "already_here.png"
    blocked_src.touch()  # src already exists -> Move.can_undo rejects it
    entries = [
        Journal.Entry(
            op="move",
            src=blocked_src,
            ext_out={},
            op_out={"dest": str(tmp_path / "cats" / "already_here.png")},
            is_dry=False,
        ),
        Journal.Entry(
            op="bogus_op", src=tmp_path / "other.png", ext_out={}, op_out={}, is_dry=False
        ),
    ]
    journal = StubJournal(entries=list(entries))

    with pytest.raises(RuntimeError) as exc_info:
        run_undos(journal, source=Path("unused"))

    message = str(exc_info.value)
    assert "already exists" in message
    assert "Unknown op: 'bogus_op'" in message
    # Neither entry was valid, so nothing should have been undone or logged
    assert journal.entries == entries


def test_run_undos_rejects_dry_run_entry(tmp_path):
    """Test run_undos refuses to undo an entry that was only planned, never executed"""
    entry = Journal.Entry(
        op="move",
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

    entry = Journal.Entry(
        op="move", src=src, ext_out={"category": "cats"}, op_out={"dest": str(dest)}, is_dry=False
    )
    journal = StubJournal(entries=[entry])
    before = len(journal.entries)

    run_undos(journal, source=Path("unused"))

    assert src.exists()
    assert not dest.exists()
    assert len(journal.entries) == before + 1
    undo_entry = journal.entries[-1]
    assert undo_entry.op == "move"
    # op_out should reflect where the file actually ended up (src), not echo the original entry
    assert undo_entry.op_out == {"dest": str(src)}


def test_run_undos_dry_run_does_not_execute(tmp_path):
    """Test run_undos in dry mode validates & logs but does not actually move any files"""
    dest = tmp_path / "cats" / "a.png"
    dest.parent.mkdir()
    dest.touch()
    src = tmp_path / "a.png"

    entry = Journal.Entry(op="move", src=src, ext_out={}, op_out={"dest": str(dest)}, is_dry=False)
    journal = StubJournal(entries=[entry])

    run_undos(journal, source=Path("unused"), is_dry=True)

    assert not src.exists()
    assert dest.exists()
    assert journal.entries[-1].op_out == entry.op_out


def test_run_undos_no_journal_files():
    """Test run_undos raises if no source is given and no journal files are available"""
    journal = StubJournal()
    with pytest.raises(RuntimeError, match="No journal files available"):
        run_undos(journal, source=None)
