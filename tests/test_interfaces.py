import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, override

import pytest

from local_img_organizer.interfaces import Extractor, Journal, Operation, Operations, OpOut, run_all

_log = logging.getLogger(__name__)


@dataclass
class StubJournal(Journal):
    entries: list[Journal.Entry] = field(default_factory=list)

    @override
    def log(self, entry: Journal.Entry) -> None:
        _log.info(entry)
        self.entries.append(entry)

    @override
    def read(self) -> Generator[Journal.Entry]:
        yield from self.entries


class StubOperation(Operation):
    op_type = Operations.RENAME

    @override
    def plan(self, data: Operation.Data) -> OpOut:
        return {"from": str(data.src), "to": f"{data.src}_renamed"}

    @override
    def run(self, data: Operation.Data, planned: OpOut) -> None:
        _log.info("would rename %s -> %s", planned["from"], planned["to"])

    @override
    def undo(self, og_data: Operation.Data, og_out: OpOut) -> None:
        _log.info("would undo %s -> %s", og_out["to"], og_out["from"])


class StubExtractor(Extractor):
    label: ClassVar[str] = "test_ext_label"

    @override
    def run(self, img_dir: Path, *, is_dry: bool) -> Generator[Callable[[], Journal.Entry]]:
        for file in img_dir.iterdir():
            op_in = Operation.Data(src=file, is_dry=is_dry, ext_data={"label": self.label})
            for op in self.ops:
                yield op.prepare(op_in)


def test_run_all(tmp_path):
    """Test set up & running run_all"""
    files = [tmp_path / f"test_file_{i}.png" for i in range(5)]
    for f in files:
        f.touch()

    journal = StubJournal()
    op = StubOperation()
    run_all(tmp_path, journal, [StubExtractor(ops=[op])])
    entries = list(journal.read())

    # Verify all files were processed
    assert len(entries) == len(files)

    # Verify entry data
    for entry in entries:
        assert entry.op == Operations.RENAME
        assert entry.src in files
        assert entry.op_in == {"label": StubExtractor.label}
        assert entry.op_out == {"from": str(entry.src), "to": f"{entry.src}_renamed"}

    # Verify undo
    for entry in entries:
        undo_entry = op.prepare_undo(entry)()
        assert undo_entry.op == Operations.RENAME
        assert undo_entry.op_out == entry.op_out


def test_bad_op_no_name():
    """Test catching if a new Operation does not define op_type"""
    with pytest.raises(TypeError, match="must define op_type"):

        class BadOperation(Operation):
            pass


def test_bad_op_run():
    """Test avoiding bubbling up operation exceptions, but ensure they are journaled"""

    class FailingOp(Operation):
        op_type = Operations.RENAME

        @override
        def plan(self, data: Operation.Data) -> OpOut:
            return {"planned": "something"}

        @override
        def run(self, data: Operation.Data, planned: OpOut) -> None:
            msg = "something went wrong"
            raise RuntimeError(msg)

        @override
        def undo(self, og_data: Operation.Data, og_out: OpOut) -> None:
            pass

    data = Operation.Data(src=Path("fake.png"), is_dry=False, ext_data={"label": "x"})
    entry = FailingOp().prepare(data)()

    # Verify error was captured, not raised
    assert entry.op_out == {"error": "something went wrong"}
