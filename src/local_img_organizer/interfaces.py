"""Defines interfaces for implementations"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class Operations(StrEnum):
    """Names of Operations that can be ran"""

    RENAME = auto()
    MOVE = auto()
    TAG = auto()


type ExtOut = dict[str, Any]  # Extractor data out - fed into Operation
type OpOut = dict[str, Any]  # Operation data out - used to indicate what was done


class Journal(ABC):
    """Journal to track what has been done for debugging & undoing"""

    @dataclass
    class Entry:
        """Needed data for each journal entry"""

        op: Operations
        src: Path
        op_in: ExtOut
        op_out: OpOut

    @abstractmethod
    def log(self, entry: Entry) -> None:
        """Write out the entry"""

    @abstractmethod
    def read(self) -> Generator[Entry]:
        """Return each entry"""


class Operation(ABC):
    """Runnable operation to execute an action based on Extractor data"""

    op_type: Operations  # subclasses must set this

    @dataclass
    class Data:
        """Input data to run with"""

        src: Path
        is_dry: bool  # do not actually execute
        ext_data: ExtOut

    @abstractmethod
    def plan(self, data: Data) -> OpOut:
        """Compute and return what this operation would do"""

    @abstractmethod
    def run(self, data: Data, planned: OpOut) -> None:
        """Execute the planned operation's side effects"""

    @abstractmethod
    def undo(self, og_data: Data, og_out: OpOut) -> None:
        """Reverse a previously executed operation"""

    def _safe(
        self,
        action: Callable[[], None],
        data: Data,
        planned: OpOut,
    ) -> OpOut:
        """Wrap an action with error handling, return the OpOut or an error dict"""
        try:
            action()
        except Exception as ex:  # noqa: BLE001
            _log.exception(f"Error for {self.op_type} - in: {data}, out: {ex}")
            return {"error": str(ex)}
        return planned

    def prepare(self, data: Data) -> Callable[[], Journal.Entry]:
        """Return callable that will plan & run the operation, returning a Journal.Entry"""

        def run_get_entry() -> Journal.Entry:
            planned = self.plan(data)
            op_out = (
                planned
                if data.is_dry
                else self._safe(lambda: self.run(data, planned), data, planned)
            )
            return Journal.Entry(
                op=self.op_type,
                src=data.src,
                op_in=data.ext_data,
                op_out=op_out,
            )

        return run_get_entry

    def prepare_undo(self, entry: Journal.Entry) -> Callable[[], Journal.Entry]:
        """Return callable that will undo a previously journaled operation"""

        def undo_get_entry() -> Journal.Entry:
            og_data = Operation.Data(src=entry.src, is_dry=False, ext_data=entry.op_in)
            op_out = self._safe(lambda: self.undo(og_data, entry.op_out), og_data, entry.op_out)
            return Journal.Entry(
                op=self.op_type,
                src=entry.src,
                op_in=entry.op_in,
                op_out=op_out,
            )

        return undo_get_entry

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify subclass contract at definition time"""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "op_type", None):
            raise TypeError(f"{cls.__name__} must define op_type")


@dataclass
class Extractor(ABC):
    """Extracts data to be fed into an Operation"""

    ops: list[Operation]  # configured operations

    @abstractmethod
    def run(self, img_dir: Path, *, is_dry: bool) -> Generator[Callable[[], Journal.Entry]]:
        """Run the extractor to get all of its metadata, then for each of its assigned Operations,
        yield the prepared op for each located img
        """


def run_all(
    img_dir: Path,
    journal: Journal,
    extractors: list[Extractor],
    *,
    is_dry: bool = False,
) -> None:
    """Top level function to tie all the interfaces together
    :param img_dir: Dir with the images to run on (recursively)
    :param journal: Journal implementation
    :param extractors: Extractors to set up & execute Operations for
    :param is_dry: Do not actually execute operations
    """
    for ext in extractors:
        for op in ext.run(img_dir, is_dry=is_dry):
            journal.log(op())
