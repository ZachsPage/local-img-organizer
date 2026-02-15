"""Defines interfaces for implementations"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any


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
    def run(self, data: Data) -> OpOut:
        """Return OpOut - which should have data around what was done by this Operation"""

    @abstractmethod
    def undo(self, og_data: Data, og_data_out: OpOut) -> OpOut:
        """Return what was done - is fed in what was originally done as indicated by the journal"""

    def prepare(self, data: Data) -> Callable[[], Journal.Entry]:
        """Return callable that will run the operation & return its data in a Journal.Entry"""

        def run_get_entry() -> Journal.Entry:
            # TODO - need error handling here - if 'run' fails, how should that be handled - log
            # error & skip journal, or journal as a failure?
            return Journal.Entry(
                op=self.op_type,
                src=data.src,
                op_in=data.ext_data,
                op_out=self.run(data),
            )

        return run_get_entry

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify subclass contract at definition time"""
        super().__init_subclass__(**kwargs)
        if not cls.op_type:
            raise TypeError(f"{cls.__name__} must define op_type")


@dataclass
class Extractor(ABC):
    """Extracts data to be fed into an Operation"""

    ops: list[Operation]  # configured operations

    @abstractmethod
    def run(self, img_dir: Path) -> Generator[Callable[[], Journal.Entry]]:
        """Run the extractor to get all of its metadata, then for each of its assigned Operations,
        yield the prepared op for each located img
        """


def run(img_dir: Path, journal: Journal, extractors: list[Extractor]) -> None:
    """Top level function to tie all the interfaces together
    :param img_dir: Dir with the images to run on (recursively)
    :param journal: Journal implementation
    :param extractors: Extractors to set up & execute Operations for
    """
    for ext in extractors:
        for op in ext.run(img_dir):
            journal.log(op())
