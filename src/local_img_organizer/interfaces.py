"""Defines interfaces for implementations"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

_log = logging.getLogger(__name__)


type ExtOut = dict[str, Any]  # Indicates what the extractor found
type OpOut = dict[str, Any]  # Indicates what the op


class Journal(ABC):
    """Journal to track what has been done for debugging & undoing"""

    @dataclass
    class Entry:
        """Needed data for each journal entry"""

        op: str
        src: Path
        ext_out: ExtOut
        op_out: OpOut

    @abstractmethod
    def log(self, entry: Entry) -> None:
        """Write out the entry"""

    @abstractmethod
    def read(self) -> Generator[Entry]:
        """Return each entry"""


class CfgModel(BaseModel):
    """Base for all Cfg inner classes — enforces strict user-facing YAML validation"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class Operation(ABC):
    """Runnable operation to execute an action based on Extractor data"""

    class Cfg(CfgModel):
        """Base config for all operations — subclasses add their own fields"""

        op: str

    @dataclass
    class Data:
        """Input data to run with"""

        src: Path
        is_dry: bool  # do not actually execute

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify subclass contract at definition time"""
        super().__init_subclass__(**kwargs)
        if "Cfg" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a Cfg inner class")

    @classmethod
    def from_cfg(cls, data: dict[str, Any]) -> Self:
        """Build an Operation from raw YAML config data - override if more complicated"""
        return cls(cfg=cls.Cfg.model_validate(data))  # type: ignore[call-arg]

    @abstractmethod
    def plan(self, data: Data) -> OpOut:
        """Compute and return what this operation would do"""

    @abstractmethod
    def run(self, data: Data, planned: OpOut) -> None:
        """Execute the planned operation's side effects"""

    @abstractmethod
    def undo(self, og_data: Data, og_planned: OpOut) -> None:
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
            _log.exception(f"Error for {type(self).__name__} - in: {data}, out: {ex}")
            return {"error": str(ex)}
        return planned

    def prepare(self, data: Data, ext_data: ExtOut | None = None) -> Callable[[], Journal.Entry]:
        """Return callable that will plan & run the operation, returning a Journal.Entry"""

        def run_get_entry() -> Journal.Entry:
            planned = self.plan(data)
            op_out = (
                planned
                if data.is_dry
                else self._safe(lambda: self.run(data, planned), data, planned)
            )
            return Journal.Entry(
                op=type(self).__name__.lower(),
                src=data.src,
                ext_out=ext_data or {},
                op_out=op_out,
            )

        return run_get_entry

    # TODO: Need a top-level run_undo() to mirror run_all() — reads a run journal, calls
    # prepare_undo(entry) for each entry, logs results to a separate undo journal.
    # The journal stays agnostic (just sees log(entry) calls either way); the CLI owns the
    # distinction between run and undo journals. No redo needed — re-running the original
    # operation from scratch is equivalent.
    def prepare_undo(self, entry: Journal.Entry) -> Callable[[], Journal.Entry]:
        """Return callable that will undo a previously journaled operation"""

        def undo_get_entry() -> Journal.Entry:
            og_data = Operation.Data(src=entry.src, is_dry=False)
            op_out = self._safe(lambda: self.undo(og_data, entry.op_out), og_data, entry.op_out)
            return Journal.Entry(
                op=type(self).__name__.lower(),
                src=entry.src,
                ext_out=entry.ext_out,
                op_out=op_out,
            )

        return undo_get_entry


@dataclass
class Extractor(ABC):
    """Extracts data to be fed into an Operation"""

    class Cfg(CfgModel):
        """Base config for all extractors — subclasses add their own fields"""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify subclass contract at definition time"""
        super().__init_subclass__(**kwargs)
        if "Cfg" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define a Cfg inner class")

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
    :param extractors: Extractors to set up & execute Operations for
    :param journal: Journal implementation
    :param is_dry: Do not actually execute operations
    """
    if not extractors:
        raise RuntimeError("No extractors configured")
    for ext in extractors:
        for op in ext.run(img_dir, is_dry=is_dry):
            journal.log(op())
