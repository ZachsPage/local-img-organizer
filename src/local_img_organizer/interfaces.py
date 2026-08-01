"""Defines interfaces for implementations"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from local_img_organizer.utils import defer_exceptions, import_cls

_log = logging.getLogger(__name__)


type ExtOut = dict[str, Any]  # Indicates what the extractor found
type OpOut = dict[str, Any]  # Indicates what the op will do


@dataclass
class Journal(ABC):
    """Journal to track what has been done for debugging & undoing"""

    @dataclass
    class Entry:
        """Needed data for each journal entry"""

        op: str
        src: Path
        ext_out: ExtOut
        op_out: OpOut
        is_dry: bool

    @abstractmethod
    def log(self, entry: Entry) -> None:
        """Write out the entry"""

    def get_files_for_undo(self) -> list[Path]:
        """Return journal files available for undo"""
        return []

    @abstractmethod
    def read(self, source: Path | None = None) -> Generator[Entry]:
        """Return each entry; source selects a specific file"""


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
        """Compute and return what this operation will do - raise if invalid"""

    @abstractmethod
    def run(self, data: Data, planned: OpOut) -> None:
        """Execute the planned operation's side effects"""

    @classmethod
    @abstractmethod
    def can_undo(cls, entry: Journal.Entry) -> None:
        """Raise with a reason if this entry's undo is invalid - ex. missing files"""

    @classmethod
    @abstractmethod
    def undo(cls, og_data: Data, og_planned: OpOut) -> OpOut:
        """Reverse a previously executed operation, returning what was actually restored"""

    @classmethod
    def _safe(cls, action: Callable[[], OpOut], data: Data) -> OpOut:
        """Wrap an action with error handling, returning its OpOut or an error dict"""
        try:
            return action()
        except Exception as ex:  # noqa: BLE001
            _log.exception(f"Error for {cls.__name__} - in: {data}, out: {ex}")
            return {"error": str(ex)}

    def prepare(self, data: Data, ext_data: ExtOut | None = None) -> Callable[[], Journal.Entry]:
        """Return callable that will plan & run the operation, returning a Journal.Entry"""

        def run_get_entry() -> Journal.Entry:
            planned = self.plan(data)

            def run_and_get_out() -> OpOut:
                self.run(data, planned)
                return planned

            op_out = planned if data.is_dry else self._safe(run_and_get_out, data)
            return Journal.Entry(
                op=type(self).__name__.lower(),
                src=data.src,
                ext_out=ext_data or {},
                op_out=op_out,
                is_dry=data.is_dry,
            )

        return run_get_entry

    @classmethod
    def prepare_undo(
        cls, entry: Journal.Entry, *, is_dry: bool = False
    ) -> Callable[[], Journal.Entry]:
        """Return callable that will undo a previously journaled operation"""

        def undo_get_entry() -> Journal.Entry:
            og_data = Operation.Data(src=entry.src, is_dry=is_dry)
            op_out = (
                entry.op_out
                if is_dry
                else cls._safe(lambda: cls.undo(og_data, entry.op_out), og_data)
            )
            return Journal.Entry(
                op=entry.op,
                src=entry.src,
                ext_out=entry.ext_out,
                op_out=op_out,
                is_dry=is_dry,
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


def run_ops(
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
    # Resolve to absolute so journaled entries (src, and any op-specific paths derived from it,
    # ex. Move's dest) remain valid for undo regardless of the cwd at undo time.
    img_dir = img_dir.resolve()
    for ext in extractors:
        for op in ext.run(img_dir, is_dry=is_dry):
            journal.log(op())


def run_undos(
    journal: Journal,
    *,
    source: Path | None = None,
    is_dry: bool = False,
) -> None:
    """Top level function to validate / run undos
    :param journal: Journal to read undo source from and log undo results to
    :param source: Specific journal file to undo; if None, user is prompted to select from available
    :param is_dry: Plan undos without executing them
    """
    if not source:
        options = journal.get_files_for_undo()
        if not options:
            raise RuntimeError("No journal files available to undo")
        for i, f in enumerate(options):
            print(f"  [{i}] {f.name}")
        source = options[int(input("Select journal to undo: "))]

    entries = list(journal.read(source))

    def check(entry: Journal.Entry) -> None:
        if entry.is_dry:
            raise ValueError(f"{entry.src}: entry was a dry run, nothing was executed to undo")
        import_cls(f"local_img_organizer.ops.{entry.op}", entry.op, kind="op").can_undo(entry)

    defer_exceptions([partial(check, entry) for entry in entries])

    for entry in entries:
        op_cls = import_cls(f"local_img_organizer.ops.{entry.op}", entry.op, kind="op")
        journal.log(op_cls.prepare_undo(entry, is_dry=is_dry)())
