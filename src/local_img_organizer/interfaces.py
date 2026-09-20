"""Defines interfaces for implementations"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from local_img_organizer.img_file import ImgFile
from local_img_organizer.utils import find_images, import_cls

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
        img_uid: str
        src: Path
        ext_out: ExtOut
        op_out: OpOut
        is_dry: bool

        @property
        def dest(self) -> Path | None:
            """Where this entry's op left the file, or None if it did not move one"""
            return Path(self.op_out["dest"]) if "dest" in self.op_out else None

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

        src: ImgFile
        is_dry: bool  # do not actually execute
        ext: ExtOut = field(default_factory=dict)

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
        """Compute and return what this operation will do - used for dry-run - raise if invalid"""

    @abstractmethod
    def run(self, data: Data, planned: OpOut) -> None:
        """Execute the planned operation's side effects"""

    @classmethod
    @abstractmethod
    def plan_undo(cls, entry: Journal.Entry, img: ImgFile) -> OpOut:
        """Validate this entry's undo against where the image now sits & claim whatever the undo
        needs, returning what it will do - `{}` when this operation changes nothing to reverse
        """

    @classmethod
    @abstractmethod
    def undo(cls, img: ImgFile, planned: OpOut) -> None:
        """Reverse a previously executed operation - only called with a non-empty `plan_undo`"""

    def plan_entry(self, data: Data) -> Journal.Entry | None:
        """Plan this operation & return the entry to journal, or None with nothing to record"""
        # Read before planning - `plan` may advance the image's path for the next op in a chain
        src = data.src.path
        if not (planned := self.plan(data)):
            return None
        return Journal.Entry(
            op=type(self).__name__.lower(),
            img_uid=data.src.id,
            src=src,
            ext_out=data.ext,
            op_out=planned,
            is_dry=data.is_dry,
        )


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
    def run(
        self, images: list[ImgFile], *, is_dry: bool
    ) -> Generator[tuple[Operation, Operation.Data]]:
        """Run the extractor to get all of its metadata, then for each of its assigned Operations,
        yield it with the data to run it on, for each img
        """


def run_ops(
    img_dir: Path,
    journal: Journal,
    extractors: list[Extractor],
    *,
    exclusions: list[str] | None = None,
    is_dry: bool = False,
) -> None:
    """Top level function to tie all the interfaces together
    :param img_dir: Dir with the images to run on (recursively)
    :param extractors: Extractors to set up & execute Operations for
    :param journal: Journal implementation
    :param exclusions: Names / wildcards under `img_dir` to skip - see `find_images`
    :param is_dry: Do not actually execute operations
    """
    if not extractors:
        raise RuntimeError("No extractors configured")
    # Resolve to absolute so journaled entries (src, and any op-specific paths derived from it,
    # ex. Move's dest) remain valid for undo regardless of the cwd at undo time.
    img_dir = img_dir.resolve()
    # Built once & shared, so an op later in a chain plans against where the file will be, and
    # so every op on the same image journals the same img_uid for undo to group by
    images = ImgFile.collect(find_images(img_dir, exclusions=exclusions or []))
    for ext in extractors:
        for op, data in ext.run(images, is_dry=is_dry):
            _log_then_run(op, data, journal)


def _log_then_run(op: Operation, data: Operation.Data, journal: Journal) -> None:
    """Journal what the operation plans, then execute it

    The entry is written *before* the operation runs, so a failed run - or a crash between the
    two - can leave an entry for something that did not happen, but never a change with no entry.
    `ImgFile.from_journal_for_undo` tells those apart by where the file actually is. A failed run
    journals the error & re-raises to stop the run, leaving everything already done undoable.
    """
    if (entry := op.plan_entry(data)) is None:
        return
    journal.log(entry)
    if data.is_dry:
        return
    try:
        op.run(data, entry.op_out)
    except Exception as ex:
        _log.exception(f"Error for {entry.op} - in: {data}, out: {ex}")
        journal.log(replace(entry, op_out={"error": str(ex)}))
        raise


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

    for op_cls, entry, img, out in _plan_undos(list(journal.read(source))):
        journal.log(replace(entry, op_out=out, is_dry=is_dry))
        if not is_dry:
            op_cls.undo(img, out)


type _PlannedUndo = tuple[type[Operation], Journal.Entry, ImgFile, OpOut]


def _plan_undos(entries: list[Journal.Entry]) -> list[_PlannedUndo]:
    """Return every undo to run, newest entry first, or raise with all the reasons it cannot

    Planning all of them before any runs means a journal that cannot be fully unwound changes
    nothing. Each image gets one ImgFile, created where the latest entry left it - a chain's
    earlier entries name paths that only exist again once its later ones are undone, and each
    planned undo moves the ImgFile back for the entry before it.
    """
    images: dict[str, ImgFile] = {}
    planned: list[_PlannedUndo] = []
    errors: list[str] = []
    for entry in reversed(entries):
        if entry.is_dry:
            errors.append(f"{entry.src}: entry was a dry run, nothing was executed to undo")
            continue
        try:
            op_cls = import_cls(f"local_img_organizer.ops.{entry.op}", entry.op, kind="op")
            if (img := images.get(entry.img_uid)) is None:
                img = images[entry.img_uid] = ImgFile.from_journal_for_undo(entry)
            if out := op_cls.plan_undo(entry, img):
                planned.append((op_cls, entry, img, out))
        except Exception as ex:  # noqa: BLE001
            errors.append(str(ex))
    if errors:
        raise RuntimeError("\n".join(errors))
    return planned
