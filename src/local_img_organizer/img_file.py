"""Tracks one image through a run so chained operations act on where the file actually is"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from local_img_organizer.utils import get_logger, move_file

_log = get_logger(__name__)

if TYPE_CHECKING:  # Annotation only - interfaces imports this module
    from local_img_organizer.interfaces import Journal


@dataclass
class ImgFile:
    """An image's identity & paths for a single run

    `path` advances as each operation plans its change, so the next operation in a chain plans
    against where the file will be. `disk_path` only moves once an operation executes - the two
    differ during a dry run, and in between an operation's `plan` & `run`.
    """

    path: Path
    id: str = field(default_factory=lambda: uuid4().hex)
    # Paths claimed by a plan this run, shared by every ImgFile so two images cannot plan the same
    # name. A dry run moves nothing, so the filesystem alone cannot answer this
    taken: set[Path] = field(default_factory=set)
    disk_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Start the real path where the planned one starts"""
        self.disk_path = self.path

    @classmethod
    def collect(cls, paths: list[Path]) -> list["ImgFile"]:
        """Return an ImgFile per path, all sharing one set of claimed paths"""
        taken: set[Path] = set()
        return [cls(path=path, taken=taken) for path in paths]

    def plan_path_change(self, dest: Path) -> dict[str, str]:
        """Claim `dest` & advance the planned path, returning what the operation journals"""
        if self.is_taken(dest):
            raise ValueError(f"Dest {dest} already exists?")
        self.taken.add(dest)
        self.path = dest
        return {"dest": str(dest)}

    def apply_path_change(self, dest: Path) -> None:
        """Move the real file to `dest` - raises if the filesystem no longer matches the plan"""
        move_file(self.disk_path, dest)
        self.disk_path = dest

    def is_taken(self, path: Path) -> bool:
        """Return whether `path` is on disk or already claimed by a plan this run"""
        return path in self.taken or path.exists()

    @classmethod
    def from_journal_for_undo(cls, entry: "Journal.Entry") -> "ImgFile":
        """Return the image a journaled op acted on, at whichever path it really sits now

        Only valid for the latest undo entry of an image - every earlier entry of a chain names
        paths that exist again once the later ones are undone, so `path` is tracked from here on.
        """
        if (dest := entry.dest) is None:
            return cls(path=entry.src)  # nothing was moved - ex. a noop, or a run that failed
        src_found, dest_found = entry.src.exists(), dest.exists()
        if src_found and dest_found:
            raise ValueError(f"{entry.src}: {dest} also exists, cannot tell whether the op ran")
        if not src_found and not dest_found:
            raise ValueError(f"{entry.src}: neither it nor {dest} exist, cannot undo")
        if dest_found:  # happy path
            return cls(path=dest)
        # The entry is journaled before its op runs, so a failed run - or a crash between the two -
        # leaves the file here. Undoing the entries before it still works from there
        _log.warning(f"{entry.src}: never moved to {dest}, undoing from where it actually is")
        return cls(path=entry.src)

    def plan_move_back(self, entry: "Journal.Entry") -> dict[str, str]:
        """Claim the path this entry's op moved the file away from, returning what its undo does

        `{}` when there is nothing to undo - the op moved nothing, or it was journaled but never
        ran. Raises if the file is not where the journal says it should be.
        """
        if (dest := entry.dest) is None:
            return {}  # the op moved nothing - ex. a noop, or a run that failed
        if self.path == entry.src:
            return {}  # `from_journal_for_undo` found the file here, so the op never ran
        if self.path != dest:
            raise ValueError(f"{entry.src}: expected the file at {dest}, found {self.path}")
        return self.plan_path_change(entry.src)
