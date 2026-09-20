"""Noop operation - journals an extractor's findings without touching the file"""

from dataclasses import dataclass, field
from typing import override

from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Journal, Operation, OpOut

type _Data = Operation.Data


@dataclass
class Noop(Operation):
    """Does nothing to the image - useful to see what an extractor found before acting on it"""

    class Cfg(Operation.Cfg):
        """Noop operation configuration"""

        op: str = "noop"

    cfg: Cfg = field(default_factory=Cfg)

    @override
    def plan(self, data: _Data) -> OpOut:
        # Nothing to do, but something to journal - recording what the extractor found is the
        # whole point of this op, and an empty plan is never journaled
        return {"noop": True}

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        return

    @classmethod
    @override
    def plan_undo(cls, entry: Journal.Entry, img: ImgFile) -> OpOut:
        return {}  # nothing was done to reverse

    @classmethod
    @override
    def undo(cls, img: ImgFile, planned: OpOut) -> None:
        return
