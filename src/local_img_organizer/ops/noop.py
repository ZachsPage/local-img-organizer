"""Noop operation - journals an extractor's findings without touching the file"""

from dataclasses import dataclass, field
from typing import override

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
        return {}

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        return

    @classmethod
    @override
    def can_undo(cls, entry: Journal.Entry) -> None:
        return

    @classmethod
    @override
    def undo(cls, og_data: _Data, og_planned: OpOut) -> OpOut:
        return {}
