"""Move operation - moves an image into a named subdirectory"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from local_img_organizer.interfaces import Operation, OpOut

type _Data = Operation.Data


@dataclass
class Move(Operation):
    """Moves an image file into a named subdirectory alongside the source"""

    class Cfg(Operation.Cfg):
        """Move operation configuration"""

        subdir_name: str

    cfg: Cfg

    @override
    def plan(self, data: _Data) -> OpOut:
        """Compute and return what this operation would do"""
        if not data.src.is_file():
            raise ValueError(f"{data.src} is not a file")
        if data.src.parent.name == self.cfg.subdir_name:
            return {}
        dest = data.src.parent / self.cfg.subdir_name / data.src.name
        return {"dest": str(dest)}

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        """Execute the planned operation's side effects"""
        if not planned:
            return
        dest = Path(planned["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        data.src.rename(dest)

    @override
    def undo(self, og_data: _Data, og_planned: OpOut) -> None:
        """Reverse a previously executed operation"""
        if not og_planned:
            return
        Path(og_planned["dest"]).rename(og_data.src)
