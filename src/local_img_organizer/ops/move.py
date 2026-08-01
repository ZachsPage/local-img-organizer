"""Move operation - moves an image into a named subdirectory"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from local_img_organizer.interfaces import Journal, Operation, OpOut

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
        if not data.src.is_file():
            raise ValueError(f"{data.src} is not a file")
        if data.src.parent.name == self.cfg.subdir_name:
            return {}
        dest = data.src.parent / self.cfg.subdir_name / data.src.name
        if dest.exists():
            raise ValueError(f"Dest {dest} already exists?")
        return {"dest": str(dest)}

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        if not planned:
            return
        dest = Path(planned["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        data.src.rename(dest)

    @classmethod
    @override
    def can_undo(cls, entry: Journal.Entry) -> None:
        dest = entry.op_out.get("dest")
        if not dest:
            return
        if Path(entry.src).exists():
            raise ValueError(f"{entry.src}: already exists, undo would overwrite it")
        if not Path(dest).exists():
            raise ValueError(f"{entry.src}: dest {dest} is missing, cannot undo move")

    @classmethod
    @override
    def undo(cls, og_data: _Data, og_planned: OpOut) -> OpOut:
        dest = og_planned.get("dest")
        if not dest:
            return {}
        dest_path = Path(dest)
        dest_path.rename(og_data.src)
        subdir = dest_path.parent
        if not any(subdir.iterdir()):
            subdir.rmdir()
        return {"dest": str(og_data.src)}
