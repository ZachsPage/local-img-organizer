"""Move operation - moves an image into a named subdirectory"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from local_img_organizer.img_file import ImgFile
from local_img_organizer.interfaces import Journal, Operation, OpOut
from local_img_organizer.utils import get_logger

_log = get_logger(__name__)

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
        path = data.src.path
        if path.parent.name == self.cfg.subdir_name:
            _log.debug(f"{path.name}: skipping, already in {self.cfg.subdir_name}")
            return {}
        return data.src.plan_path_change(path.parent / self.cfg.subdir_name / path.name)

    @override
    def run(self, data: _Data, planned: OpOut) -> None:
        data.src.apply_path_change(Path(planned["dest"]))

    @classmethod
    @override
    def plan_undo(cls, entry: Journal.Entry, img: ImgFile) -> OpOut:
        return img.plan_move_back(entry)

    @classmethod
    @override
    def undo(cls, img: ImgFile, planned: OpOut) -> None:
        subdir = img.disk_path.parent
        img.apply_path_change(Path(planned["dest"]))
        if not any(subdir.iterdir()):
            subdir.rmdir()
