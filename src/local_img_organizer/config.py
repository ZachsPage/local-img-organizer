"""Configuration"""

from pathlib import Path
from typing import Any

import yaml

from local_img_organizer.interfaces import Extractor, Operation
from local_img_organizer.utils import import_cls


def parse_extractors(cfg_file: Path) -> list[Extractor]:
    """Parse YAML config and return ready-to-run Extractor instances"""
    with Path.open(cfg_file) as f:
        data = yaml.safe_load(f)
    result: list[Extractor] = []
    for name, ext_data in (data.get("extractors") or {}).items():
        ext_cls = import_cls(f"local_img_organizer.extractors.{name}", name, kind="extractor")
        result.append(ext_cls.from_cfg(ext_data or {}))
    return result


def parse_operations(op_list: list[dict[str, Any]]) -> list[Operation]:
    """Build Operation instances from a list of raw YAML op dicts"""
    ops: list[Operation] = []
    for op_data in op_list:
        op_name = op_data["op"]
        op_cls = import_cls(f"local_img_organizer.ops.{op_name}", op_name, kind="op")
        ops.append(op_cls.from_cfg(op_data))
    return ops
