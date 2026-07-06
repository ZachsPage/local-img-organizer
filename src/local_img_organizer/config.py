"""Configuration"""

from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from local_img_organizer.interfaces import Extractor, Operation


def parse_extractors(cfg_file: Path) -> list[Extractor]:
    """Parse YAML config and return ready-to-run Extractor instances"""
    with Path.open(cfg_file) as f:
        data = yaml.safe_load(f)
    result: list[Extractor] = []
    for name, ext_data in (data.get("extractors") or {}).items():
        try:
            mod = import_module(f"local_img_organizer.extractors.{name}")
            ext_cls = getattr(mod, name.capitalize())
        except (ModuleNotFoundError, AttributeError):
            raise ValueError(f"Unknown extractor: {name!r}") from None
        result.append(ext_cls.from_cfg(ext_data or {}))
    return result


def parse_operations(op_list: list[dict[str, Any]]) -> list[Operation]:
    """Build Operation instances from a list of raw YAML op dicts"""
    ops: list[Operation] = []
    for op_data in op_list:
        op_name = op_data["op"]
        try:
            mod = import_module(f"local_img_organizer.ops.{op_name}")
            op_cls = getattr(mod, op_name.capitalize())
        except (ModuleNotFoundError, AttributeError):
            raise ValueError(f"Unknown op: {op_name!r}") from None
        ops.append(op_cls.from_cfg(op_data))
    return ops
