"""Tests configuration"""

import tempfile
from pathlib import Path

import pytest

from local_img_organizer.config import parse_extractors
from local_img_organizer.extractors.classification import Classification
from local_img_organizer.extractors.metadata import Metadata
from local_img_organizer.ops.move import Move

_EXAMPLE_CFG = Path(__file__).parent.parent / "config" / "example_cfg.yaml"


def _example_classification() -> Classification:
    """Return the Classification extractor built from the example config"""
    found = [e for e in parse_extractors(_EXAMPLE_CFG) if isinstance(e, Classification)]
    assert len(found) == 1
    return found[0]


def test_example_config_loads() -> None:
    """parse_extractors parses example_cfg.yaml into ready-to-run extractors"""
    extractors = parse_extractors(_EXAMPLE_CFG)
    assert [type(e) for e in extractors] == [Classification, Metadata]


def test_example_config_categories() -> None:
    """Classification extractor has the expected categories and Move ops"""
    ext = _example_classification()
    assert "a recipe or cooking instructions" in ext.categories_to_ops
    assert "a receipt, bill, or document" in ext.categories_to_ops
    for ops in ext.categories_to_ops.values():
        assert len(ops) == 1
        assert isinstance(ops[0], Move)


def test_move_cfg_subdir_name() -> None:
    """Move ops carry the subdir_name from the YAML"""
    recipe_op = _example_classification().categories_to_ops["a recipe or cooking instructions"][0]
    assert isinstance(recipe_op, Move)
    assert recipe_op.cfg.subdir_name == "recipes"


def test_inline_config() -> None:
    """parse_extractors accepts any valid YAML, not just the example file"""
    threshold = 0.8
    batch_size = 8
    yaml_content = f"""
extractors:
  classification:
    categories:
      - "a dog":
        - op: move
          subdir_name: dogs
    threshold: {threshold}
    batch_size: {batch_size}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = Path(f.name)

    try:
        extractors = parse_extractors(tmp_path)
        ext = extractors[0]
        assert isinstance(ext, Classification)
        assert ext.cfg.threshold == threshold
        assert ext.cfg.batch_size == batch_size
        dog_op = ext.categories_to_ops["a dog"][0]
        assert isinstance(dog_op, Move)
        assert dog_op.cfg.subdir_name == "dogs"
    finally:
        tmp_path.unlink()


def test_empty_extractors() -> None:
    """No extractors configured returns empty list"""
    yaml_content = "extractors:\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = Path(f.name)

    try:
        assert parse_extractors(tmp_path) == []
    finally:
        tmp_path.unlink()


def test_invalid_op_type() -> None:
    """Unknown op type raises a ValueError"""
    yaml_content = """
extractors:
  classification:
    categories:
      - "a dog":
        - op: unknown_op
          subdir_name: dogs
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Unknown op"):
            parse_extractors(tmp_path)
    finally:
        tmp_path.unlink()
