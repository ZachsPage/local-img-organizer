"""Configuration"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ClassificationConfig(BaseModel):
    """Configuration for image classification"""

    categories: list[dict[str, Any]]  # Each item: {category_string: [operations]}


class ExtractorsConfig(BaseModel):
    """Container for all extractor configurations"""

    classification: ClassificationConfig | None = None


class Cfg(BaseModel):
    """Main configuration model"""

    extractors: ExtractorsConfig

    @classmethod
    def from_file(cls, cfg_file: Path) -> "Cfg":
        """Return validated configuration from YAML file"""
        with Path.open(cfg_file) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @property
    def class_cats(self) -> list[str]:
        """Return configured image classification category strings"""
        if self.extractors.classification:
            cats: list[str] = []
            for item in self.extractors.classification.categories:
                if isinstance(item, dict):
                    # Extract the category string (dict key)
                    cats.extend(item.keys())
            return cats
        return []
