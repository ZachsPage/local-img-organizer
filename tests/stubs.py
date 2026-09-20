"""Shared test doubles"""

import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from local_img_organizer.interfaces import Journal

_log = logging.getLogger(__name__)


@dataclass
class StubJournal(Journal):
    """Journal that keeps entries in memory"""

    entries: list[Journal.Entry] = field(default_factory=list)

    @override
    def log(self, entry: Journal.Entry) -> None:
        _log.info(entry)
        self.entries.append(entry)

    @override
    def read(self, source: Path | None = None) -> Generator[Journal.Entry]:
        yield from self.entries
