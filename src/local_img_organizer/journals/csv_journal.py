"""CSV journal — persists entries to a timestamped CSV file for durable run/undo history"""

import csv
import json
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import override

from local_img_organizer.interfaces import Journal

_CSV_COLUMNS = ["op", "src", "ext_out", "op_out"]


@dataclass
class CSVJournal(Journal):
    """Journal that writes to a timestamped CSV in journal_dir; reads back by file path"""

    journal_dir: Path
    _file: Path | None = field(default=None, init=False, repr=False)

    @override
    def log(self, entry: Journal.Entry) -> None:
        with self._get_or_create_file().open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    entry.op,
                    str(entry.src),
                    json.dumps(entry.ext_out),
                    json.dumps(entry.op_out),
                ]
            )

    @override
    def get_files_for_undo(self) -> list[Path]:
        if not self.journal_dir.exists():
            return []
        return sorted(self.journal_dir.glob("journal_*.csv"))

    @override
    def read(self, source: Path | None = None) -> Generator[Journal.Entry]:
        file = source or self._file
        if file is None or not file.exists():
            return
        with file.open(newline="") as f:
            for row in csv.DictReader(f):
                yield Journal.Entry(
                    op=row["op"],
                    src=Path(row["src"]),
                    ext_out=json.loads(row["ext_out"]),
                    op_out=json.loads(row["op_out"]),
                )

    def _get_or_create_file(self) -> Path:
        if self._file is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            self._file = self.journal_dir / f"journal_{timestamp}.csv"
            with self._file.open("w", newline="") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)
        return self._file
