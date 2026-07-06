"""Print journal — logs entries to stdout. TODO: replace with a persistent journal impl"""

from collections.abc import Generator
from typing import override

from local_img_organizer.interfaces import Journal


class PrintJournal(Journal):
    """Journal that prints each entry to stdout"""

    @override
    def log(self, entry: Journal.Entry) -> None:
        print(entry)

    @override
    def read(self) -> Generator[Journal.Entry]:
        yield from []
