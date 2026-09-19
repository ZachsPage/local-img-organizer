from pathlib import Path

from local_img_organizer.interfaces import Journal
from local_img_organizer.journals.csv_journal import CSVJournal


def _entry(i: int) -> Journal.Entry:
    return Journal.Entry(
        op="move",
        src=Path(f"/img/{i}.png"),
        ext_out={"category": "cats"},
        op_out={"dest": f"/img/cats/{i}.png"},
        is_dry=False,
    )


def test_log_and_read_round_trip(tmp_path):
    """Test entries logged to a CSVJournal can be read back with matching data"""
    journal = CSVJournal(journal_dir=tmp_path)
    entries = [_entry(i) for i in range(3)]
    for entry in entries:
        journal.log(entry)

    read_back = list(journal.read())
    assert read_back == entries


def test_log_and_read_round_trip_dry_run(tmp_path):
    """Test the is_dry flag survives the csv round trip"""
    journal = CSVJournal(journal_dir=tmp_path)
    entry = Journal.Entry(
        op="move",
        src=Path("/img/0.png"),
        ext_out={"category": "cats"},
        op_out={"dest": "/img/cats/0.png"},
        is_dry=True,
    )
    journal.log(entry)

    assert list(journal.read()) == [entry]


def test_log_appends_to_same_file(tmp_path):
    """Test repeated log() calls on one instance write to a single journal file"""
    journal = CSVJournal(journal_dir=tmp_path)
    journal.log(_entry(0))
    journal.log(_entry(1))
    assert len(list(tmp_path.glob(f"{CSVJournal.FILE_PREFIX}*.csv"))) == 1


def test_read_missing_file_yields_nothing(tmp_path):
    """Test reading before anything has been logged yields no entries"""
    journal = CSVJournal(journal_dir=tmp_path)
    assert list(journal.read()) == []


def test_read_specific_source(tmp_path):
    """Test read(source=...) reads a specific journal file rather than the instance's own"""
    first = CSVJournal(journal_dir=tmp_path)
    first.log(_entry(0))

    second = CSVJournal(journal_dir=tmp_path / "other")
    assert list(second.read(source=first._file)) == [_entry(0)]  # noqa: SLF001


def test_get_files_for_undo(tmp_path):
    """Test get_files_for_undo lists all journal files in journal_dir, newest first"""
    journal = CSVJournal(journal_dir=tmp_path)
    first = f"{CSVJournal.FILE_PREFIX}20260101_000000.csv"
    second = f"{CSVJournal.FILE_PREFIX}20260102_000000.csv"
    (tmp_path / first).touch()
    (tmp_path / second).touch()
    (tmp_path / "not_a_journal.csv").touch()

    files = journal.get_files_for_undo()
    assert [f.name for f in files] == [second, first]


def test_get_files_for_undo_missing_dir(tmp_path):
    """Test get_files_for_undo returns an empty list if journal_dir doesn't exist yet"""
    journal = CSVJournal(journal_dir=tmp_path / "nonexistent")
    assert journal.get_files_for_undo() == []


def test_round_trip_csv_special_characters(tmp_path):
    """Test paths/values with commas, quotes, and unicode survive the csv+json round trip"""
    entry = Journal.Entry(
        op="move",
        src=Path('/img/a, "tricky" file, 猫.png'),
        ext_out={"category": 'cats, "cute"', "tags": ['a"b', "c,d"]},
        op_out={"dest": '/img/cats/a, "tricky" file, 猫.png'},
        is_dry=False,
    )
    journal = CSVJournal(journal_dir=tmp_path)
    journal.log(entry)
    assert list(journal.read()) == [entry]
