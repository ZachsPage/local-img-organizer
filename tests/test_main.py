import sys
from pathlib import Path

import pytest

import main as main_module
from local_img_organizer.interfaces import Journal
from local_img_organizer.journals.csv_journal import CSVJournal


def test_parse_args_requires_input_and_cfg_without_undo(monkeypatch):
    """Test -i/-c are required when --undo is not set"""
    monkeypatch.setattr(sys, "argv", ["main.py", "-j", "journals"])
    with pytest.raises(SystemExit):
        main_module.parse_args()


def test_parse_args_allows_missing_input_and_cfg_with_undo(monkeypatch):
    """Test -i/-c are optional when --undo is set"""
    monkeypatch.setattr(sys, "argv", ["main.py", "--undo", "-j", "journals"])
    args = main_module.parse_args()
    assert args.undo is True
    assert args.input_dir is None
    assert args.cfg is None


def test_main_journal_dir_defaults_to_input_dir(monkeypatch, tmp_path):
    """Test omitting -j falls back to -i"""
    calls = {}
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setattr(sys, "argv", ["main.py", "-i", str(tmp_path), "-c", str(cfg)])
    monkeypatch.setattr(main_module, "parse_extractors", lambda _: [])
    monkeypatch.setattr(main_module, "parse_exclusions", lambda _: [])
    monkeypatch.setattr(
        main_module, "run_ops", lambda _dir, journal, *_, **__: calls.setdefault("journal", journal)
    )
    main_module.main()
    assert calls["journal"].journal_dir == tmp_path


def test_parse_args_journal_dir_explicit(monkeypatch, tmp_path):
    """Test an explicit -j is kept as-is rather than falling back to -i"""
    cfg = tmp_path / "cfg.yaml"
    journal_dir = tmp_path / "journals"
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "-i", str(tmp_path), "-c", str(cfg), "-j", str(journal_dir)],
    )
    args = main_module.parse_args()
    assert args.journal_dir == journal_dir


def test_main_runs_ops_when_not_undo(monkeypatch, tmp_path):
    """Test main() parses extractors & calls run_ops, not run_undos, without --undo"""
    calls = {}
    cfg = tmp_path / "cfg.yaml"
    journal_dir = tmp_path / "journals"

    def fake_parse_extractors(cfg_file):
        calls["cfg"] = cfg_file
        return ["fake_extractor"]

    def fake_run_ops(input_dir, journal, extractors, *, exclusions, is_dry):
        calls["run_ops"] = (input_dir, journal, extractors, exclusions, is_dry)

    def fake_run_undos(*args, **kwargs):
        calls["run_undos"] = (args, kwargs)

    monkeypatch.setattr(main_module, "parse_extractors", fake_parse_extractors)
    monkeypatch.setattr(main_module, "parse_exclusions", lambda _: ["Screenshots"])
    monkeypatch.setattr(main_module, "run_ops", fake_run_ops)
    monkeypatch.setattr(main_module, "run_undos", fake_run_undos)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "-i", str(tmp_path), "-c", str(cfg), "-j", str(journal_dir)],
    )

    main_module.main()

    assert "run_undos" not in calls
    assert calls["cfg"] == cfg
    input_dir, journal, extractors, exclusions, is_dry = calls["run_ops"]
    assert input_dir == tmp_path
    assert extractors == ["fake_extractor"]
    assert exclusions == ["Screenshots"]
    assert is_dry is False
    assert isinstance(journal, CSVJournal)
    assert journal.journal_dir == journal_dir


def test_main_runs_undos_when_undo_flag_set(monkeypatch, tmp_path):
    """Test main() calls run_undos, not run_ops/parse_extractors, with --undo"""
    calls = {}
    journal_dir = tmp_path / "journals"

    def fake_run_undos(journal, *, is_dry):
        calls["run_undos"] = (journal, is_dry)

    def fake_run_ops(*args, **kwargs):
        calls["run_ops"] = (args, kwargs)

    def fake_parse_extractors(cfg_file):
        calls["parse_extractors"] = cfg_file
        return []

    monkeypatch.setattr(main_module, "run_undos", fake_run_undos)
    monkeypatch.setattr(main_module, "run_ops", fake_run_ops)
    monkeypatch.setattr(main_module, "parse_extractors", fake_parse_extractors)
    monkeypatch.setattr(main_module, "parse_exclusions", fake_parse_extractors)
    monkeypatch.setattr(sys, "argv", ["main.py", "--undo", "-j", str(journal_dir), "--dry-run"])

    main_module.main()

    assert "run_ops" not in calls
    assert "parse_extractors" not in calls
    journal, is_dry = calls["run_undos"]
    assert is_dry is True
    assert isinstance(journal, CSVJournal)
    assert journal.journal_dir == journal_dir


def _entry() -> Journal.Entry:
    return Journal.Entry(
        op="move",
        img_uid="img0",
        src=Path("/img/0.png"),
        ext_out={},
        op_out={"dest": "/img/cats/0.png"},
        is_dry=False,
    )


def _no_run(monkeypatch, shown):
    """Stub out everything but the GUI handling, recording what gets shown"""
    monkeypatch.setattr(main_module, "parse_extractors", lambda _: [])
    monkeypatch.setattr(main_module, "parse_exclusions", lambda _: [])
    monkeypatch.setattr(main_module, "run_ops", lambda *_, **__: None)
    monkeypatch.setattr(main_module, "run_undos", lambda *_, **__: None)
    monkeypatch.setattr(
        main_module.html_view, "show", lambda _journal, source: shown.append(source)
    )


def test_parse_args_gui_defaults_to_after(monkeypatch, tmp_path):
    """Test the GUI is shown after a run unless asked otherwise"""
    monkeypatch.setattr(sys, "argv", ["main.py", "-i", str(tmp_path), "-c", "cfg.yaml"])
    assert main_module.parse_args().gui == "after"


def test_parse_args_gui_only_needs_a_journal_dir(monkeypatch):
    """Test --gui only still needs somewhere to look for journals"""
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui", "only"])
    with pytest.raises(SystemExit):
        main_module.parse_args()


def test_parse_args_gui_only_without_input_and_cfg(monkeypatch, tmp_path):
    """Test -i/-c are not required with --gui only, since nothing is run"""
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui", "only", "-j", str(tmp_path)])
    args = main_module.parse_args()
    assert args.cfg is None
    assert args.input_dir is None


def test_main_shows_the_journal_the_run_wrote(monkeypatch, tmp_path):
    """Test --gui after shows the journal file this run created, not an older one"""
    shown = []
    (tmp_path / f"{CSVJournal.FILE_PREFIX}20200101_000000_000000.csv").touch()  # an older run

    def fake_run_ops(_dir, journal, *_, **__):
        journal.log(_entry())

    _no_run(monkeypatch, shown)
    monkeypatch.setattr(main_module, "run_ops", fake_run_ops)
    monkeypatch.setattr(sys, "argv", ["main.py", "-i", str(tmp_path), "-c", "cfg.yaml"])

    main_module.main()

    assert len(shown) == 1
    assert "20200101" not in shown[0].name


def test_main_gui_none_shows_nothing(monkeypatch, tmp_path):
    """Test --gui none runs without opening a page"""
    shown = []
    _no_run(monkeypatch, shown)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "-i", str(tmp_path), "-c", "cfg.yaml", "--gui", "none"]
    )
    main_module.main()
    assert shown == []


def test_main_gui_after_an_empty_run_shows_nothing(monkeypatch, tmp_path):
    """Test a run that journaled nothing has no page to show"""
    shown = []
    _no_run(monkeypatch, shown)
    monkeypatch.setattr(sys, "argv", ["main.py", "-i", str(tmp_path), "-c", "cfg.yaml"])
    main_module.main()
    assert shown == []


def test_main_gui_only_shows_the_newest_journal(monkeypatch, tmp_path):
    """Test --gui only opens the newest existing journal without running anything"""
    shown = []
    ran = []
    _no_run(monkeypatch, shown)
    monkeypatch.setattr(main_module, "run_ops", lambda *_, **__: ran.append("run_ops"))
    older = tmp_path / f"{CSVJournal.FILE_PREFIX}20200101_000000_000000.csv"
    newest = tmp_path / f"{CSVJournal.FILE_PREFIX}20260101_000000_000000.csv"
    older.touch()
    newest.touch()
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui", "only", "-j", str(tmp_path)])

    main_module.main()

    assert shown == [newest]
    assert ran == []


def test_main_gui_only_without_any_journal(monkeypatch, tmp_path):
    """Test --gui only says where it looked when there is no journal to show"""
    _no_run(monkeypatch, [])
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui", "only", "-j", str(tmp_path)])
    with pytest.raises(RuntimeError, match="No journal files found"):
        main_module.main()


def test_main_shows_the_undo_journal(monkeypatch, tmp_path):
    """Test an undo run also gets a page, showing what the undo did"""
    shown = []
    _no_run(monkeypatch, shown)
    monkeypatch.setattr(main_module, "run_undos", lambda journal, **__: journal.log(_entry()))
    monkeypatch.setattr(sys, "argv", ["main.py", "--undo", "-j", str(tmp_path)])

    main_module.main()

    assert len(shown) == 1
