import sys

import pytest

import main as main_module
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


def test_parse_args_journal_dir_defaults_to_input_dir(monkeypatch, tmp_path):
    """Test omitting -j falls back to -i"""
    cfg = tmp_path / "cfg.yaml"
    monkeypatch.setattr(sys, "argv", ["main.py", "-i", str(tmp_path), "-c", str(cfg)])
    args = main_module.parse_args()
    assert args.journal_dir == tmp_path


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

    def fake_run_ops(input_dir, journal, extractors, *, is_dry):
        calls["run_ops"] = (input_dir, journal, extractors, is_dry)

    def fake_run_undos(*args, **kwargs):
        calls["run_undos"] = (args, kwargs)

    monkeypatch.setattr(main_module, "parse_extractors", fake_parse_extractors)
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
    input_dir, journal, extractors, is_dry = calls["run_ops"]
    assert input_dir == tmp_path
    assert extractors == ["fake_extractor"]
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
    monkeypatch.setattr(sys, "argv", ["main.py", "--undo", "-j", str(journal_dir), "--dry-run"])

    main_module.main()

    assert "run_ops" not in calls
    assert "parse_extractors" not in calls
    journal, is_dry = calls["run_undos"]
    assert is_dry is True
    assert isinstance(journal, CSVJournal)
    assert journal.journal_dir == journal_dir
