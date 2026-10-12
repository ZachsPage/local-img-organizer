"""Main entry - see main"""

import argparse
import logging
from pathlib import Path

from local_img_organizer.config import parse_exclusions, parse_extractors
from local_img_organizer.interfaces import run_ops, run_undos
from local_img_organizer.journals import html_view
from local_img_organizer.journals.csv_journal import CSVJournal

_log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Return the parsed arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="Top dir to look for images under",
    )
    parser.add_argument("-c", "--cfg", type=Path, help="Input yaml config")
    parser.add_argument(
        "-j",
        "--journal-dir",
        type=Path,
        help="Dir to read/write the run's CSV journal to",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Plan / journal operations without executing them",
    )
    parser.add_argument(
        "--undo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Undo from a journal - will be prompted",
    )
    parser.add_argument(
        "--gui",
        default="after",
        choices=("after", "none", "only"),
        help="Show the run's journal as an HTML page - 'after' once it finishes, 'none' never, "
        "'only' to view the newest journal without running anything",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=logging.getLevelNamesMapping(),
        help="Log level - DEBUG shows why operations skip images",
    )
    args = parser.parse_args()
    if args.gui == "only":
        if args.journal_dir is None and args.input_dir is None:
            parser.error("--journal-dir/-j or --input-dir/-i is required with --gui only")
    elif not args.undo and (args.input_dir is None or args.cfg is None):
        parser.error("--input-dir/-i and --cfg/-c are required unless --undo is set")
    return args


def main() -> None:
    """Run extractors & operations for all images, undo a previous run, or view a journal"""
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    )
    if args.journal_dir is None:
        _log.info(f"Nothing passed for --journal-dir, use --input-dir {args.input_dir}")
        args.journal_dir = args.input_dir
    journal = CSVJournal(journal_dir=args.journal_dir)
    if args.gui == "only":
        if not (files := journal.get_files()):
            raise RuntimeError(f"No journal files found in {args.journal_dir}")
        html_view.show(journal, files[0])
        return
    if args.undo:
        run_undos(journal, is_dry=args.dry_run)
    else:
        extractors = parse_extractors(args.cfg)
        exclusions = parse_exclusions(args.cfg)
        run_ops(args.input_dir, journal, extractors, exclusions=exclusions, is_dry=args.dry_run)
    if args.gui == "none":
        return
    if journal.file is None:
        _log.info("Nothing was journaled, so there is nothing to show")
        return
    html_view.show(journal, journal.file)


if __name__ == "__main__":
    main()
