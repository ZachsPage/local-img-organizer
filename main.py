"""Main entry - see main"""

import argparse
import logging
from pathlib import Path

from local_img_organizer.config import parse_extractors
from local_img_organizer.interfaces import run_ops, run_undos
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
    args = parser.parse_args()
    if not args.undo and (args.input_dir is None or args.cfg is None):
        parser.error("--input-dir/-i and --cfg/-c are required unless --undo is set")
    if args.journal_dir is None:
        _log.info(f"Nothing passed for --journal-dir, use --input-dir {args.input_dir}")
        args.journal_dir = args.input_dir
    return args


def main() -> None:
    """Run extractors & operations for all images, or undo a previous run"""
    args = parse_args()
    journal = CSVJournal(journal_dir=args.journal_dir)
    if args.undo:
        run_undos(journal, is_dry=args.dry_run)
    else:
        extractors = parse_extractors(args.cfg)
        run_ops(args.input_dir, journal, extractors, is_dry=args.dry_run)


if __name__ == "__main__":
    main()
