"""Main entry - see main"""

import argparse
from pathlib import Path

from local_img_organizer.config import parse_extractors
from local_img_organizer.interfaces import run_all
from local_img_organizer.journals.print_journal import PrintJournal


def parse_args() -> argparse.Namespace:
    """Return the parsed arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Top dir to look for images under",
    )
    parser.add_argument("-c", "--cfg", type=Path, required=True, help="Input yaml config")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Plan / journal operations without executing them",
    )
    return parser.parse_args()


def main() -> None:
    """Run extractors & operations for all images"""
    args = parse_args()
    run_all(args.input_dir, PrintJournal(), parse_extractors(args.cfg), is_dry=args.dry_run)


if __name__ == "__main__":
    main()
