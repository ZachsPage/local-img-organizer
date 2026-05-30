"""Main entry - see main"""

import argparse
from pathlib import Path

from local_img_organizer import Cfg

# TODO - maybe change the naming schema here to end in Ext & Op?
from local_img_organizer.extractors.classification import Classification


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
        "--debug",
        action="store_true",
        help="Will interactively debug classifications",
    )
    return parser.parse_args()


def main() -> None:
    """Run extractors & operations for all images"""
    args = parse_args()
    cfg = Cfg.from_file(args.cfg)
    if not cfg.class_cats:
        raise RuntimeError("No classification categories configured")
    Classification(
        ops=[],
        cfg=Classification.Cfg(
            categories=cfg.class_cats,
            debug=args.debug,
        ),
    ).run(Path(args.input_dir), is_dry=True)


if __name__ == "__main__":
    main()
