import argparse
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from local_img_organizer import Cfg, classify_folder, load_model


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    cfg = Cfg.from_file(args.cfg)
    if not (categories := cfg.class_cats):
        print("No categories configured")
        return

    print("Loading model...")
    model, processor = load_model()
    print(f"Classifying images into {len(categories)} categories: {categories}...")
    start_ns = time.time_ns()
    path_to_cats = classify_folder(
        folder_path=args.input_dir,
        labels=cfg.class_cats,
        model=model,
        processor=processor,
        threshold=0.95,
        batch_size=16,
    )
    elapsed_s = (time.time_ns() - start_ns) / 1e9
    num_with_classes = len([x for x in path_to_cats.values() if x])
    print(f"Classified {num_with_classes}/{len(path_to_cats)} images in {elapsed_s:.2f}s")

    if args.debug:  # Interactively display images grouped by category for manual verification
        categorized = defaultdict(list)
        for path, category in path_to_cats.items():
            categorized[category if category else "[no match]"].append(path)
        for category in sorted(categorized.keys()):
            images = categorized[category]
            print(f"\nShowing {len(images)} imgs classified as '{category}'")
            try:
                for path in images:
                    try:
                        subprocess.run(
                            ["xdg-open", str(path)], check=True, stderr=subprocess.DEVNULL
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"  Error opening {path}: {e}")
            except KeyboardInterrupt:
                print("\n  Skipping to next category...")
                continue


if __name__ == "__main__":
    main()
