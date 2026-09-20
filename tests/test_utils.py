"""Tests utils"""

from pathlib import Path

from local_img_organizer.utils import find_images


def _make_tree(root: Path) -> None:
    """Create a nested tree of image & non-image files to search"""
    for rel in (
        "top.jpg",
        "notes.txt",
        "clip.gif",
        "IMG_1_edited.jpg",
        "Screenshots/shot.png",
        "Screenshots/nested/deep.jpg",
        "trip/beach.JPG",
        "trip/beach.gif",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_find_images_recurses_subdirs(tmp_path: Path) -> None:
    """Images in subdirs are found, non-image extensions are not"""
    _make_tree(tmp_path)
    found = {p.relative_to(tmp_path).as_posix() for p in find_images(tmp_path)}
    assert found == {
        "top.jpg",
        "clip.gif",
        "IMG_1_edited.jpg",
        "Screenshots/shot.png",
        "Screenshots/nested/deep.jpg",
        "trip/beach.JPG",
        "trip/beach.gif",
    }


def test_find_images_exclusions(tmp_path: Path) -> None:
    """A dir name skips everything under it, wildcards match file names case-sensitively"""
    _make_tree(tmp_path)
    exclusions = ["Screenshots", "*.gif", "IMG_*_edited.jpg", "*.JPG"]
    found = {
        p.relative_to(tmp_path).as_posix() for p in find_images(tmp_path, exclusions=exclusions)
    }
    assert found == {"top.jpg"}
