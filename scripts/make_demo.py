"""Builds a folder of drawn stand-in photos & a config to demo the organizer against

AI generated hardcoded images to show example of classification / operations based on image content.

Usage: `uv run scripts/make_demo.py <out_dir>`, then `uv run main.py -i <out_dir>`
"""

import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageColor, ImageDraw, ImageFont

CFG = """\
exclusions:
  - "Screenshots"
  - "IMG_*_edited.jpg"

extractors:
  classification:
    threshold: 0.99
    categories:
      - "a recipe or cooking instructions":
        - op: move
          subdir_name: recipes
      - "a receipt, bill, or document":
        - op: move
          subdir_name: documents
      - "a vehicle":
        - op: tag
          name: type
          value: vehicle
      - "a dog or cat, or other pet":
        - op: tag
          name: type
          value: pet
  metadata:
    lookup_location: true
    operations:
      - op: rename
      - op: tag
        date: true
"""

SIZE = (800, 600)


class Photo(NamedTuple):
    """One file to generate - `exif_date` & `gps` are only written when set"""

    name: str
    draw: Callable[[], Image.Image]
    modified: str
    exif_date: str | None = None
    gps: tuple[float, float] | None = None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _scene(sky_top: str, sky_bottom: str, ground: str, seed: int) -> Callable[[], Image.Image]:
    """Return a drawer for a sky, sun & two ridgelines - `seed` varies where they sit"""

    def draw_scene() -> Image.Image:
        top, bottom, land = (ImageColor.getrgb(c)[:3] for c in (sky_top, sky_bottom, ground))
        w, h = SIZE
        img = Image.new("RGB", SIZE)
        draw = ImageDraw.Draw(img)
        for y in range(h):
            mix = y / h
            color = tuple(int(a + (b - a) * mix) for a, b in zip(top, bottom, strict=True))
            draw.line([(0, y), (w, y)], fill=color)
        sun_x, sun_y = 150 + (seed * 137) % 500, 140 + (seed * 53) % 160
        draw.ellipse([sun_x - 55, sun_y - 55, sun_x + 55, sun_y + 55], fill="#fff4c2")
        for layer, shade in enumerate((0.75, 1.0)):
            peak = 330 + layer * 70
            ridge = [(x, peak + ((x * (seed + 3 + layer)) % 97) - 48) for x in range(0, w + 1, 40)]
            fill = tuple(int(c * shade) for c in land)
            draw.polygon([(0, h), *ridge, (w, h)], fill=fill)
        return img

    return draw_scene


def _receipt() -> Image.Image:
    img = Image.new("RGB", (600, 800), "#6b6b6b")
    draw = ImageDraw.Draw(img)
    draw.rectangle([130, 40, 470, 760], fill="#fbfbf6")
    draw.text((300, 80), "GROCERY MART", fill="#111", font=_font(30), anchor="mm")
    draw.text((300, 115), "RECEIPT #004417", fill="#333", font=_font(18), anchor="mm")
    items = [
        ("Milk 1 gal", "3.49"),
        ("Eggs dozen", "4.29"),
        ("Bread", "2.99"),
        ("Coffee", "9.99"),
        ("Apples 3 lb", "5.47"),
        ("Chicken", "11.80"),
        ("Rice 5 lb", "6.25"),
    ]
    for i, (name, price) in enumerate(items):
        y = 170 + i * 44
        draw.text((155, y), name, fill="#222", font=_font(20))
        draw.text((445, y), f"${price}", fill="#222", font=_font(20), anchor="ra")
    draw.line([155, 490, 445, 490], fill="#222", width=2)
    totals = [("SUBTOTAL", "44.28"), ("TAX", "2.66"), ("TOTAL", "46.94")]
    for i, (name, price) in enumerate(totals):
        y = 510 + i * 44
        draw.text((155, y), name, fill="#111", font=_font(24))
        draw.text((445, y), f"${price}", fill="#111", font=_font(24), anchor="ra")
    draw.text((300, 700), "THANK YOU FOR SHOPPING", fill="#333", font=_font(16), anchor="mm")
    return img


def _recipe() -> Image.Image:
    img = Image.new("RGB", SIZE, "#fff8e7")
    draw = ImageDraw.Draw(img)
    brown = "#7a3e1d"
    draw.rectangle([20, 20, 780, 580], outline="#c97b4a", width=4)
    draw.text((400, 70), "RECIPE: Buttermilk Pancakes", fill=brown, font=_font(38), anchor="mm")
    columns = {
        "Ingredients": [
            "- 2 cups flour",
            "- 2 tbsp sugar",
            "- 2 tsp baking powder",
            "- 2 eggs",
            "- 2 cups buttermilk",
            "- 4 tbsp melted butter",
        ],
        "Instructions": [
            "1. Whisk dry ingredients",
            "2. Mix in eggs & milk",
            "3. Stir in butter",
            "4. Cook on a hot griddle",
            "5. Flip when bubbly",
            "6. Serve with syrup",
        ],
    }
    for col, (title, lines) in enumerate(columns.items()):
        x = 60 + col * 360
        draw.text((x, 130), title, fill=brown, font=_font(28))
        for i, line in enumerate(lines):
            draw.text((x + 20, 180 + i * 34), line, fill="#333", font=_font(22))
    footer = "Prep 10 min   Cook 15 min   Serves 4"
    draw.text((400, 520), footer, fill=brown, font=_font(22), anchor="mm")
    return img


def _car() -> Image.Image:
    img = Image.new("RGB", SIZE, "#a8d8ff")
    draw = ImageDraw.Draw(img)
    red, glass = "#d62828", "#dff3ff"
    draw.rectangle([0, 420, 800, 600], fill="#4a4a4a")
    for x in range(20, 800, 120):
        draw.rectangle([x, 505, x + 60, 515], fill="#f2f2f2")
    draw.rounded_rectangle([120, 290, 690, 420], radius=40, fill=red)
    draw.polygon([(250, 295), (310, 200), (520, 200), (600, 295)], fill=red)
    draw.polygon([(275, 290), (322, 216), (400, 216), (400, 290)], fill=glass)
    draw.polygon([(415, 290), (415, 216), (510, 216), (570, 290)], fill=glass)
    draw.ellipse([655, 320, 690, 350], fill="#ffe66d")
    draw.rectangle([120, 330, 135, 360], fill="#8d0801")
    for cx in (240, 570):
        draw.ellipse([cx - 62, 358, cx + 62, 482], fill="#1b1b1b")
        draw.ellipse([cx - 30, 390, cx + 30, 450], fill="#c0c0c0")
    return img


def _cat() -> Image.Image:
    img = Image.new("RGB", SIZE, "#e8d5b7")
    draw = ImageDraw.Draw(img)
    fur, dark, pink = "#e08e45", "#8a4f1d", "#f7c6c7"
    draw.ellipse([250, 330, 550, 640], fill=fur)
    draw.polygon([(270, 210), (285, 70), (390, 170)], fill=fur)
    draw.polygon([(530, 210), (515, 70), (410, 170)], fill=fur)
    draw.polygon([(295, 180), (300, 110), (350, 165)], fill=pink)
    draw.polygon([(505, 180), (500, 110), (450, 165)], fill=pink)
    draw.ellipse([250, 130, 550, 400], fill=fur)
    for cx in (335, 465):
        draw.ellipse([cx - 32, 225, cx + 32, 285], fill="#b8e986")
        draw.ellipse([cx - 9, 228, cx + 9, 282], fill="#111")
    draw.polygon([(385, 300), (415, 300), (400, 320)], fill="#f28482")
    draw.arc([370, 318, 400, 350], 0, 180, fill=dark, width=4)
    draw.arc([400, 318, 430, 350], 0, 180, fill=dark, width=4)
    for dy in (-12, 4, 20):
        draw.line([370, 320 + dy // 3, 240, 310 + dy * 2], fill=dark, width=3)
        draw.line([430, 320 + dy // 3, 560, 310 + dy * 2], fill=dark, width=3)
    for x in (330, 400, 470):
        draw.line([x, 140, x, 180], fill=dark, width=8)
    return img


_DUSK = ("#33214d", "#e86a92", "#3b3355")

PHOTOS = [
    Photo("IMG_4411.jpg", _receipt, "2024-03-11 12:40:02"),
    Photo("IMG_4412.jpg", _recipe, "2024-03-12 17:05:44"),
    Photo("IMG_4420.jpg", _car, "2024-04-02 09:12:30"),
    Photo("IMG_4431.jpg", _cat, "2024-04-19 20:31:15"),
    Photo(
        "DSC_0412.jpg",
        _scene("#1b2a6b", "#f6a25c", "#2f4f3a", 0),
        "2024-01-05 10:00:00",
        "2023:06:14 18:42:10",
        (39.7392, -104.9903),
    ),
    Photo(
        "DSC_0413.jpg",
        _scene("#5aa9e6", "#d9f0ff", "#c9b458", 1),
        "2024-01-05 10:00:01",
        "2023:06:15 09:03:55",
        (48.8566, 2.3522),
    ),
    Photo(
        "trip/P1010023.jpg",
        _scene("#86c5da", "#eaf6ff", "#588157", 5),
        "2022-10-01 08:00:00",
        "2022:09:30 12:15:00",
        (35.6762, 139.6503),
    ),
    Photo("IMG_3074.jpg", _scene(*_DUSK, 2), "2019-03-02 14:21:07"),
    Photo(
        "IMG_20170709_213627.jpg",
        _scene("#0b132b", "#3a506b", "#1c2541", 3),
        "2017-07-09 21:36:27",
    ),
    Photo("beach_sunset.png", _scene("#ff7b54", "#ffd56b", "#2a6f97", 4), "2021-08-20 19:55:40"),
    Photo(
        "Screenshots/Screenshot_1.png",
        _scene("#222222", "#444444", "#111111", 6),
        "2023-02-02 02:02:02",
    ),
    Photo("IMG_3075_edited.jpg", _scene(*_DUSK, 2), "2019-03-02 15:00:00"),
]


def _exif_args(photo: Photo) -> list[str]:
    """Return the exiftool arguments that write `photo`'s capture time & coordinates"""
    args = []
    if photo.exif_date:
        args += [f"-DateTimeOriginal={photo.exif_date}", f"-CreateDate={photo.exif_date}"]
    if photo.gps:
        lat, lon = photo.gps
        args += [
            f"-GPSLatitude={abs(lat)}",
            f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
            f"-GPSLongitude={abs(lon)}",
            f"-GPSLongitudeRef={'E' if lon >= 0 else 'W'}",
        ]
    return args


def main() -> None:
    """Write the demo config & photos into the dir given as the only argument"""
    if len(sys.argv) != 2:  # noqa: PLR2004
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True)
    (out / "local_img_org.yaml").write_text(CFG)
    for photo in PHOTOS:
        path = out / photo.name
        path.parent.mkdir(exist_ok=True)
        photo.draw().save(path, quality=85)
        if args := _exif_args(photo):
            subprocess.run(["exiftool", "-q", "-overwrite_original", *args, str(path)], check=True)
        stamp = datetime.strptime(photo.modified, "%Y-%m-%d %H:%M:%S").timestamp()  # noqa: DTZ007
        os.utime(path, (stamp, stamp))
    print(f"Wrote {len(PHOTOS)} images to {out} - now run: uv run main.py -i {out}")


if __name__ == "__main__":
    main()
