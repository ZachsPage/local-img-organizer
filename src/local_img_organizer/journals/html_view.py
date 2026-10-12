"""Renders a journal into one self-contained HTML page so a run can be reviewed in a browser"""

import base64
import io
import webbrowser
from collections import Counter
from collections.abc import Iterable
from html import escape
from os.path import commonpath
from pathlib import Path

from PIL import Image

from local_img_organizer.interfaces import Journal
from local_img_organizer.utils import get_logger

_log = get_logger(__name__)

# Rendered at 2x the CSS size below so the thumbnails stay sharp on a hidpi screen
THUMB_SIZE = (240, 240)

_STYLE = """
:root { color-scheme: light dark; --line: #8884; --dim: #8888; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 16px; }
h1 { font-size: 18px; margin: 0 0 4px; word-break: break-all; }
.sub { color: var(--dim); margin-bottom: 12px; word-break: break-all; }
.banner { margin-bottom: 16px; padding: 8px 12px; border-radius: 6px; border-left: 4px solid; }
.banner b { display: block; font-size: 16px; }
.planned { background: #f5a62322; border-color: #f5a623; }
.executed { background: #2e9e4f22; border-color: #2e9e4f; }
table { border-collapse: collapse; width: 100%; }
th { text-align: left; border-bottom: 1px solid var(--line); padding: 6px 8px; }
td { border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }
td.thumb { width: 136px; }
img { max-width: 120px; max-height: 120px; border-radius: 4px; display: block; }
.missing { width: 120px; height: 90px; border-radius: 4px; background: var(--line); }
.name { font-weight: 600; word-break: break-all; }
.ext { color: var(--dim); font-size: 12px; word-break: break-all; }
.op { word-break: break-all; }
.op + .op { margin-top: 6px; }
.op code { font-size: 13px; }
"""


def show(journal: Journal, source: Path) -> Path:
    """Write `source`'s HTML page next to it, open it in a browser & return where it was written"""
    out = source.with_suffix(".html")
    out.write_text(render(journal.read(source), source), encoding="utf-8")
    _log.info(f"Wrote {out}")
    webbrowser.open(out.resolve().as_uri())
    return out


def render(entries: Iterable[Journal.Entry], source: Path) -> str:
    """Return a full HTML page showing a thumbnail & the operation chain for every image"""
    grouped = _group(entries)
    flat = [entry for group in grouped.values() for entry in group]
    counts = Counter(entry.op for entry in flat)
    summary = ", ".join(f"{n} {op}" for op, n in sorted(counts.items())) or "no entries"
    root = _common_root(flat)
    under = f" under <code>{escape(str(root))}</code>" if root else ""
    rows = "".join(_row(group, root) for group in grouped.values())
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(source.name)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{escape(source.name)}</h1>"
        f"<div class='sub'>{len(grouped)} images, {escape(summary)}{under}</div>"
        f"{_banner(flat)}"
        "<table><tr><th>Image</th><th>File</th><th>Operations</th></tr>"
        f"{rows}</table></body></html>\n"
    )


def _banner(entries: list[Journal.Entry]) -> str:
    """Return the page's dry-run or executed banner

    A journal only ever holds one run, so every entry in it is dry or not together - saying it
    once up top beats repeating it on every operation.
    """
    if any(entry.is_dry for entry in entries):
        return (
            "<div class='banner planned'><b>Planned - dry run</b>"
            "Nothing was changed. These are the operations a real run would execute.</div>"
        )
    return (
        "<div class='banner executed'><b>Executed</b>"
        "These operations ran. Undo them with <code>--undo</code>.</div>"
    )


def _common_root(entries: list[Journal.Entry]) -> Path | None:
    """Return the dir every journaled path sits under, so each row can leave it off"""
    parents = {path.parent for entry in entries for path in (entry.src, entry.dest) if path}
    if not parents:
        return None
    try:
        return Path(commonpath(parents))
    except ValueError:  # no shared root - ex. mixed absolute & relative paths, or two drives
        return None


def _rel(path: Path, root: Path | None) -> str:
    """Return `path` without the shared root the page already shows up top"""
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def _group(entries: Iterable[Journal.Entry]) -> dict[str, list[Journal.Entry]]:
    """Return the entries per image, both in the order they were journaled

    Every operation on an image shares its `img_uid`, but a second extractor's entries for it land
    further down the journal, so the entries of a chain are not necessarily next to each other.
    """
    grouped: dict[str, list[Journal.Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.img_uid, []).append(entry)
    return grouped


def _row(entries: list[Journal.Entry], root: Path | None) -> str:
    """Return the table row for one image & everything its operations did, or plan to do"""
    src = entries[0].src
    thumb = _thumb(entries)
    cell = f"<img src='data:image/jpeg;base64,{thumb}'>" if thumb else "<div class='missing'></div>"
    ops = "".join(
        f"<div class='op'><code>{escape(entry.op)}</code> {escape(_describe(entry, root))}"
        f"<div class='ext'>{escape(_fields(entry.ext_out))}</div></div>"
        for entry in entries
    )
    return (
        f"<tr><td class='thumb'>{cell}</td>"
        f"<td><div class='name'>{escape(_rel(src, root))}</div></td>"
        f"<td>{ops}</td></tr>"
    )


def _thumb(entries: list[Journal.Entry]) -> str:
    """Return a base64 JPEG thumbnail from the first path that holds an image, or "" if none do

    A dry run moves nothing, so its planned `dest` paths do not exist & the original `src` is what
    gets read - after a real run it is the other way around, hence trying the newest path first.
    """
    for path in [entry.dest for entry in reversed(entries) if entry.dest] + [entries[0].src]:
        try:
            with Image.open(path) as img:
                img.thumbnail(THUMB_SIZE)
                buffer = io.BytesIO()
                img.convert("RGB").save(buffer, "JPEG", quality=70)
        except OSError as ex:  # missing, or not readable as an image - try the next path
            _log.debug(f"{path}: no thumbnail - {ex}")
            continue
        return base64.b64encode(buffer.getvalue()).decode()
    return ""


def _describe(entry: Journal.Entry, root: Path | None) -> str:
    """Return what this entry's operation did, or planned to do, in one line"""
    if entry.dest is not None:
        return _rel(entry.dest, root)
    if "error" in entry.op_out:
        return f"error: {entry.op_out['error']}"
    return _fields(entry.op_out)


def _fields(data: dict[str, object]) -> str:
    """Return a `key: value` listing of an extractor's or operation's output"""
    return ", ".join(f"{key}: {value}" for key, value in data.items())
