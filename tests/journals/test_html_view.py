from dataclasses import replace
from pathlib import Path

from PIL import Image

from local_img_organizer.interfaces import Journal
from local_img_organizer.journals import html_view
from local_img_organizer.journals.csv_journal import CSVJournal


def _entry(op="move", src="/img/0.png", op_out=None, *, is_dry=False, uid="img0", ext=None):
    return Journal.Entry(
        op=op,
        img_uid=uid,
        src=Path(src),
        ext_out={"category": "cats"} if ext is None else ext,
        op_out={"dest": "/img/cats/0.png"} if op_out is None else op_out,
        is_dry=is_dry,
    )


def _make_img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 300), "red").save(path)
    return path


def test_render_groups_a_chain_into_one_row(tmp_path):
    """Test every entry of one image renders as a single row listing each op in journal order"""
    entries = [
        _entry(op="rename", op_out={"dest": "/img/IMG20240101120000.png"}),
        _entry(op="move", src="/img/IMG20240101120000.png", op_out={"dest": "/img/cats/IMG.png"}),
    ]
    page = html_view.render(entries, tmp_path / "journal.csv")

    assert page.count("<tr><td") == 1  # one row for the image, however many ops it has
    assert "<code>rename</code>" in page
    assert "<code>move</code>" in page
    assert "cats/IMG.png" in page
    assert "1 images, 1 move, 1 rename" in page


def test_render_separate_images_get_separate_rows(tmp_path):
    """Test entries with different img_uids are not grouped together"""
    entries = [_entry(uid="img0"), _entry(uid="img1", src="/img/1.png")]
    page = html_view.render(entries, tmp_path / "journal.csv")
    assert page.count("<tr><td") == len({entry.img_uid for entry in entries})


def test_render_banners_the_whole_journal_once(tmp_path):
    """Test dry/real is called out once up top - a journal is one run, so it is all or none"""
    entries = [_entry(op="rename"), _entry(op="move", uid="img1")]
    dry = [replace(entry, is_dry=True) for entry in entries]

    page = html_view.render(dry, tmp_path / "journal.csv")
    assert page.count("class='banner planned'") == 1
    assert "Nothing was changed" in page
    assert "banner executed" not in page

    page = html_view.render(entries, tmp_path / "journal.csv")
    assert page.count("class='banner executed'") == 1
    assert "These operations ran" in page
    assert "banner planned" not in page


def test_render_shows_the_shared_root_once(tmp_path):
    """Test the dir every path sits under is shown up top & left off of each row"""
    entries = [
        _entry(src="/photos/0.png", op_out={"dest": "/photos/cats/0.png"}),
        _entry(uid="img1", src="/photos/trip/1.png", op_out={"dest": "/photos/cats/1.png"}),
    ]
    page = html_view.render(entries, tmp_path / "journal.csv")

    assert "under <code>/photos</code>" in page
    assert "/photos/" not in page.split("</div>", 2)[2]  # only the header carries the full path
    assert "trip/1.png" in page
    assert "cats/1.png" in page


def test_render_without_a_shared_root(tmp_path):
    """Test paths with nothing in common are shown in full rather than raising"""
    entries = [_entry(src="relative.png", op_out={"dest": "/absolute/relative.png"})]
    page = html_view.render(entries, tmp_path / "journal.csv")

    assert "under <code>" not in page
    assert "/absolute/relative.png" in page


def test_render_escapes_paths_and_extractor_output(tmp_path):
    """Test journaled values are escaped rather than injected into the page as markup"""
    entries = [_entry(src="/img/<script>.png", ext={"category": "a & b"})]
    page = html_view.render(entries, tmp_path / "journal.csv")
    assert "<script>" not in page
    assert "&lt;script&gt;.png" in page
    assert "a &amp; b" in page


def test_render_thumbnail_from_src_for_a_dry_run(tmp_path):
    """Test a dry run thumbnails the original file, since its planned dest does not exist yet"""
    src = _make_img(tmp_path / "0.png")
    entries = [_entry(src=str(src), op_out={"dest": str(tmp_path / "cats/0.png")}, is_dry=True)]
    page = html_view.render(entries, tmp_path / "journal.csv")
    assert "data:image/jpeg;base64," in page
    assert "class='missing'" not in page


def test_render_thumbnail_from_dest_after_a_real_run(tmp_path):
    """Test an executed run thumbnails the dest, since the file no longer sits at src"""
    dest = _make_img(tmp_path / "cats/0.png")
    entries = [_entry(src=str(tmp_path / "0.png"), op_out={"dest": str(dest)})]
    page = html_view.render(entries, tmp_path / "journal.csv")
    assert "data:image/jpeg;base64," in page


def test_render_placeholder_when_no_path_holds_an_image(tmp_path):
    """Test an image that is gone from both src & dest renders a placeholder, not a broken page"""
    entries = [_entry(src=str(tmp_path / "gone.png"), op_out={"dest": str(tmp_path / "x.png")})]
    page = html_view.render(entries, tmp_path / "journal.csv")
    assert "class='missing'" in page
    assert "base64" not in page


def test_render_shows_an_error_entry(tmp_path):
    """Test the entry journaled for a failed run shows its error"""
    entries = [_entry(op_out={"error": "boom"})]
    page = html_view.render(entries, tmp_path / "journal.csv")
    assert "error: boom" in page


def test_render_shows_op_output_without_a_dest(tmp_path):
    """Test an op that moves nothing still shows what it recorded - ex. noop & tag"""
    page = html_view.render([_entry(op="noop", op_out={"noop": True})], tmp_path / "journal.csv")
    assert "noop: True" in page


def test_render_no_entries(tmp_path):
    """Test an empty journal renders a page saying so rather than raising"""
    page = html_view.render([], tmp_path / "journal.csv")
    assert "0 images, no entries" in page


def test_show_writes_next_to_the_journal_and_opens_it(tmp_path, monkeypatch):
    """Test show() reads the journal, writes a sibling .html & opens that file in a browser"""
    opened = []
    monkeypatch.setattr(html_view.webbrowser, "open", opened.append)
    journal = CSVJournal(journal_dir=tmp_path)
    journal.log(_entry())

    out = html_view.show(journal, journal.file)

    assert out == journal.file.with_suffix(".html")
    assert "cats/0.png" in out.read_text()
    assert opened == [out.resolve().as_uri()]
