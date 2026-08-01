# TODO

## Tasks

* Get basic model running & classifying images
    * Done - on an RTX 4070, runs on 100 images in about 10s
* First draft layout for `extractor` / `operation` / `journal` interfaces
    * Done
* `operation` - `move` - done
    * Add test coverage report - done
* `extractor` - `classification` - done
* Revisit interface - done
* Generic config parsing - done
    * Replace `Classification.Cfg` dataclass with Pydantic `BaseModel`
    * Add `OpCfg` discriminated union in `config.py` (`MoveCfg`, etc.) keyed on `op:` field
    * Rewrite `ClassificationConfig` to `categories: dict[str, list[OpCfg]]`
    * Add `build() -> Classification` on `ClassificationConfig` — instantiates real `Operation` objects
    * `Cfg.from_file()` returns `list[Extractor]` ready for `run_all`
* Implement CSV journal - done
    * Consider new interfaces - done
        * Creation to write to a specific directory - done, `CSVJournal(journal_dir=...)`
        * Filenames - should be in order of execution time - done,
          `local_img_org_journal_{UTC timestamp}.csv`
        * How to list all journals for optional undo - done, `get_files_for_undo()`
        * Storing metadata? Or are file paths always absolute? - done, `run_ops` resolves
          `img_dir` to absolute so journaled paths (and anything ops derive from them, ex.
          Move's `dest`) are undo-safe regardless of cwd at undo time
* Implement undos using journal - done
    * Ensure all undos are valid before running any of them to maintain state - done, `can_undo`
      is checked for every entry via `defer_exceptions` before any entry is undone
* Fixes from more testing - dry-run, run, undo, edge cases
  * Wire up `run_undos` in `main.py` - done, `--undo` flag
* `extractor` - `metadata` - done
    * Reads EXIF via Pillow - `date_taken` (+ sub-seconds) and `gps`, plus `file_modified` from
      the filesystem. Only what an op has a use for - the raw tag dump and GPS altitude were cut
    * `file_modified` is always reported and never folded into `date_taken`, so a guessed date and
      a real one stay distinguishable in the journal. `rename` falls back to it when there is no
      capture time, and `tag` can write it into `DateTimeOriginal` to make the fix permanent
    * `location` - `gps` is reverse geocoded to a city / state / country by `reverse_geocode`,
      which is offline (bundled GeoNames dataset). It matches the *nearest* populated place, so
      it can be wrong for a photo taken far from one - country / state hold up better than city
    * With no `operations` configured every image runs `noop`, so findings still land in the
      journal - see below
    * `require` gates the operations on keys being present - ex. only `move` photos with a
      `location`, leaving the rest to `noop`
* `operation` - `noop` - done
    * Journals an extractor's findings without touching the file, so an extractor can be run on
      its own to see what it finds. Undo of a `noop` is always valid and does nothing
* Think through the pipeline when one image gets more than one `operation`
    * Every op in a list is prepared against the *original* path - `Extractor.run` builds one
      `Operation.Data(src=path)` per op up front, so the 2nd op never learns that the 1st moved
      or renamed the file. Confirmed with `move` then `move`, which dies in the 2nd `plan` with
      `ValueError: .../a.jpg is not a file`
    * Worse, `prepare` only wraps `run` in `_safe` - a `plan` that raises escapes `run_ops` and
      kills the whole run partway through, so some images are done and some are untouched. The
      journal does hold everything that happened before the raise, so the partial run is undoable
    * `tag` will hit this from the other side - it does not move the file, so putting it before
      `rename` / `move` happens to work, but by luck rather than by design
    * Undo has the matching problem - `run_undos` replays entries in journal order, but a chain
      has to unwind in reverse (undo the `rename`, *then* the `move`). `can_undo` is also checked
      for every entry up front against the current filesystem, so a chained entry can look invalid
      only because the entry after it has not been undone yet
    * Options to weigh:
        * Thread the path forward - each op reports where the file ended up (`move` already
          returns `dest`) and the runner feeds that into the next op's `Data`. Means building
          `Data` per op at execution time instead of capturing it in `prepare`. Most flexible,
          most interface churn
        * Declare the intent on the op - ex. a `mutates_path` class attr, then either force those
          to run last or reject a config with more than one of them at parse time. Cheap, and
          fails fast on a bad config instead of halfway through a run
        * Fix the order - always `tag` -> `rename` -> `move`, ignoring the order in the config.
          Simplest, but implicit, and it breaks as soon as an op does not fit those buckets
        * Plan everything first, then execute - one pass computing each op's plan against the
          projected path, a second pass to run them. Would also make `--dry-run` honest about
          what a chain is going to do, since today it only ever plans against the original path
    * Whichever way this goes, decide whether a failed `plan` should abort the run (current
      behavior) or just skip that image and journal the error the way a failed `run` does
* `operation` - `rename`
* `operation` - `tag`
    * Write EXIF with `exiftool`, not Pillow - `Image.save(exif=...)` silently drops the tags on
      gif / bmp / tiff (measured, Pillow 12.0.0) and re-encodes the pixels. `piexif` writes in
      place but only covers jpg / webp. `exiftool` handles every format read here plus HEIC,
      errors loudly, and has `-P` to preserve mtime - cost is a binary dependency for write ops
* Any op that rewrites file bytes has to preserve mtime
    * A rename leaves mtime alone, but a content write bumps it - so tagging a photo that still
      has no `DateTimeOriginal` makes a later run see the rewrite time as its `file_modified`,
      silently. Wrap writes in a `preserve_mtime` helper in `utils.py`

## Working Notes
