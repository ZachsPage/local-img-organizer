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
* `operation` - `noop` - done
* Think through the pipeline when one image gets more than one `operation`
    * Ex. `move` then `rename` - the 2nd op's `plan` is built against the original path, so it
      fails once the 1st op has moved the file. Not considered in the original design
    * Undo has the matching problem - a chain has to unwind in reverse (undo `rename`, *then*
      `move`), but `can_undo` is checked for every entry up front, before any undo has run
    * Options to weigh:
        * Give each image a UUID and track its current path in a file registry that ops
          read/update, so the next op looks up where the file is instead of it being threaded
          through
        * Fix the order - always `tag` -> `rename` -> `move`, ignoring the order in the config.
          Simplest, but implicit, and it breaks as soon as an op does not fit those buckets
    * Decide whether a failed `plan` should abort the run (current behavior) or just skip that
      image and journal the error the way a failed `run` does
* `operation` - `rename`
    * Needs a way to route extractor metadata (ex. `date_taken`) into the new name
    * `tag` needs the same metadata access, but should only write it when the tag doesn't already
      exist on the file (ex. skip if `DateTimeOriginal` is already set)
    * Also want to run an op only when a metadata value matches something specific - ex. `move`
      only when `location` is a given city. `rename` is the first case needing this, but think it
      through generally rather than one-off for `rename`
* `operation` - `tag`
    * Use `exiftool` (`-P` to preserve mtime) to tag since it's in place, not Pillow as it can
      drop tags & re-encode pixels
* Any op that rewrites file bytes has to preserve mtime
    * A rename leaves mtime alone, but a content write bumps it - so tagging a photo that still
      has no `DateTimeOriginal` makes a later run see the rewrite time as its `file_modified`,
      silently. Wrap writes in a `preserve_mtime` helper in `utils.py`

## Working Notes
