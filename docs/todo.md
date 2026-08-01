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
* `extractor` - `metadata`
* `operation` - `rename`
* `operation` - `tag`

## Working Notes
