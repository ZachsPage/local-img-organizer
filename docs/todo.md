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
* Generic config parsing
    * Replace `Classification.Cfg` dataclass with Pydantic `BaseModel`
    * Add `OpCfg` discriminated union in `config.py` (`MoveCfg`, etc.) keyed on `op:` field
    * Rewrite `ClassificationConfig` to `categories: dict[str, list[OpCfg]]`
    * Add `build() -> Classification` on `ClassificationConfig` — instantiates real `Operation` objects
    * `Cfg.from_file()` returns `list[Extractor]` ready for `run_all`
* Implement CSV journal
* Implement undos using journal
    * Ensure all undos are valid before running any of them to maintain state
* Fixes from more testing - dry-run, run, undo, edge cases
* `extractor` - `metadata`
* `operation` - `rename`
* `operation` - `tag`

## Working Notes