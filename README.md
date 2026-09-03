# Local Image Organizer

This project serves a few purposes:
- Help me organize my personal photos
- Provide me experience working with local machine learning models
- Maybe provide a simple tool to help others organize their personal photos as well

## Features

**This project is a work in progress - this README lists the project's goals, see [todo.md](./docs/todo.md) for pending work**

A config file can be provided like [example_cfg.yaml](./config/example_cfg.yaml) to run 
`extractors` on the local photos, and use their data to execute `operations` (`ops`).

The available `extractors` are:
- `classification`
    - Extracts an images classification based on use defined buckets - or `None` of it does not match any
- `metadata`
    - Extracts photo metadata - `date_taken`, `file_modified`, and `gps`
    - `date_taken` only carries a UTC offset when the camera recorded one (EXIF 2.31's
      `OffsetTime*` tags) - otherwise it stays a naive local wall-clock rather than being guessed
    - `file_modified` is always reported, and is kept separate from `date_taken` so a real capture
      time is never confused with a guess at one - an `operation` decides whether to fall back to it
    - `gps` is also reverse geocoded offline into a `location` (city / state / country)

The available `operations` to execute (once fed output data from an `extractor`):
- `rename`
    - Unifies naming of pictures to align with the format `IMG_{YYYY}{MM}{DD}_{HH}{MM}{SS}{MS_}`
    - If the `metadata` extractor fails, tries to infer from file name
- `move`
    - Moves file to a new location - ex. a subfolder for more nested organization
- `tag`
    - Adds new tag / label entries to the photo metadata
- `noop`
    - Does nothing to the file - records what the extractor found in the journal
    - Used automatically when an extractor has no operations configured

A large focus of this project is to provide `undo` functionality:
- While dry-runs are supported, maybe an incorrect `rename` or `subfolder` operation slipped through
- The user should be able to find what was done, and undo the whole operation

## Photo Organizing Goals

My goals are to have all photos & videos...:
- Named as above so they are organized by "date taken" when sorting by name
- A large main folder with most of my photos & videos
- Have sub-folders For specific larger categories - like `screenshots` / `documents` / specific events
- Add photo metadata to add additional tags that could be useful for sorting like:
    - `type: {people / landscape / pets / cars / indoors}`
    - `event: {name of event}`

Other useful tools discovered during this project:
- [digiKam](https://www.digikam.org/) - a very cool & feature complete photo viewing tool
    - Also uses local models to do classification & tagging
    - Certainly a better choice to do this kind of work, but then I wouldn't have a project

## Development

Uses [uv](https://github.com/astral-sh/uv) as a project manager:
- See their website for install if needed, then run `uv sync`

```bash
# Run the project
uv run main.py
# Type check
uv run mypy src/
# Lint & format
uv run ruff format && uv run ruff check
# Fix linting
uv run ruff check --fix
# Run all tests
uv run pytest
# For testing / an example, move some images into ./tests/my_data/ then:
uv run main.py --input-dir ./tests/my_data/ -c ./config/example_cfg.yaml --dry-run
```