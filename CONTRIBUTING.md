# Contributing

Bug reports, new data sources, doc fixes and feature ideas are all welcome. Open an issue or a pull request.

## Setup

```bash
git clone https://github.com/jensbremnes/geobn.git
cd geobn
uv sync --frozen
uv run --frozen --with pytest python -m pytest tests/
```

Without uv, run `pip install -e ".[dev]"` and then `pytest tests/`.

## Before opening a PR

- The tests pass.
- New behaviour has tests.
- The docs are updated if the public API changed.

## Adding a new data source

1. Create `src/geobn/sources/my_source.py`, subclassing `DataSource`.
2. Export it from `src/geobn/sources/__init__.py` and `src/geobn/__init__.py`.
3. Add tests to `tests/test_sources.py`.
4. Add a `:::` mkdocstrings directive to the right page in `docs/api/sources/`.
