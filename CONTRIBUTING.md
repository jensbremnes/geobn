# Contributing

Bug reports, new data sources, documentation fixes, and feature ideas are all welcome — feel free to open an issue or PR.

## Setup

```bash
git clone https://github.com/jensebr/geobn.git
cd geobn
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

## Before submitting a PR

- All tests pass (`uv run pytest tests/ -v`)
- New behaviour is covered by tests
- Docs updated if the public API changed

## Adding a new data source

1. Create `src/geobn/sources/my_source.py` following the `DataSource` ABC.
2. Export from `src/geobn/sources/__init__.py` and `src/geobn/__init__.py`.
3. Add tests to `tests/test_sources.py`.
4. Add a `:::` mkdocstrings directive in the appropriate `docs/api/sources/` page.
