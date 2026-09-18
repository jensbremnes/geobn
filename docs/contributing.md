# Contributing

## Setup

```bash
git clone https://github.com/jensbremnes/geobn.git
cd geobn
uv sync --frozen
```

Without uv, use `pip install -e ".[dev]"`.

## Running tests

```bash
uv run --frozen --with pytest python -m pytest tests/
```

The tests run offline. HTTP sources are mocked with `unittest.mock.patch`, and shared
fixtures live in `tests/conftest.py`.

## Adding a new data source

1. Create `src/geobn/sources/my_source.py`, subclassing `DataSource`.
2. Export it from `src/geobn/sources/__init__.py` and `src/geobn/__init__.py`.
3. Add tests to `tests/test_sources.py`.
4. Add a `:::` mkdocstrings directive to the right page in `docs/api/sources/`.

A source implements one method:

```python
class MySource(DataSource):
    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        ...
```

If the source needs credentials, check them in `__init__()` so that a missing key
fails early instead of inside `fetch()`.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve   # browse at http://127.0.0.1:8000
```

Before pushing doc changes, run:

```bash
mkdocs build --strict
```

`--strict` turns mkdocstrings warnings (missing symbols, broken links) into errors.

## Pull requests

Work on a branch and open a pull request against `main`. Stage only the files that
belong to the change, and keep commit messages short and in the present tense
(`add WCSSource`, `fix nodata sentinel`). The docs site is deployed by GitHub Actions
when changes land on `main`.
