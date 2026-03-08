# Contributing

## Dev setup

```bash
git clone https://github.com/jensebr/geobn.git
cd geobn
uv pip install -e ".[dev]"
```

The `dev` extra adds pytest to the standard install.

## Running tests

```bash
uv run pytest tests/ -v
```

All 125 tests must pass before committing. Tests are fully offline — no real network
calls are made (HTTP sources are mocked with `unittest.mock.patch`).

### Test conventions

- Fixtures live in `tests/conftest.py`.
- Use `unittest.mock.patch("requests.get", ...)` to mock HTTP sources.

## Adding a new data source

1. Create `src/geobn/sources/my_source.py` following the `DataSource` ABC.
2. Export from `src/geobn/sources/__init__.py` and `src/geobn/__init__.py`.
3. Add tests to `tests/test_new_sources.py` (or a new file).

Every source must implement:

```python
class MySource(DataSource):
    def fetch(self, grid: GridSpec | None = None) -> RasterData:
        ...
```

If the source requires credentials, validate them in `__init__()` before `fetch()` is called.

## Building docs locally

```bash
uv pip install -e ".[docs]"
mkdocs serve
# Browse at http://127.0.0.1:8000
```

Build static site and check for broken links:

```bash
mkdocs build --strict
```

The `--strict` flag treats mkdocstrings warnings (missing symbols, broken links) as
errors.

## Git workflow

After completing any meaningful unit of work, commit and push:

```bash
git add <specific files>
git commit -m "concise present-tense description"
git push origin main
```

Rules:
- Stage only relevant files (never `git add -A` blindly)
- Write concise, present-tense commit messages: `"add WCSSource"`, `"fix nodata sentinel"`
- Always run `uv run pytest tests/ -v` before committing

GitHub Actions deploys docs automatically on push to `main`.
