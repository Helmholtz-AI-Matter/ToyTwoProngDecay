## Contributing

Thanks for contributing to `ToyTwoProngDecay`.

This project aspires to support both `uv` workflows and plain `python`/`pip`
workflows.

This package assumes a few basic software engineering practices for all changes:

- add or update unit tests for behavior changes
- run linting with `ruff`
- keep version-controlled changes focused and reviewable
- provide and maintain type hints for Python code
- run type checking before opening a pull request

## Local workflow

Install the development environment:

```bash
uv sync --all-groups
```

Or with plain Python and pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,docs]'
```

## Build docs locally

The documentation site is notebook-driven. Marimo notebooks under
`src/notebooks/*.py` are the canonical source, and the published site only
includes a curated subset of them. Generated `ipynb` files under
`docs/notebooks/` are build artifacts and should not be committed.

Build the docs locally with `uv`:

```bash
uv run python scripts/build_docs.py build
```

Preview the docs locally with a live server:

```bash
uv run python scripts/build_docs.py serve
```

Or with plain Python:

```bash
python scripts/build_docs.py build
python scripts/build_docs.py serve
```

The local preview defaults to `http://127.0.0.1:8000`.

Before submitting changes, run:

```bash
uv run ruff check src tests
uv run mypy src/ttpd tests
uv run pytest
```

Or with plain Python:

```bash
python -m ruff check src tests
python -m mypy src/ttpd tests
python -m pytest
```

## Scope and style

- keep production code in `src/ttpd`
- keep tests in `tests`
- treat `src/notebooks` as documentation artifacts unless you are intentionally updating demos
- follow the existing NumPy-style docstrings and physics naming conventions

## Things to remember

- CLI entry point: `ttpd`
- Package import: `ttpd`
- Install dev dependencies with `uv`: `uv sync --all-groups`
- Install dev dependencies with `pip`: `python -m pip install -e '.[dev,docs]'`
- Run tests with `uv`: `uv run pytest`
- Run tests with plain Python: `python -m pytest`
- Run Ruff with `uv`: `uv run ruff check src tests`
- Run Ruff with plain Python: `python -m ruff check src tests`
- Run mypy with `uv`: `uv run mypy src/ttpd tests`
- Run mypy with plain Python: `python -m mypy src/ttpd tests`
- Build curated docs with `uv`: `uv run python scripts/build_docs.py build`
- Build curated docs with plain Python: `python scripts/build_docs.py build`
- Preview curated docs with `uv`: `uv run python scripts/build_docs.py serve`
- Preview curated docs with plain Python: `python scripts/build_docs.py serve`
- GitHub Actions runs the unit test suite on every push and pull request for
  Python `3.11`, `3.12`, and `3.13`
- GitHub Pages publishes a curated docs site from selected marimo notebooks
  without committing generated `ipynb` artifacts to the repository
