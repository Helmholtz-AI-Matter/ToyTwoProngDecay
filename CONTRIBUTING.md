## Contributing

Thanks for contributing to `ToyTwoProngDecay`.

**Note**: This project aspires to support both `uv` workflows and plain `python`/`pip`
workflows.

This package assumes a few basic software engineering practices for all changes or contributions:

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
python -m pip install -e '.[dev]'
```

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

## Development

- CLI entry point: `ttpd`
- Package import: `ttpd`
- Install dev dependencies with `uv`: `uv sync --all-groups`
- Install dev dependencies with `pip`: `python -m pip install -e '.[dev]'`
- Run tests with `uv`: `uv run pytest`
- Run tests with plain Python: `python -m pytest`
- Run Ruff with `uv`: `uv run ruff check src tests`
- Run Ruff with plain Python: `python -m ruff check src tests`
- Run mypy with `uv`: `uv run mypy src/ttpd tests`
- Run mypy with plain Python: `python -m mypy src/ttpd tests`
- GitHub Actions runs the unit test suite on every push and pull request for
  Python `3.11`, `3.12`, and `3.13`
