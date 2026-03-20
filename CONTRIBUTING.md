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
