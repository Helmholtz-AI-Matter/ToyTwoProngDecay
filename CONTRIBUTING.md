## Contributing

Thanks for contributing to `ToyTwoProngDecay`.

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

Before submitting changes, run:

```bash
uv run ruff check src tests
uv run mypy src/ttpd tests
uv run pytest
```

## Scope and style

- keep production code in `src/ttpd`
- keep tests in `tests`
- treat `src/notebooks` as documentation artifacts unless you are intentionally updating demos
- follow the existing NumPy-style docstrings and physics naming conventions
