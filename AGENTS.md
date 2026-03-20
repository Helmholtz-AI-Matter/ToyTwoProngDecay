# AGENTS GUIDE

## Purpose
- Give every agent entering this tree the context they need before editing anything: this repo is an open-source toy two-prong decay simulator built with Python, NumPy, PyTorch, and SBI.
- Most logic belongs in `src/ttpd`, `tests`, and the historical `src/notebooks` artifacts. Keep the instructions below in sync as you add commands, linting, or style expectations.

## Cursor & Copilot instructions
- No `.cursor` or `.cursorrules` directories were found under this tree, so no extra cursor rules apply today.
- The repository also lacks `.github/copilot-instructions.md`; assume Copilot behaves with its default heuristics unless you add such a file later.

## Environment setup
- Python 3.11 or newer is required per `pyproject.toml`; prefer to recreate a local interpreter with `python -m venv .venv` or reuse `py313` by running `source py313/bin/activate`.
- Upgrade pip/setuptools before installing dependencies: `python -m pip install --upgrade pip setuptools wheel`.
- Install runtime and dev requirements with `uv sync --all-groups` (it reads `pyproject.toml` and `uv.lock`).
- Plain `pip` workflows should also work: use `python -m pip install .` for runtime installs or `python -m pip install -e '.[dev,docs]'` for editable development installs.
- For repeatable shells, keep `uv.lock` in sync and rerun `uv sync --all-groups` whenever dependencies change.
- After every dependency change, rerun `uv lock` and `uv sync --all-groups`, verify `uv.lock`, and commit both `pyproject.toml`/`uv.lock` together so the lock file never lags.
- This project aspires to support both `uv` workflows and plain `python`/`pip` workflows; keep both paths working when changing packaging metadata or contributor docs.
- Activate the same interpreter you use for CI/UV (`py313/bin/activate`) before invoking tests or builds, which keeps CUDA/CPU detection consistent.

## Build, packaging, and execution
- `uv build` produces a wheel or sdist depending on the `uv` defaults; inspect `dist/` afterwards to verify metadata.
- `python -m build` is the plain-Python alternative for producing sdists and wheels; use it when verifying the non-`uv` packaging path.
- `uv run ttpd` launches the CLI entry point defined under `[project.scripts]`; `python -m ttpd` is the matching module entry point if you add one later.
- `uv run --` lets you pass arbitrary commands through the `uv` shim; e.g., `uv run -- python -m module` picks up the locked interpreter and dependencies from this workspace.
- Treat artifacts under `src/notebooks` as documentation: rerun them locally, then export new HTML/JSON when the narrative needs updating.
- `uv build` is also the recommended way to exercise packaging metadata; if you override `uv` defaults, note the override in this file for future agents.

## Documentation site
- The GitHub Pages site is notebook-driven: canonical sources live in `src/notebooks/*.py`, with marimo notebooks acting as the authoritative docs source.
- The initial public docs build currently includes only `src/notebooks/example_simulate_factory.py`; do not publish `src/notebooks/mnpe-demo.py`, `src/notebooks/toy2decay-simulator.py`, or their derived artifacts unless the docs scope changes.
- Generated notebook artifacts under `docs/notebooks/*.ipynb` are temporary build outputs for MkDocs and must not be committed.
- Build the curated docs site with `uv run python scripts/build_docs.py build` or `python scripts/build_docs.py build`.
- Preview the site locally with `uv run python scripts/build_docs.py serve` or `python scripts/build_docs.py serve`.
- The MkDocs configuration lives in `mkdocs.yml`; keep the notebook export script and docs navigation aligned when adding or removing published notebooks.

## Testing
- `uv run pytest` runs the entire Pytest suite located under `tests/`; it respects the `dev` dependency group and uses whichever interpreter `uv` selects.
- To focus on a single file, run `uv run pytest tests/test_generator.py` or `python -m pytest tests/test_generator.py` when you need extra Pytest flags.
- To run a single case, append the node id: `uv run pytest tests/test_generator.py::test_signal_pt_conservation` or use `python -m pytest tests/test_generator.py -k pt_conservation`.
- Reuse the `DEFAULT_DEVICE` constant from `tests/test_generator.py` when writing new tests so you benefit from the CUDA/CPU fallback already captured there.
- Seed determinism with `torch.manual_seed` and `np.random.seed`; reset seeds before each test if reproducibility matters for your change.
- Tests now live in `tests/test_generator.py` and use Pytest signatures and NumPy-style docstrings; refer to that file for examples of how to seed randomness and assert invariants about momentum/invariant mass.

## Generator & simulation
- `src/ttpd/kinematics.py` now hosts the physics helpers (`to_cartesian`, `to_ptphieta`, `ptphieta_to_epxyz`, `invariant_mass_from_epxyz`, etc.) plus the shared constants `mZ0` and `mMu`, while `src/ttpd/generator.py` keeps the abstractions `TwoProngDecay` and the `SimulateFactory` dataclass.
- `TwoProngDecay` exposes `generate`, `smear`, and `simulate` methods with configurable product mass, smearing function, device placement, and optional seeds to keep samplers deterministic.
- The default smearing function mirrors the notebook’s `smear_ptphieta` (pt/dphi/eta jitter, independent phi/eta resets); inject a custom callable when you need a different detector model.
- `SimulateFactory.create(...)` returns a dataclass that keeps the `TwoProngDecay` instance on the `decay` attribute while providing the single-argument `simulate(theta)` callable shown in the notebook. Pass `generation_seed`/`smear_seed` through the factory to reproduce specific draws or share random states across runs.
- When updating these modules, maintain NumPy-style docstrings and keep helper exports stable so that notebooks/tests relying on `to_ptphieta` or `invariant_mass_from_ptphieta` continue working.
- When tests need to target CPU-only or GPU-specific behavior, pass `device=DEFAULT_DEVICE` explicitly and document the assumption in the test docstring comment.
- PyTorch distributions (e.g., `torch.distributions.uniform.Uniform`) are already part of the helpers; if you add new samples, keep naming consistent and test the sampling shapes.

## Linting & formatting
- Ruff is part of the `dev` dependency group; run `uv run ruff check src tests` before finishing Python changes.
- The plain-Python alternative is `python -m ruff check src tests`; keep it working for contributors who do not use `uv`.
- Mypy is part of the `dev` dependency group; run `uv run mypy src/ttpd tests` when you add or modify typed Python code.
- The plain-Python alternative is `python -m mypy src/ttpd tests`; keep it working alongside the `uv` path.
- Keep line length near 88 characters, keep 4-space indentation, and prefer trailing commas in multi-line collections (the existing torch setup in `tests/test_generator.py` already follows these habits).
- Ruff configuration lives in `pyproject.toml`; notebook artifacts under `src/notebooks` are excluded from linting and type checking.
- Document any formatter you add by updating this file with how to run it so future agents won’t have to guess the command.
- When adding new modules, run `uv run ruff check src tests` or `python -m ruff check src tests` locally before committing, and treat `uv run pytest` or `python -m pytest` as the final verification step.

## Repository layout & artifacts
- `pyproject.toml` + `uv.lock` drive dependency resolution and lock PyTorch/NumPy versions required for the simulator.
- Production code lives under `src/ttpd`; add new modules alongside `generator.py` once you start fleshing out the package.
- Legacy or illustrative notebooks sit in `src/notebooks`; treat the `.py`, `.ipynb`, `.html`, and `sbi-logs` artifacts as documentation that only changes when you regenerate them deliberately.
- Tests live in `tests/` and currently focus on the `generate_decay_event` helper; any new features should ship with a test that mirrors the existing torch-based style.

## Physics constants & units
- The physics in `tests/test_generator.py` back-propagates constants such as `mZ0 = 91.1876` GeV and `mMu = 0.105658` GeV; keep these values centralized when adding new generators.
- Document the units for any new fields you introduce (GeV, radians, dimensionless flags) so downstream users can interpret the tensors without guessing.
- If you need to adjust masses beyond the PDG values, call out the assumption with a comment (e.g., `# TODO: load PDG mass from table once available`).
- Background offsets (phi, eta, pt) are treated as perturbations that vanish for signal events; follow the same pattern so backgrounds remain controlled by the `theta[:,1]` flag.

## Code style guidelines

### Imports
- Group imports in three sections: standard library, third-party, and local packages, each separated by a single blank line.
- Use parentheses for multi-line imports and prefer explicit `from typing import Tuple` statements when typing tensors or tuples of tensors.
- Local imports should align with the `ttpd` package structure once you introduce more modules; avoid importing from `src` directly.

### Formatting
- Follow PEP 8 defaults: 4 spaces per indent level, blank lines between top-level functions, and hanging indents for wrapped arguments (notice how `torch.hstack` arguments are indented in `tests/test_generator.py`).
- Aim for ~88 characters per line; breaks longer than 88 should favor natural tensor segments or function arguments, not arbitrary halves.
- Keep expressions readable by breaking them before binary operators and using parentheses when chaining multiple tensor operations.

### Typing & signatures
- Annotate every helper with precise return types like `-> torch.Tensor` or `-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`.
- Use `torch.Tensor` explicitly instead of bare `Tensor` or `Any`; document shape expectations in docstrings when necessary.
- Prefer typed arguments rather than leaving configuration bits untyped; for example, `theta: torch.Tensor` and `device: torch.device` are better than plain `tensor` arguments.
- Keep newly added production helpers fully typed so `uv run mypy src/ttpd tests` stays clean.
- Use NumPy-style docstrings for all public helpers (Parameters/Returns sections) so the new generator/tests stay consistent.

### Documentation & comments

### Naming conventions
- Stick to `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for module-level constants like `DEFAULT_DEVICE`, `mZ0`, or `mMu`.
- Prefix helper names with `_` if they are truly private to the module and never used externally (this keeps public APIs clean in `__init__`).
- Use descriptive names for physics inputs (`theta`, `phi1`, `background_eta_offset`) rather than abbreviations so future readers immediately know what each tensor represents.

### Documentation & comments
- All public helpers should have docstrings following the NumPy-style sections (`Parameters`, `Returns`, etc.), like the current `generate_decay_event` docstring.
- Reserve inline comments for clarifying physics assumptions or non-obvious tensor reshapes; avoid restating what the code already expresses.
- When you leave a TODO, explain the missing context (e.g., `# TODO: load PDG mass from table once available`).

### Error handling & validation
- Raise explicit exceptions (`ValueError`, `RuntimeError`) for invalid inputs rather than relying on PyTorch to fail deep inside the computation graph.
- Use `assert` for internal invariants (as in the existing shape checks) and include informative f-strings so the failure message reveals what went wrong.
- Validate tensor shapes and devices early (e.g., `theta = theta.to(device)`); prefer clarity over clever shaping logic.

### Torch & numerics
- Stick with PyTorch primitives (`torch.sqrt`, `torch.cos`, `torch.distributions`) for all physics math; avoid NumPy unless you intentionally convert tensors to arrays.
- Keep device-aware constants at module scope so device selection lives in one place (`DEFAULT_DEVICE` is the canonical example).
- Use `torch.distributions` when sampling and store a distribution instance if you sample repeatedly so seeding remains consistent.

### Device & dtype awareness
- Default tensors to the precision that matches your physics budget (float32 today); avoid mixing float64 unless you account for the performance impact on CUDA.
- Keep all tensors on the same device before arithmetic: `theta = theta.to(device)` is the pattern used in `generate_decay_event`.
- When creating tensors like `torch.ones_like`, pass `dtype` and `device` explicitly if you are cloning from a different source, so PyTorch does not copy back to CPU.

### Randomness & reproducibility
- Seed both NumPy and PyTorch once at the module level (`np.random.seed(42)` / `torch.random.manual_seed(42)`), and reseed inside tests where deterministic pairs are expected.
- Avoid using `torch.manual_seed` in production logic; reserve it for test modules or experiment scripts so the generator remains statistical by default.
- Document in comments when draw order, shuffling, or offsets depend on multiple distributions so future agents know which seed affects which sampler.

### Testing conventions
- Seed randomness at the top of the test module with `np.random.seed(42)` and `torch.random.manual_seed(42)` and reseed where determinism matters inside tests.
- Use `torch.allclose` with tolerance arguments (`atol=1e-5` or `1e-4`) instead of bare equality when comparing floats.
- Keep functional tests focused on meaningful invariants—momentum/tensor conservation, phi separation, invariant mass, reproducibility—as the existing suite does.
- Prefer tensor helpers (e.g., `to_ptphieta`) to keep tests readable and reusable.

### Notebook and artifact handling
- Treat `src/notebooks` as documentation, not production code; edit them only when you need to explain new physics or workflow details.
- Regenerate `.html`/`.json` exports when you change the corresponding `.ipynb` or `.py` demo so agents can trace the source-to-artifact relationship.
- Keep the `src/notebooks/sbi-logs` directory untouched unless you explicitly rerun the simulator and want to update logs.

## Running new experiments
- Note any new experiment commands in this section (e.g., `PYTORCH_CUDA_ALLOC_CONF=... uv run -- python experiments/run.py`) so other agents can reproduce the setup.
- If an experiment needs special env vars or GPU flags, capture them here along with the reason they are required.
- Log the random seeds and configuration files you used for each experiment so rerunning it produces the same logs.
- Keep experiment-specific scripts under a dedicated directory (e.g., `experiments/`) and document how they depend on the rest of the package.

## Documentation & onboarding
- When you introduce new abstractions (like `TwoProngDecay` or `SimulateFactory`), document their usage both in this guide and via inline docstrings so future agents can rely on nominal behavior.
- Update the README or add short markdown notes if you expose new CLI commands or dataset-generation scripts.
- Keep `CONTRIBUTING.md` aligned with the current expectations for tests, linting, version control, and typed Python code.
- Capture any new physics plots or notebook snippets as documentation under `src/notebooks`; do not merge large binary files without a justification.

## Reminders for future agents
- Update this document whenever you add build/test commands, new dependency groups, or a linter/formatter configuration.
- Document new entry points in the `Build, packaging, and execution` section so other agents know how to run them.
- Keep new Python logic inside `src/ttpd` and `tests` unless you have a strong justification to expand the package layout.
