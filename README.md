## ToyTwoProngDecay

`ToyTwoProngDecay` (`ttpd`) is a fast, simple simulator for two-particle decays.
It provides a configurable factory interface for generating and smearing paired
final-state kinematics for toy Monte Carlo studies.

The simulator is intentionally approximate. Its goal is not precision event
generation, but a representation that is close enough to physically sensible
two-prong decays for method-development studies such as unfolding.

## Why this project exists

- fast iteration on toy Monte Carlo studies
- simple factory-based interface for configurable generators
- approximate detector smearing for downstream inference workflows
- useful baseline simulator for unfolding and related methodology work

## Installation

From a local checkout with plain `pip`:

```bash
pip install .
```

From a local checkout with `uv`:

```bash
uv sync --all-groups
```

Once published on PyPI, installation will also work as:

```bash
pip install ToyTwoProngDecay
uv add ToyTwoProngDecay
```

## Quick start

```python
import torch

from ttpd.generator import SimulateFactory, invariant_mass_from_ptphieta, mZ0

factory = SimulateFactory.create(device=torch.device("cpu"))
theta = torch.tensor([[mZ0, 0.0], [85.0, 1.0]])
events = factory.simulate(theta, generation_seed=123, smear_seed=321)
masses = invariant_mass_from_ptphieta(events)
```

The `theta` tensor uses two columns:

- `theta[:, 0]`: parent mass in GeV
- `theta[:, 1]`: signal/background flag

## Development

- CLI entry point: `ttpd`
- Package import: `ttpd`
- Run tests locally: `uv run pytest`
- GitHub Actions runs the unit test suite on every push and pull request for
  Python `3.11`, `3.12`, and `3.13`
