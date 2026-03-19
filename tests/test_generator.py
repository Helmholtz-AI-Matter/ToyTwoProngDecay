"""Pytest suite covering the two-prong decay generator."""

import numpy as np
import torch
from typing import Optional

from ttpd import generator


np.random.seed(42)
torch.manual_seed(42)

DEFAULT_DEVICE = generator.DEFAULT_DEVICE


def identity_smear(batch: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """Pass through the inputs unchanged (useful for invariant tests)."""
    return batch


def test_simulate_shape_and_finite() -> None:
    factory = generator.SimulateFactory.create(
        product_mass=generator.mMu, smear_fn=identity_smear
    )
    theta = torch.tensor([[91.1876, 0.0], [91.1876, 1.0]], device=DEFAULT_DEVICE)
    output = factory.simulate(theta, generation_seed=10, smear_seed=20)
    assert output.shape == (2, 8)
    assert output.device == theta.device
    assert torch.isfinite(output).all()


def test_signal_back_to_back_phi() -> None:
    factory = generator.SimulateFactory.create(smear_fn=identity_smear)
    theta = torch.tensor([[91.1876, 0.0]] * 128, device=DEFAULT_DEVICE)
    events = factory.simulate(theta, generation_seed=123, smear_seed=456)
    phi1 = events[:, 1]
    phi2 = events[:, 5]
    dphi = torch.remainder(phi2 - phi1, 2 * torch.pi)
    assert torch.allclose(dphi, torch.full_like(dphi, torch.pi), atol=1e-5)


def test_signal_pt_and_pz_conservation() -> None:
    factory = generator.SimulateFactory.create(smear_fn=identity_smear)
    theta = torch.tensor([[91.1876, 0.0]] * 256, device=DEFAULT_DEVICE)
    events = factory.simulate(theta, generation_seed=321, smear_seed=654)
    pt1, phi1 = events[:, 0], events[:, 1]
    pt2, phi2 = events[:, 4], events[:, 5]
    eta1, eta2 = events[:, 2], events[:, 6]
    px_sum = pt1 * torch.cos(phi1) + pt2 * torch.cos(phi2)
    py_sum = pt1 * torch.sin(phi1) + pt2 * torch.sin(phi2)
    pz_sum = pt1 * torch.sinh(eta1) + pt2 * torch.sinh(eta2)
    zeros = torch.zeros_like(px_sum)
    assert torch.allclose(px_sum, zeros, atol=1e-5)
    assert torch.allclose(py_sum, zeros, atol=1e-5)
    assert torch.allclose(pz_sum, zeros, atol=1e-5)


def test_signal_invariant_mass_matches_theta() -> None:
    factory = generator.SimulateFactory.create(smear_fn=identity_smear)
    theta = torch.tensor([[91.1876, 0.0]] * 512, device=DEFAULT_DEVICE)
    events = factory.simulate(theta, generation_seed=111, smear_seed=222)
    invariant_mass = generator.invariant_mass_from_ptphieta(events)
    assert invariant_mass.shape == theta[:, 0].unsqueeze(1).shape
    assert torch.allclose(invariant_mass, theta[:, 0].unsqueeze(1), atol=1e-4)


def test_background_not_back_to_back() -> None:
    factory = generator.SimulateFactory.create()
    theta = torch.tensor([[91.1876, 1.0]] * 256, device=DEFAULT_DEVICE)
    events = factory.simulate(theta, generation_seed=11, smear_seed=13)
    dphi = torch.abs(events[:, 5] - events[:, 1])
    assert not torch.allclose(dphi, torch.full_like(dphi, torch.pi), atol=1e-3)


def test_reproducibility_with_seeds() -> None:
    factory = generator.SimulateFactory.create(smear_fn=identity_smear)
    theta = torch.tensor([[91.1876, 0.0]] * 64, device=DEFAULT_DEVICE)
    first = factory.simulate(theta, generation_seed=7, smear_seed=17)
    second = factory.simulate(theta, generation_seed=7, smear_seed=17)
    assert torch.allclose(first, second)


def test_create_simulator_matches_simulate() -> None:
    factory = generator.SimulateFactory.create(smear_fn=identity_smear)
    theta = torch.tensor([[91.1876, 0.0]] * 32, device=DEFAULT_DEVICE)

    simulator = factory.create_simulator(generation_seed=9, smear_seed=19)
    simulated = simulator(theta)
    direct = factory.simulate(theta, generation_seed=9, smear_seed=19)

    assert torch.allclose(simulated, direct)


def test_create_simulator_device_override() -> None:
    factory = generator.SimulateFactory.create(smear_fn=identity_smear)
    theta = torch.tensor([[91.1876, 0.0]] * 16, device=DEFAULT_DEVICE)
    override_device = torch.device("cpu")

    simulator = factory.create_simulator(
        generation_seed=13, smear_seed=17, device=override_device
    )
    events = simulator(theta)

    assert events.device == override_device
    assert torch.allclose(
        events,
        factory.simulate(
            theta,
            generation_seed=13,
            smear_seed=17,
            device=override_device,
        ),
    )


def test_custom_mass_and_smear_affects_pt() -> None:
    custom_pt_scale = 0.9

    def custom_smear(batch: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        scaled = batch.clone()
        scaled[:, 0] *= custom_pt_scale
        scaled[:, 4] *= custom_pt_scale
        return scaled

    decay = generator.TwoProngDecay(product_mass=0.211, smear_fn=custom_smear)
    theta = torch.tensor([[82.0, 0.0]] * 32, device=DEFAULT_DEVICE)
    events = decay.simulate(theta, generation_seed=5)
    assert torch.allclose(events[:, 0], events[:, 4], atol=1e-4)
    assert torch.all(events[:, 0] < torch.full_like(events[:, 0], 82.0))


def test_helper_round_trip_conversions() -> None:
    pt = torch.tensor([[40.0], [50.0]], device=DEFAULT_DEVICE)
    phi = torch.tensor([[0.1], [-0.2]], device=DEFAULT_DEVICE)
    eta = torch.tensor([[0.3], [-0.4]], device=DEFAULT_DEVICE)
    mass = torch.tensor([[0.105658], [0.105658]], device=DEFAULT_DEVICE)
    energy, px, py, pz = generator.to_cartesian(pt, phi, eta, mass)
    pt_rt, phi_rt, eta_rt, mass_rt = generator.to_ptphieta(energy, px, py, pz)
    assert torch.allclose(pt, pt_rt)
    assert torch.allclose(phi, phi_rt)
    assert torch.allclose(eta, eta_rt)
    assert torch.allclose(mass, mass_rt, atol=5e-3)
