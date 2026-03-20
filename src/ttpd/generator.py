"""Physics helpers and generators for toy MNPE Monte Carlo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Default device for generators (GPU if available, CPU otherwise)."""

mZ0 = 91.1876  # GeV, TODO perhaps retrieve this from pdg
"""Z boson mass used in signal generation (GeV)."""

mMu = 0.105658  # GeV, TODO perhaps retrieve this from pdg
"""Muon mass used for signal decay products (GeV)."""

TensorQuad: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

SmearFn = Callable[[torch.Tensor, int | None], torch.Tensor]
"""Signature for smear functions (batch, seed) -> smeared batch."""


def to_cartesian(
    pt: torch.Tensor, phi: torch.Tensor, eta: torch.Tensor, mass: torch.Tensor
) -> TensorQuad:
    """Convert (pt, phi, eta, mass) to (energy, px, py, pz).

    Parameters
    ----------
    pt: torch.Tensor
        Transverse momentum per prong.
    phi: torch.Tensor
        Azimuthal angle per prong.
    eta: torch.Tensor
        Pseudorapidity per prong.
    mass: torch.Tensor
        Mass per prong.

    Returns
    -------
    TensorQuad
        Energy, px, py, pz tensors with the same shape as the inputs.
    """
    pabs = pt * torch.cosh(eta)
    energy = torch.sqrt(pabs**2 + mass**2)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return energy, px, py, pz


def to_ptphieta(
    energy: torch.Tensor, px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor
) -> TensorQuad:
    """Convert (energy, px, py, pz) to (pt, phi, eta, mass).

    Parameters
    ----------
    energy, px, py, pz: torch.Tensor
        Four-vector components for a batch of particles.

    Returns
    -------
    TensorQuad
        pt, phi, eta, mass tensors with the same shape as the inputs.
    """
    pt = torch.sqrt(px**2 + py**2)
    pabs = torch.sqrt(px**2 + py**2 + pz**2)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / pt)
    mass = torch.sqrt(energy**2 - pabs**2)
    return pt, phi, eta, mass


def ptphieta_to_epxyz(batch_decay_vectors: torch.Tensor) -> torch.Tensor:
    """Translate pt/phi/eta/mass vectors into energy/momentum coordinates.

    Parameters
    ----------
    batch_decay_vectors: torch.Tensor
        Batched vectors with shape (B, 8):
        [mu1 pt, mu1 phi, mu1 eta, mu1 mass, mu2 pt, mu2 phi, mu2 eta, mu2 mass].

    Returns
    -------
    torch.Tensor
        Batched vectors with shape (B, 8):
        [mu1 E, mu1 px, mu1 py, mu1 pz, mu2 E, mu2 px, mu2 py, mu2 pz].
    """
    mu1pt, mu1phi, mu1eta, mu1m = (
        batch_decay_vectors[:, 0].unsqueeze(1),
        batch_decay_vectors[:, 1].unsqueeze(1),
        batch_decay_vectors[:, 2].unsqueeze(1),
        batch_decay_vectors[:, 3].unsqueeze(1),
    )
    mu2pt, mu2phi, mu2eta, mu2m = (
        batch_decay_vectors[:, 4].unsqueeze(1),
        batch_decay_vectors[:, 5].unsqueeze(1),
        batch_decay_vectors[:, 6].unsqueeze(1),
        batch_decay_vectors[:, 7].unsqueeze(1),
    )

    mu1e, mu1px, mu1py, mu1pz = to_cartesian(mu1pt, mu1phi, mu1eta, mu1m)
    mu2e, mu2px, mu2py, mu2pz = to_cartesian(mu2pt, mu2phi, mu2eta, mu2m)

    return torch.hstack([mu1e, mu1px, mu1py, mu1pz, mu2e, mu2px, mu2py, mu2pz])


def invariant_mass_from_epxyz(
    batch_decay_vectors: torch.Tensor, prong_mass: float = mMu
) -> torch.Tensor:
    """Calculate invariant mass from energy/momentum vectors.

    Parameters
    ----------
    batch_decay_vectors: torch.Tensor
        Batched vectors with shape (B, 8) representing two particles.
        Encoding is expected as:
        mu1e, mu1px, mu1py, mu1pz, mu2e, mu2px, mu2py, mu2pz
    prong_mass: float
        mass of the decay product to reconstruct mass from

    Returns
    -------
    torch.Tensor
        Invariant mass per event of shape (B, 1).
    """
    mu1e, mu1px, mu1py, mu1pz = list(range(4))
    mu2e, mu2px, mu2py, mu2pz = list(range(4, 8))

    mu1E = batch_decay_vectors[:, mu1e]
    mu2E = batch_decay_vectors[:, mu2e]

    mu1Px = batch_decay_vectors[:, mu1px]
    mu2Px = batch_decay_vectors[:, mu2px]

    mu1Py = batch_decay_vectors[:, mu1py]
    mu2Py = batch_decay_vectors[:, mu2py]

    mu1Pz = batch_decay_vectors[:, mu1pz]
    mu2Pz = batch_decay_vectors[:, mu2pz]

    value = torch.zeros(
        [batch_decay_vectors.shape[0], 1], device=batch_decay_vectors.device
    )

    value[:, 0] = 2 * (prong_mass**2)
    value[:, 0] += 2 * (mu1E * mu2E)
    value[:, 0] -= 2 * (mu1Px * mu2Px + mu1Py * mu2Py + mu1Pz * mu2Pz)
    return torch.sqrt(value)


def invariant_mass_from_ptphieta(batch_decay_vectors: torch.Tensor) -> torch.Tensor:
    """Calculate invariant mass from pt/phi/eta/mass vectors.

    Parameters
    ----------
    batch_decay_vectors: torch.Tensor
        Batched vectors with shape (B, 8).
        Encoding is expected as:
        [mu1 pt, mu1 phi, mu1 eta, mu1 mass, mu2 pt, mu2 phi, mu2 eta, mu2 mass]

    Returns
    -------
    torch.Tensor
        Invariant mass per event of shape (B, 1).
    """
    return invariant_mass_from_epxyz(ptphieta_to_epxyz(batch_decay_vectors))


def _kaellen_function(
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    return (
        alpha**2
        + beta**2
        + gamma**2
        - 2 * alpha * beta
        - 2 * alpha * gamma
        - 2 * beta * gamma
    )


def default_smear_ptphieta(
    batch_decay_vectors: torch.Tensor, seed: int | None = None
) -> torch.Tensor:
    """Apply detector-inspired smearing to decay outputs.

    Parameters
    ----------
    batch_decay_vectors: torch.Tensor
        Batched decay vectors with shape (B, 8).
    seed: int | None
        Optional torch manual seed for reproducibility.

    Returns
    -------
    torch.Tensor
        Smeared decay vectors with the same shape as the input.

    some inspiration was taken from
    [1] https://ar5iv.labs.arxiv.org/html/2212.07338
    """
    device = batch_decay_vectors.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    mu1pt, mu1phi, mu1eta = 0, 1, 2
    mu2pt, mu2phi, mu2eta = 4, 5, 6

    value = batch_decay_vectors.clone()

    #uniform range approximated from [1] fig 10
    sigmas = (
        torch.rand(value.shape[0], generator=generator, device=device) * 0.03 + 0.01
    )

    pt1_smear = torch.normal(
        mean=torch.ones_like(value[:, mu1pt]), std=sigmas, generator=generator
    )
    pt2_smear = torch.normal(
        mean=torch.ones_like(value[:, mu2pt]), std=sigmas, generator=generator
    )
    value[:, mu1pt] *= pt1_smear
    value[:, mu2pt] *= pt2_smear

    phi_smear = torch.normal(
        mean=torch.ones_like(value[:, mu1phi]),
        std=torch.ones_like(sigmas) * 0.01,
        generator=generator,
    )
    value[:, mu1phi] *= phi_smear
    # reverse phi_smear
    value[:, mu2phi] *= torch.flip(phi_smear, dims=(0,))

    eta_smear = torch.normal(
        mean=torch.ones_like(value[:, mu1eta]),
        std=torch.ones_like(sigmas) * 0.01,
        generator=generator,
    )
    value[:, mu1eta] *= eta_smear
    # reverse phi_smear
    value[:, mu2eta] *= torch.flip(eta_smear, dims=(0,))

    return value


class TwoProngDecay:
    """Generator for two-prong decays with configurable smearing."""

    def __init__(
        self,
        product_mass: float = mMu,
        smear_fn: SmearFn = default_smear_ptphieta,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        self.product_mass = product_mass
        self.smear_fn = smear_fn
        self.device = device

    def generate(
        self,
        theta: torch.Tensor,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate unsmeared decay kinematics for each theta entry. Use
        section 49.4.2 of PDG booklet for kinematics algebra.

        Parameters
        ----------
        theta: torch.Tensor
            Tensor of shape ``(N, 2)`` where ``theta[:, 0]`` is the parent
            mass and ``theta[:, 1]`` is the signal/background flag (`0` is
            assumed to represent signal and `1` is assumed to represent
            background)
        device: torch.device | None
            Optional override for the target device.
        seed: int | None
            Optional torch seed used for sampling; reused for all internal
            distributions.

        Returns
        -------
        torch.Tensor
            Batched decay vectors with shape (N, 8).
        """
        target_device = device or self.device
        theta = theta.to(target_device)
        generator = torch.Generator(device=target_device)
        if seed is not None:
            generator.manual_seed(seed)

        n_samples = theta.shape[0]
        Mtot = theta[:, 0].unsqueeze(-1)
        Mdecay = torch.tensor(self.product_mass, device=target_device)

        # compute momentum magnitude using the Kallen function for a 2-body
        # decay at rest
        ptotal = (1.0 / (2.0 * Mtot)) * torch.sqrt(
            _kaellen_function(Mtot**2, Mdecay**2, Mdecay**2)
        )

        # sample isotropic directions for the first prong
        # uniform distribution from [-1,1]
        cos_theta1 = -1.0 + 2.0 * torch.rand(
            (n_samples, 1), generator=generator, device=target_device
        )
        sin_theta1 = torch.sqrt(1.0 - cos_theta1**2)
        # uniform distribution from [0,2pi]
        phi1 = (
            2.0
            * torch.pi
            * torch.rand((n_samples, 1), generator=generator, device=target_device)
        )

        # build four-momenta for the first prong
        en1 = Mtot / 2
        px1 = ptotal * sin_theta1 * torch.cos(phi1)
        py1 = ptotal * sin_theta1 * torch.sin(phi1)
        pz1 = ptotal * cos_theta1
        pt1, phi1, eta1, mass1 = to_ptphieta(en1, px1, py1, pz1)

        mu1 = torch.hstack([pt1, phi1, eta1, mass1])
        assert mu1.shape == torch.Size([n_samples, 4]), "mu1 shape mismatch"

        # the second prong is emitted back-to-back in the parent rest frame
        en2 = torch.clone(en1)
        px2 = -px1
        py2 = -py1
        pz2 = -pz1
        pt2, phi2, eta2, mass2 = to_ptphieta(en2, px2, py2, pz2)

        assert pt2.shape == pt1.shape, "pt mismatch"

        # add background offsets only when theta[:, 1] > 0
        background_pt_offset = theta[:, 1] * torch.rand(
            (n_samples,), generator=generator, device=target_device
        )
        background_pt_offset *= 0.25

        background_phi_offset = theta[:, 1] * torch.normal(
            mean=torch.zeros(n_samples, device=target_device),
            std=torch.pi / 4,
            generator=generator,
        )
        background_eta_offset = theta[:, 1] * (
            torch.rand((n_samples,), generator=generator, device=target_device) - 0.5
        )
        background_eta_offset *= 0.5

        pt2 *= 1.0 - background_pt_offset.unsqueeze(1)
        phi2 *= 1.0 - background_phi_offset.unsqueeze(1)
        eta2 *= 1.0 - background_eta_offset.unsqueeze(1)

        mu2 = torch.hstack([pt2, phi2, eta2, mass2])

        assert mu1.shape == mu2.shape, "mu1/mu2 shape mismatch"
        assert mu2.shape == torch.Size([n_samples, 4]), "mu2 shape mismatch"

        return torch.hstack([mu1, mu2])

    def smear(self, events: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """Apply the configured smear function to the generated events.

        Parameters
        ----------
        events: torch.Tensor
            Batched decay vectors with shape (B, 8).
        seed: int | None
            Optional seed passed to the smear function.

        Returns
        -------
        torch.Tensor
            Smeared events with shape (B, 8).
        """
        return self.smear_fn(events, seed)

    def simulate(
        self,
        theta: torch.Tensor,
        device: torch.device | None = None,
        generation_seed: int | None = None,
        smear_seed: int | None = None,
    ) -> torch.Tensor:
        """Generate and smear decay events from theta.

        Parameters
        ----------
        theta: torch.Tensor
            Input theta array shaped (N, 2).
        device: torch.device | None
            Optional device override for the generation step.
        generation_seed: int | None
            Seed used during kinematic sampling.
        smear_seed: int | None
            Seed used when smearing the produced events.

        Returns
        -------
        torch.Tensor
            Smeared batch of shape (N, 8).
        """
        events = self.generate(theta, device=device, seed=generation_seed)
        return self.smear(events, seed=smear_seed)


@dataclass
class SimulateFactory:
    """Factory that exposes a callable simulate function backed by TwoProngDecay."""

    decay: TwoProngDecay

    @classmethod
    def create(
        cls,
        product_mass: float = mMu,
        smear_fn: SmearFn = default_smear_ptphieta,
        device: torch.device = DEFAULT_DEVICE,
    ) -> SimulateFactory:
        """Create a factory with a new TwoProngDecay instance."""
        return cls(
            decay=TwoProngDecay(
                product_mass=product_mass, smear_fn=smear_fn, device=device
            )
        )

    def simulate(
        self,
        theta: torch.Tensor,
        device: torch.device | None = None,
        generation_seed: int | None = None,
        smear_seed: int | None = None,
    ) -> torch.Tensor:
        """Callable that mirrors the notebook simulate interface.

        Parameters
        ----------
        theta: torch.Tensor
            Input tensor describing masses and signal/background flags.
        device: torch.device | None
            Optional override for generation.
        generation_seed: int | None
            Optional seed for kinematic sampling.
        smear_seed: int | None
            Optional seed for the smear function.

        Returns
        -------
        torch.Tensor
            Smeared decay vectors with shape (N, 8).
        """
        return self.decay.simulate(
            theta,
            device=device,
            generation_seed=generation_seed,
            smear_seed=smear_seed,
        )

    def create_simulator(
        self,
        generation_seed: int | None = None,
        smear_seed: int | None = None,
        device: torch.device | None = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a simulator callable that reuses this factory's decay.

        Parameters
        ----------
        generation_seed: int | None
            Optional seed passed to `self.simulate` so the generated kinematics
            stay deterministic when the callable is reused.
        smear_seed: int | None
            Optional seed passed to the smear function.
        device: torch.device | None
            Optional override for the generation device; defaults to the factory's
            own device if ``None``.

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            Function that accepts ``theta`` and returns smeared events.
        """

        def simulate(theta: torch.Tensor) -> torch.Tensor:
            return self.simulate(
                theta,
                device=device,
                generation_seed=generation_seed,
                smear_seed=smear_seed,
            )

        return simulate

    @property
    def simulate_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a callable bound to this factory for function-style use."""
        return self.create_simulator()
