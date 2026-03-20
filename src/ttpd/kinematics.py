"""Kinematic constants and helper functions for two-prong decay studies."""

from __future__ import annotations

from typing import TypeAlias

import torch

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


def to_cartesian(
    pt: torch.Tensor,
    phi: torch.Tensor,
    eta: torch.Tensor,
    mass: torch.Tensor,
) -> TensorQuad:
    """Convert ``(pt, phi, eta, mass)`` to ``(energy, px, py, pz)``.

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
    energy: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
) -> TensorQuad:
    """Convert ``(energy, px, py, pz)`` to ``(pt, phi, eta, mass)``.

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
        Batched vectors with shape ``(B, 8)``:
        ``[mu1 pt, mu1 phi, mu1 eta, mu1 mass, mu2 pt, mu2 phi, mu2 eta, mu2 mass]``.

    Returns
    -------
    torch.Tensor
        Batched vectors with shape ``(B, 8)``:
        ``[mu1 E, mu1 px, mu1 py, mu1 pz, mu2 E, mu2 px, mu2 py, mu2 pz]``.
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
    batch_decay_vectors: torch.Tensor,
    prong_mass: float = mMu,
) -> torch.Tensor:
    """Calculate invariant mass from energy/momentum vectors.

    Parameters
    ----------
    batch_decay_vectors: torch.Tensor
        Batched vectors with shape ``(B, 8)`` representing two particles.
        Encoding is expected as
        ``mu1e, mu1px, mu1py, mu1pz, mu2e, mu2px, mu2py, mu2pz``.
    prong_mass: float
        Mass of the decay product to reconstruct mass from.

    Returns
    -------
    torch.Tensor
        Invariant mass per event of shape ``(B, 1)``.
    """
    mu1e, mu1px, mu1py, mu1pz = list(range(4))
    mu2e, mu2px, mu2py, mu2pz = list(range(4, 8))

    mu1_energy = batch_decay_vectors[:, mu1e]
    mu2_energy = batch_decay_vectors[:, mu2e]

    mu1_px = batch_decay_vectors[:, mu1px]
    mu2_px = batch_decay_vectors[:, mu2px]

    mu1_py = batch_decay_vectors[:, mu1py]
    mu2_py = batch_decay_vectors[:, mu2py]

    mu1_pz = batch_decay_vectors[:, mu1pz]
    mu2_pz = batch_decay_vectors[:, mu2pz]

    value = torch.zeros(
        [batch_decay_vectors.shape[0], 1],
        device=batch_decay_vectors.device,
    )

    value[:, 0] = 2 * (prong_mass**2)
    value[:, 0] += 2 * (mu1_energy * mu2_energy)
    value[:, 0] -= 2 * (mu1_px * mu2_px + mu1_py * mu2_py + mu1_pz * mu2_pz)
    return torch.sqrt(value)


def invariant_mass_from_ptphieta(batch_decay_vectors: torch.Tensor) -> torch.Tensor:
    """Calculate invariant mass from pt/phi/eta/mass vectors.

    Parameters
    ----------
    batch_decay_vectors: torch.Tensor
        Batched vectors with shape ``(B, 8)``.
        Encoding is expected as
        ``[mu1 pt, mu1 phi, mu1 eta, mu1 mass, mu2 pt, mu2 phi, mu2 eta, mu2 mass]``.

    Returns
    -------
    torch.Tensor
        Invariant mass per event of shape ``(B, 1)``.
    """
    return invariant_mass_from_epxyz(ptphieta_to_epxyz(batch_decay_vectors))
