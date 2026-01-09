import numpy as np
import torch
from typing import Tuple

np.random.seed(42)
torch.random.manual_seed(42)

# TODO: check how easy it is to obtain this from the PDG booklet
mZ0 = 91.1876    # GeV
mMu = 0.105658  # GeV

# Use GPU if available
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_cartesian(pt, phi, eta, mass) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    pabs = pt*torch.cosh(eta)

    energy = torch.sqrt(pabs**2 + mass**2)

    px = pt*torch.cos(phi)
    py = pt*torch.sin(phi)
    pz = pt*torch.sinh(eta)

    return energy, px, py, pz

def to_ptphieta(energy, px, py, pz) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    pt = torch.sqrt(px**2 + py**2)
    pabs = torch.sqrt(px**2 + py**2 + pz**2)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz/pt)
    mass = torch.sqrt(energy**2 - pabs**2)
    return pt, phi, eta, mass

def generate_decay_event(theta, device=DEFAULT_DEVICE):
    """
    generate toyMC true events. This function assumes that the signal events are generated from a Z0 boson at rest, any backround stems from an arbitrary combination of kinematic variables

    Parameters
    ----------
    theta : torch.Tensor of shape (N, 2)
        - theta[:,0]: available mass in GeV
        - theta[:,1]: flag from which process the mass is generated
        (0 = signal, nonzero = background)

    Returns
    -------
    batch_of_4vecs : torch.Tensor
        shaped as (B, [Muon1 pt, Muon1 phi, Muon1 eta, Muon2 pt, Muon2 phi, Muon2 eta]) where B is the batch dimension
    """
    theta = theta.to(device) #theta in relation to sbi nomenclature
    n_samples = theta.shape[0]

    # split available energy, decay at rest
    E_mu = theta[:,0] / 2.
    p = torch.sqrt(E_mu**2 - mMu**2).unsqueeze(1)

    # prong 1
    cos_theta1_ = torch.distributions.uniform.Uniform(-1,1)
    cos_theta1 = cos_theta1_.sample((n_samples,1))
    theta1_angle = torch.arccos(cos_theta1)

    eta1 = -torch.log(torch.tan(theta1_angle / 2))

    phi1_ = torch.distributions.uniform.Uniform(0,2*torch.pi)
    phi1_angle = phi1_.sample((n_samples,1))

    pT1 = p / torch.cosh(eta1)
    mu1 = torch.hstack([pT1,
                        phi1_angle,
                        eta1,
                        mMu*torch.ones_like(eta1)]
                      )

    en1, px1, py1, pz1 = to_cartesian(pT1, phi1_angle, eta1, mMu)
    # prong 2: back-to-back muon 2
    background_phi_offset_ = torch.distributions.normal.Normal(0,torch.pi/4)
    background_phi_offset = theta[:,1]*background_phi_offset_.sample((phi1_angle.shape[0],)) # should be 0 for signal

    #phi2_angle = phi1_angle + torch.pi + background_phi_offset.unsqueeze(1)
    #phi2_mask = phi2_angle >= 2*torch.pi
    #phi2_angle[phi2_mask] -= 2.*torch.pi

    # populate missing kinematics according to assumptions
    background_eta_offset_ = torch.distributions.uniform.Uniform(-.25,.25)
    background_eta_offset  = theta[:,1]*background_eta_offset_.sample((eta1.shape[0],)) # should be 0 for signal
    eta2 = -eta1 + background_eta_offset.unsqueeze(1)     # follows from θ → π − θ for signal, add offset for background

    assert eta2.shape == eta1.shape

    background_pt_offset_ = torch.distributions.uniform.Uniform(0,.25)
    background_pt_offset = theta[:,1]*background_pt_offset_.sample((pT1.shape[0],)) # should be 0 for signal

    px2 = -px1
    py2 = -py1

    pT2 = torch.sqrt(px2**2 + py2**2)*(1. - background_pt_offset.unsqueeze(1))
    phi2_angle = torch.atan2(py2, px2)*(1. - background_phi_offset.unsqueeze(1))

    assert pT2.shape == pT1.shape

    mu2 = torch.hstack([pT2,
                        phi2_angle,
                        eta2,
                        mMu*torch.ones_like(eta2)]
                      )

    # defend against some misaligned assumptions
    assert mu1.shape == mu2.shape, f"[1] {mu1.shape} [2] {mu2.shape}"
    assert mu2.shape == torch.Size([n_samples,4])

    return torch.hstack([mu1,mu2])


def test_generate_decay_event_shape_and_finite():
    theta = torch.tensor([[91.1876, 0.0],
                          [91.1876, 1.0]], device=DEFAULT_DEVICE)

    out = generate_decay_event(theta)

    assert out.shape == (2, 8)
    assert out.device == theta.device
    assert torch.isfinite(out).all()

def test_signal_back_to_back_phi():
    theta = torch.tensor([[91.1876, 0.0]] * 100, device=DEFAULT_DEVICE)
    out = generate_decay_event(theta)

    phi1 = out[:, 1]
    phi2 = out[:, 5]

    dphi = torch.remainder(phi2 - phi1, 2 * torch.pi)
    assert torch.allclose(dphi, torch.full_like(dphi, torch.pi), atol=1e-6)

def test_signal_pz_conservation():
    theta = torch.tensor([[91.1876, 0.0]] * 1000, device=DEFAULT_DEVICE)
    out = generate_decay_event(theta)

    pT1, eta1 = out[:, 0], out[:, 2]
    pT2, eta2 = out[:, 4], out[:, 6]

    pz_sum = pT1 * torch.sinh(eta1) + pT2 * torch.sinh(eta2)
    assert torch.allclose(pz_sum, torch.zeros_like(pz_sum), atol=1e-5)

def test_signal_pt_conservation():
    theta = torch.tensor([[91.1876, 0.0]] * 1000, device=DEFAULT_DEVICE)
    out = generate_decay_event(theta)

    pT1, phi1 = out[:, 0], out[:, 1]
    pT2, phi2 = out[:, 4], out[:, 5]

    px_sum = pT1 * torch.cos(phi1) + pT2 * torch.cos(phi2)
    py_sum = pT1 * torch.sin(phi1) + pT2 * torch.sin(phi2)

    assert torch.allclose(px_sum, torch.zeros_like(px_sum), atol=1e-5)
    assert torch.allclose(py_sum, torch.zeros_like(py_sum), atol=1e-5)

def test_signal_invariant_mass():
    theta = torch.tensor([[91.1876, 0.0]] * 500, device=DEFAULT_DEVICE)
    out = generate_decay_event(theta)

    # unpack
    pT1, phi1, eta1 = out[:, 0], out[:, 1], out[:, 2]
    pT2, phi2, eta2 = out[:, 4], out[:, 5], out[:, 6]

    # energies
    E1 = torch.sqrt((pT1 * torch.cosh(eta1))**2 + mMu**2)
    E2 = torch.sqrt((pT2 * torch.cosh(eta2))**2 + mMu**2)

    # momenta
    px1, py1, pz1 = pT1 * torch.cos(phi1), pT1 * torch.sin(phi1), pT1 * torch.sinh(eta1)
    px2, py2, pz2 = pT2 * torch.cos(phi2), pT2 * torch.sin(phi2), pT2 * torch.sinh(eta2)

    E = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    mass = torch.sqrt(E**2 - px**2 - py**2 - pz**2)

    assert torch.allclose(mass, theta[:, 0], atol=1e-4)

def test_background_not_back_to_back():
    theta = torch.tensor([[91.1876, 1.0]] * 500, device=DEFAULT_DEVICE)
    out = generate_decay_event(theta)

    phi1 = out[:, 1]
    phi2 = out[:, 5]

    dphi = torch.remainder(phi2 - phi1, 2 * torch.pi)
    assert not torch.allclose(dphi, torch.full_like(dphi, torch.pi), atol=1e-3)

def test_reproducibility():
    torch.manual_seed(42)
    theta = torch.tensor([[91.1876, 0.0]] * 10)
    out1 = generate_decay_event(theta)

    torch.manual_seed(42)
    out2 = generate_decay_event(theta)

    assert torch.allclose(out1, out2)
