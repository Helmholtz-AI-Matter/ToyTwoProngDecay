import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from typing import Tuple

    np.random.seed(42)
    torch.random.manual_seed(42)

    # TODO: check how easy it is to obtain this from the PDG booklet
    mZ0 = 91.1876    # GeV
    mMu = 0.105658  # GeV

    # Use GPU if available
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEFAULT_DEVICE, Tuple, mMu, np, plt, torch


@app.cell
def _(Tuple, torch):
    # reconstruction of the invariant mass
    def to_cartesian(pt, phi, eta, mass) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """ convert a 4-vector encoded as (pt, phi, eta, mass) into (energy, px, py, pz) """
        pabs = pt*torch.cosh(eta)

        energy =  torch.sqrt(pabs**2 + mass**2)

        px = pt*torch.cos(phi)
        py = pt*torch.sin(phi)
        pz = pt*torch.sinh(eta)

        return energy, px, py, pz

    def to_ptphieta(energy, px, py, pz) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """ convert a 4-vector encoded as (energy, px, py, pz) to (pt, phi, eta, mass) """

        pt = torch.sqrt(px**2 + py**2)
        pabs = torch.sqrt(px**2 + py**2 + pz**2)
        phi = torch.atan2(py, px)
        eta = torch.asinh(pz/pt)
        mass = torch.sqrt(energy**2 - pabs**2)
        return pt, phi, eta, mass

    # check roundtrip
    pt_, phi_, eta_, mass_ = torch.ones(2)*40., torch.as_tensor([1.,-1.])*.5, torch.as_tensor([1.,-1.])*-.25, torch.ones(2)*.102
    pt_rt, phi_rt, eta_rt, mass_rt = to_ptphieta(*to_cartesian(pt_, phi_, eta_, mass_))

    assert torch.allclose(pt_, pt_rt), f"pts don't match {pt_,pt_rt}"
    assert torch.allclose(phi_,phi_rt), f"phis don't match {phi_,phi_rt}"
    assert torch.allclose(eta_, eta_rt), f"etas don't match {eta_,eta_rt}"
    return to_cartesian, to_ptphieta


@app.cell
def _(DEFAULT_DEVICE, mMu, phit1, to_ptphieta, torch):
    def kaellen_function(alpha, beta, gamma):
        """ helper function adoped from section 49.4.2 of PDG booklet for kinematics """
        return alpha**2 + beta**2 + gamma**2 - 2*alpha*beta - 2*alpha*gamma - 2*beta*gamma

    def generate_decay_event(theta, device=DEFAULT_DEVICE):
        """
        generate toyMC true events. This function assumes that the signal events are generated from a Z0 boson at rest, any backround stems from an arbitrary combination of kinematic variables

        Parameters
        ----------
        theta : torch.Tensor of shape (N, 2)
            - theta[:,0]: available mass in GeV 
            - theta[:,1]: flag from which process the mass is generated 
            (0 = signal i.e. Z0 decay, nonzero = background)

        Returns
        -------
        batch_of_4vecs : torch.Tensor
            shaped as (B, [Muon1 pt, Muon1 phi, Muon1 eta, Muon2 pt, Muon2 phi, Muon2 eta]) where B is the batch dimension
        """
        theta = theta.to(device) #theta in relation to sbi nomenclature
        n_samples = theta.shape[0]

        # use section 49.4.2 of PDG booklet for kinematics
        # split available energy, decay at rest
        Mtot = theta[:,0].unsqueeze(-1)
        Mdecay = mMu #could be configurable later

        #########
        # prong 1
        Edecay1 = Mtot/2
        ptotal1 = (1/(2.*Mtot))*torch.sqrt(kaellen_function(Mtot**2, Mdecay**2, Mdecay**2))#torch.sqrt(E_mu**2 - mMu**2).unsqueeze(1)

        # dice theta angle
        cos_theta1_ = torch.distributions.uniform.Uniform(-1,1)
        cos_theta1 = cos_theta1_.sample((n_samples,1))
        sin_theta1 = torch.sqrt(1 - cos_theta1**2)

        # dice phi angle
        phi1_ = torch.distributions.uniform.Uniform(0,2*torch.pi)
        phi1 = phi1_.sample((n_samples,1))


        # calculate kinematics
        en1 = Edecay1
        px1 = ptotal1 * sin_theta1 * torch.cos(phi1)
        py1 = ptotal1 * sin_theta1 * torch.sin(phi1)
        pz1 = ptotal1 * cos_theta1
        pt1, phi1, eta1, mass1 = to_ptphieta(en1, px1, py1, pz1)

        assert pt1.shape == torch.Size([n_samples,1]), f"pt1: {pt1.shape} != {n_samples,1}"
        assert phi1.shape == torch.Size([n_samples,1]), f"phi1: {phit1.shape} != {n_samples,1}"

        mu1 = torch.hstack([pt1,
                            phi1,
                            eta1,
                            mass1])

        ##########
        # prong 2: back-to-back decay product 2
        en2 = torch.clone(en1)
        px2 = -px1 # back-to-back
        py2 = -py1 # back-to-back
        pz2 = -pz1 # back-to-back

        pt2, phi2, eta2, mass2 = to_ptphieta(en2, px2, py2, pz2)

        assert eta2.shape == eta1.shape
        assert phi2.shape == phi1.shape
        assert pt2.shape == pt1.shape

        background_pt_offset_ = torch.distributions.uniform.Uniform(0,.25)
        background_pt_offset = theta[:,1]*background_pt_offset_.sample((pt1.shape[0],)) # should be 0 for signal

        background_phi_offset_ = torch.distributions.normal.Normal(0,torch.pi/4)
        background_phi_offset = theta[:,1]*background_phi_offset_.sample((phi1.shape[0],)) # should be 0 for signal

        background_eta_offset_ = torch.distributions.uniform.Uniform(-.25,.25)
        background_eta_offset  = theta[:,1]*background_eta_offset_.sample((eta1.shape[0],)) # should be 0 for signal

        pt2 *= (1. - background_pt_offset.unsqueeze(1))
        phi2 *= (1. - background_phi_offset.unsqueeze(1))
        eta2 *= (1. - background_eta_offset.unsqueeze(1))

        mu2 = torch.hstack([pt2,
                            phi2,
                            eta2,
                            mass2]
                          )

        # defend against some misaligned assumptions
        assert mu1.shape == mu2.shape, f"[1] {mu1.shape} [2] {mu2.shape}"
        assert mu2.shape == torch.Size([n_samples,4]), f"{mu2.shape} != {n_samples,4}"

        return torch.hstack([mu1,mu2])

    return (generate_decay_event,)


@app.cell
def _(generate_decay_event, mMu, torch):
    # generate some dummy data

    toy_signal_masses = torch.linspace(62,122,50).unsqueeze(1)
    toy_signal_labels = torch.zeros_like(toy_signal_masses)
    toy_backrd_labels = torch.ones_like(toy_signal_masses)

    # one unit testing asserts
    assert toy_signal_labels.shape == torch.Size([50,1]), f"misaligned shape {toy_signal_labels.shape}"

    toy_signal_thetas = torch.cat([toy_signal_masses, toy_signal_labels], dim=1).squeeze()
    toy_signal_events = generate_decay_event(toy_signal_thetas)

    # some unit testing asserts
    assert toy_signal_events.shape == torch.Size([50,8]), f"misaligned shape {toy_signal_events.shape}"
    assert torch.abs(toy_signal_events[0,3] - mMu) < .0005, f"mass0 {toy_signal_events[0,3]} != {mMu}"
    assert torch.abs(toy_signal_events[0,-1] - mMu) < .0005, f"mass1 {toy_signal_events[0,-1]} != {mMu}"

    toy_backrd_thetas = torch.cat([toy_signal_masses, toy_backrd_labels], dim=1).squeeze()
    toy_backrd_events = generate_decay_event(toy_backrd_thetas)
    return toy_backrd_events, toy_signal_events


@app.cell
def _(plt, toy_backrd_events, toy_signal_events):
    # plot the kinematics of each decay product

    figk, axk = plt.subplots(2,3, tight_layout=True, figsize=(10,10))

    axk[0,0].hist(toy_signal_events[:,0])
    axk[0,0].hist(toy_backrd_events[:,0])
    axk[0,0].set_xlabel("$p_T$ of muon 1 / GeV")

    axk[0,1].hist(toy_signal_events[:,1])
    axk[0,1].hist(toy_backrd_events[:,1])
    axk[0,1].set_xlabel("$\phi$ of muon 1 / a.u.")

    axk[0,2].hist(toy_signal_events[:,2],label="signal")
    axk[0,2].hist(toy_backrd_events[:,2],label="backrd")
    axk[0,2].set_xlabel("$\eta$ of muon 1 / a.u.")
    axk[0,2].legend()

    axk[1,0].hist(toy_signal_events[:,4])
    axk[1,0].hist(toy_backrd_events[:,4])
    axk[1,0].set_xlabel("$p_T$ of muon 2 / GeV")

    axk[1,1].hist(toy_signal_events[:,5])
    axk[1,1].hist(toy_backrd_events[:,5])
    axk[1,1].set_xlabel("$\phi$ of muon 2 / a.u.")

    axk[1,2].hist(toy_signal_events[:,6],label="signal")
    axk[1,2].hist(toy_backrd_events[:,6],label="backrd")
    axk[1,2].set_xlabel("$\eta$ of muon 2 / a.u.")
    axk[1,2].legend()

    figk
    return


@app.cell
def _(plt, torch, toy_signal_events):
    # plot delta phi to make sure all decay prongs are back-to-back

    figdp, axdp = plt.subplots(1,1, tight_layout=True)

    dphi = toy_signal_events[:,5] - toy_signal_events[:,1]
    axdp.hist(dphi)
    axdp.set_xlabel("$\Delta\phi$ ")

    adphi = torch.abs(toy_signal_events[:,5] - toy_signal_events[:,1])
    assert torch.allclose(adphi,
                         torch.pi*torch.ones_like(adphi),
                          atol=.01), f"{adphi}"
    figdp
    return


@app.cell
def _(mMu, to_cartesian, torch):
    def ptphieta_to_epxyz(batch_decay_vectors: torch.Tensor, common_mass: float = mMu) -> torch.Tensor:

        """ transform batched decay vectors (B, [Muon1 pt, Muon1 phi, Muon1 eta, Muon1 mass, Muon2 pt, Muon2 phi, Muon2 eta, Muon2 mass]) into (B, [Muon1 E, Muon1 px, Muon1 py, Muon1 pz, Muon2 E, Muon2 px, Muon2 py, Muon2 pz])"""

        bdv = batch_decay_vectors
        mu1pt, mu1phi, mu1eta, mu1m = bdv[:,0].unsqueeze(1), bdv[:,1].unsqueeze(1), bdv[:,2].unsqueeze(1), bdv[:,3].unsqueeze(1)
        mu2pt, mu2phi, mu2eta, mu2m = bdv[:,4].unsqueeze(1), bdv[:,5].unsqueeze(1), bdv[:,6].unsqueeze(1), bdv[:,7].unsqueeze(1)

        mu1e, mu1px, mu1py, mu1pz = to_cartesian(mu1pt, mu1phi, mu1eta, mu1m)
        mu2e, mu2px, mu2py, mu2pz = to_cartesian(mu2pt, mu2phi, mu2eta, mu2m)

        value = torch.hstack([mu1e, mu1px, mu1py, mu1pz, mu2e, mu2px, mu2py, mu2pz])

        return value

    test_ptphieta = torch.ones([2,8])
    obs_epxyz = ptphieta_to_epxyz(test_ptphieta)
    assert obs_epxyz.shape == test_ptphieta.shape, f"{obs_epxyz.shape} != {test_ptphieta.shape}"
    return (ptphieta_to_epxyz,)


@app.cell
def _(mMu, torch):
    def invariant_mass_from_epxyz(batch_decay_vectors: torch.Tensor, common_mass: float = mMu) -> torch.Tensor:
        """ calculate invariant mass from batch_decay_vectors in shape (B, [Muon1 E, Muon1 px, Muon1 py, Muon1 pz, Muon2 E, Muon2 px, Muon2 py, Muon2 pz]), see also https://en.wikipedia.org/wiki/Invariant_mass#Example:_two-particle_collision """

        bdv = batch_decay_vectors
        value = torch.zeros([bdv.shape[0],1])

        # indices of bdv
        mu1E, mu1px, mu1py, mu1pz = list(range(4))
        mu2E, mu2px, mu2py, mu2pz = list(range(4,8))

        value[:,0] = 2*(common_mass**2) 
        value[:,0] += 2*bdv[:,mu1E]*bdv[:,mu2E] 
        value[:,0] -= 2*(bdv[:,mu1px]*bdv[:,mu2px] + bdv[:,mu1py]*bdv[:,mu2py] + bdv[:,mu1pz]*bdv[:,mu2pz])

        return torch.sqrt(value)

    return (invariant_mass_from_epxyz,)


@app.cell
def _(
    generate_decay_event,
    invariant_mass_from_epxyz,
    ptphieta_to_epxyz,
    torch,
):
    # unit testing cell

    inbound_mass = 91.2
    theta_z0 = torch.hstack([torch.ones((32,1))*91.2, torch.zeros((32,1))])
    assert theta_z0.shape == (32,2),f"{theta_z0.shape}"

    z0_decays = generate_decay_event(theta_z0)
    z0_epxyz = ptphieta_to_epxyz(z0_decays)

    assert torch.allclose(z0_epxyz[:,0], torch.ones_like(z0_epxyz[:,0])*(inbound_mass/2),atol=.001)
    assert torch.allclose(z0_epxyz[:,4], torch.ones_like(z0_epxyz[:,4])*(inbound_mass/2),atol=.001)

    pdecay1 = z0_epxyz[:,1]**2 + z0_epxyz[:,2]**2 + z0_epxyz[:,3]**2
    pdecay2 = z0_epxyz[:,-3]**2 + z0_epxyz[:,-2]**2 + z0_epxyz[:,-1]**2

    assert torch.allclose(pdecay1,pdecay2)

    z0_mass = invariant_mass_from_epxyz(z0_epxyz)

    assert z0_mass.shape == theta_z0[:,0].unsqueeze(1).shape, f"{z0_mass.shape, theta_z0[:,0].shape}"
    assert torch.allclose(theta_z0[:,0],z0_mass), f"{theta_z0[:,0]} != \n{z0_mass.flatten()}"
    return


@app.cell
def _(torch):
    # TODO: smear the MC particles to emulate detector reaction
    # def resolution(pt):
    #     slope = -.014 #-.01342
    #     xoffset = 1.503
    #     yoffset = .01
    #     sigmas = (slope*pt + xoffset)**2 + yoffset #low masses are more smeared than higher ones
    #     return sigmas

    def smear_ptphieta(batch_decay_vectors: torch.Tensor) -> torch.Tensor:
        """ apply detector resolution effects to mc particles. particles in batched decay vectors are assumed to comply to (B, [Muon1 pt, Muon1 phi, Muon1 eta, Muon1 mass, Muon2 pt, Muon2 phi, Muon2 eta, Muon2 mass]); the same format is returned

        some inspiration was taken from 
        [1] https://ar5iv.labs.arxiv.org/html/2212.07338
        """

        bdv = batch_decay_vectors
        # indices of bdv
        mu1pt, mu1phi, mu1eta, mu1m = list(range(4))
        mu2pt, mu2phi, mu2eta, mu2m = list(range(4,8))

        value = torch.clone(bdv)

        sigmas = torch.distributions.uniform.Uniform(.01,.04).sample((value.shape[0],)) #range approximated from [1] fig 10
        pt1_smear_factor = torch.normal(mean=torch.ones_like(bdv[:,mu1pt]), std=sigmas)
        pt2_smear_factor = torch.normal(mean=torch.ones_like(bdv[:,mu2pt]), std=sigmas)

        value[:,mu1pt] *= pt1_smear_factor
        value[:,mu2pt] *= pt2_smear_factor

        #TODO: smear phi and eta too, should be dependent on phi/eta respectively
        phi1_smear_factor = torch.normal(
                                mean=torch.ones_like(bdv[:,mu1phi]), 
                                std=torch.ones_like(sigmas)*.01
                            )
        phi2_smear_factor = torch.flip(phi1_smear_factor,dims=(0,)) #reverse
        value[:,mu1phi] *= phi1_smear_factor
        value[:,mu2phi] *= phi2_smear_factor

        eta1_smear_factor = torch.normal(
                                mean=torch.ones_like(bdv[:,mu1eta]), 
                                std=torch.ones_like(sigmas)*.01
                            )
        eta2_smear_factor = torch.flip(eta1_smear_factor, dims=(0,))
        value[:,mu1eta] *= eta1_smear_factor
        value[:,mu2eta] *= eta2_smear_factor

        return value


    return (smear_ptphieta,)


@app.cell
def _(
    invariant_mass_from_epxyz,
    ptphieta_to_epxyz,
    smear_ptphieta,
    toy_backrd_events,
    toy_signal_events,
):
    # in sbi lingo: the table with 8+1 columns made from the events are the observations x 
    dtoy_signal_events, dtoy_backrd_events = smear_ptphieta(toy_signal_events), smear_ptphieta(toy_backrd_events)

    dtoy_signal_epxyz, dtoy_backrd_epxyz = ptphieta_to_epxyz(dtoy_signal_events), ptphieta_to_epxyz(dtoy_backrd_events)

    toy_signal_dinvm, toy_backrd_dinvm = invariant_mass_from_epxyz(dtoy_signal_epxyz), invariant_mass_from_epxyz(dtoy_backrd_epxyz)
    return


@app.cell
def _(
    DEFAULT_DEVICE,
    generate_decay_event,
    invariant_mass_from_epxyz,
    ptphieta_to_epxyz,
    smear_ptphieta,
    torch,
):
    def simulate(thetas: torch.Tensor, device = DEFAULT_DEVICE) -> torch.Tensor :
        """ take theta which is encoded as This function assumes that the signal events are generated from a Z0 boson at rest, any backround stems from an arbitrary combination of kinematic variables; this method applies a detector specific smearing to the produced decay products

        Parameters
        ----------
        theta : torch.Tensor of shape (N, 2)
            - theta[:,0]: available mass in GeV 
            - theta[:,1]: flag from which process the mass is generated 
            (0 = signal i.e. Z0 decay, nonzero = background)

        Returns
        -------
        batch_of_4vecs : torch.Tensor
            shaped as (B, [Muon1 pt, Muon1 phi, Muon1 eta, Muon2 pt, Muon2 phi, Muon2 eta]) where B is the batch dimension
        """

        # generate MC ground truth
        events = generate_decay_event(thetas)

        # smear to emulate detector effects
        value = smear_ptphieta(events)

        return value

    def invariant_mass_from_ptphieta(batch_decay_vectors: torch.Tensor):
        """ given a ptphieta encoded vector of shape (B, [Muon1 pt, Muon1 phi, Muon1 eta, Muon1 mass, Muon2 pt, Muon2 phi, Muon2 eta, Muon2 mass]), calculate the invariant mass from either decay products"""

        bdv = batch_decay_vectors
        #assert torch.allclose(bdv[:,3],bdv[:,-1],atol=1e-4), f"masses are not compatible {bdv[0,3],bdv[-1,-1]}"

        # convert to (E, px, py, pz) coordinates
        epxyz = ptphieta_to_epxyz(bdv)

        # calculate invariant mass of decay produces
        value = invariant_mass_from_epxyz(epxyz)

        return value

    return invariant_mass_from_ptphieta, simulate


@app.cell
def _(torch):
    ## the full data generating pipeline ############################################################
    # generate the mass distribution, i.e. theta, from our priors
    num_signal = 2000
    num_bkrd = 200
    num_total = num_signal + num_bkrd

    # define the priors for signal and background
    signal_prior = torch.distributions.Cauchy(loc=92.1, scale=.05)
    bkrd_prior = torch.distributions.Uniform(62.,122.)

    # create prior object
    theta = torch.empty((num_signal+num_bkrd,2))

    # generate signal with label 0
    theta[:num_signal,0] = signal_prior.sample((num_signal,))
    theta[:num_signal,1] = torch.zeros_like(theta[:num_signal,0])

    # generate background with label 1
    theta[num_signal:,0] = bkrd_prior.sample((num_bkrd,))
    theta[num_signal:,1] = torch.ones_like(theta[num_signal:,0])
    return num_signal, num_total, theta


@app.cell
def _(num_total, simulate, theta):
    # simulate some observations x
    x = simulate(theta) 

    # minimal unit tests
    assert x.shape == (num_total,8)
    return (x,)


@app.cell
def _(generate_decay_event, mMu, ptphieta_to_epxyz, theta, torch):
    mc_events = generate_decay_event(theta)
    mc_epxyz = ptphieta_to_epxyz(mc_events)

    # minimal unit tests
    mc_E1_ = torch.sqrt(mc_epxyz[:,1]**2 + mc_epxyz[:,2]**2 + mc_epxyz[:,3]**2 + mMu**2)
    assert torch.allclose(mc_E1_,mc_epxyz[:,0], atol=1e-4)

    mc_E2_ = torch.sqrt(mc_epxyz[:,5]**2 + mc_epxyz[:,6]**2 + mc_epxyz[:,7]**2 + mMu**2)
    assert torch.allclose(mc_E2_,mc_epxyz[:,4], atol=1e-4)
    return mc_epxyz, mc_events


@app.cell(hide_code=True)
def _(
    invariant_mass_from_epxyz,
    invariant_mass_from_ptphieta,
    mc_epxyz,
    mc_events,
    np,
    num_signal,
    plt,
    theta,
    x,
):
    mc_inv   = invariant_mass_from_epxyz(mc_epxyz)
    mc_x = invariant_mass_from_ptphieta(mc_events)

    pseudo_x = invariant_mass_from_ptphieta(x)


    fig2, axs_ = plt.subplots(2,4, figsize=(12,8),tight_layout=True)

    axs_[0,0].hist([theta[:num_signal,0], theta[num_signal:,0]], bins=np.arange(62,122,1), stacked=True, label=["signal", "background"] )
    axs_[0,0].set_title("`theta` from toy prior")
    axs_[0,0].set_xlabel("input mass / 'GeV'")
    axs_[0,0].set_ylabel("count / 1 GeV")
    axs_[0,0].set_yscale("log")


    axs_[0,1].hist([mc_x[:num_signal,0], mc_x[num_signal:,0]], bins=np.arange(62,122,1.), stacked=True, label=["signal", "background"] )
    axs_[0,1].set_xlabel( "mass / 'GeV'")
    axs_[0,1].set_ylabel("count / 1 GeV")
    axs_[0,1].set_title("mass calculated from x w/o smearing (ptphieta)")
    axs_[0,1].set_yscale("log")
    #axs_[1].set_yscale("log")

    axs_[0,2].hist([pseudo_x[:num_signal,0], pseudo_x[num_signal:,0]], bins=np.arange(62,122,1.), stacked=True, label=["signal", "background"] )
    axs_[0,2].set_xlabel("(rec) mass / 'GeV'")
    axs_[0,2].set_ylabel("count / 1 GeV")
    axs_[0,2].set_title("mass calculated from x w/ smearing")
    axs_[0,2].set_yscale("log")

    axs_[0,3].hist([mc_inv[:num_signal,0], mc_inv[num_signal:,0]], bins=np.arange(62,122,1.), stacked=True, label=["signal", "background"] )
    axs_[0,3].set_xlabel("mass / 'GeV'")
    axs_[0,3].set_ylabel("count / 1 GeV")
    axs_[0,3].set_title("mass calculated from x w/o smearing (epxyz)")
    axs_[0,3].set_yscale("log")

    #axs_[1].set_ylim(1,10_000)

    axs_[0,3].legend()

    axs_[1,0].hist(theta[:num_signal,0], bins=np.arange(62,122,1), color="darkorange", label=["signal"] )
    axs_[1,0].set_title("`theta` from toy prior")
    axs_[1,0].set_xlabel("input mass / 'GeV'")
    axs_[1,0].set_ylabel("count / 1 GeV")
    axs_[1,0].set_yscale("log")

    axs_[1,1].hist(mc_x[:num_signal,0], bins=np.arange(62,122,1.), color="darkorange", label=["signal"] )
    axs_[1,1].set_xlabel("(rec) mass / 'GeV'")
    axs_[1,1].set_ylabel("count / 1 GeV")
    axs_[1,1].set_title("mass calculated from x w/ smearing")
    axs_[1,1].set_yscale("log")

    axs_[1,2].hist(theta[:num_signal,0], bins=20, color="darkorange", label=["signal"] )
    axs_[1,2].set_title("`theta` from toy prior")
    axs_[1,2].set_xlabel("input mass / 'GeV'")
    axs_[1,2].set_ylabel("count / 1 GeV")
    axs_[1,2].set_yscale("log")

    axs_[1,3].hist(pseudo_x[:num_signal,0], bins=20, color="darkorange", label=["signal"] )
    axs_[1,3].set_title("signal mass calculated from x w/ smearing")
    axs_[1,3].set_xlabel("input mass / 'GeV'")
    axs_[1,3].set_ylabel("count / 1 GeV")
    axs_[1,3].set_yscale("log")


    fig2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
