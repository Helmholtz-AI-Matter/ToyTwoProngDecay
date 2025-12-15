import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    np.random.seed(42)
    torch.random.manual_seed(42)

    # TODO: check how easy it is to obtain this from the PDG booklet
    mZ0 = 91.1876    # GeV
    mMu = 0.105658  # GeV

    # Use GPU if available
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEFAULT_DEVICE, mMu, np, plt, torch


@app.cell
def _(DEFAULT_DEVICE, mMu, torch):
    def generate_decay_event(theta, device=DEFAULT_DEVICE):
        """
        generate toyMC true event.

        Parameters
        ----------
        theta : torch.Tensor of shape (N, 2)
            - theta[:,0]: available mass in GeV 
            - theta[:,1]: flag (0 = signal, nonzero = background)

        Returns
        -------
        batch_of_4vecs : torch.Tensor
            two event level four-vectors in a batch 
        """
        theta = theta.to(device) #theta in relation to sbi nomenclature
        n_samples = theta.shape[0]

        # split available energy, decay at rest
        E_mu = theta[:,0] / 2.
        p = torch.sqrt(E_mu**2 - mMu**2).unsqueeze(1)

        # prong 1
        cos_theta1_ = torch.distributions.uniform.Uniform(-1,1)
        theta1_angle = torch.arccos(cos_theta1_.sample((n_samples,1)))
        eta1 = -torch.log(torch.tan(theta1_angle / 2))

        phi1_ = torch.distributions.uniform.Uniform(0,2*torch.pi)
        phi1_angle = phi1_.sample((n_samples,1))

        pT1 = p * torch.sin(theta1_angle)
        mu1 = torch.hstack([pT1, 
                            phi1_angle, 
                            eta1, 
                            mMu*torch.ones_like(eta1)]
                          )

        # prong 2: back-to-back muon 2
        background_phi_offset_ = torch.distributions.normal.Normal(0,torch.pi/4)
        background_phi_offset = theta[:,1]*background_phi_offset_.sample((phi1_angle.shape[0],)) # should be 0 for signal
    
        phi2_angle = phi1_angle + torch.pi + background_phi_offset.unsqueeze(1)
        phi2_mask = phi2_angle >= 2*torch.pi
        phi2_angle[phi2_mask] -= 2.*torch.pi

    
        # populate missing kinematics according to assumptions
        background_eta_offset_ = torch.distributions.uniform.Uniform(-.25,.25)
        background_eta_offset  = theta[:,1]*background_eta_offset_.sample((eta1.shape[0],)) # should be 0 for signal
        eta2 = -eta1 + background_eta_offset.unsqueeze(1)     # follows from θ → π − θ for signal, add offset for background

        assert eta2.shape == eta1.shape
    
        background_pt_offset_ = torch.distributions.uniform.Uniform(0,.25)
        background_pt_offset = theta[:,1]*background_pt_offset_.sample((pT1.shape[0],)) # should be 0 for signal
        pT2 = pT1*(1. - background_pt_offset.unsqueeze(1))          # same pT as decay happens at rest for signal, different pT for background

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

    return (generate_decay_event,)


@app.cell
def _(generate_decay_event, torch):
    toy_signal_masses = torch.linspace(62,122,50).unsqueeze(1)
    toy_signal_labels = torch.zeros_like(toy_signal_masses)
    toy_backrd_labels = torch.ones_like(toy_signal_masses)

    assert toy_signal_labels.shape == torch.Size([50,1]), f"misaligned shape {toy_signal_labels.shape}"

    toy_signal_thetas = torch.cat([toy_signal_masses, toy_signal_labels], dim=1).squeeze()
    toy_signal_events = generate_decay_event(toy_signal_thetas)
    assert toy_signal_events.shape == torch.Size([50,8]), f"misaligned shape {toy_signal_events.shape}"

    toy_backrd_thetas = torch.cat([toy_signal_masses, toy_backrd_labels], dim=1).squeeze()
    toy_backrd_events = generate_decay_event(toy_backrd_thetas)

    return toy_backrd_events, toy_signal_events


@app.cell
def _(plt, toy_backrd_events, toy_signal_events):
    figk, axk = plt.subplots(2,3, tight_layout=True, figsize=(10,10))

    axk[0,0].hist(toy_signal_events[:,0])
    axk[0,0].set_xlabel("$p_T$ of muon 1 / GeV")

    axk[0,1].hist(toy_signal_events[:,1])
    axk[0,1].set_xlabel("$\phi$ of muon 1 / a.u.")

    axk[0,2].hist(toy_signal_events[:,2])
    axk[0,2].set_xlabel("$\eta$ of muon 1 / a.u.")

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
def _(DEFAULT_DEVICE, generate_decay_event, torch):

    def smear(fourvec):
        """"""
        smeared_ = torch.distributions.normal.Normal(fourvec[:,0], fourvec[:,0]*.03)
        smeared = smeared_.sample((fourvec.shape[0],))

        detector_fourvec = torch.clone(fourvec)
        detector_fourvec[:,0] = smeared
    
        return detector_fourvec

    def simulate(theta, device=DEFAULT_DEVICE):
        """
        Main toy simulation entry point.

        Parameters
        ----------
        theta : torch.Tensor of shape (N, 2)
            - theta[:,0]: available mass in GeV 
            - theta[:,1]: flag (0 = signal, nonzero = background)

        Returns
        -------
        detector_mass : torch.Tensor
            Detector-level simulated mass values
        """
        events = generate_decay_event(theta, device=device)
    
        # to emulate detector response due to different momentum profiles
        leftvalue = smear(events[:,:4])
        rightvalue = smear(events[:,4:])

        value = torch.hstack([leftvalue, rightvalue])
    
        return value
    return (simulate,)


@app.cell
def _(torch):
    #create signal and background
    num_signal = 2000
    num_bkrd = 200
    num_total = num_signal + num_bkrd

    signal_prior = torch.distributions.Cauchy(loc=92.1, scale=.05)
    bkrd_prior = torch.distributions.Uniform(62.,122.)

    theta = torch.empty((num_signal+num_bkrd,2))
    theta[:num_signal,0] = signal_prior.sample((num_signal,))
    theta[:num_signal,1] = torch.zeros_like(theta[:num_signal,0])

    theta[num_signal:,0] = bkrd_prior.sample((num_bkrd,))
    theta[num_signal:,1] = torch.ones_like(theta[num_signal:,0])
    return num_signal, num_total, theta


@app.cell
def _(num_total, simulate, theta):
    x = simulate(theta) 
    assert x.shape == (num_total,8)
    return (x,)


@app.cell
def _(np, num_signal, plt, theta):
    fig2, axs_ = plt.subplots(1,2, tight_layout=True)

    axs_[0].hist([theta[:num_signal,0], theta[num_signal:,0]], bins=np.arange(62,122,1), stacked=True, label=["signal", "background"] )
    axs_[0].set_title("`theta` from toy prior")
    axs_[0].set_xlabel("input mass / 'GeV'")
    axs_[0].set_ylabel("count / 1 GeV")


    axs_[1].hist([theta[:num_signal,0], theta[num_signal:,0]], bins=np.arange(62,122,1.), stacked=True, label=["signal", "background"] )
    axs_[1].set_xlabel("input mass / 'GeV'")
    axs_[1].set_ylabel("count / 1 GeV")
    axs_[1].set_ylim(1,10_000)

    axs_[1].set_yscale("log")
    axs_[1].legend()
    return


@app.cell
def _(np, num_signal, plt, theta, x):
    fig3, axs = plt.subplots(1,2, tight_layout=True)

    axs[0].hist([x[:num_signal], x[num_signal:]], bins=np.arange(62,122,1), stacked=True, label=["rec signal", "rec background"] )
    axs[0].hist(theta[:,0], bins=np.arange(62,122,1.), histtype='step', label=["ground truth"] )
    axs[0].set_title("reconstructed `x` for `theta`")
    axs[0].set_xlabel("input mass / 'GeV'")
    axs[0].set_ylabel("count / 1 GeV")


    axs[1].hist([x[:num_signal], x[num_signal:]], bins=np.arange(62,122,1.), stacked=True, 
                label=["rec signal", "rec background"] )
    axs[1].hist(theta[:,0], bins=np.arange(62,122,1.), histtype='step', label=["sim ground truth"] )

    axs[1].set_xlabel("input mass / 'GeV'")
    axs[1].set_ylabel("count / 1 GeV")
    axs[1].set_ylim(1,50_000)

    axs[1].set_yscale("log")
    axs[1].legend()
    return


@app.cell
def _(theta, x):
    # Now do MNPE demo
    # https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.inference.MNPE.html
    from sbi.inference import MNPE

    num_sims = theta.shape[0]

    theta[:,1] = theta[:,1].int()

    thetas = theta[1:,...]
    xs = x[1:,...].unsqueeze(-1)

    print(thetas.shape, xs.shape)

    inference = MNPE()
    _ = inference.append_simulations(thetas, xs).train()
    return (inference,)


@app.cell
def _(inference, x):

    posterior = inference.build_posterior()

    x_o = x[:1,...].unsqueeze(-1)
    samples = posterior.sample((100,), x=x_o)
    return


@app.cell
def _():
    # from sbi.analysis import pairplot
    # from sbi.analysis.plotting_classes import HistDiagOptions

    # true = theta[:1,...]
    # print(samples.shape, samples.min(), samples.mean(), samples.max(), true)

    # fig, axes = pairplot(
    #     samples,
    #     limits=[[70,120],[0,2]],
    #     figsize=(10, 10),
    #     points=true,
    #     labels=["deconv mass","estimate label"],
    #     diag_kwargs=HistDiagOptions(
    #         mpl_kwargs={
    #             "bins": np.arange(62,122,1),
    #         }
    #     )
    # )
    return


@app.cell
def _(torch):
    ar1 = torch.ones([6,1])
    ar2 = torch.zeros([6,1])

    merged = torch.hstack([ar1, ar2]).squeeze()

    print(merged.shape, merged)

    arr1 = torch.ones([6,2])
    arr2 = torch.zeros([6,2])

    merrged = torch.hstack([arr1, arr2]).squeeze()

    print(merrged.shape, merrged)

    return ar1, ar2


@app.cell
def _(ar1, ar2, torch):
    torch.cat([ar1, ar2],dim=1).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
