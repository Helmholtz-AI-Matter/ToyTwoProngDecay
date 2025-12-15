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
def _(DEFAULT_DEVICE, eta1_angle, eta2, mMu, np, phi2, torch):
    def resolution(mass):
        slope = -.014 #-.01342
        xoffset = 1.503
        yoffset = .01
        sigmas = (slope*mass + xoffset)**2 + yoffset #low masses are more smeared than higher ones
        return sigmas


    def smear(masses):
        """
        Detector smearing model:
        Multiply mass by a Gaussian-distributed random factor.
        Gaussian mean = 1.0, sigma = 0.01 * log(mass).

        Parameters
        ----------
        masses : torch.Tensor
            batched masses

        Returns
        -------
        torch.Tensor: detector-level smeared mass
        """
        sigmas = 0.1*resolution(masses) #low masses are more smeared than higher ones
        assert sigmas.shape == masses.shape
        smear_factor = torch.normal(mean=torch.ones_like(masses), std=sigmas)
        detector_mass = masses * smear_factor
        return detector_mass

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
    
        # split available energy
        E_mu = theta[:,0] / 2.
        p = np.sqrt(E_mu**2 - mMu**2).unsqueeze(1)

        # prong 1
        cos_theta1_ = torch.distributions.uniform.Uniform(-1,1)
        theta1_angle = torch.arccos(cos_theta1_.sample((n_samples,1)))
        eta1 = -np.log(np.tan(theta1_angle / 2))
    
        phi1_ = torch.distributions.uniform.Uniform(0,2*torch.pi)
        phi1_angle = phi1_.sample((n_samples,1))

        pT1 = p * torch.sin(theta1_angle)
        mu1 = torch.stack([pT1, phi1_angle, eta1, mMu*torch.ones_like(eta1)],dim=1).squeeze()

        # prong 2: back-to-back muon 2
        phi2_angle = phi1_angle + torch.pi
        phi2_mask = phi2 >= 2*torch.pi
        phi2_angle[phi2_mask, :] -= 2.*torch.pi

        eta2_angle = -eta1_angle       # follows from θ → π − θ
        pT2 = pT1          # same pT

        mu2 = torch.stack([pT2, phi2_angle, eta2, mMu*torch.ones_like(eta1)],dim=1).squeeze()

        return torch.stack([mu1,mu2], dim=1).squeeze()

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
        # could smear depending on signal or background (theta[:,1])
        # to emulate detector response due to different momentum profiles
        detector_mass = smear(theta[:,0])

        return detector_mass
    return resolution, simulate, smear


@app.cell
def _(plt, resolution, smear, torch):
    smearing_inputs = torch.linspace(62,122,50)
    smearing_outputs = smear(smearing_inputs)

    fig1, ax = plt.subplots(1,3, tight_layout=True, figsize=(10,5))
    ax1, ax2, ax3 = ax

    ax1.plot(smearing_inputs, .1*resolution(smearing_inputs))
    ax1.set_title("sigma for fwd conv given input mass")
    ax1.set_xlabel("input mass / 'GeV'")
    ax1.set_ylabel("sigma / a.u.")

    ax2.plot(smearing_inputs, smearing_outputs)
    ax2.set_title("after fwd conv")
    ax2.set_xlabel("input mass / 'GeV'")
    ax2.set_ylabel("output mass / 'GeV'")


    ax3.plot(smearing_inputs, torch.normal(mean=torch.ones_like(smearing_inputs), std=.1*resolution(smearing_inputs)))
    ax3.set_title("smear factor to multiply input mass with")

    #fig1.show()
    return


@app.cell
def _(torch):
    #create signal and background
    num_signal = 2000
    num_bkrd = 200

    signal_prior = torch.distributions.Cauchy(loc=92.1, scale=.05)
    bkrd_prior = torch.distributions.Uniform(62.,122.)

    theta = torch.empty((num_signal+num_bkrd,2))
    theta[:num_signal,0] = signal_prior.sample((num_signal,))
    theta[:num_signal,1] = torch.zeros_like(theta[:num_signal,0])

    theta[num_signal:,0] = bkrd_prior.sample((num_bkrd,))
    theta[num_signal:,1] = torch.ones_like(theta[num_signal:,0])
    return num_signal, theta


@app.cell
def _(simulate, theta):
    x = simulate(theta) # would be great to have 4-vectors for real decays in x
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
    arr1 = torch.ones([6,1])
    arr2 = torch.zeros([6,1])

    merged = torch.stack([arr1, arr2], dim=1).squeeze()

    print(merged.shape, merged)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
