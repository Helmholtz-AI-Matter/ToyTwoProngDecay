import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    np.random.seed(42)
    torch.random.manual_seed(42)

    # Use GPU if available
    DEFAULT_DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"using {DEFAULT_DEVICE} as device")
    return DEFAULT_DEVICE, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate some signal

    We first use `ttpd` to generate some signal data. We populate one signal channel (cauchy distribution) and one background channel (uniform distribution).
    """)
    return


@app.cell
def _(DEFAULT_DEVICE):
    from ttpd.generator import SimulateFactory
    from ttpd.kinematics import invariant_mass_from_ptphieta, mZ0

    # build the simulator
    factory = SimulateFactory.create(device=DEFAULT_DEVICE)
    sim = factory.create_simulator(generation_seed=1137, smear_seed=1237)
    return mZ0, sim


@app.cell
def _(mZ0, np, torch):
    #create signal and background
    num_signal = 10_000
    num_bkrd = 2_000

    signal_prior = torch.distributions.Cauchy(loc=mZ0, scale=.1)

    theta = torch.empty((num_signal+num_bkrd,2), dtype=torch.float32)
    theta[:num_signal,0] = signal_prior.sample((num_signal,))
    theta[:num_signal,1] = torch.zeros_like(theta[:num_signal,0])

    q05, q95 = torch.quantile(theta[:num_signal,0], 
                              torch.from_numpy(np.asarray([.05,.95], dtype=np.float32))
                             )

    #we cannot allow negative masses, long tails of cauchy sometimes sample below 0. 
    lo,hi = float(max(q05.numpy(),0.)), float(q95)
    bkrd_prior = torch.distributions.Uniform(low=lo, high=hi)

    theta[num_signal:,0] = bkrd_prior.sample((num_bkrd,))
    theta[num_signal:,1] = torch.ones_like(theta[num_signal:,0])
    return bkrd_prior, hi, lo, num_signal, signal_prior, theta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For visual quality control, we plot the generated masses which fall into $\vartheta_0$.
    """)
    return


@app.cell
def _(hi, lo, num_signal, plt, theta):
    plt.hist(theta[:num_signal,0],bins=30, range=(lo,hi), label="signal")
    plt.hist(theta[num_signal:,0],bins=30, range=(lo,hi), label="background")
    plt.title("prior mass values")
    plt.xlabel("mass / a.u.")
    plt.yscale("log")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We perform simulations. This takes the input invariant masses and constructs the 4-vectors of 2 daughter particles.
    """)
    return


@app.cell
def _(sim, theta):
    x = sim(theta)
    print(x.shape, x.dtype)
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training an MNPE density estimator

    The `sbi` toolbox is capable of learning a density estimator, which can estimate $p(\vartheta|x)$, i.e. the posterior density of $vartheta$ given the observed data $x$.
    """)
    return


@app.cell
def _(DEFAULT_DEVICE, theta, x):
    # Now do MNPE demo
    # https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.inference.MNPE.html
    from sbi.inference import MNPE

    num_sims = theta.shape[0]
    theta[:,1] = theta[:,1].int() #convert to int to signal categorical column

    inference = MNPE(device=DEFAULT_DEVICE)
    _ = inference.append_simulations(theta.to(DEFAULT_DEVICE), x.to(DEFAULT_DEVICE)).train()
    return (inference,)


@app.cell
def _(inference):

    posterior = inference.build_posterior()
    return (posterior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation data

    In order to draw independent conclusions, let's sample some more data and use that to check the validity of our trained density estimator.
    """)
    return


@app.cell
def _(bkrd_prior, signal_prior, torch):
    # validation data
    num_val_signal = 150
    num_val_backgd = 50
    num_val = num_val_backgd + num_val_signal

    theta_val = torch.zeros((num_val,2))
    theta_val[:num_val_signal,0] = signal_prior.sample((num_val_signal,))
    theta_val[:num_val_signal,1] = torch.zeros_like(theta_val[:num_val_signal,0])

    theta_val[num_val_signal:,0] = bkrd_prior.sample((num_val_backgd,))
    theta_val[num_val_signal:,1] = torch.ones_like(theta_val[num_val_signal:,0])
    return (theta_val,)


@app.cell
def _(sim, theta_val):
    x_val = sim(theta_val)
    return (x_val,)


@app.cell
def _(np, posterior, theta_val, x_val):
    from sbi.analysis import pairplot
    from sbi.analysis.plotting_classes import HistDiagOptions

    first_theta = theta_val[:1,...]
    first_x = x_val[:1,...]

    num_posterior_samples = 200
    samples = posterior.sample((num_posterior_samples,),x=first_x)

    fig, axes = pairplot(
        samples.cpu(),
        limits=[[70,120],[-1,2]],
        figsize=(5, 5),
        points=first_theta.cpu(),
        labels=["deconv mass","estimate label"],
        diag_kwargs=HistDiagOptions(
            mpl_kwargs={
                "bins": np.arange(62,122,1),
            }
        )
    )
    fig.suptitle("validation prediction for signal type")
    fig
    return HistDiagOptions, num_posterior_samples, pairplot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This single item from the validation set comes out right ontop of the truth. The problem appears to be very easy for MNPE. Let's try the same with a background event. Here the challenge is to fit a uniform distribution.
    """)
    return


@app.cell
def _(
    HistDiagOptions,
    np,
    num_posterior_samples,
    pairplot,
    posterior,
    theta_val,
    x_val,
):
    last_theta = theta_val[-1:,...]
    last_x = x_val[-1:,...]

    samples_ = posterior.sample((num_posterior_samples,),x=last_x)

    fig_, axes_ = pairplot(
        samples_.cpu(),
        limits=[[70,120],[-1,2]],
        figsize=(5, 5),
        points=last_theta.cpu(),
        labels=["deconv mass","estimate label"],
        diag_kwargs=HistDiagOptions(
            mpl_kwargs={
                "bins": np.arange(62,122,1),
            }
        )
    )
    fig_.suptitle("validation prediction for signal type")
    fig_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior Checks

    [TBA](https://sbi.readthedocs.io/en/stable/advanced_tutorials/10_diagnostics_posterior_predictive_checks.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Critique

    [TBA](https://sbi.readthedocs.io/en/stable/advanced_tutorials/11_diagnostics_simulation_based_calibration.html#posterior-calibration-with-tarp-lemos-et-al-2023)
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
