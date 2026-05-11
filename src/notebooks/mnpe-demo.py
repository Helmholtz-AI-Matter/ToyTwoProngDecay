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

    We first use `ttpd` to generate some signal data.
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
    return hi, lo, num_signal, signal_prior, theta


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


@app.cell
def _(sim, theta):
    x = sim(theta)
    print(x.shape, x.dtype)
    return (x,)


@app.cell
def _(DEFAULT_DEVICE, theta, x):
    # Now do MNPE demo
    # https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.inference.MNPE.html
    from sbi.inference import MNPE

    num_sims = theta.shape[0]
    theta[:,1] = theta[:,1].int() #convert to int to signal categorical column

    # thetas = theta[1:,...]
    # xs = x[1:,...].unsqueeze(-1)

    inference = MNPE(device=DEFAULT_DEVICE)
    _ = inference.append_simulations(theta.to(DEFAULT_DEVICE), x.to(DEFAULT_DEVICE)).train()
    return (inference,)


@app.cell
def _(inference):

    posterior = inference.build_posterior()

    return


@app.cell
def _(backgd_prior, signal_prior, torch):
    # validation data
    num_val_signal = 150
    num_val_backgd = 50
    num_val = num_val_backgd + num_val_signal

    theta_val = torch.zeros((num_val,2))
    theta_val[:num_val_signal,0] = signal_prior.sample((num_val_signal,))
    theta_val[:num_val_signal,1] = torch.zeros_like(theta_val[:num_val_signal,0])

    theta_val[num_val_signal:,0] = backgd_prior.sample((num_val_backgd,))
    theta_val[num_val_signal:,1] = torch.ones_like(theta_val[num_val_signal:,0])

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
