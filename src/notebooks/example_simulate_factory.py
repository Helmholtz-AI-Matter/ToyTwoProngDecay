import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SimulateFactory example
    This notebook demonstrates how to configure `SimulateFactory` for Z⁰→μ⁺μ⁻ signal generations plus background processes,
    then sketches the resulting invariant mass distributions for the prior, the smeared observations, and their difference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure the simulator
    The first step is to import the physics helpers and instantiate a
    `SimulateFactory` on CPU so the example remains reproducible across local and
    CI documentation builds.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch

    from ttpd.generator import SimulateFactory
    from ttpd.kinematics import invariant_mass_from_ptphieta, mZ0

    # create simulator factory
    factory = SimulateFactory.create(device=torch.device("cpu"))
    return factory, invariant_mass_from_ptphieta, mZ0, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's generate some signal and background prior samples.
    """)
    return


@app.cell
def _(mZ0, torch):
    signal_theta = torch.hstack([torch.full((16, 1), mZ0), torch.zeros((16, 1))])
    background_theta_1 = torch.hstack(
        [
            torch.linspace(70.0, 72.0, 4).unsqueeze(1),
            torch.ones((4, 1)),
        ]
    )
    background_theta_2 = torch.hstack(
        [
            torch.linspace(115.0, 118.0, 4).unsqueeze(1),
            torch.ones((4, 1)),
        ]
    )
    theta = torch.vstack([signal_theta, background_theta_1, background_theta_2])
    return (theta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate events and inspect reconstructed masses

    We now bootstrap the simulate function from the factory object. This has the benefit that we can integrate useful parameters like seeds or smearing functions or target devices into the simulate call directly.
    """)
    return


@app.cell
def _(factory, theta, torch):

    simulate = factory.create_simulator(
        generation_seed=123, smear_seed=321, device=torch.device("cpu")
    )
    events = simulate(theta)
    print("batch shape:", events.shape)
    print("prior labels:", theta[:, 1].unique())

    return events, simulate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simulate function returns 4-vectors of the two decay prongs only. We can now continue and calculate the invariant masses of the events provided.
    """)
    return


@app.cell
def _(events, invariant_mass_from_ptphieta, plt, theta):
    masses = invariant_mass_from_ptphieta(events)
    delta = masses - theta[:, 0].unsqueeze(1)

    print("first 5 reconstructed masses:", masses[:5].flatten())
    print("delta mean:", delta.mean().item())

    plt.close("all")
    return (masses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare prior and reconstructed spectra
    The following plot compares the sampled parent-mass prior with the reconstructed invariant masses obtained from the smeared decay products.
    """)
    return


@app.cell
def _(masses, plt, theta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    prior = theta[:, 0].numpy()
    obs = masses.flatten().numpy()

    axes[0].hist(prior, bins=20, color="tab:blue", alpha=0.7)
    axes[0].set_title("Prior mass (theta[:,0])")
    axes[0].set_xlabel("Mass / GeV")

    axes[1].hist(obs, bins=20, color="tab:orange", alpha=0.7)
    axes[1].set_title("Reconstructed mass from events smeared events")
    axes[1].set_xlabel("Mass / GeV")

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More real world generation

    In the cell below, we generate a set of signal events and one background channel only. For each channel, we select a distribution of mass values, but have to assign the channel ID by virtue of an integer (`0` for signal, `1` for background).
    """)
    return


@app.cell
def _(mZ0, torch):
    # prepare channel id
    num_signal = 5_000
    num_backgd = 1_000
    theta_ids = torch.concat(
        [torch.zeros((num_signal, 1)), torch.ones((num_backgd, 1))]
    )

    # prepare pdf or prior for each channel
    signal_pdf = torch.distributions.Cauchy(loc=mZ0, scale=0.05)
    backgd_pdf = torch.distributions.Uniform(mZ0 - 30, mZ0 + 30)
    theta_masses = torch.concat(
        [signal_pdf.sample((num_signal, 1)), backgd_pdf.sample((num_backgd, 1))]
    )

    # join everything into one tensor
    thetas = torch.hstack([theta_masses, theta_ids])
    return num_backgd, num_signal, thetas


@app.cell
def _(thetas):
    # what it looks like
    print(
        f"created simulation parameter samples of shape {thetas.shape}\n and type {thetas.dtype}\n"
    )
    print(f"the first entries contain signal 'events'\n{thetas[:5, ...]}")
    print(f"the last entries contain background 'events'\n{thetas[-5:, ...]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we can go forward and generate some observations `xs`.
    """)
    return


@app.cell
def _(simulate, thetas):
    # let's simulate
    xs = simulate(thetas)
    return (xs,)


@app.cell
def _(
    invariant_mass_from_ptphieta,
    mZ0,
    num_backgd,
    num_signal,
    plt,
    thetas,
    torch,
    xs,
):
    # let's inspect
    figa, axesa = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    masses_ = invariant_mass_from_ptphieta(xs)

    axesa[0].hist(
        [thetas[:num_signal, 0], thetas[-num_backgd:, 0]],
        bins=torch.arange(mZ0 - 20, mZ0 + 20, 1),
        stacked=True,
        label=["signal", "background"],
    )
    axesa[0].set_title("Prior/generated mass")
    axesa[0].set_xlabel("Mass / GeV")

    axesa[1].hist(
        [masses_[:num_signal, 0], masses_[-num_backgd:, 0]],
        bins=torch.arange(mZ0 - 20, mZ0 + 20, 1),
        stacked=True,
        label=["signal", "background"],
    )
    axesa[1].set_title("invariant mass from smeared events")
    axesa[1].set_xlabel("Mass / GeV")
    axesa[1].legend()
    figa
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above plot shows the generated samples (left) and smeared+reconstructed on the right. We see that the cauchy distribution has become wider as we smeared the kinematics of the decay products.
    """)
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
