import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


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

    from ttpd.generator import (
        SimulateFactory,
        invariant_mass_from_ptphieta,
        mZ0,
    )

    # create simulator factory
    factory = SimulateFactory.create(device=torch.device("cpu"))

    return factory, invariant_mass_from_ptphieta, mZ0, plt, torch


@app.cell
def _(mZ0, torch):
    # generate signal and background prior samples
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
    The simulator is seeded so the notebook always produces the same example
    batch when the documentation site is rebuilt.
    """)
    return


@app.cell
def _(factory, invariant_mass_from_ptphieta, plt, theta, torch):

    simulate = factory.create_simulator(
        generation_seed=123, smear_seed=321, device=torch.device("cpu")
    )
    events = simulate(theta)
    masses = invariant_mass_from_ptphieta(events)
    delta = masses - theta[:, 0].unsqueeze(1)

    print("batch shape:", events.shape)
    print("prior labels:", theta[:, 1].unique())
    print("first 5 reconstructed masses:", masses[:5].flatten())
    print("delta mean:", delta.mean().item())

    plt.close("all")
    return delta, masses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare prior and reconstructed spectra
    The final plot compares the sampled parent-mass prior with the reconstructed
    invariant masses obtained from the smeared decay products.
    """)
    return


@app.cell
def _(delta, masses, plt, theta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    prior = theta[:, 0].numpy()
    obs = masses.flatten().numpy()

    axes[0].hist(prior, bins=20, color="tab:blue", alpha=0.7)
    axes[0].set_title("Prior mass (theta[:,0])")
    axes[0].set_xlabel("Mass / GeV")

    axes[1].hist(obs, bins=20, color="tab:orange", alpha=0.7)
    axes[1].set_title("Reconstructed mass from events smeared events")
    axes[1].set_xlabel("Mass / GeV")

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
