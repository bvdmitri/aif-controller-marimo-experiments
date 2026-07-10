"""Landing page for the AIF-controller traffic experiments.

A markdown-only marimo notebook: no simulation, no parameters, no Run button.
It explains what this repository is, the two-layer model, the pluggable
controller, the controller-to-traveller communication channels, and how the four experiment
notebooks are organised.
"""

import marimo

__generated_with = "0.23.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Active Inference traffic control: experiment companion

        This repository is the interactive companion to the paper extending our
        IWAI route-choice work with a **macro-layer signal controller**. The
        question it explores: how should an Active Inference signal controller
        *communicate* with Active Inference travellers, which information is most
        useful to share, and what happens when travellers **ignore** it?

        The network is a single signalised intersection with a bypass. Travellers
        from $A$ to $B$ choose between

        * the **intersection route** $\alpha$ (through the signalised junction), and
        * the **bypass route** $\beta$ (longer but uncongestable),

        while a competing $C\to D$ stream $\gamma$ shares the junction and competes
        for green time. A signal controller allocates the green-time split between
        the two competing movements (link 2 for $A\!-\!B$, link 6 for $C\!-\!D$).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Two layers

        * **Micro layer (travellers).** Decentralised Active Inference agents,
          reused from the IWAI demonstration. Each agent holds a Gaussian belief
          over its route's latent state $(F, C, L, \phi)$ (free-flow time,
          capacity, queue, and the expected **green split** $\phi$), observes
          only the route it took, and chooses by minimising Expected Free
          Energy over its predicted travel time.

        * **Macro layer (controller).** A **pluggable** signal controller. Four
          controllers share one interface so they can be swapped and compared:

          * **fixed-time**: a constant split (non-adaptive);
          * **reactive**: shifts green toward the longer queue (SCOOT-like);
          * **anticipatory**: predictive grid search over the split;
          * **AIF**: the Active Inference controller (implemented): a Gaussian
            belief over the junction queues $(L_2, L_6)$ and split selection by
            minimising the Expected Free Energy toward a preferred *empty,
            balanced* queue observation.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Communication and compliance

        The controller has a network-wide view and can share it through **two
        distinct channels**:

        * **Extra observations** (CG / SN): travellers natively see only the
          route they took; the controller relays the *true realised* route
          congestion $L_r$ (**CG**) and/or signal split $\phi_r$ (**SN**) of the
          routes they did *not* take, folded into their end-of-day belief update.
          Reaches every traveller and works with any controller.
        * **Controller-belief broadcast** (QB / SP): the AIF controller shares
          its own *belief* before travellers choose: its forward-predicted queue
          belief $\mathcal N(\hat L,\widehat{\mathrm{var}})$ (**QB**) and/or its
          planned green split $\hat\phi$ (**SP**). A compliant traveller *fuses*
          that distribution with its own posterior to decide; the fusion is
          transient and never enters the smoother.

        A per-cohort **compliance fraction** sets how many travellers fuse the
        controller's belief (the QB/SP channel); the rest choose on their own
        posterior alone. Extra observations are not gated by compliance.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The four experiments

        Each notebook isolates one mechanism, and the storyline builds on the
        previous one:

        1. **`01_coordination_mechanism`**: fix the AIF controller and a single
           traveller population, and show how the two layers **co-adapt** through
           the shared network within a day and across days. Establishes the
           behavioural baseline.
        2. **`02_controller_benchmark`**: compare the AIF controller against
           fixed-time, reactive, and anticipatory controllers (cost, queues,
           signal stability). The core result: AIF outperforms the baselines.
        3. **`03_information_communication`**: vary *what travellers learn about
           the network*. By default, relay the true realised congestion/split of
           the routes they did not take (BL / CG / SN / CG+SN, the **extra
           observations** channel); optionally instead share the AIF controller's
           own forecast belief (BL / QB / SP / QB+SP, fused at decision time).
           Measure the value of information.
        4. **`05_robustness`**: fix the AIF controller and re-run the coupled
           system across **traffic-demand scales**, checking that the two layers
           keep coordinating rather than relying on a single fixed operating
           point as the network fills up.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to run

        * **Tests**: `uv run --extra dev pytest tests/ -q`
        * **Headless smoke** (exercises the coupled pipeline for every controller):
          `uv run python scripts/smoke_notebooks.py`
        * **This notebook**: `uv run marimo edit notebooks/00_introduction.py`

        Inference is closed-form and deterministic (a rolling-window Gaussian
        smoother and closed-form Expected Free Energy; no Monte-Carlo sampling), so
        a given configuration and seed reproduces exactly.
        """
    )
    return


if __name__ == "__main__":
    app.run()
