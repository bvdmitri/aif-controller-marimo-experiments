"""Landing page for the AIF-controller traffic experiments.

A markdown-only marimo notebook: no simulation, no parameters, no Run button.
It explains what this repository is, the two-layer model, the pluggable
controller, and how the experiment notebooks (added later) will be organised.
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
          over its route's latent travel-time state $(F, C, L)$, observes only the
          route it took, and chooses by minimising Expected Free Energy. A *social
          internalisation* $\theta$ controls how much of the broadcast congestion
          externality each agent folds into its perceived route cost.

        * **Macro layer (controller).** A **pluggable** signal controller. Several
          controllers share one interface so they can be swapped and compared:

          * **fixed-time** -- a constant split (non-adaptive);
          * **reactive** -- shifts green toward the longer queue (SCOOT-like);
          * **anticipatory** -- predictive grid search over the split;
          * **AIF** -- the Active Inference controller. *Its internal model is an
            open design question;* for now it is a placeholder that conforms to the
            interface so the whole pipeline runs end to end.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Communication and compliance

        The controller has a network-wide view and broadcasts an information
        signal that travellers may fold into their perceived route cost
        $\zeta_r = TT_r + \theta\, E_r$. Candidate signals: travel time, congestion
        (queue), congestion externality $E_r$, or marginal social cost. A per-cohort
        **compliance fraction** sets how many travellers actually read the
        broadcast; the rest ignore it and choose on private travel time alone.

        Which signal is most effective, and how robust coordination is to
        travellers ignoring it, are the two questions the experiments will probe.

        > **Status.** The repository currently ships the *structure*: the network,
        > the traveller model, the controller abstraction with its baselines, the
        > communication mechanism, and tests. The experiment notebooks and the
        > Active Inference controller's formulation are developed next, in step with
        > the paper's methodology section.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to run

        * **Tests** -- `uv run --extra dev pytest tests/ -q`
        * **Headless smoke** (exercises the coupled pipeline for every controller) --
          `uv run python scripts/smoke_notebooks.py`
        * **This notebook** -- `uv run marimo edit notebooks/00_introduction.py`

        Inference is closed-form and deterministic (a rolling-window Gaussian
        smoother and closed-form Expected Free Energy; no Monte-Carlo sampling), so
        a given configuration and seed reproduces exactly.
        """
    )
    return


if __name__ == "__main__":
    app.run()
