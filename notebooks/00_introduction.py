"""Landing page for the AIF-controller traffic experiments.

A markdown-only marimo notebook: no simulation, no parameters, no Run button.
It explains what this repository is, the two-layer model, the pluggable
controller, the two communication channels, and how the four experiment
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
          over its route's latent state $(F, C, L, \phi)$ -- free-flow time,
          capacity, queue, and the expected **green split** $\phi$ -- observes
          only the route it took, and chooses by minimising Expected Free
          Energy. A *social internalisation* $\theta$ controls how much of the
          congestion externality each agent folds into its perceived route cost.

        * **Macro layer (controller).** A **pluggable** signal controller. Four
          controllers share one interface so they can be swapped and compared:

          * **fixed-time** -- a constant split (non-adaptive);
          * **reactive** -- shifts green toward the longer queue (SCOOT-like);
          * **anticipatory** -- predictive grid search over the split;
          * **AIF** -- the Active Inference controller (implemented): a Gaussian
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

        * **Cost-offset advisory** ($\theta$) -- a per-route signal (e.g. the
          congestion externality $E_r$) folded into the perceived cost
          $\zeta_r = TT_r + \theta\, E_r$. This shifts route *choice* only.
        * **Controller-belief broadcast** (QB / SP) -- the controller shares its
          own *belief* before travellers choose: its forward-predicted queue
          belief $\mathcal N(\hat L,\widehat{\mathrm{var}})$ (**QB**) and/or its
          planned green split $\hat\phi$ (**SP**). A compliant traveller *fuses*
          that distribution with its own posterior to decide; the fusion is
          transient and never enters the smoother.

        A per-cohort **compliance fraction** sets how many travellers fuse the
        controller's belief; the rest choose on their own posterior alone.
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

        1. **`01_social_internalisation`** — fix the AIF controller, sweep the
           traveller social internalisation $\theta\in\{0,0.25,0.5,0.75,1\}$,
           tracing the user-equilibrium to system-optimum spectrum. Establishes
           the behavioural baseline.
        2. **`02_controller_benchmark`** — compare the AIF controller against
           fixed-time, reactive, and anticipatory controllers (cost, queues,
           signal stability). The core result: AIF outperforms the baselines.
        3. **`03_information_communication`** — having shown AIF performs best,
           investigate *that* controller further: what should it share from its
           own belief? Compare BL / QB / SP / QB+SP (queue belief and/or planned
           split, fused at decision time) and measure the value of information.
        4. **`04_compliance_robustness`** — fix the AIF controller and share its
           full belief (QB+SP), then sweep the **compliance fraction**: how the
           coordination effect changes as fewer travellers fuse the controller's
           belief. Tests whether it degrades *gracefully*.
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
