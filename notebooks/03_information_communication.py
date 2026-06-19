"""Experiment 3 -- System information communication.

Fixes the AIF signal controller and a single traveller population, and varies
only the **belief-informing broadcast**: what the controller shares with
travellers about routes they did not take.

* **BL** -- baseline: travellers observe only their own realised travel time.
* **CG** -- the controller broadcasts route queues ``L_hat_r``.
* **SN** -- the controller broadcasts the green split ``phi_hat_r``.
* **CG+SN** -- both.

Compliant travellers fold the broadcast into their belief about the unchosen
routes, so richer information should sharpen beliefs (lower posterior
uncertainty), smooth route choice, and lower system cost. CG+SN is expected to
be best. This notebook runs the four settings and overlays the outcomes.
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
        # Experiment 3 — System information communication

        The AIF controller and the traveller population are fixed; we vary only
        what the controller **broadcasts** to travellers about routes they did
        not take that day. A compliant traveller folds the broadcast queue
        ($\hat L_r$, **CG**) and/or green split ($\hat\phi_r$, **SN**) into its
        belief, so it can learn about a route without driving it. The first-hand
        observation on the route actually taken always wins (no double
        counting), and non-compliant travellers ignore the broadcast.

        Set the parameters, click **Run**, and read the overlay below. The
        expectation is that **CG+SN** gives the most stable route choice and the
        lowest system cost.
        """
    )
    return


@app.cell
def _():
    from dataclasses import replace

    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.parameters import (
        AIFControllerSpec,
        BeliefSignal,
        DemandParams,
        Params,
        SimParams,
    )
    from aif_traffic.plotting import (
        figure_placeholder,
        plot_sweep_metrics,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        BeliefSignal,
        DemandParams,
        Params,
        SimParams,
        explainer_pointer,
        figure_placeholder,
        notebook_explainer,
        plot_sweep_metrics,
        replace,
        run_experiment,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo):
    days = mo.ui.slider(10, 180, value=90, label="days")
    seed = mo.ui.slider(0, 100, value=42, label="seed")
    demand_scale = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="demand scale")
    theta = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="theta")
    compliance = mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="compliance")

    run_btn = mo.ui.run_button(label="Run all communication settings")

    def _row(widget, desc):
        return mo.hstack([widget, mo.md(desc)], widths=[2, 3], align="center", gap=1)

    controls = mo.vstack([
        mo.md("### Parameters you can play with"),
        _row(days, "Total days to simulate (the first warm-up days are discarded)."),
        _row(seed, "Master seed; redraws all stochastic elements."),
        _row(demand_scale,
             r"Scales peak A--B and C--D demand. $>1$ loads the junction and "
             r"sharpens the differences between communication settings."),
        _row(theta,
             r"Social internalisation $\theta$, held fixed across the four "
             r"settings so the comparison isolates the broadcast."),
        _row(compliance,
             r"Fraction of travellers that read the broadcast. At $0$ every "
             r"setting collapses onto the baseline."),
        run_btn,
    ], gap=0.5)
    controls
    return compliance, days, demand_scale, run_btn, seed, theta


@app.cell
def _(
    AIFControllerSpec,
    BeliefSignal,
    DemandParams,
    Params,
    SimParams,
    compliance,
    days,
    demand_scale,
    mo,
    replace,
    run_btn,
    run_experiment,
    seed,
    theta,
):
    if not run_btn.value:
        results_by_setting = None
    else:
        base_d = DemandParams()
        _scale = float(demand_scale.value)
        _demand = replace(
            base_d,
            d_AB_max=base_d.d_AB_max * _scale,
            d_CD_max=base_d.d_CD_max * _scale,
        )
        _base = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
            controller=AIFControllerSpec(),
            demand=_demand,
        ).with_theta(float(theta.value)).with_compliance(float(compliance.value))

        _settings = {
            "BL": _base.with_belief_signals(),
            "CG": _base.with_belief_signals(BeliefSignal.CONGESTION),
            "SN": _base.with_belief_signals(BeliefSignal.GREEN_SPLIT),
            "CG+SN": _base.with_belief_signals(
                BeliefSignal.CONGESTION, BeliefSignal.GREEN_SPLIT
            ),
        }
        results_by_setting = {}
        for _name, _p in mo.status.progress_bar(
            list(_settings.items()), title="communication settings"
        ):
            results_by_setting[_name] = run_experiment(_p, seeds=[int(seed.value)])
    return (results_by_setting,)


@app.cell
def _(figure_placeholder, plot_sweep_metrics, results_by_setting):
    fig_comm = (
        figure_placeholder("Communication settings overlay")
        if results_by_setting is None
        else plot_sweep_metrics(results_by_setting)
    )
    fig_comm
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("information_communication"))
    return


if __name__ == "__main__":
    app.run()
