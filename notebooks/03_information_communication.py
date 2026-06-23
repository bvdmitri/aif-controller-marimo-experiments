"""Experiment 3 -- System information communication.

Fixes the AIF signal controller and a single traveller population, and varies
only **what the controller shares from its own belief** before travellers
choose.

* **BL** -- baseline: the controller shares nothing.
* **QB** -- the controller shares its forward-predicted queue belief
  ``N(L_hat, var)``.
* **SP** -- the controller shares its planned green split ``phi_hat``.
* **QB+SP** -- both.

A compliant traveller fuses the shared Gaussian into its own posterior at
decision time (a transient fusion that never enters the smoother); richer
information sharpens its anticipation of the day. This notebook runs the four
settings at full compliance and overlays the outcomes.
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
        **what the controller shares from its own belief** before travellers
        choose. The controller forward-predicts the day and shares its queue
        belief ($\mathcal N(\hat L,\widehat{\mathrm{var}})$, **QB**) and/or its
        planned green split ($\hat\phi$, **SP**). A compliant traveller *fuses*
        that distribution into its own posterior to decide — a transient fusion
        that never enters its smoother; non-compliant travellers ignore it.

        Set the parameters, click **Run**, and read the overlay below. The
        question is whether richer shared anticipation gives more stable route
        choice and lower system cost.
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
        plot_route_choice_heatmaps,
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
        plot_route_choice_heatmaps,
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
            "QB": _base.with_belief_signals(BeliefSignal.QUEUE_BELIEF),
            "SP": _base.with_belief_signals(BeliefSignal.SPLIT_PLAN),
            "QB+SP": _base.with_belief_signals(
                BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
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
def _(mo):
    mo.md(
        r"""
        ### Route-choice patterns within the day

        The overlay above shows *what* each setting converges to; the heatmaps
        below show *when within the day* travellers pick the intersection route
        $\alpha$ (share $P_\alpha$ over day $\times$ time-of-day), one column per
        setting. Richer information should settle the pattern faster across the
        learning days and reduce the mid-day diversion swing.
        """
    )
    return


@app.cell
def _(figure_placeholder, plot_route_choice_heatmaps, results_by_setting):
    fig_routes = (
        figure_placeholder("Route-choice heatmaps")
        if results_by_setting is None
        else plot_route_choice_heatmaps(results_by_setting)
    )
    fig_routes
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("information_communication"))
    return


if __name__ == "__main__":
    app.run()
