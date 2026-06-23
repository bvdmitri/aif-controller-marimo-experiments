"""Experiment 4 -- Traveller compliance / non-compliance robustness.

Fixes the AIF signal controller and the traveller population, and shares the
controller's full belief (**QB+SP** -- its forward-predicted queue belief and
its planned green split) before travellers choose. We then vary only the
**compliance fraction**: how many travellers actually fuse the controller's
belief into their decision.

* compliance ``0.0`` -- nobody fuses, so the broadcast is an exact no-op:
  bit-identical to the baseline.
* compliance ``1.0`` -- every traveller anticipates the day using the
  controller's shared belief.

The paper claims the coordination effect **degrades gracefully** as travellers
ignore the controller. This notebook runs the compliance sweep and overlays the
outcomes so that claim can be read off directly.
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
        # Experiment 4 — Traveller compliance robustness

        The AIF controller and the traveller population are fixed, and the
        controller shares its full belief (**QB+SP**: its forward-predicted
        queue belief and its planned green split) before travellers choose. We
        vary only the **compliance fraction**: the share of travellers that
        actually *fuse* the controller's belief into their decision; the rest
        decide on their own posterior alone.

        At compliance $0$ nobody fuses, so the broadcast is an exact no-op and
        the run is bit-identical to the baseline; at compliance $1$ every
        traveller anticipates the day with the controller's shared belief. The
        question is whether the effect degrades **gracefully** as compliance
        falls — smoothly, with no cliff.
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
    traveller_window = mo.ui.slider(1, 30, value=10, label="traveller window [days]")
    controller_window = mo.ui.slider(1, 30, value=10, label="controller window [days]")

    run_btn = mo.ui.run_button(label="Run all compliance settings")

    def _row(widget, desc):
        return mo.hstack([widget, mo.md(desc)], widths=[2, 3], align="center", gap=1)

    controls = mo.vstack([
        mo.md("### Parameters you can play with"),
        _row(days, "Total days to simulate (the first warm-up days are discarded)."),
        _row(seed, "Master seed; redraws all stochastic elements."),
        _row(demand_scale,
             r"Scales peak A--B and C--D demand. $>1$ loads the junction and "
             r"sharpens the gap between high- and low-compliance outcomes."),
        _row(traveller_window,
             "Days each traveller's rolling-window smoother remembers when "
             "forming route beliefs (held fixed across settings)."),
        _row(controller_window,
             "Days of past queue observations the AIF controller smooths over "
             "before acting and broadcasting its belief."),
        run_btn,
    ], gap=0.5)
    controls
    return (
        controller_window,
        days,
        demand_scale,
        run_btn,
        seed,
        traveller_window,
    )


@app.cell
def _(
    AIFControllerSpec,
    BeliefSignal,
    DemandParams,
    Params,
    SimParams,
    controller_window,
    days,
    demand_scale,
    mo,
    replace,
    run_btn,
    run_experiment,
    seed,
    traveller_window,
):
    if not run_btn.value:
        results_by_compliance = None
    else:
        base_d = DemandParams()
        _scale = float(demand_scale.value)
        _demand = replace(
            base_d,
            d_AB_max=base_d.d_AB_max * _scale,
            d_CD_max=base_d.d_CD_max * _scale,
        )
        # Fix the AIF controller and share its full belief (QB+SP); the only
        # thing that changes across settings is the compliance fraction (who
        # fuses the controller's belief into their decision).
        _base = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
            controller=AIFControllerSpec(
                controller_window_size=int(controller_window.value)),
            demand=_demand,
        ).with_belief_signals(
            BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
        ).with_window_size(int(traveller_window.value))

        _fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        _settings = {
            f"{int(round(f * 100))}%": _base.with_compliance(f) for f in _fractions
        }
        results_by_compliance = {}
        for _name, _p in mo.status.progress_bar(
            list(_settings.items()), title="compliance settings"
        ):
            results_by_compliance[_name] = run_experiment(_p, seeds=[int(seed.value)])
    return (results_by_compliance,)


@app.cell
def _(figure_placeholder, plot_sweep_metrics, results_by_compliance):
    fig_compliance = (
        figure_placeholder("Compliance settings overlay")
        if results_by_compliance is None
        else plot_sweep_metrics(results_by_compliance)
    )
    fig_compliance
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("compliance_robustness"))
    return


if __name__ == "__main__":
    app.run()
