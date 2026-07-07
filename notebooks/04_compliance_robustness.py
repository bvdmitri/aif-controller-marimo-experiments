"""Compliance / non-compliance robustness (exploratory -- not a paper experiment).

This notebook exercises the **belief-sharing** channel, which is parked for a
future "heterogeneity" paper and kept off by default elsewhere; it is **not**
part of the current paper's experiment set (the paper's communication story is
the "extra observations" relay of notebook 03). It is retained here as an
exploratory artifact for the belief-sharing + compliance idea.

It fixes the AIF signal controller and the traveller population, and shares the
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

    from aif_traffic import notebook_controls as nc
    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.notebook_io import figure_block, sweep_progress_bar
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
        figure_block,
        figure_placeholder,
        nc,
        notebook_explainer,
        plot_sweep_metrics,
        replace,
        run_experiment,
        sweep_progress_bar,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo, nc):
    # All controls come from aif_traffic.notebook_controls (shared across the
    # experiments; see CLAUDE.md). compliance is the swept variable here, so it
    # is not a slider; theta is not used (no externality channel).
    days = nc.days()
    warmup = nc.warmup()
    seed = nc.seed()
    control_interval = nc.control_interval()
    demand_scale = nc.demand_scale()
    learn_noise = nc.learn_noise()
    noise_regime = nc.noise_regime()
    stationary = nc.stationary()
    time_step = nc.time_step()
    bypass_capacity_scale = nc.bypass_capacity_scale()

    run_btn = mo.ui.run_button(label="Run all compliance settings")
    return (
        bypass_capacity_scale,
        control_interval,
        days,
        demand_scale,
        learn_noise,
        noise_regime,
        run_btn,
        seed,
        stationary,
        time_step,
        warmup,
    )


@app.cell
def _(bypass_capacity_scale, control_interval, days, demand_scale, learn_noise,
      nc, noise_regime, run_btn, seed, stationary, time_step, warmup):
    # Window sliders under (and disabled by) the stationary toggle.
    traveller_window = nc.traveller_window(disabled=stationary.value)
    controller_window = nc.controller_window(disabled=stationary.value)

    controls = nc.standard_panel({
        "days": days, "warmup": warmup, "seed": seed,
        "time_step": time_step, "control_interval": control_interval,
        "demand_scale": demand_scale,
        "bypass_capacity_scale": bypass_capacity_scale,
        "learn_noise": learn_noise, "noise_regime": noise_regime,
        "stationary": stationary, "traveller_window": traveller_window,
        "controller_window": controller_window,
    }, run_btn)
    controls
    return controller_window, traveller_window


@app.cell
def _(
    AIFControllerSpec,
    BeliefSignal,
    DemandParams,
    Params,
    SimParams,
    bypass_capacity_scale,
    control_interval,
    controller_window,
    days,
    demand_scale,
    learn_noise,
    noise_regime,
    replace,
    run_btn,
    run_experiment,
    seed,
    stationary,
    sweep_progress_bar,
    time_step,
    traveller_window,
    warmup,
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
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value),
                        burn_in=int(warmup.value), dt_min=int(time_step.value)),
            controller=AIFControllerSpec(
                control_interval_min=int(control_interval.value),
                horizon_min=int(control_interval.value),
                controller_window_size=int(controller_window.value)),
            demand=_demand,
        ).with_belief_signals(
            BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
        ).with_window_size(int(traveller_window.value)).with_learn_obs_noise(
            bool(learn_noise.value)
        ).with_stationary(bool(stationary.value)).with_noise_regime(
            noise_regime.value
        ).with_bypass_capacity_scale(float(bypass_capacity_scale.value))

        _fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        _settings = {
            f"{int(round(f * 100))}%": _base.with_compliance(f) for f in _fractions
        }
        results_by_compliance = {}
        with sweep_progress_bar(
            len(_settings), _base.sim, title="compliance settings"
        ) as _bar:
            for _name, _p in _settings.items():
                results_by_compliance[_name] = run_experiment(
                    _p, seeds=[int(seed.value)], on_step=_bar.update)
    return (results_by_compliance,)


@app.cell
def _(figure_block, figure_placeholder, plot_sweep_metrics, results_by_compliance):
    fig_compliance = (
        figure_placeholder("Compliance settings overlay")
        if results_by_compliance is None
        else plot_sweep_metrics(results_by_compliance)
    )
    figure_block("plot_sweep_metrics", fig_compliance)
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("compliance_robustness"))
    return


if __name__ == "__main__":
    app.run()
