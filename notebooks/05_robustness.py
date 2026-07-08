"""Robustness of the coupled two-layer system to traffic demand.

Fixes the AIF signal controller (sequential-from-belief externality advisory on,
full compliance) and re-runs the coupled system at several **traffic-demand
scales**, multiplying the peak A--B and C--D demand by {0.8, 1.0, 1.2, 1.4}. It
asks whether travellers and the controller keep coordinating as the network fills
up rather than relying on a single fixed operating point: within one day the
route flows and green split should shift together with the load, and across days
each load should re-settle to its own stable route split and signal policy.
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
        # Experiment 5 — Robustness to traffic demand

        Does the coupled two-layer Active Inference framework keep coordinating as
        the network load changes, or does it rely on a single fixed operating
        point? Here the AIF controller is fixed (sequential-from-belief
        externality advisory on, full compliance) and the coupled system is re-run
        at several **traffic-demand scales**, multiplying the peak A--B and C--D
        demand by $\{0.8, 1.0, 1.2, 1.4\}$; one coloured line per scale.

        - **Within-day.** At a representative day, the intersection-route flow
          $Q_\alpha$, the bypass-route flow $Q_\beta$ and the controller's green
          split $\phi_2$ across the day. As demand grows, more travellers should
          divert to the bypass at the peak while the controller allocates more
          green time, the two adapting together.
        - **Across-day.** The daily route share $P_\alpha$, the daily mean green
          split $\phi_2$, and the daily total system cost (with the controller's
          cost-belief SD on a right axis). Each load should re-settle to its own
          stable split and policy, with cost and controller uncertainty falling
          together.
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
        DemandParams,
        Params,
        SignalType,
        SimParams,
    )
    from aif_traffic.plotting import (
        figure_placeholder,
        plot_across_day_by_demand,
        plot_within_day_by_demand,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        DemandParams,
        Params,
        SignalType,
        SimParams,
        explainer_pointer,
        figure_block,
        figure_placeholder,
        nc,
        notebook_explainer,
        plot_across_day_by_demand,
        plot_within_day_by_demand,
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
    # experiments; see CLAUDE.md). Traffic demand is swept (hard-coded list
    # below), so it is not a panel slider; compliance is fixed at 1.0 so the
    # advisory acts. The advisory mechanism / seed default to the sequential
    # externality seeded from belief (the good default; the raw single-value
    # externality herds and is a poor baseline).
    days = nc.days()
    warmup = nc.warmup()
    seed = nc.seed()
    control_interval = nc.control_interval()
    signal_mechanism = nc.signal_mechanism()
    sequential_increments = nc.sequential_increments()
    sequential_seed = nc.sequential_seed()
    advisory_smoothing = nc.advisory_smoothing()
    learn_noise = nc.learn_noise()
    noise_regime = nc.noise_regime()
    stationary = nc.stationary()
    time_step = nc.time_step()

    run_btn = mo.ui.run_button(label="Run the demand-robustness sweep")
    return (
        advisory_smoothing,
        control_interval,
        days,
        learn_noise,
        noise_regime,
        run_btn,
        seed,
        sequential_increments,
        sequential_seed,
        signal_mechanism,
        stationary,
        time_step,
        warmup,
    )


@app.cell
def _(advisory_smoothing, control_interval, days, learn_noise, nc,
      noise_regime, run_btn, seed, sequential_increments, sequential_seed,
      signal_mechanism, stationary, time_step, warmup):
    # Window sliders under (and disabled by) the stationary toggle.
    traveller_window = nc.traveller_window(disabled=stationary.value)
    controller_window = nc.controller_window(disabled=stationary.value)

    controls = nc.standard_panel({
        "days": days, "warmup": warmup, "seed": seed,
        "time_step": time_step, "control_interval": control_interval,
        "learn_noise": learn_noise, "noise_regime": noise_regime,
        "stationary": stationary, "traveller_window": traveller_window,
        "controller_window": controller_window,
        "signal_mechanism": signal_mechanism,
        "sequential_increments": sequential_increments,
        "sequential_seed": sequential_seed,
        "advisory_smoothing": advisory_smoothing,
    }, run_btn)
    controls
    return controller_window, traveller_window


@app.cell
def _(
    AIFControllerSpec,
    Params,
    SignalType,
    SimParams,
    advisory_smoothing,
    control_interval,
    controller_window,
    days,
    learn_noise,
    noise_regime,
    replace,
    seed,
    sequential_increments,
    sequential_seed,
    signal_mechanism,
    stationary,
    time_step,
    traveller_window,
    warmup,
):
    # Base params: fixed AIF controller with the externality advisory on at full
    # compliance. The advisory-mechanism dropdown picks the raw (single, herd-
    # inducing) externality vs the per-traveller sequential externality (the good
    # default, seeded from belief); traffic demand is scaled per run in the sweep
    # below.
    _base = replace(
        Params(),
        sim=replace(SimParams(), days=int(days.value), seed=int(seed.value),
                    burn_in=int(warmup.value), dt_min=int(time_step.value)),
        controller=AIFControllerSpec(
            control_interval_min=int(control_interval.value),
            horizon_min=int(control_interval.value),
            controller_window_size=int(controller_window.value)),
    )
    if signal_mechanism.value == "Sequential externality":
        _seed = "belief" if sequential_seed.value == "From belief" else "empty"
        _base = (_base.with_comm(SignalType.EXTERNALITY_SEQUENTIAL)
                 .with_sequential_increments(int(sequential_increments.value))
                 .with_sequential_seed(_seed))
    else:
        _base = _base.with_comm(SignalType.EXTERNALITY)
    base_params = (
        _base
        .with_compliance(1.0)
        .with_advisory_smoothing(int(advisory_smoothing.value))
        .with_window_size(int(traveller_window.value))
        .with_learn_obs_noise(bool(learn_noise.value))
        .with_stationary(bool(stationary.value))
        .with_noise_regime(noise_regime.value)
    )
    DEMAND_SCALES = [0.8, 1.0, 1.2, 1.4]
    return DEMAND_SCALES, base_params


@app.cell
def _(DEMAND_SCALES, DemandParams, base_params, replace, run_btn,
      run_experiment, seed, sweep_progress_bar):
    # Demand sweep: re-run the coupled system at each peak-demand scale.
    # Result shape: {scale_label: ExperimentResult}.
    if not run_btn.value:
        results_by_demand = None
    else:
        results_by_demand = {}
        with sweep_progress_bar(len(DEMAND_SCALES), base_params.sim,
                                title="demand robustness") as _bar:
            for _s in DEMAND_SCALES:
                _demand = replace(DemandParams(),
                                  d_AB_max=DemandParams().d_AB_max * _s,
                                  d_CD_max=DemandParams().d_CD_max * _s)
                _p = replace(base_params, demand=_demand)
                results_by_demand[f"{_s:g}x"] = run_experiment(
                    _p, seeds=[int(seed.value)], on_step=_bar.update)
    return (results_by_demand,)


@app.cell
def _(figure_block, figure_placeholder, plot_within_day_by_demand,
      results_by_demand):
    fig_within = (
        figure_placeholder("Within-day adaptation, one line per demand scale")
        if results_by_demand is None
        else plot_within_day_by_demand(results_by_demand)
    )
    figure_block("plot_within_day_by_demand", fig_within)
    return


@app.cell
def _(figure_block, figure_placeholder, plot_across_day_by_demand,
      results_by_demand):
    fig_across = (
        figure_placeholder("Across-day learning, one line per demand scale")
        if results_by_demand is None
        else plot_across_day_by_demand(results_by_demand)
    )
    figure_block("plot_across_day_by_demand", fig_across)
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("robustness"))
    return


if __name__ == "__main__":
    app.run()
