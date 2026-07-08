"""Capacity sensitivity of social internalisation, and the advisory cobweb.

Fixes the AIF signal controller (externality advisory on, full compliance) and
sweeps social internalisation ``theta`` across several **bypass-capacity scales**
(link 5 saturation flow x1.0/0.5/0.25). The default network's bypass is
uncongestable spare capacity, so ``theta`` barely moves system cost; throttling
the bypass turns it into a real bottleneck and gives internalisation something to
redistribute.

The catch is a route-choice **cobweb**. The cost-offset advisory is built from a
day's realised state and acted on the next day; once the bypass is congestible
this one-day-stale signal drives a day-to-day oscillation that sends system cost
far above the ``theta=0`` baseline (``theta`` appears to backfire). Two levers
address it: the **advisory smoothing window ``W``** averages the advisory over the
last ``W`` days (past ~25 days ``theta`` helps again), and the **advisory
mechanism** dropdown swaps the raw (single, herd-inducing) externality for a
per-traveller **sequential** externality that assigns demand incrementally so the
population splits toward the system optimum instead of herding, collapsing the
cobweb even at ``W=1``.
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
        # Experiment 4 — Capacity sensitivity of social internalisation

        Sweeping social internalisation $\theta$ barely moves system cost on the
        default network: the bypass route $\beta$ is high-capacity spare, so
        diverting off the intersection is nearly free and the congestion
        externality $\theta$ acts on is tiny. Here the AIF controller is fixed
        (externality advisory on, full compliance) and we sweep $\theta$ across
        **bypass-capacity scales** (link 5 $\times\,1.0, 0.5, 0.25$), throttling
        the bypass into a genuine bottleneck.

        Two levers address the resulting **cobweb**. The cost-offset advisory is
        one day stale (built today, acted on tomorrow); with the bypass throttled,
        the **raw** advisory broadcasts one identical per-route value to everyone,
        so all compliant travellers herd onto the cheaper route and the route
        share oscillates day to day, making $\theta$ backfire.

        - **Advisory smoothing $W$** (slider): travellers act on the mean advisory
          over the last $W$ days. Larger $W$ damps the cobweb; past $W\approx25$
          days $\theta$ helps again. $W=1$ is the raw act-on-yesterday advisory.
        - **Advisory mechanism** (dropdown): switch from *Raw externality* to
          *Sequential externality*. The controller incrementally assigns demand to
          the cheaper route and re-computes the marginal cost as each route fills,
          handing different travellers different advisories (indexed by rank). The
          population then splits toward the system optimum instead of herding, so
          the cobweb collapses even at $W=1$. The **sequential increments $M$**
          slider sets how finely the schedule is built.

        Move $W$ and the mechanism dropdown and watch the cobweb appear and vanish
        in the day-series chart.
        """
    )
    return


@app.cell
def _():
    from dataclasses import replace

    from aif_traffic import notebook_controls as nc
    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.notebook_io import (
        figure_block,
        sweep_progress_bar,
        table_block,
    )
    from aif_traffic.parameters import (
        AIFControllerSpec,
        DemandParams,
        Params,
        SignalType,
        SimParams,
    )
    from aif_traffic.plotting import (
        capacity_theta_summary,
        figure_placeholder,
        plot_cost_vs_theta_by_capacity,
        plot_sweep_metrics,
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
        capacity_theta_summary,
        explainer_pointer,
        figure_block,
        figure_placeholder,
        nc,
        notebook_explainer,
        plot_cost_vs_theta_by_capacity,
        plot_sweep_metrics,
        replace,
        run_experiment,
        sweep_progress_bar,
        table_block,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo, nc):
    # All controls come from aif_traffic.notebook_controls (shared across the
    # experiments; see CLAUDE.md). theta and the bypass scale are swept
    # (hard-coded lists below), so they are not panel sliders; compliance is
    # fixed at 1.0 so the externality advisory acts. The advisory-smoothing W is
    # a slider (default 25). The bypass_capacity_scale slider picks the single
    # scale used for the raw-vs-smoothed cobweb comparison.
    days = nc.days()
    warmup = nc.warmup()
    seed = nc.seed()
    control_interval = nc.control_interval()
    demand_scale = nc.demand_scale()
    bypass_capacity_scale = nc.bypass_capacity_scale()
    signal_mechanism = nc.signal_mechanism()
    sequential_increments = nc.sequential_increments()
    advisory_smoothing = nc.advisory_smoothing()
    learn_noise = nc.learn_noise()
    noise_regime = nc.noise_regime()
    stationary = nc.stationary()
    time_step = nc.time_step()

    run_btn = mo.ui.run_button(label="Run the capacity x theta sweep")
    return (
        advisory_smoothing,
        bypass_capacity_scale,
        control_interval,
        days,
        demand_scale,
        learn_noise,
        noise_regime,
        run_btn,
        seed,
        sequential_increments,
        signal_mechanism,
        stationary,
        time_step,
        warmup,
    )


@app.cell
def _(advisory_smoothing, bypass_capacity_scale, control_interval, days,
      demand_scale, learn_noise, nc, noise_regime, run_btn, seed,
      sequential_increments, signal_mechanism, stationary,
      time_step, warmup):
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
        "signal_mechanism": signal_mechanism,
        "sequential_increments": sequential_increments,
        "advisory_smoothing": advisory_smoothing,
    }, run_btn)
    controls
    return controller_window, traveller_window


@app.cell
def _(
    AIFControllerSpec,
    DemandParams,
    Params,
    SignalType,
    SimParams,
    control_interval,
    controller_window,
    days,
    demand_scale,
    learn_noise,
    noise_regime,
    replace,
    seed,
    sequential_increments,
    signal_mechanism,
    stationary,
    time_step,
    traveller_window,
    warmup,
):
    # Base params: fixed AIF controller with the externality advisory on at full
    # compliance (so theta acts). theta / bypass scale / advisory-smoothing W are
    # applied per run in the sweeps below. The advisory-mechanism dropdown picks
    # the raw (single, herd-inducing) externality vs the per-traveller sequential
    # externality (M increment bins); the latter is the cobweb fix.
    _scale = float(demand_scale.value)
    _demand = replace(DemandParams(),
                      d_AB_max=DemandParams().d_AB_max * _scale,
                      d_CD_max=DemandParams().d_CD_max * _scale)
    _base = replace(
        Params(),
        sim=replace(SimParams(), days=int(days.value), seed=int(seed.value),
                    burn_in=int(warmup.value), dt_min=int(time_step.value)),
        controller=AIFControllerSpec(
            control_interval_min=int(control_interval.value),
            horizon_min=int(control_interval.value),
            controller_window_size=int(controller_window.value)),
        demand=_demand,
    )
    if signal_mechanism.value == "Sequential externality":
        _base = (_base.with_comm(SignalType.EXTERNALITY_SEQUENTIAL)
                 .with_sequential_increments(int(sequential_increments.value)))
    else:
        _base = _base.with_comm(SignalType.EXTERNALITY)
    base_params = (
        _base
        .with_compliance(1.0)
        .with_window_size(int(traveller_window.value))
        .with_learn_obs_noise(bool(learn_noise.value))
        .with_stationary(bool(stationary.value))
        .with_noise_regime(noise_regime.value)
    )
    THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
    SCALES = [1.0, 0.5, 0.25]
    return SCALES, THETAS, base_params


@app.cell
def _(SCALES, THETAS, advisory_smoothing, base_params, run_btn, run_experiment,
      seed, sweep_progress_bar):
    # Main sweep: theta x bypass-capacity scale, at the chosen advisory-smoothing
    # window W. Result shape: {scale_label: {theta: ExperimentResult}}.
    if not run_btn.value:
        results_by_scale_theta = None
    else:
        _W = int(advisory_smoothing.value)
        results_by_scale_theta = {}
        with sweep_progress_bar(len(SCALES) * len(THETAS), base_params.sim,
                                title="capacity x theta") as _bar:
            for _s in SCALES:
                _row = {}
                for _t in THETAS:
                    _p = (base_params.with_bypass_capacity_scale(_s)
                          .with_theta(_t).with_advisory_smoothing(_W))
                    _row[_t] = run_experiment(_p, seeds=[int(seed.value)],
                                              on_step=_bar.update)
                results_by_scale_theta[f"bypass x{_s:g}"] = _row
    return (results_by_scale_theta,)


@app.cell
def _(advisory_smoothing, figure_block, figure_placeholder,
      plot_cost_vs_theta_by_capacity, results_by_scale_theta):
    fig_cost = (
        figure_placeholder("Cost vs theta, one line per bypass capacity scale")
        if results_by_scale_theta is None
        else plot_cost_vs_theta_by_capacity(results_by_scale_theta)
    )
    figure_block(
        "plot_cost_vs_theta_by_capacity", fig_cost,
        extra=(f"Advisory smoothing W = {int(advisory_smoothing.value)} days. "
               "Raise W toward ~25 to damp the cobweb and see the throttled "
               "curves bend back down."),
    )
    return


@app.cell
def _(capacity_theta_summary, results_by_scale_theta, table_block):
    cap_df = (None if results_by_scale_theta is None
              else capacity_theta_summary(results_by_scale_theta))
    table_block("capacity_theta_summary", cap_df)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The cobweb and its two fixes

        At the bypass scale set by the **bypass capacity scale** slider, this
        overlays $\theta=1$ under three advisories: the **raw stale** advisory
        ($W=1$), the **raw smoothed** advisory ($W$ from the slider), and the
        **sequential** advisory ($W=1$). Watch the intersection-share panel: the
        raw stale advisory oscillates day to day (the cobweb); smoothing it settles
        it after a lag; the sequential advisory settles it immediately, with no
        smoothing, by handing each traveller a different rank-indexed nudge.
        """
    )
    return


@app.cell
def _(SignalType, advisory_smoothing, base_params, bypass_capacity_scale,
      run_btn, run_experiment, seed, sequential_increments, sweep_progress_bar):
    # The two cobweb fixes side by side at theta=1, for the single bypass scale on
    # the slider: the raw stale advisory (W=1, the cobweb), the raw advisory
    # smoothed over W days, and the per-traveller sequential advisory (W=1). This
    # overlay always builds all three regardless of the dropdown, so both fixes
    # are visible together.
    if not run_btn.value:
        results_by_advisory = None
    else:
        _s = float(bypass_capacity_scale.value)
        _W = int(advisory_smoothing.value)
        _M = int(sequential_increments.value)
        _base = base_params.with_bypass_capacity_scale(_s).with_theta(1.0)
        _variants = {
            "raw stale (W=1)": _base.with_comm(SignalType.EXTERNALITY)
                                    .with_advisory_smoothing(1),
            f"raw smoothed (W={_W})": _base.with_comm(SignalType.EXTERNALITY)
                                          .with_advisory_smoothing(_W),
            "sequential (W=1)": _base.with_comm(SignalType.EXTERNALITY_SEQUENTIAL)
                                     .with_sequential_increments(_M)
                                     .with_advisory_smoothing(1),
        }
        results_by_advisory = {}
        with sweep_progress_bar(len(_variants), base_params.sim,
                                title="advisory cobweb fixes") as _bar:
            for _name, _p in _variants.items():
                results_by_advisory[_name] = run_experiment(
                    _p, seeds=[int(seed.value)], on_step=_bar.update)
    return (results_by_advisory,)


@app.cell
def _(bypass_capacity_scale, figure_block, figure_placeholder,
      plot_sweep_metrics, results_by_advisory):
    fig_cobweb = (
        figure_placeholder("Raw vs smoothed advisory (day series)")
        if results_by_advisory is None
        else plot_sweep_metrics(results_by_advisory)
    )
    figure_block(
        "plot_sweep_metrics", fig_cobweb,
        extra=(f"At bypass scale x{float(bypass_capacity_scale.value):g}, "
               "theta=1. The intersection-share panel shows the raw advisory's "
               "day-to-day cobweb, and how both smoothing and the sequential "
               "advisory collapse it."),
    )
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("capacity_sensitivity"))
    return


if __name__ == "__main__":
    app.run()
