"""Experiment 3 -- System information communication.

Fixes the signal controller and a single traveller population, and varies only
**what information travellers receive about the network** before/after they
choose. The ``communication mechanism`` dropdown selects which channel is swept:

* **Extra observations** (default) -- travellers natively see only the route
  they took; the controller relays the *true realised* route congestion (**CG**,
  queue ``L_r``) and/or signal green split (**SN**, ``phi_r``) of the routes they
  did *not* take, folded into their end-of-day belief update. Settings
  BL/CG/SN/CG+SN. Works with any controller and reaches every traveller.
* **Belief sharing** -- the AIF controller shares its own forward-predicted
  belief: its queue belief (**QB**, ``N(L_hat, var)``) and/or planned green split
  (**SP**). A compliant traveller fuses that Gaussian into its posterior at
  decision time (transient, never entering the smoother). Settings BL/QB/SP/QB+SP,
  run at full compliance.
* **Both** -- each channel alone and combined (BL/CG+SN/QB+SP/CG+SN+QB+SP).
* **Disable** -- the no-information baseline only.

The notebook runs the selected settings and overlays the outcomes.
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

        The controller and the traveller population are fixed; we vary only
        **what information travellers receive about the network**. Pick the
        channel with the **communication mechanism** dropdown:

        * **Extra observations** (default): travellers see only the route they
          took, so the controller relays the *true realised* congestion (**CG**,
          queue $L_r$) and/or signal split (**SN**, $\phi_r$) of the routes they
          did *not* take, folded into their end-of-day belief update. This works
          with any controller and reaches everyone.
        * **Belief sharing**: the AIF controller shares its own forecast belief
          ($\mathcal N(\hat L,\widehat{\mathrm{var}})$, **QB**) and/or planned
          split ($\hat\phi$, **SP**); a compliant traveller *fuses* it into its
          posterior to decide (transient, never entering its smoother).

        Set the parameters, click **Run**, and read the overlay below. The
        question is whether richer information gives more stable route choice and
        lower system cost.
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
        ObservationSignal,
        Params,
        SimParams,
    )
    from aif_traffic.plotting import (
        figure_placeholder,
        plot_day_overview_grid,
        plot_queue_belief_day,
        plot_route_choice_heatmaps,
        plot_sweep_metrics,
        plot_within_day_by_setting,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        BeliefSignal,
        DemandParams,
        ObservationSignal,
        Params,
        SimParams,
        explainer_pointer,
        figure_block,
        figure_placeholder,
        nc,
        notebook_explainer,
        plot_day_overview_grid,
        plot_queue_belief_day,
        plot_route_choice_heatmaps,
        plot_sweep_metrics,
        plot_within_day_by_setting,
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
    # experiments; see CLAUDE.md). The comm_mechanism dropdown selects which
    # information channel is swept. compliance gates the belief-sharing channel
    # only (extra observations reach everyone); theta (the externality channel)
    # and the AIF-tuning knobs are not used here.
    comm_mechanism = nc.comm_mechanism()
    days = nc.days()
    warmup = nc.warmup()
    seed = nc.seed()
    control_interval = nc.control_interval()
    demand_scale = nc.demand_scale()
    learn_noise = nc.learn_noise()
    noise_regime = nc.noise_regime()
    stationary = nc.stationary()
    compliance = nc.compliance()

    run_btn = mo.ui.run_button(label="Run all communication settings")
    return (
        comm_mechanism,
        compliance,
        control_interval,
        days,
        demand_scale,
        learn_noise,
        noise_regime,
        run_btn,
        seed,
        stationary,
        warmup,
    )


@app.cell
def _(comm_mechanism, compliance, control_interval, days, demand_scale,
      learn_noise, nc, noise_regime, run_btn, seed, stationary, warmup):
    # Window sliders under (and disabled by) the stationary toggle -- see the
    # note in Experiment 1's control panel.
    traveller_window = nc.traveller_window(disabled=stationary.value)
    controller_window = nc.controller_window(disabled=stationary.value)

    controls = nc.standard_panel({
        "comm_mechanism": comm_mechanism,
        "days": days, "warmup": warmup, "seed": seed,
        "control_interval": control_interval, "demand_scale": demand_scale,
        "learn_noise": learn_noise, "noise_regime": noise_regime,
        "stationary": stationary, "traveller_window": traveller_window,
        "controller_window": controller_window,
        "compliance": compliance,
    }, run_btn)
    controls
    return controller_window, traveller_window


@app.cell
def _(
    AIFControllerSpec,
    BeliefSignal,
    DemandParams,
    ObservationSignal,
    Params,
    SimParams,
    comm_mechanism,
    compliance,
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
    traveller_window,
    warmup,
):
    if not run_btn.value:
        results_by_setting = None
        sweep_params = None
    else:
        base_d = DemandParams()
        _scale = float(demand_scale.value)
        _demand = replace(
            base_d,
            d_AB_max=base_d.d_AB_max * _scale,
            d_CD_max=base_d.d_CD_max * _scale,
        )
        # theta stays at its default 0 (the externality channel is not used
        # here). compliance gates the belief-sharing channel only; extra
        # observations reach every traveller regardless.
        _base = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value),
                        burn_in=int(warmup.value)),
            controller=AIFControllerSpec(
                control_interval_min=int(control_interval.value),
                horizon_min=int(control_interval.value),
                controller_window_size=int(controller_window.value)),
            demand=_demand,
        ).with_compliance(
            float(compliance.value)
        ).with_window_size(int(traveller_window.value)).with_learn_obs_noise(
            bool(learn_noise.value)
        ).with_stationary(bool(stationary.value)).with_noise_regime(
            noise_regime.value
        )

        _CG = ObservationSignal.ROUTE_CONGESTION
        _SN = ObservationSignal.SIGNAL_CONTROL
        _QB = BeliefSignal.QUEUE_BELIEF
        _SP = BeliefSignal.SPLIT_PLAN
        _mech = comm_mechanism.value
        if _mech == "Disable":
            _settings = {"BL": _base}
        elif _mech == "Belief sharing":
            # Belief sharing is studied at full compliance (the swept content is
            # what the controller shares, not who listens).
            _bb = _base.with_compliance(1.0)
            _settings = {
                "BL": _bb.with_belief_signals(),
                "QB": _bb.with_belief_signals(_QB),
                "SP": _bb.with_belief_signals(_SP),
                "QB+SP": _bb.with_belief_signals(_QB, _SP),
            }
        elif _mech == "Both":
            _bb = _base.with_compliance(1.0)
            _settings = {
                "BL": _bb,
                "CG+SN": _bb.with_extra_observations(_CG, _SN),
                "QB+SP": _bb.with_belief_signals(_QB, _SP),
                "CG+SN+QB+SP": _bb.with_extra_observations(_CG, _SN).with_belief_signals(_QB, _SP),
            }
        else:  # "Extra observations" (default)
            _settings = {
                "BL": _base.with_extra_observations(),
                "CG": _base.with_extra_observations(_CG),
                "SN": _base.with_extra_observations(_SN),
                "CG+SN": _base.with_extra_observations(_CG, _SN),
            }
        results_by_setting = {}
        sweep_params = _base
        with sweep_progress_bar(
            len(_settings), _base.sim, title="communication settings"
        ) as _bar:
            for _name, _p in _settings.items():
                # Snapshot every day so the per-setting belief-vs-realised chart
                # has the travellers' posterior on the days it overlays.
                results_by_setting[_name] = run_experiment(
                    _p, seeds=[int(seed.value)], on_step=_bar.update,
                    snapshot_days=range(int(days.value)))
    return results_by_setting, sweep_params


@app.cell
def _(figure_block, figure_placeholder, plot_sweep_metrics, results_by_setting):
    # 2x2 grid layout (Xue's Experiment-3 Figure 1).
    fig_comm = (
        figure_placeholder("Communication settings overlay")
        if results_by_setting is None
        else plot_sweep_metrics(results_by_setting, layout="grid")
    )
    figure_block("plot_sweep_metrics", fig_comm)
    return


@app.cell
def _(figure_block, figure_placeholder, plot_within_day_by_setting,
      results_by_setting, sweep_params):
    fig_by_setting = (
        figure_placeholder("Within-day belief vs reality by setting")
        if results_by_setting is None
        else plot_within_day_by_setting(results_by_setting, sweep_params)
    )
    figure_block("plot_within_day_by_setting", fig_by_setting)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### What the controller believes vs what actually happened

        The AIF controller maintains a rolling-window smoother posterior over the
        within-day queue. Below, for one chosen setting and day, that belief
        (mean $\pm 1\sigma$) is overlaid on the day's realised queue, so you can
        read the controller's estimate against reality (and, under belief sharing,
        *what gets broadcast*).
        """
    )
    return


@app.cell
def _(figure_block, figure_placeholder, plot_day_overview_grid,
      results_by_setting):
    # At-a-glance comparison of the first / middle / last day (shared Y per row),
    # so the day-to-day change is visible without moving the slider below. Shown
    # for the richest (last) setting of the sweep; the dropdown below drills into
    # any setting/day.
    _ov_key = (
        None if results_by_setting is None
        else list(results_by_setting.keys())[-1]
    )
    fig_overview = (
        figure_placeholder("Multi-day overview")
        if results_by_setting is None
        else plot_day_overview_grid(results_by_setting[_ov_key].step)
    )
    figure_block(
        "plot_day_overview_grid", fig_overview,
        extra=(
            None if _ov_key is None
            else f"_Shown for the **{_ov_key}** setting; use the dropdown below "
            "to drill into any setting and day._"
        ),
    )
    return (fig_overview,)


@app.cell
def _(days, mo, results_by_setting):
    # Defined once; displayed as a synced copy directly above the belief chart.
    # Options follow whichever settings the selected mechanism actually swept.
    _labels = list(results_by_setting.keys()) if results_by_setting else ["BL"]
    setting_sel = mo.ui.dropdown(
        options=_labels, value=_labels[-1], label="setting",
    )
    day_sel = mo.ui.slider(
        0, max(int(days.value) - 1, 0),
        value=max(int(days.value) - 1, 0),
        label="inspect day",
    )
    return day_sel, setting_sel


@app.cell
def _(day_sel, figure_block, figure_placeholder, mo, plot_queue_belief_day,
      results_by_setting, setting_sel):
    fig_belief = (
        figure_placeholder("Controller belief vs realised queue")
        if results_by_setting is None
        else plot_queue_belief_day(
            results_by_setting[setting_sel.value].step, day=int(day_sel.value)
        )
    )
    _out = figure_block("plot_queue_belief_day", fig_belief,
                        extra="Pick the **setting** and **inspect day** above.")
    _controls = mo.hstack([setting_sel, day_sel], justify="start", gap=2)
    mo.vstack([_controls, _out]) if results_by_setting is not None else _out
    return (fig_belief,)


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
def _(figure_block, figure_placeholder, plot_route_choice_heatmaps,
      results_by_setting):
    fig_routes = (
        figure_placeholder("Route-choice heatmaps")
        if results_by_setting is None
        else plot_route_choice_heatmaps(results_by_setting)
    )
    figure_block("plot_route_choice_heatmaps", fig_routes)
    return


@app.cell
def _(figure_block, figure_placeholder, plot_route_choice_heatmaps,
      results_by_setting):
    # Across-day queue patterns (L_2) alongside the route-choice patterns above.
    fig_queue_hm = (
        figure_placeholder("Queue heatmaps by setting")
        if results_by_setting is None
        else plot_route_choice_heatmaps(results_by_setting, value="L2")
    )
    figure_block(
        "plot_route_choice_heatmaps", fig_queue_hm,
        extra="_Showing the A--B queue $L_2$ instead of the route share._")
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("information_communication"))
    return


if __name__ == "__main__":
    app.run()
