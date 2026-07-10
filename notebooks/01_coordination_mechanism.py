"""Experiment 1: Understanding the coordination mechanism.

Fixes the **AIF signal controller** and a single traveller population, and shows
how the two layers **co-adapt** through the shared network with no external
coordination signal. The single run shows the coupled traveller/controller
adaptation within a day (route flows, believed vs realised travel time and queue,
green split) and across days (route shares, green-split policy, system cost and
belief uncertainty converging together). This establishes the behavioural
baseline for Experiments 2 and 3.
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
        # Experiment 1: Understanding the coordination mechanism

        The AIF signal controller and a single traveller population are fixed;
        this experiment shows how the two layers **co-adapt** through the shared
        network, with no external coordination signal.

        The single run below shows the coupled adaptation within a day (route
        flows, believed vs realised travel time and queue, the green split) and
        across days (route shares and the controller's green-split policy
        stabilising as system cost and belief uncertainty fall together).

        Set the parameters, click **Run**, and read the charts below.
        """
    )
    return


@app.cell
def _():
    from dataclasses import replace
    from pathlib import Path

    from aif_traffic import notebook_controls as nc
    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.notebook_io import (
        figure_block,
        is_deployed,
        outputs_dir,
        table_block,
    )
    from aif_traffic.parameters import (
        AIFControllerSpec,
        DemandParams,
        Params,
        SimParams,
    )
    from aif_traffic.plotting import (
        animate_days,
        animate_network_state,
        animate_route_flows,
        figure_placeholder,
        plot_belief_reality_queues,
        plot_co_adaptation,
        plot_coupled_within_day,
        plot_daily_system_cost,
        plot_day_overview_grid,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_learned_obs_noise,
        plot_learning_uncertainty,
        plot_network_state,
        plot_queue_belief_day,
        plot_route_flows,
        plot_route_share_over_days,
        plot_signal_day,
        plot_within_day_tt_vs_belief,
        run_summary_table,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        DemandParams,
        Params,
        Path,
        SimParams,
        animate_days,
        animate_network_state,
        animate_route_flows,
        explainer_pointer,
        figure_block,
        figure_placeholder,
        is_deployed,
        nc,
        notebook_explainer,
        outputs_dir,
        plot_belief_reality_queues,
        plot_co_adaptation,
        plot_coupled_within_day,
        plot_daily_system_cost,
        plot_day_overview_grid,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_learned_obs_noise,
        plot_learning_uncertainty,
        plot_network_state,
        plot_queue_belief_day,
        plot_route_flows,
        plot_route_share_over_days,
        plot_signal_day,
        plot_within_day_tt_vs_belief,
        replace,
        run_experiment,
        run_summary_table,
        table_block,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo, nc):
    # All controls come from aif_traffic.notebook_controls (shared across the
    # experiments; see CLAUDE.md). This coordination-mechanism notebook exposes
    # the simulation and AIF-controller knobs (no controller->traveller signal is
    # used here, so no communication controls).
    days = nc.days()
    warmup = nc.warmup()
    seed = nc.seed()
    control_interval = nc.control_interval()
    demand_scale = nc.demand_scale()
    learn_noise = nc.learn_noise()
    noise_regime = nc.noise_regime()
    stationary = nc.stationary()
    gamma = nc.gamma()
    omega = nc.omega()
    sigma_pref = nc.sigma_pref()
    phi_grid = nc.phi_grid()
    time_step = nc.time_step()
    bypass_capacity_scale = nc.bypass_capacity_scale()

    run_btn = mo.ui.run_button(label="Run experiment")
    return (
        bypass_capacity_scale,
        control_interval,
        days,
        demand_scale,
        gamma,
        learn_noise,
        noise_regime,
        omega,
        phi_grid,
        run_btn,
        seed,
        sigma_pref,
        stationary,
        time_step,
        warmup,
    )


@app.cell
def _(bypass_capacity_scale, control_interval, days, demand_scale,
      gamma, learn_noise, nc, noise_regime, omega, phi_grid, run_btn, seed,
      sigma_pref, stationary, time_step, warmup):
    # The rolling-window sliders live under (and are disabled by) the stationary
    # toggle: they only bite in the non-stationary mode. Defined here, downstream
    # of `stationary`, so toggling it re-renders them (updating `disabled`)
    # without resetting the other controls.
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
        "gamma": gamma, "omega": omega, "sigma_pref": sigma_pref,
        "phi_grid": phi_grid,
    }, run_btn)
    controls
    return controller_window, traveller_window


@app.cell
def _(
    AIFControllerSpec,
    DemandParams,
    Params,
    SimParams,
    bypass_capacity_scale,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    learn_noise,
    mo,
    noise_regime,
    omega,
    phi_grid,
    replace,
    run_btn,
    run_experiment,
    seed,
    sigma_pref,
    stationary,
    time_step,
    traveller_window,
    warmup,
):
    if not run_btn.value:
        params = None
        results = None
    else:
        spec = AIFControllerSpec(
            control_interval_min=int(control_interval.value),
            horizon_min=int(control_interval.value),
            gamma=float(gamma.value),
            omega=float(omega.value),
            sigma_pref=float(sigma_pref.value),
            phi_grid_size=int(phi_grid.value),
            controller_window_size=int(controller_window.value),
        )
        base_d = DemandParams()
        scale = float(demand_scale.value)
        demand = replace(
            base_d,
            d_AB_max=base_d.d_AB_max * scale,
            d_CD_max=base_d.d_CD_max * scale,
        )
        # The controller and travellers co-adapt through the shared network with
        # no controller->traveller signal here (that is Experiment 3).
        params = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value),
                        burn_in=int(warmup.value), dt_min=int(time_step.value)),
            controller=spec,
            demand=demand,
        ).with_window_size(int(traveller_window.value)).with_learn_obs_noise(
            bool(learn_noise.value)
        ).with_stationary(bool(stationary.value)).with_noise_regime(
            noise_regime.value
        ).with_bypass_capacity_scale(float(bypass_capacity_scale.value))
        # Snapshot every day so the belief-vs-realised charts have the per-agent
        # posterior on the days they overlay.
        results = run_experiment(
            params, seeds=[int(seed.value)], progress=mo.status.progress_bar,
            snapshot_days=range(int(days.value)),
        )
    return params, results


@app.cell
def _(figure_block, figure_placeholder, params, plot_demand_profile, results):
    fig_demand = (
        figure_placeholder("Demand profile")
        if results is None
        else plot_demand_profile(params)
    )
    figure_block("plot_demand_profile", fig_demand)
    return (fig_demand,)


@app.cell
def _(figure_block, figure_placeholder, plot_day_overview_grid, results):
    # At-a-glance comparison of the first / middle / last day, on a shared Y per
    # row, so the day-to-day change is visible without moving the slider below.
    fig_overview = (
        figure_placeholder("Multi-day overview")
        if results is None
        else plot_day_overview_grid(results.step)
    )
    figure_block("plot_day_overview_grid", fig_overview)
    return (fig_overview,)


@app.cell
def _(figure_block, figure_placeholder, params, plot_within_day_tt_vs_belief,
      results):
    fig_tt_belief = (
        figure_placeholder("Within-day travel time: realised vs belief")
        if results is None
        else plot_within_day_tt_vs_belief(results.step, results.snapshots, params)
    )
    figure_block("plot_within_day_tt_vs_belief", fig_tt_belief)
    return (fig_tt_belief,)


@app.cell
def _(figure_block, figure_placeholder, params, plot_coupled_within_day,
      results):
    fig_coupled = (
        figure_placeholder("Coupled within-day: flow & green split")
        if results is None
        else plot_coupled_within_day(results.step, params)
    )
    figure_block("plot_coupled_within_day", fig_coupled)
    return (fig_coupled,)


@app.cell
def _(days, mo):
    # Defined once; displayed as a synced copy above each day-dependent chart
    # below (marimo keeps every instance of the same element in sync), so the
    # inspect-day control is always next to the chart you are looking at.
    day_sel = mo.ui.slider(
        0, max(int(days.value) - 1, 0),
        value=max(int(days.value) - 1, 0),
        label="inspect day",
    )
    return (day_sel,)


@app.cell
def _(day_sel, figure_block, figure_placeholder, mo, plot_signal_day, results):
    fig_signal = (
        figure_placeholder("Within-day queues and green split")
        if results is None
        else plot_signal_day(results.step, day=int(day_sel.value))
    )
    _out = figure_block("plot_signal_day", fig_signal)
    mo.vstack([day_sel, _out]) if results is not None else _out
    return (fig_signal,)


@app.cell
def _(day_sel, figure_block, figure_placeholder, mo, plot_queue_belief_day,
      results):
    fig_belief = (
        figure_placeholder("Controller belief vs realised queue")
        if results is None
        else plot_queue_belief_day(results.step, day=int(day_sel.value))
    )
    _out = figure_block("plot_queue_belief_day", fig_belief)
    mo.vstack([day_sel, _out]) if results is not None else _out
    return (fig_belief,)


@app.cell
def _(day_sel, figure_block, figure_placeholder, mo, params,
      plot_belief_reality_queues, results):
    fig_bel_real = (
        figure_placeholder("Belief vs realised queue")
        if results is None
        else plot_belief_reality_queues(
            results.step, results.snapshots, params, day=int(day_sel.value))
    )
    _out = figure_block("plot_belief_reality_queues", fig_bel_real)
    mo.vstack([day_sel, _out]) if results is not None else _out
    return (fig_bel_real,)


@app.cell
def _(figure_block, figure_placeholder, learn_noise, plot_learned_obs_noise,
      results):
    fig_obs_noise = (
        plot_learned_obs_noise(results.controller)
        if (results is not None and bool(learn_noise.value))
        else figure_placeholder("Learned observation noise (enable the checkbox)")
    )
    figure_block("plot_learned_obs_noise", fig_obs_noise)
    return (fig_obs_noise,)


@app.cell
def _(figure_block, figure_placeholder, plot_learning_uncertainty, results):
    fig_uncert = (
        figure_placeholder("Learning uncertainty over days")
        if results is None
        else plot_learning_uncertainty(results.cohort, results.controller)
    )
    figure_block("plot_learning_uncertainty", fig_uncert)
    return (fig_uncert,)


@app.cell
def _(day_sel, figure_block, figure_placeholder, mo, plot_route_flows, results):
    fig_routes = (
        figure_placeholder("Per-route traveller flow")
        if results is None
        else plot_route_flows(results.step, day=int(day_sel.value))
    )
    _out = figure_block("plot_route_flows", fig_routes)
    mo.vstack([day_sel, _out]) if results is not None else _out
    return (fig_routes,)


@app.cell
def _(SimParams, mo):
    # Defined once; displayed as a synced copy above the network-state chart.
    tod_sel = mo.ui.slider(
        0, SimParams().h_min, value=150, step=5, label="time of day [min]",
    )
    color_metric = mo.ui.radio(
        options=["travellers", "queue"], value="travellers",
        label="colour links by", inline=True,
    )
    return color_metric, tod_sel


@app.cell
def _(color_metric, day_sel, figure_block, figure_placeholder, mo, params,
      plot_network_state, results, tod_sel):
    fig_network = (
        figure_placeholder("Network state at a time of day")
        if results is None
        else plot_network_state(
            results.step, params.network,
            day=int(day_sel.value), tau=int(tod_sel.value),
            color_by=color_metric.value,
        )
    )
    _out = figure_block("plot_network_state", fig_network)
    _controls = mo.hstack([day_sel, tod_sel, color_metric],
                          justify="start", gap=2)
    mo.vstack([_controls, _out]) if results is not None else _out
    return (fig_network,)


@app.cell
def _(is_deployed, mo):
    make_net_gif = mo.ui.checkbox(value=False, label="Render network-flow gif")
    make_net_gif if not is_deployed() else mo.md("")
    return (make_net_gif,)


@app.cell
def _(animate_network_state, color_metric, day_sel, figure_block, is_deployed,
      make_net_gif, mo, outputs_dir, params, results):
    if results is None or is_deployed() or not make_net_gif.value:
        net_gif_view = mo.md("")
    else:
        net_gif_path = animate_network_state(
            results.step, params.network,
            outputs_dir() / "aif_network_flow.gif",
            day=int(day_sel.value), color_by=color_metric.value,
        )
        net_gif_view = figure_block("animate_network_state",
                                    mo.image(str(net_gif_path)))
    net_gif_view
    return


@app.cell
def _(figure_block, figure_placeholder, plot_green_split_heatmap, results):
    fig_phi_hm = (
        figure_placeholder("Green split phi2 over (day x time)")
        if results is None
        else plot_green_split_heatmap(results.step, value="phi2")
    )
    figure_block("plot_green_split_heatmap", fig_phi_hm,
                 extra="_Showing the green split $\\phi_2$._")
    return (fig_phi_hm,)


@app.cell
def _(figure_block, figure_placeholder, mo, plot_green_split_heatmap, results):
    if results is None:
        fig_queues = figure_placeholder("Queue heatmaps (L2, L6)")
    else:
        fig_queues = mo.hstack([
            plot_green_split_heatmap(results.step, value="L2"),
            plot_green_split_heatmap(results.step, value="L6"),
        ])
    figure_block("plot_green_split_heatmap", fig_queues,
                 extra="_Showing the queues $L_2$ and $L_6$ instead of the split._")
    return (fig_queues,)


@app.cell
def _(figure_block, figure_placeholder, plot_daily_system_cost, results):
    fig_cost = (
        figure_placeholder("Day-to-day system cost")
        if results is None
        else plot_daily_system_cost(results.step)
    )
    figure_block("plot_daily_system_cost", fig_cost)
    return (fig_cost,)


@app.cell
def _(figure_block, figure_placeholder, plot_route_share_over_days, results):
    fig_share = (
        figure_placeholder("Day-to-day route share")
        if results is None
        else plot_route_share_over_days(results.step)
    )
    figure_block("plot_route_share_over_days", fig_share)
    return (fig_share,)


@app.cell
def _(figure_block, figure_placeholder, plot_co_adaptation, results):
    fig_coadapt = (
        figure_placeholder("Day-to-day co-adaptation")
        if results is None
        else plot_co_adaptation(results.step, results.controller)
    )
    figure_block("plot_co_adaptation", fig_coadapt)
    return (fig_coadapt,)


@app.cell
def _(results, run_summary_table, table_block):
    # The steady-state numbers behind the day-series charts for this single run.
    run_df = None if results is None else run_summary_table(results)
    table_block("run_summary_table", run_df)
    return


@app.cell
def _(is_deployed, mo):
    make_gif = mo.ui.checkbox(value=False, label="Render per-day gif")
    make_gif if not is_deployed() else mo.md("")
    return (make_gif,)


@app.cell
def _(animate_days, figure_block, is_deployed, make_gif, mo, outputs_dir,
      results):
    if results is None or is_deployed() or not make_gif.value:
        gif_view = mo.md("")
    else:
        gif_path = animate_days(results.step, outputs_dir() / "aif_controller_days.gif")
        gif_view = figure_block("animate_days", mo.image(str(gif_path)))
    gif_view
    return


@app.cell
def _(is_deployed, mo):
    make_flow_gif = mo.ui.checkbox(
        value=False, label="Render per-day traveller-flow gif")
    make_flow_gif if not is_deployed() else mo.md("")
    return (make_flow_gif,)


@app.cell
def _(animate_route_flows, figure_block, is_deployed, make_flow_gif, mo,
      outputs_dir, results):
    if results is None or is_deployed() or not make_flow_gif.value:
        flow_gif_view = mo.md("")
    else:
        flow_gif_path = animate_route_flows(
            results.step, outputs_dir() / "aif_traveller_days.gif")
        flow_gif_view = figure_block(
            "animate_route_flows", mo.image(str(flow_gif_path)))
    flow_gif_view
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("coordination_mechanism"))
    return


if __name__ == "__main__":
    app.run()
