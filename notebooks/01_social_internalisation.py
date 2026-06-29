"""Experiment 1 -- Traveller social internalisation.

Fixes the **AIF signal controller** and varies how cooperative the travellers
are. The single run (default ``theta = 0``, the user equilibrium) shows the coupled
traveller/controller adaptation within a day and across days; the sweep section
then runs ``theta in {0, 0.25, 0.5, 0.75, 1}`` -- the spectrum from the user
equilibrium to the system optimum -- and overlays the resulting route shares,
system cost, queues, and belief uncertainty. The congestion externality ``E_r``
is communicated so ``theta`` can act on it (perceived cost
``zeta_r = TT_r + theta * E_r``); the belief-informing CG/SN broadcasts of
Experiment 3 are not used here. This establishes the behavioural baseline for
Experiments 2 and 3.
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
        # Experiment 1 — Traveller social internalisation

        The AIF signal controller is fixed; we vary how cooperative the
        travellers are through their **social internalisation** $\theta$. A
        traveller perceives a route as $\zeta_r = TT_r + \theta\,E_r$, where
        $E_r$ is the congestion externality it imposes: $\theta=0$ is the user
        equilibrium (purely selfish), $\theta=1$ the system optimum (fully
        cooperative).

        The single run below shows the coupled adaptation at a chosen $\theta$
        (default $0.5$). The **sweep** section then runs
        $\theta\in\{0,0.25,0.5,0.75,1\}$ and overlays the outcomes. The
        externality $E_r$ is communicated so $\theta$ can act on it; the
        belief-informing broadcasts (CG/SN) of Experiment 3 are *not* used here.

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
        sweep_progress_bar,
    )
    from aif_traffic.parameters import (
        AIFControllerSpec,
        DemandParams,
        Params,
        SignalType,
        SimParams,
    )
    from aif_traffic.plotting import (
        animate_days,
        animate_network_state,
        figure_placeholder,
        plot_daily_system_cost,
        plot_day_overview_grid,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_learned_obs_noise,
        plot_network_state,
        plot_queue_belief_day,
        plot_route_flows,
        plot_route_share_over_days,
        plot_signal_day,
        plot_sweep_metrics,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        DemandParams,
        Params,
        Path,
        SignalType,
        SimParams,
        animate_days,
        animate_network_state,
        explainer_pointer,
        figure_block,
        figure_placeholder,
        is_deployed,
        nc,
        notebook_explainer,
        outputs_dir,
        plot_daily_system_cost,
        plot_day_overview_grid,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_learned_obs_noise,
        plot_network_state,
        plot_queue_belief_day,
        plot_route_flows,
        plot_route_share_over_days,
        plot_signal_day,
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
    # experiments; see CLAUDE.md). Experiment 1 exposes the full set: the social
    # knobs theta + compliance (the externality is broadcast, so they bite) and
    # the AIF-controller knobs.
    days = nc.days()
    seed = nc.seed()
    control_interval = nc.control_interval()
    demand_scale = nc.demand_scale()
    traveller_window = nc.traveller_window()
    controller_window = nc.controller_window()
    learn_noise = nc.learn_noise()
    theta = nc.theta()
    compliance = nc.compliance()
    gamma = nc.gamma()
    omega = nc.omega()
    sigma_pref = nc.sigma_pref()
    phi_grid = nc.phi_grid()

    run_btn = mo.ui.run_button(label="Run experiment")

    controls = nc.standard_panel({
        "days": days, "seed": seed, "control_interval": control_interval,
        "demand_scale": demand_scale, "traveller_window": traveller_window,
        "controller_window": controller_window, "learn_noise": learn_noise,
        "theta": theta, "compliance": compliance,
        "gamma": gamma, "omega": omega, "sigma_pref": sigma_pref,
        "phi_grid": phi_grid,
    }, run_btn)
    controls
    return (
        compliance,
        control_interval,
        controller_window,
        days,
        demand_scale,
        gamma,
        learn_noise,
        omega,
        phi_grid,
        run_btn,
        seed,
        sigma_pref,
        theta,
        traveller_window,
    )


@app.cell
def _(
    AIFControllerSpec,
    DemandParams,
    Params,
    SimParams,
    compliance,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    learn_noise,
    mo,
    omega,
    phi_grid,
    replace,
    run_btn,
    run_experiment,
    seed,
    sigma_pref,
    SignalType,
    theta,
    traveller_window,
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
        # theta enters the perceived cost as zeta_r = TT_r + theta * E_r, so it
        # only changes behaviour when the externality E_r is communicated. We
        # broadcast EXTERNALITY at full compliance; with no such signal theta
        # multiplies a zero offset and every theta gives an identical result.
        params = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
            controller=spec,
            demand=demand,
        ).with_comm(SignalType.EXTERNALITY).with_compliance(
            float(compliance.value)
        ).with_theta(
            float(theta.value)
        ).with_window_size(int(traveller_window.value)).with_learn_obs_noise(
            bool(learn_noise.value)
        )
        results = run_experiment(
            params, seeds=[int(seed.value)], progress=mo.status.progress_bar,
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
def _(mo):
    mo.md(
        r"""
        ## Sweep social internalisation $\theta$

        Re-runs the same network and AIF controller for
        $\theta\in\{0,0.25,0.5,0.75,1\}$ and overlays the day-to-day outcomes.
        Higher $\theta$ is expected to spread demand more evenly and lower the
        system cost. Each run broadcasts the externality at full compliance (so
        $\theta$ actually bites — see the note on the $\theta$ slider); with no
        externality channel every curve would coincide. This is heavy (five
        full experiments, each re-rolling the day per minute), so it is gated
        behind its own button.
        """
    )
    return


@app.cell
def _(mo):
    sweep_btn = mo.ui.run_button(label="Run theta sweep")
    sweep_btn
    return (sweep_btn,)


@app.cell
def _(figure_block, mo, params, plot_sweep_metrics, run_experiment,
      sweep_btn, sweep_progress_bar):
    if not sweep_btn.value or params is None:
        fig_sweep = mo.md(
            "_Run the single experiment above, then click **Run theta sweep**._"
        )
    else:
        _thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
        _results_by_theta = {}
        with sweep_progress_bar(
            len(_thetas), params.sim, title="theta sweep"
        ) as _bar:
            for _th in _thetas:
                _results_by_theta[f"theta={_th:g}"] = run_experiment(
                    params.with_theta(_th), seeds=[params.sim.seed],
                    on_step=_bar.update,
                )
        fig_sweep = figure_block(
            "plot_sweep_metrics", plot_sweep_metrics(_results_by_theta)
        )
    fig_sweep
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("social_internalisation"))
    return


if __name__ == "__main__":
    app.run()
