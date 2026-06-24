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

    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.notebook_io import is_deployed, outputs_dir
    from aif_traffic.parameters import (
        AIFControllerSpec,
        DemandParams,
        Params,
        SignalType,
        SimParams,
    )
    from aif_traffic.plotting import (
        animate_days,
        figure_placeholder,
        plot_daily_system_cost,
        plot_demand_profile,
        plot_green_split_heatmap,
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
        explainer_pointer,
        figure_placeholder,
        is_deployed,
        notebook_explainer,
        outputs_dir,
        plot_daily_system_cost,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_network_state,
        plot_queue_belief_day,
        plot_route_flows,
        plot_route_share_over_days,
        plot_signal_day,
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
    control_interval = mo.ui.slider(1, 30, value=10, label="control interval [min]")
    demand_scale = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="demand scale")
    theta = mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="theta")
    traveller_window = mo.ui.slider(0, 60, value=30, label="traveller window [days]")

    gamma = mo.ui.slider(0.5, 20.0, step=0.5, value=4.0, label="gamma")
    omega = mo.ui.slider(0.0, 0.2, step=0.005, value=0.02, label="omega")
    sigma_pref = mo.ui.slider(5.0, 60.0, step=1.0, value=20.0, label="sigma_pref [veh]")
    phi_grid = mo.ui.slider(3, 21, value=9, label="candidate splits K")
    controller_window = mo.ui.slider(0, 60, value=30, label="controller window [days]")

    run_btn = mo.ui.run_button(label="Run experiment")

    def _row(widget, desc):
        return mo.hstack([widget, mo.md(desc)], widths=[2, 3], align="center", gap=1)

    controls = mo.vstack([
        mo.md("### Parameters you can play with"),
        _row(days, "Total days to simulate (the first warm-up days are discarded)."),
        _row(seed, "Master seed; redraws all stochastic elements."),
        _row(control_interval,
             "Minutes between green-split decisions, and the controller's "
             "prediction horizon for scoring each split."),
        _row(demand_scale,
             r"Scales peak A--B and C--D demand. $>1$ pushes the junction "
             r"toward saturation and makes the control problem harder."),
        _row(theta,
             r"Social internalisation $\theta$ for the single run below: how "
             r"much travellers add the congestion externality $E_r$ to their "
             r"perceived cost $\zeta_r = TT_r + \theta E_r$. $0$ = selfish, "
             r"$1$ = fully cooperative. (The externality is broadcast at full "
             r"compliance so $\theta$ has an $E_r$ to act on; this re-rolls the "
             r"day each step, so runs take longer.)"),
        _row(traveller_window,
             "Days each traveller's rolling-window smoother remembers when "
             r"forming route beliefs. Shorter $\to$ more reactive day-to-day "
             "choices; longer → steadier but slower to adapt."),
        mo.md("---"),
        mo.md("### Controller (Active Inference)"),
        _row(gamma,
             r"Action precision $\gamma^c$. Higher $\to$ a sharper preference "
             r"for the lowest-EFE split (more decisive control)."),
        _row(omega,
             r"Balance weight in the preference $\Sigma^c_{\mathrm{pref}}$: "
             r"penalises capacity-normalised queue imbalance between the two "
             r"movements. $0$ = only total queue matters."),
        _row(sigma_pref,
             r"Preferred-queue tolerance (veh): the SD of the *empty queues* "
             r"preference. Smaller $\to$ less tolerant of any residual queue."),
        _row(phi_grid, "Number of candidate green splits scored each decision."),
        _row(controller_window,
             "Days of past queue observations the macro AIF controller smooths "
             r"over before acting. Shorter $\to$ reacts faster but noisier; "
             "longer → steadier."),
        run_btn,
    ], gap=0.5)
    controls
    return (
        control_interval,
        controller_window,
        days,
        demand_scale,
        gamma,
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
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
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
        ).with_comm(SignalType.EXTERNALITY).with_compliance(1.0).with_theta(
            float(theta.value)
        ).with_window_size(int(traveller_window.value))
        results = run_experiment(
            params, seeds=[int(seed.value)], progress=mo.status.progress_bar,
        )
    return params, results


@app.cell
def _(figure_placeholder, params, plot_demand_profile, results):
    fig_demand = (
        figure_placeholder("Demand profile")
        if results is None
        else plot_demand_profile(params)
    )
    fig_demand
    return (fig_demand,)


@app.cell
def _(days, mo, results):
    day_sel = mo.ui.slider(
        0, max(int(days.value) - 1, 0),
        value=max(int(days.value) - 1, 0),
        label="inspect day",
    )
    day_sel if results is not None else mo.md("")
    return (day_sel,)


@app.cell
def _(day_sel, figure_placeholder, plot_signal_day, results):
    fig_signal = (
        figure_placeholder("Within-day queues and green split")
        if results is None
        else plot_signal_day(results.step, day=int(day_sel.value))
    )
    fig_signal
    return (fig_signal,)


@app.cell
def _(day_sel, figure_placeholder, plot_queue_belief_day, results):
    # The controller's learned belief (smoother posterior, mean +/- 1 sigma)
    # against the day's realised queue: the band narrows on later days as the
    # rolling window fills.
    fig_belief = (
        figure_placeholder("Controller belief vs realised queue")
        if results is None
        else plot_queue_belief_day(results.step, day=int(day_sel.value))
    )
    fig_belief
    return (fig_belief,)


@app.cell
def _(day_sel, figure_placeholder, plot_route_flows, results):
    fig_routes = (
        figure_placeholder("Per-route traveller flow")
        if results is None
        else plot_route_flows(results.step, day=int(day_sel.value))
    )
    fig_routes
    return (fig_routes,)


@app.cell
def _(SimParams, mo, results):
    tod_sel = mo.ui.slider(
        0, SimParams().h_min, value=150, step=5, label="time of day [min]",
    )
    color_metric = mo.ui.radio(
        options=["travellers", "queue"], value="travellers",
        label="colour links by", inline=True,
    )
    (mo.hstack([tod_sel, color_metric], justify="start", gap=2)
     if results is not None else mo.md(""))
    return color_metric, tod_sel


@app.cell
def _(color_metric, day_sel, figure_placeholder, params, plot_network_state,
      results, tod_sel):
    fig_network = (
        figure_placeholder("Network state at a time of day")
        if results is None
        else plot_network_state(
            results.step, params.network,
            day=int(day_sel.value), tau=int(tod_sel.value),
            color_by=color_metric.value,
        )
    )
    fig_network
    return (fig_network,)


@app.cell
def _(figure_placeholder, plot_green_split_heatmap, results):
    fig_phi_hm = (
        figure_placeholder("Green split phi2 over (day x time)")
        if results is None
        else plot_green_split_heatmap(results.step, value="phi2")
    )
    fig_phi_hm
    return (fig_phi_hm,)


@app.cell
def _(mo, plot_green_split_heatmap, results):
    if results is None:
        fig_queues = mo.md("")
    else:
        fig_queues = mo.hstack([
            plot_green_split_heatmap(results.step, value="L2"),
            plot_green_split_heatmap(results.step, value="L6"),
        ])
    fig_queues
    return (fig_queues,)


@app.cell
def _(figure_placeholder, plot_daily_system_cost, results):
    fig_cost = (
        figure_placeholder("Day-to-day system cost")
        if results is None
        else plot_daily_system_cost(results.step)
    )
    fig_cost
    return (fig_cost,)


@app.cell
def _(figure_placeholder, plot_route_share_over_days, results):
    fig_share = (
        figure_placeholder("Day-to-day route share")
        if results is None
        else plot_route_share_over_days(results.step)
    )
    fig_share
    return (fig_share,)


@app.cell
def _(is_deployed, mo):
    make_gif = mo.ui.checkbox(value=False, label="Render per-day gif")
    make_gif if not is_deployed() else mo.md("")
    return (make_gif,)


@app.cell
def _(animate_days, is_deployed, make_gif, mo, outputs_dir, results):
    if results is None or is_deployed() or not make_gif.value:
        gif_view = mo.md("")
    else:
        gif_path = animate_days(results.step, outputs_dir() / "aif_controller_days.gif")
        gif_view = mo.image(str(gif_path))
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
def _(mo, params, plot_sweep_metrics, run_experiment, sweep_btn):
    if not sweep_btn.value or params is None:
        fig_sweep = mo.md(
            "_Run the single experiment above, then click **Run theta sweep**._"
        )
    else:
        _thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
        _results_by_theta = {}
        for _th in mo.status.progress_bar(_thetas, title="theta sweep"):
            _results_by_theta[f"theta={_th:g}"] = run_experiment(
                params.with_theta(_th), seeds=[params.sim.seed],
            )
        fig_sweep = plot_sweep_metrics(_results_by_theta)
    fig_sweep
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("social_internalisation"))
    return


if __name__ == "__main__":
    app.run()
