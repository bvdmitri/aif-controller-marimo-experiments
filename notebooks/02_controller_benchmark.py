"""Experiment 2 -- Controller benchmark comparison.

Runs the same network and demand under all four signal controllers
(fixed-time, reactive, anticipatory, and the Active Inference controller) and
compares the most relevant outcomes: total system cost, peak queues, and how
steadily each controller drives the green split. Scalar day-series are overlaid
on one chart; the green-split policy is shown one column per controller. This is
the core contribution: does the AIF controller outperform established
non-adaptive, reactive, and predictive strategies?
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
        # Experiment 2 — Controller benchmark comparison

        Four controllers, same network and demand, swapping only the control
        strategy: **fixed-time**, **reactive** (SCOOT-like), **anticipatory**
        (predictive), and the **AIF** controller. We keep only the metrics the
        paper's controller benchmark cares about: total system cost,
        critical-link queues, and the stability of the signal policy.

        Set the parameters, click **Run**, and read the comparison below. Push
        *demand scale* above 1 to load the junction and make the controllers
        differ more clearly.
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
        AnticipatoryControllerSpec,
        FixedTimeControllerSpec,
        Params,
        ReactiveControllerSpec,
        SignalType,
        SimParams,
    )
    from aif_traffic.plotting import (
        animate_controller_comparison,
        controller_summary,
        figure_placeholder,
        plot_controller_metrics,
        plot_controller_theta_grid,
        plot_green_split_heatmaps_by_controller,
        plot_learned_obs_noise,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        AnticipatoryControllerSpec,
        FixedTimeControllerSpec,
        Params,
        Path,
        ReactiveControllerSpec,
        SignalType,
        SimParams,
        animate_controller_comparison,
        controller_summary,
        explainer_pointer,
        figure_block,
        figure_placeholder,
        is_deployed,
        nc,
        notebook_explainer,
        outputs_dir,
        plot_controller_metrics,
        plot_controller_theta_grid,
        plot_green_split_heatmaps_by_controller,
        plot_learned_obs_noise,
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
    # experiments; see CLAUDE.md). theta + compliance are live here because the
    # benchmark broadcasts the externality advisory (so the controllers are
    # compared at a chosen social-internalisation level, matching the theta-grid).
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
    k_L = nc.k_L()

    run_btn = mo.ui.run_button(label="Run experiment")

    controls = nc.standard_panel({
        "days": days, "seed": seed, "control_interval": control_interval,
        "demand_scale": demand_scale, "traveller_window": traveller_window,
        "controller_window": controller_window, "learn_noise": learn_noise,
        "theta": theta, "compliance": compliance,
        "gamma": gamma, "omega": omega, "sigma_pref": sigma_pref,
        "phi_grid": phi_grid, "k_L": k_L,
    }, run_btn)
    controls
    return (
        compliance,
        control_interval,
        controller_window,
        days,
        demand_scale,
        gamma,
        k_L,
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
    AnticipatoryControllerSpec,
    FixedTimeControllerSpec,
    Params,
    ReactiveControllerSpec,
    SignalType,
    SimParams,
    compliance,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    k_L,
    learn_noise,
    omega,
    phi_grid,
    replace,
    run_btn,
    run_experiment,
    seed,
    sigma_pref,
    sweep_progress_bar,
    theta,
    traveller_window,
):
    if not run_btn.value:
        results_by_ctrl = None
    else:
        ci = int(control_interval.value)
        specs = {
            "fixed_time": FixedTimeControllerSpec(control_interval_min=ci),
            "reactive": ReactiveControllerSpec(
                control_interval_min=ci, k_L=float(k_L.value)),
            "anticipatory": AnticipatoryControllerSpec(control_interval_min=ci),
            "aif": AIFControllerSpec(
                control_interval_min=ci, horizon_min=ci,
                gamma=float(gamma.value), omega=float(omega.value),
                sigma_pref=float(sigma_pref.value),
                phi_grid_size=int(phi_grid.value),
                controller_window_size=int(controller_window.value)),
        }
        base = Params()
        scale = float(demand_scale.value)
        demand = replace(
            base.demand,
            d_AB_max=base.demand.d_AB_max * scale,
            d_CD_max=base.demand.d_CD_max * scale,
        )
        results_by_ctrl = {}
        _sim = replace(SimParams(), days=int(days.value), seed=int(seed.value))
        with sweep_progress_bar(len(specs), _sim, title="controllers") as _bar:
            for _name, _spec in specs.items():
                # Broadcast the externality advisory at the chosen theta/compliance
                # so the controllers are compared at a chosen social-internalisation
                # level (theta is inert without it); learn-noise per the checkbox.
                _p = replace(
                    base,
                    sim=_sim,
                    controller=_spec,
                    demand=demand,
                ).with_comm(SignalType.EXTERNALITY).with_compliance(
                    float(compliance.value)
                ).with_theta(float(theta.value)).with_window_size(
                    int(traveller_window.value)
                ).with_learn_obs_noise(bool(learn_noise.value))
                results_by_ctrl[_name] = run_experiment(
                    _p, seeds=[int(seed.value)], on_step=_bar.update,
                )
    return (results_by_ctrl,)


@app.cell
def _(figure_block, figure_placeholder, plot_controller_metrics, results_by_ctrl):
    fig_metrics = (
        figure_placeholder("System cost, peak queue, signal variation")
        if results_by_ctrl is None
        else plot_controller_metrics(results_by_ctrl)
    )
    figure_block("plot_controller_metrics", fig_metrics)
    return


@app.cell
def _(figure_block, figure_placeholder, learn_noise, plot_learned_obs_noise,
      results_by_ctrl):
    # The AIF controller's learned observation noise (only when the VB checkbox
    # is on; the baselines have no queue belief to learn noise for).
    fig_obs_noise = (
        plot_learned_obs_noise(results_by_ctrl["aif"].controller)
        if (results_by_ctrl is not None and bool(learn_noise.value))
        else figure_placeholder("Learned observation noise (enable the checkbox)")
    )
    figure_block("plot_learned_obs_noise", fig_obs_noise)
    return


@app.cell
def _(figure_block, figure_placeholder, plot_green_split_heatmaps_by_controller,
      results_by_ctrl):
    fig_heatmaps = (
        figure_placeholder("Green split by controller")
        if results_by_ctrl is None
        else plot_green_split_heatmaps_by_controller(results_by_ctrl)
    )
    figure_block("plot_green_split_heatmaps_by_controller", fig_heatmaps)
    return


@app.cell
def _(controller_summary, mo, results_by_ctrl):
    if results_by_ctrl is None:
        summary_view = mo.md("*Run to see the summary table.*")
    else:
        summary_view = mo.ui.table(
            controller_summary(results_by_ctrl).round(1), selection=None,
        )
    summary_view
    return


@app.cell
def _(is_deployed, mo):
    make_gif = mo.ui.checkbox(value=False, label="Render per-day comparison gif")
    make_gif if not is_deployed() else mo.md("")
    return (make_gif,)


@app.cell
def _(
    animate_controller_comparison,
    figure_block,
    is_deployed,
    make_gif,
    mo,
    outputs_dir,
    results_by_ctrl,
):
    if results_by_ctrl is None or is_deployed() or not make_gif.value:
        gif_view = mo.md("")
    else:
        gif_path = animate_controller_comparison(
            results_by_ctrl, outputs_dir() / "controller_comparison_days.gif",
        )
        gif_view = figure_block("animate_controller_comparison",
                                mo.image(str(gif_path)))
    gif_view
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Sensitivity to social internalisation $\theta$

        The comparison above fixes the travellers' behaviour. This grid asks a
        sharper question: does the AIF controller's advantage survive across the
        whole user-equilibrium-to-system-optimum spectrum? It re-runs **every
        controller at every** $\theta\in\{0,0.25,0.5,0.75,1\}$ and reports the
        steady-state system cost as a heatmap (lower = better).

        This is **20 full runs**, so it has its own button.
        """
    )
    return


@app.cell
def _(mo):
    grid_btn = mo.ui.run_button(label="Run theta x controller grid (20 runs)")
    grid_btn
    return (grid_btn,)


@app.cell
def _(
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    FixedTimeControllerSpec,
    Params,
    ReactiveControllerSpec,
    SignalType,
    SimParams,
    compliance,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    grid_btn,
    k_L,
    learn_noise,
    omega,
    replace,
    run_experiment,
    seed,
    sigma_pref,
    sweep_progress_bar,
    traveller_window,
):
    if not grid_btn.value:
        results_by_ctrl_theta = None
    else:
        _ci = int(control_interval.value)
        _specs = {
            "fixed_time": FixedTimeControllerSpec(control_interval_min=_ci),
            "reactive": ReactiveControllerSpec(
                control_interval_min=_ci, k_L=float(k_L.value)),
            "anticipatory": AnticipatoryControllerSpec(control_interval_min=_ci),
            "aif": AIFControllerSpec(
                control_interval_min=_ci, horizon_min=_ci,
                gamma=float(gamma.value), omega=float(omega.value),
                sigma_pref=float(sigma_pref.value),
                controller_window_size=int(controller_window.value)),
        }
        _base = Params()
        _scale = float(demand_scale.value)
        _demand = replace(
            _base.demand,
            d_AB_max=_base.demand.d_AB_max * _scale,
            d_CD_max=_base.demand.d_CD_max * _scale,
        )
        _thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
        _cells = [(n, s, t) for n, s in _specs.items() for t in _thetas]
        results_by_ctrl_theta = {n: {} for n in _specs}
        _sim = replace(SimParams(), days=int(days.value), seed=int(seed.value))
        with sweep_progress_bar(
            len(_cells), _sim, title="theta x controller",
        ) as _bar:
            for _name, _spec, _theta in _cells:
                _p = replace(
                    _base,
                    sim=_sim,
                    controller=_spec,
                    demand=_demand,
                ).with_comm(SignalType.EXTERNALITY).with_compliance(
                    float(compliance.value)
                ).with_theta(_theta).with_window_size(
                    int(traveller_window.value)
                ).with_learn_obs_noise(bool(learn_noise.value))
                results_by_ctrl_theta[_name][_theta] = run_experiment(
                    _p, seeds=[int(seed.value)], on_step=_bar.update,
                )
    return (results_by_ctrl_theta,)


@app.cell
def _(figure_block, figure_placeholder, plot_controller_theta_grid,
      results_by_ctrl_theta):
    fig_theta_grid = (
        figure_placeholder("System cost over theta x controller")
        if results_by_ctrl_theta is None
        else plot_controller_theta_grid(results_by_ctrl_theta)
    )
    figure_block("plot_controller_theta_grid", fig_theta_grid)
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("controller_benchmark"))
    return


if __name__ == "__main__":
    app.run()
