"""Experiment 2: Controller benchmark comparison.

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
        # Experiment 2: Controller benchmark comparison

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
        table_block,
    )
    from aif_traffic.parameters import (
        AIFControllerSpec,
        AnticipatoryControllerSpec,
        FixedTimeControllerSpec,
        Params,
        ReactiveControllerSpec,
        SimParams,
    )
    from aif_traffic.plotting import (
        animate_controller_comparison,
        controller_summary,
        figure_placeholder,
        plot_controller_metrics,
        plot_controller_queue_comparison,
        plot_green_split_heatmaps_by_controller,
        plot_learned_obs_noise,
        plot_within_day_queue_by_controller,
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
        plot_controller_queue_comparison,
        plot_green_split_heatmaps_by_controller,
        plot_learned_obs_noise,
        plot_within_day_queue_by_controller,
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
    # experiments; see CLAUDE.md). No controller->traveller signal is used in the
    # benchmark, so there are no communication controls.
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
    k_L = nc.k_L()
    time_step = nc.time_step()
    bypass_capacity_scale = nc.bypass_capacity_scale()

    run_btn = mo.ui.run_button(label="Run experiment")
    return (
        bypass_capacity_scale,
        control_interval,
        days,
        demand_scale,
        gamma,
        k_L,
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
      gamma, k_L, learn_noise, nc, noise_regime, omega, phi_grid, run_btn, seed,
      sigma_pref, stationary, time_step, warmup):
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
        "gamma": gamma, "omega": omega, "sigma_pref": sigma_pref,
        "phi_grid": phi_grid, "k_L": k_L,
    }, run_btn)
    controls
    return controller_window, traveller_window


@app.cell
def _(
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    FixedTimeControllerSpec,
    Params,
    ReactiveControllerSpec,
    SimParams,
    bypass_capacity_scale,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    k_L,
    learn_noise,
    noise_regime,
    stationary,
    omega,
    warmup,
    phi_grid,
    replace,
    run_btn,
    run_experiment,
    seed,
    sigma_pref,
    sweep_progress_bar,
    time_step,
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
        _sim = replace(SimParams(), days=int(days.value), seed=int(seed.value),
                       burn_in=int(warmup.value), dt_min=int(time_step.value))
        with sweep_progress_bar(len(specs), _sim, title="controllers") as _bar:
            for _name, _spec in specs.items():
                # Each controller is compared on the same coupled network; no
                # controller->traveller signal is used. Learn-noise per checkbox.
                _p = replace(
                    base,
                    sim=_sim,
                    controller=_spec,
                    demand=demand,
                ).with_window_size(
                    int(traveller_window.value)
                ).with_learn_obs_noise(bool(learn_noise.value)).with_stationary(
                    bool(stationary.value)).with_noise_regime(
                    noise_regime.value
                ).with_bypass_capacity_scale(float(bypass_capacity_scale.value))
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
def _(figure_block, figure_placeholder, plot_controller_queue_comparison,
      results_by_ctrl):
    # The paper's controller-comparison row: system cost plus the daily total
    # network queue L2+L5+L6, mean + within-day range.
    fig_queue_cmp = (
        figure_placeholder("System cost & total queue by controller")
        if results_by_ctrl is None
        else plot_controller_queue_comparison(results_by_ctrl)
    )
    figure_block("plot_controller_queue_comparison", fig_queue_cmp)
    return


@app.cell
def _(figure_block, figure_placeholder, plot_within_day_queue_by_controller,
      results_by_ctrl):
    # Within-day realised queue on L2 / L5 / L6 for each controller, one square
    # panel per strategy at the representative day.
    fig_within_q = (
        figure_placeholder("Within-day queue on L2/L5/L6 by controller")
        if results_by_ctrl is None
        else plot_within_day_queue_by_controller(results_by_ctrl)
    )
    figure_block("plot_within_day_queue_by_controller", fig_within_q)
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
def _(controller_summary, results_by_ctrl, table_block):
    summary_df = (
        None if results_by_ctrl is None
        else controller_summary(results_by_ctrl)
    )
    table_block("controller_summary", summary_df)
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
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("controller_benchmark"))
    return


if __name__ == "__main__":
    app.run()
