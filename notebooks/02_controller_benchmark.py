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

    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.notebook_io import is_deployed, outputs_dir
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
        plot_controller_theta_grid,
        plot_green_split_heatmaps_by_controller,
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
        figure_placeholder,
        is_deployed,
        notebook_explainer,
        outputs_dir,
        plot_controller_metrics,
        plot_controller_theta_grid,
        plot_green_split_heatmaps_by_controller,
        replace,
        run_experiment,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo):
    days = mo.ui.slider(20, 180, step=10, value=90, label="days")
    seed = mo.ui.slider(0, 100, value=42, label="seed")
    control_interval = mo.ui.slider(1, 30, value=10, label="control interval [min]")
    demand_scale = mo.ui.slider(0.5, 2.5, step=0.1, value=1.0, label="demand scale")
    traveller_window = mo.ui.slider(0, 60, value=30, label="traveller window [days]")

    gamma = mo.ui.slider(0.5, 20.0, step=0.5, value=4.0, label="AIF gamma")
    omega = mo.ui.slider(0.0, 0.2, step=0.005, value=0.02, label="AIF omega")
    sigma_pref = mo.ui.slider(5.0, 60.0, step=1.0, value=20.0, label="AIF sigma_pref [veh]")
    k_L = mo.ui.slider(1e-4, 5e-3, step=1e-4, value=1e-3, label="reactive k_L")
    controller_window = mo.ui.slider(0, 60, value=30, label="AIF controller window [days]")

    run_btn = mo.ui.run_button(label="Run experiment")

    def _row(widget, desc):
        return mo.hstack([widget, mo.md(desc)], widths=[2, 3], align="center", gap=1)

    controls = mo.vstack([
        mo.md("### Shared parameters"),
        _row(days, "Total days simulated (first warm-up days are discarded)."),
        _row(seed, "Master seed; redraws all stochastic elements."),
        _row(control_interval,
             "Minutes between green-split decisions (applied to every controller)."),
        _row(demand_scale,
             r"Scales peak A--B and C--D demand. $>1$ loads the junction; the "
             r"controllers differ most under load."),
        _row(traveller_window,
             "Days each traveller's rolling-window smoother remembers when "
             r"forming route beliefs (applied to every controller)."),
        mo.md("---"),
        mo.md("### Controller knobs (baselines otherwise use their defaults)"),
        _row(gamma, r"AIF action precision $\gamma^c$."),
        _row(omega, r"AIF balance weight in the preference $\Sigma^c_{\mathrm{pref}}$."),
        _row(sigma_pref, r"AIF preferred-queue tolerance (veh)."),
        _row(k_L, r"Reactive feedback gain on the queue imbalance $L_2-L_6$."),
        _row(controller_window,
             "Days of past queue observations the AIF controller smooths over "
             "before acting (AIF controller only)."),
        run_btn,
    ], gap=0.5)
    controls
    return (
        control_interval,
        controller_window,
        days,
        demand_scale,
        gamma,
        k_L,
        omega,
        run_btn,
        seed,
        sigma_pref,
        traveller_window,
    )


@app.cell
def _(
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    FixedTimeControllerSpec,
    Params,
    ReactiveControllerSpec,
    SimParams,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    k_L,
    mo,
    omega,
    replace,
    run_btn,
    run_experiment,
    seed,
    sigma_pref,
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
        for _name, _spec in specs.items():
            _p = replace(
                base,
                sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
                controller=_spec,
                demand=demand,
            ).with_window_size(int(traveller_window.value))
            results_by_ctrl[_name] = run_experiment(
                _p, seeds=[int(seed.value)], progress=mo.status.progress_bar,
            )
    return (results_by_ctrl,)


@app.cell
def _(figure_placeholder, plot_controller_metrics, results_by_ctrl):
    fig_metrics = (
        figure_placeholder("System cost, peak queue, signal variation")
        if results_by_ctrl is None
        else plot_controller_metrics(results_by_ctrl)
    )
    fig_metrics
    return


@app.cell
def _(figure_placeholder, plot_green_split_heatmaps_by_controller, results_by_ctrl):
    fig_heatmaps = (
        figure_placeholder("Green split by controller")
        if results_by_ctrl is None
        else plot_green_split_heatmaps_by_controller(results_by_ctrl)
    )
    fig_heatmaps
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
        gif_view = mo.image(str(gif_path))
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
    SimParams,
    control_interval,
    controller_window,
    days,
    demand_scale,
    gamma,
    grid_btn,
    k_L,
    mo,
    omega,
    replace,
    run_experiment,
    seed,
    sigma_pref,
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
        for _name, _spec, _theta in mo.status.progress_bar(
            _cells, title="theta x controller",
        ):
            _p = replace(
                _base,
                sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
                controller=_spec,
                demand=_demand,
            ).with_theta(_theta).with_window_size(int(traveller_window.value))
            results_by_ctrl_theta[_name][_theta] = run_experiment(
                _p, seeds=[int(seed.value)],
            )
    return (results_by_ctrl_theta,)


@app.cell
def _(figure_placeholder, plot_controller_theta_grid, results_by_ctrl_theta):
    fig_theta_grid = (
        figure_placeholder("System cost over theta x controller")
        if results_by_ctrl_theta is None
        else plot_controller_theta_grid(results_by_ctrl_theta)
    )
    fig_theta_grid
    return


@app.cell
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("controller_benchmark"))
    return


if __name__ == "__main__":
    app.run()
