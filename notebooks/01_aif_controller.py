"""Active Inference signal controller: a first look.

Runs the coupled traveller/controller simulation with the **AIF signal
controller** and visualises how it sets the green split and how the junction
queues evolve, both within a day and across days. Comparison against the other
controllers is out of scope here; this notebook is about seeing the AIF
controller work and reading its behaviour.
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
        # Active Inference signal controller

        The controller keeps a Gaussian belief over the two signalised queues
        $(L_2, L_6)$, predicts them one control interval ahead under each
        candidate green split, and picks the split by minimising the **fixed**
        Expected Free Energy. Its only designed object is a *preferred
        observation* $\tilde p(o^c)=\mathcal N(0,\Sigma^c_{\mathrm{pref}})$
        ("prefer empty queues"); the low-and-balanced goal lives inside
        $\Sigma^c_{\mathrm{pref}}$, not in a hand-built cost.

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
        SimParams,
    )
    from aif_traffic.plotting import (
        animate_days,
        figure_placeholder,
        plot_daily_system_cost,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_route_share_over_days,
        plot_signal_day,
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
        explainer_pointer,
        figure_placeholder,
        is_deployed,
        notebook_explainer,
        outputs_dir,
        plot_daily_system_cost,
        plot_demand_profile,
        plot_green_split_heatmap,
        plot_route_share_over_days,
        plot_signal_day,
        replace,
        run_experiment,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo):
    days = mo.ui.slider(10, 120, value=30, label="days")
    seed = mo.ui.slider(0, 100, value=42, label="seed")
    control_interval = mo.ui.slider(1, 30, value=10, label="control interval [min]")
    demand_scale = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="demand scale")

    gamma = mo.ui.slider(0.5, 20.0, step=0.5, value=4.0, label="gamma")
    omega = mo.ui.slider(0.0, 0.2, step=0.005, value=0.02, label="omega")
    sigma_pref = mo.ui.slider(5.0, 60.0, step=1.0, value=20.0, label="sigma_pref [veh]")
    phi_grid = mo.ui.slider(3, 21, value=9, label="candidate splits K")

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
        run_btn,
    ], gap=0.5)
    controls
    return (
        control_interval,
        days,
        demand_scale,
        gamma,
        omega,
        phi_grid,
        run_btn,
        seed,
        sigma_pref,
    )


@app.cell
def _(
    AIFControllerSpec,
    DemandParams,
    Params,
    SimParams,
    control_interval,
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
        )
        base_d = DemandParams()
        scale = float(demand_scale.value)
        demand = replace(
            base_d,
            d_AB_max=base_d.d_AB_max * scale,
            d_CD_max=base_d.d_CD_max * scale,
        )
        params = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
            controller=spec,
            demand=demand,
        )
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
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("aif_controller"))
    return


if __name__ == "__main__":
    app.run()
