"""Experiment 3 -- System information communication.

Fixes the AIF signal controller and a single traveller population, and varies
only **what the controller shares from its own belief** before travellers
choose.

* **BL** -- baseline: the controller shares nothing.
* **QB** -- the controller shares its forward-predicted queue belief
  ``N(L_hat, var)``.
* **SP** -- the controller shares its planned green split ``phi_hat``.
* **QB+SP** -- both.

A compliant traveller fuses the shared Gaussian into its own posterior at
decision time (a transient fusion that never enters the smoother); richer
information sharpens its anticipation of the day. This notebook runs the four
settings at full compliance and overlays the outcomes.
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

        The AIF controller and the traveller population are fixed; we vary only
        **what the controller shares from its own belief** before travellers
        choose. The controller forward-predicts the day and shares its queue
        belief ($\mathcal N(\hat L,\widehat{\mathrm{var}})$, **QB**) and/or its
        planned green split ($\hat\phi$, **SP**). A compliant traveller *fuses*
        that distribution into its own posterior to decide — a transient fusion
        that never enters its smoother; non-compliant travellers ignore it.

        Set the parameters, click **Run**, and read the overlay below. The
        question is whether richer shared anticipation gives more stable route
        choice and lower system cost.
        """
    )
    return


@app.cell
def _():
    from dataclasses import replace

    from aif_traffic import notebook_controls as nc
    from aif_traffic.explainers import explainer_pointer, notebook_explainer
    from aif_traffic.notebook_io import figure_block
    from aif_traffic.parameters import (
        AIFControllerSpec,
        BeliefSignal,
        DemandParams,
        Params,
        SimParams,
    )
    from aif_traffic.plotting import (
        figure_placeholder,
        plot_day_overview_grid,
        plot_queue_belief_day,
        plot_route_choice_heatmaps,
        plot_sweep_metrics,
        setup_style,
    )
    from aif_traffic.simulator import run_experiment

    setup_style()
    return (
        AIFControllerSpec,
        BeliefSignal,
        DemandParams,
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
        replace,
        run_experiment,
    )


@app.cell
def _(explainer_pointer, mo):
    mo.md(explainer_pointer())
    return


@app.cell
def _(mo, nc):
    # All controls come from aif_traffic.notebook_controls (shared across the
    # experiments; see CLAUDE.md). This experiment is about the belief broadcast,
    # so it exposes compliance (gates the belief fusion) but not theta (the
    # externality channel is not used here) nor the AIF-tuning knobs.
    days = nc.days()
    seed = nc.seed()
    control_interval = nc.control_interval()
    demand_scale = nc.demand_scale()
    traveller_window = nc.traveller_window()
    controller_window = nc.controller_window()
    learn_noise = nc.learn_noise()
    compliance = nc.compliance()

    run_btn = mo.ui.run_button(label="Run all communication settings")

    controls = nc.standard_panel({
        "days": days, "seed": seed, "control_interval": control_interval,
        "demand_scale": demand_scale, "traveller_window": traveller_window,
        "controller_window": controller_window, "learn_noise": learn_noise,
        "compliance": compliance,
    }, run_btn)
    controls
    return (
        compliance,
        control_interval,
        controller_window,
        days,
        demand_scale,
        learn_noise,
        run_btn,
        seed,
        traveller_window,
    )


@app.cell
def _(
    AIFControllerSpec,
    BeliefSignal,
    DemandParams,
    Params,
    SimParams,
    compliance,
    control_interval,
    controller_window,
    days,
    demand_scale,
    learn_noise,
    mo,
    replace,
    run_btn,
    run_experiment,
    seed,
    traveller_window,
):
    if not run_btn.value:
        results_by_setting = None
    else:
        base_d = DemandParams()
        _scale = float(demand_scale.value)
        _demand = replace(
            base_d,
            d_AB_max=base_d.d_AB_max * _scale,
            d_CD_max=base_d.d_CD_max * _scale,
        )
        # This experiment is about the belief broadcast, so theta stays at its
        # default 0 (the externality channel is not used here); compliance gates
        # the belief fusion.
        _base = replace(
            Params(),
            sim=replace(SimParams(), days=int(days.value), seed=int(seed.value)),
            controller=AIFControllerSpec(
                control_interval_min=int(control_interval.value),
                horizon_min=int(control_interval.value),
                controller_window_size=int(controller_window.value)),
            demand=_demand,
        ).with_compliance(
            float(compliance.value)
        ).with_window_size(int(traveller_window.value)).with_learn_obs_noise(
            bool(learn_noise.value)
        )

        _settings = {
            "BL": _base.with_belief_signals(),
            "QB": _base.with_belief_signals(BeliefSignal.QUEUE_BELIEF),
            "SP": _base.with_belief_signals(BeliefSignal.SPLIT_PLAN),
            "QB+SP": _base.with_belief_signals(
                BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
            ),
        }
        results_by_setting = {}
        for _name, _p in mo.status.progress_bar(
            list(_settings.items()), title="communication settings"
        ):
            results_by_setting[_name] = run_experiment(_p, seeds=[int(seed.value)])
    return (results_by_setting,)


@app.cell
def _(figure_block, figure_placeholder, plot_sweep_metrics, results_by_setting):
    fig_comm = (
        figure_placeholder("Communication settings overlay")
        if results_by_setting is None
        else plot_sweep_metrics(results_by_setting)
    )
    figure_block("plot_sweep_metrics", fig_comm)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### What the controller believes vs what actually happened

        The controller shares its rolling-window smoother posterior over the
        within-day queue (that is the QB channel). Below, for one chosen setting
        and day, that belief (mean $\pm 1\sigma$) is overlaid on the day's
        realised queue, so you can read *what gets broadcast* against reality.
        """
    )
    return


@app.cell
def _(figure_block, figure_placeholder, plot_day_overview_grid,
      results_by_setting):
    # At-a-glance comparison of the first / middle / last day (shared Y per row),
    # so the day-to-day change is visible without moving the slider below. Shown
    # for the full-information QB+SP setting; the dropdown below drills into any
    # setting/day.
    fig_overview = (
        figure_placeholder("Multi-day overview")
        if results_by_setting is None
        else plot_day_overview_grid(results_by_setting["QB+SP"].step)
    )
    figure_block(
        "plot_day_overview_grid", fig_overview,
        extra="_Shown for the **QB+SP** setting; use the dropdown below to drill "
        "into any setting and day._",
    )
    return (fig_overview,)


@app.cell
def _(days, mo):
    # Defined once; displayed as a synced copy directly above the belief chart.
    setting_sel = mo.ui.dropdown(
        options=["BL", "QB", "SP", "QB+SP"], value="QB", label="setting",
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
def _(mo, notebook_explainer):
    mo.md(notebook_explainer("information_communication"))
    return


if __name__ == "__main__":
    app.run()
