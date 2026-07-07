"""Headless smoke runner for the AIF-controller experiments.

Rather than drive the marimo notebooks (00 landing page, 01 AIF controller)
through the runtime, this script exercises the *core coupled pipeline*
directly: it runs a short simulation for every controller type and every
communication signal, and renders the available plots. Any logic error
surfaces here.

Run with::

    uv run python scripts/smoke_notebooks.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from aif_traffic.aggregation import build_daily_and_summary
from aif_traffic.parameters import (
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    BeliefSignal,
    CohortSpec,
    FixedTimeControllerSpec,
    NoiseParams,
    ObservationSignal,
    Params,
    PopulationParams,
    ReactiveControllerSpec,
    SignalType,
    SimParams,
)
from aif_traffic.plotting import (
    plot_belief_reality_queues,
    plot_co_adaptation,
    plot_controller_theta_grid,
    plot_coupled_within_day,
    plot_demand_profile,
    plot_learned_obs_noise,
    plot_learning_uncertainty,
    plot_network_state,
    plot_queue_belief_day,
    plot_route_choice_heatmaps,
    plot_route_flows,
    plot_route_share_over_days,
    plot_signal_day,
    plot_sweep_metrics,
    plot_theta_route_choice,
    plot_theta_summary,
    plot_within_day_by_setting,
    plot_within_day_tt_vs_belief,
    setup_style,
)
from aif_traffic.simulator import run_experiment

OUT_DIR = Path("/tmp/aif_controller_smoke")

CONTROLLERS = {
    "fixed_time": FixedTimeControllerSpec(),
    "reactive": ReactiveControllerSpec(),
    "anticipatory": AnticipatoryControllerSpec(phi_grid_size=5),
    "aif": AIFControllerSpec(),
}


def _save_figure(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches="tight", dpi=80)


def _small_params(controller_spec, signal_type=SignalType.NONE) -> Params:
    return replace(
        Params.default(),
        sim=SimParams(days=6, burn_in=1, h_min=60, dt_min=5, seed=42,
                      selected_days=(0, 2, 5)),
        population=PopulationParams(cohorts=(CohortSpec(n_agents=200, window_size=3),)),
        noise=NoiseParams(obs_noise_sd=1.0),
        controller=controller_spec,
    ).with_comm(signal_type)


def smoke_controllers() -> None:
    """Run the coupled pipeline under each controller; render diagnostics."""
    for name, spec in CONTROLLERS.items():
        params = _small_params(spec)
        res = run_experiment(params, snapshot_days=range(params.sim.days))
        daily, summary = build_daily_and_summary(res.step, params)
        assert not res.step.empty, f"{name}: empty step frame"
        assert (res.step["phi2"] + res.step["phi6"]).sub(
            params.signal.phi_sat).abs().max() < 1e-9, f"{name}: cycle constraint"
        _save_figure(plot_signal_day(res.step), f"signal_{name}.png")
        _save_figure(plot_queue_belief_day(res.step), f"belief_{name}.png")
        _save_figure(plot_route_flows(res.step), f"route_flows_{name}.png")
        _save_figure(plot_network_state(res.step, params.network, color_by="travellers"),
                     f"network_flow_{name}.png")
        _save_figure(plot_network_state(res.step, params.network, color_by="queue"),
                     f"network_queue_{name}.png")
        _save_figure(plot_route_share_over_days(res.step), f"route_share_{name}.png")


def smoke_mechanism() -> None:
    """Experiment-1 mechanism composites: coupled within-day, belief-vs-reality
    queue, co-adaptation, and learning-uncertainty figures, plus the
    within-day travel-time-vs-belief chart (needs per-agent snapshots)."""
    params = _small_params(AIFControllerSpec())
    res = run_experiment(params, snapshot_days=range(params.sim.days))
    _save_figure(plot_coupled_within_day(res.step, params), "coupled_within_day.png")
    _save_figure(plot_within_day_tt_vs_belief(res.step, res.snapshots, params),
                 "within_day_tt_vs_belief.png")
    _save_figure(plot_belief_reality_queues(res.step, res.snapshots, params),
                 "belief_reality_queues.png")
    _save_figure(plot_co_adaptation(res.step), "co_adaptation.png")
    _save_figure(plot_learning_uncertainty(res.cohort, res.controller),
                 "learning_uncertainty.png")


def smoke_stationary() -> None:
    """Stationary (continuous filtering, the default) vs the rolling window:
    the same short experiment should end with a TIGHTER traveller posterior
    under stationary (evidence accumulates instead of being forgotten)."""
    import numpy as np

    base = _small_params(AIFControllerSpec())

    def late_sigma(res):
        s = res.cohort.groupby("day")["sigma_alpha_post"].mean()
        return float(s.iloc[-2:].mean())

    st = run_experiment(base.with_stationary(True))
    wd = run_experiment(base.with_stationary(False))
    st_sig, wd_sig = late_sigma(st), late_sigma(wd)
    assert np.isfinite(st_sig) and np.isfinite(wd_sig)
    assert st_sig <= wd_sig + 1e-6, (
        f"stationary posterior not tighter: stationary={st_sig:.3f} "
        f"windowed={wd_sig:.3f}")


def smoke_tables() -> None:
    """Build every summary table on a small run and check they are non-empty."""
    from aif_traffic.plotting import (
        communication_summary_table,
        controller_summary,
        run_summary_table,
        theta_summary_table,
    )

    base = _small_params(AIFControllerSpec())
    res = run_experiment(base)
    assert not run_summary_table(res, n_last=2).empty
    assert not controller_summary({"aif": res}).empty

    grid = {"aif": {0.0: res},
            "fixed_time": {0.0: run_experiment(_small_params(FixedTimeControllerSpec()))}}
    assert not theta_summary_table(grid, n_last=2).empty

    ct = communication_summary_table({
        "BL": run_experiment(base.with_extra_observations()),
        "SN": run_experiment(base.with_extra_observations(ObservationSignal.SIGNAL_CONTROL)),
    }, n_last=2)
    assert list(ct["setting"]) == ["BL", "SN"]


def smoke_communication() -> None:
    """Run every cost-offset broadcast signal type with a compliant cohort."""
    for st in SignalType:
        params = _small_params(ReactiveControllerSpec(), signal_type=st)
        res = run_experiment(params)
        assert not res.step.empty, f"{st}: empty step frame"


def smoke_extra_observations() -> None:
    """Run the four extra-observation settings (BL/CG/SN/CG+SN, Experiment 3
    default) and render the sweep overlay. Extra observations are ungated by
    compliance, so this also checks the baseline is bit-identical to relaying
    no observations."""
    base = _small_params(AIFControllerSpec())
    settings = {
        "BL": base.with_extra_observations(),
        "CG": base.with_extra_observations(ObservationSignal.ROUTE_CONGESTION),
        "SN": base.with_extra_observations(ObservationSignal.SIGNAL_CONTROL),
        "CG+SN": base.with_extra_observations(
            ObservationSignal.ROUTE_CONGESTION, ObservationSignal.SIGNAL_CONTROL
        ),
    }
    results = {}
    for name, params in settings.items():
        res = run_experiment(params, snapshot_days=range(params.sim.days))
        assert not res.step.empty, f"extra-obs {name}: empty step frame"
        results[name] = res

    import numpy as np

    default = run_experiment(base)
    assert np.allclose(
        results["BL"].step["P_alpha"].values, default.step["P_alpha"].values
    ), "BL extra-obs is not bit-identical to no information"

    _save_figure(plot_sweep_metrics(results, layout="grid"),
                 "extra_observations_sweep.png")
    _save_figure(plot_route_choice_heatmaps(results), "extra_observations_route_choice.png")
    _save_figure(plot_route_choice_heatmaps(results, value="L2"),
                 "extra_observations_queue_heatmap.png")
    _save_figure(plot_within_day_by_setting(results, base),
                 "extra_observations_by_setting.png")


def smoke_belief_communication() -> None:
    """Run the four controller-belief settings (BL/QB/SP/QB+SP, Experiment 3)
    with a fully-compliant cohort and render the sweep overlay. Also checks
    that the baseline is bit-identical to sharing no belief signals."""
    base = _small_params(AIFControllerSpec()).with_compliance(1.0)
    settings = {
        "BL": base.with_belief_signals(),
        "QB": base.with_belief_signals(BeliefSignal.QUEUE_BELIEF),
        "SP": base.with_belief_signals(BeliefSignal.SPLIT_PLAN),
        "QB+SP": base.with_belief_signals(
            BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
        ),
    }
    results = {}
    for name, params in settings.items():
        res = run_experiment(params)
        assert not res.step.empty, f"belief {name}: empty step frame"
        results[name] = res

    # Baseline (empty belief set) must be bit-identical to the default comm.
    import numpy as np

    default = run_experiment(base)
    assert np.allclose(
        results["BL"].step["P_alpha"].values, default.step["P_alpha"].values
    ), "BL belief broadcast is not bit-identical to no information"

    _save_figure(plot_sweep_metrics(results), "belief_communication_sweep.png")
    _save_figure(plot_route_choice_heatmaps(results), "belief_route_choice.png")


def smoke_compliance() -> None:
    """Experiment 4: the controller's shared belief (QB+SP) swept over compliance
    fractions. Renders the sweep overlay and checks that zero compliance is
    bit-identical to the baseline (nobody fuses the broadcast)."""
    import numpy as np

    base = _small_params(AIFControllerSpec()).with_belief_signals(
        BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
    )
    results = {
        f"{int(round(f * 100))}%": run_experiment(base.with_compliance(f))
        for f in (0.0, 0.5, 1.0)
    }
    for name, res in results.items():
        assert not res.step.empty, f"compliance {name}: empty step frame"
    _save_figure(plot_sweep_metrics(results), "compliance_sweep.png")

    # Zero compliance must collapse onto the no-broadcast baseline.
    none = run_experiment(_small_params(AIFControllerSpec()).with_compliance(0.0))
    assert np.array_equal(
        results["0%"].step["P_alpha"].values, none.step["P_alpha"].values
    ), "zero-compliance belief broadcast is not bit-identical to baseline"


def smoke_theta_grid() -> None:
    """Experiment 2 extension: steady-state cost over (theta x controller).
    A small 2x2 grid exercises the heatmap end-to-end."""
    grid = {}
    for cname, spec in (("fixed_time", FixedTimeControllerSpec()),
                        ("aif", AIFControllerSpec())):
        grid[cname] = {}
        for theta in (0.0, 1.0):
            grid[cname][theta] = run_experiment(
                _small_params(spec).with_theta(theta)
            )
    _save_figure(plot_controller_theta_grid(grid), "theta_controller_grid.png")
    _save_figure(plot_theta_summary(grid, n_last=3), "theta_summary.png")
    _save_figure(plot_theta_route_choice(grid, n_last=3), "theta_route_choice.png")


def smoke_learn_obs_noise() -> None:
    """Variational observation-noise learning on both layers (controller per
    movement, travellers per agent). Exercises the VB path end-to-end and checks
    the learned controller sigma_obs is finite and positive."""
    import numpy as np

    params = _small_params(AIFControllerSpec()).with_learn_obs_noise(True)
    res = run_experiment(params, snapshot_days=range(params.sim.days))
    assert not res.step.empty, "learn_obs_noise: empty step frame"
    c = res.controller
    assert {"sigma_obs_l2", "sigma_obs_l6"} <= set(c.columns)
    last = c[c["day"] == c["day"].max()]
    assert np.isfinite(last["sigma_obs_l2"]).all() and (last["sigma_obs_l2"] > 0).all()
    _save_figure(plot_learned_obs_noise(res.controller), "learned_obs_noise.png")


def smoke_controls() -> None:
    """Build each experiment's parameter panel from the shared
    ``notebook_controls`` module (catches drift/breakage without the marimo
    runtime). Mirrors the per-notebook control sets."""
    from aif_traffic import notebook_controls as nc

    panels = {
        "exp1": ["days", "warmup", "seed", "control_interval", "demand_scale",
                 "traveller_window", "controller_window", "learn_noise",
                 "stationary", "noise_regime", "theta", "compliance", "gamma",
                 "omega", "sigma_pref", "phi_grid"],
        "exp2": ["days", "warmup", "seed", "control_interval", "demand_scale",
                 "traveller_window", "controller_window", "learn_noise",
                 "stationary", "noise_regime", "theta", "compliance", "gamma",
                 "omega", "sigma_pref", "phi_grid", "k_L"],
        "exp3": ["days", "warmup", "seed", "control_interval", "demand_scale",
                 "traveller_window", "controller_window", "learn_noise",
                 "stationary", "noise_regime", "compliance"],
        "exp4": ["days", "warmup", "seed", "control_interval", "demand_scale",
                 "traveller_window", "controller_window", "learn_noise",
                 "stationary", "noise_regime"],
    }
    import marimo as mo

    for name, keys in panels.items():
        widgets = {k: getattr(nc, k)() for k in keys}
        panel = nc.standard_panel(widgets, mo.ui.run_button(label="Run"))
        assert panel is not None, f"{name}: standard_panel returned None"


def smoke_demand() -> None:
    params = _small_params(FixedTimeControllerSpec())
    _save_figure(plot_demand_profile(params), "demand.png")


def main() -> int:
    setup_style()
    smokes = {
        "demand": smoke_demand,
        "controllers": smoke_controllers,
        "mechanism": smoke_mechanism,
        "tables": smoke_tables,
        "stationary": smoke_stationary,
        "communication": smoke_communication,
        "extra_observations": smoke_extra_observations,
        "belief_communication": smoke_belief_communication,
        "compliance": smoke_compliance,
        "theta_grid": smoke_theta_grid,
        "learn_obs_noise": smoke_learn_obs_noise,
        "controls": smoke_controls,
    }
    failures: list[str] = []
    for name, fn in smokes.items():
        try:
            fn()
            print(f"[OK] {name}")
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} smokes failed: {failures}")
        return 1
    print(f"\nAll {len(smokes)} smokes passed. Figures in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
