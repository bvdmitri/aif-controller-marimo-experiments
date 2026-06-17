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
    CohortSpec,
    FixedTimeControllerSpec,
    NoiseParams,
    Params,
    PopulationParams,
    ReactiveControllerSpec,
    SignalType,
    SimParams,
)
from aif_traffic.plotting import (
    plot_demand_profile,
    plot_route_share_over_days,
    plot_signal_day,
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
        sim=SimParams(days=6, burn_in=1, h_min=60, dt_min=1, seed=42,
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
        _save_figure(plot_route_share_over_days(res.step), f"route_share_{name}.png")


def smoke_communication() -> None:
    """Run every broadcast signal type with a compliant cohort."""
    for st in SignalType:
        params = _small_params(ReactiveControllerSpec(), signal_type=st)
        res = run_experiment(params)
        assert not res.step.empty, f"{st}: empty step frame"


def smoke_demand() -> None:
    params = _small_params(FixedTimeControllerSpec())
    _save_figure(plot_demand_profile(params), "demand.png")


def main() -> int:
    setup_style()
    smokes = {
        "demand": smoke_demand,
        "controllers": smoke_controllers,
        "communication": smoke_communication,
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
