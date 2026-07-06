"""Render every paper figure in the publication (``"paper"``) style and write
them as vector PDFs to ``paper_figures/``.

This is the CI-exportable companion to the marimo notebooks: it applies the
Elsevier-`elsarticle`-3p paper style (IWAI-figure Arial/sans-serif with STIX
math, ~6.72 in text width, vector
PDF, colourblind/greyscale-safe palette) and re-renders the figures the paper
uses, deterministically (noise off, stationary on, fixed seed), so the author
can drop the PDFs straight into the LaTeX. It reuses the same chart functions
and experiment builders as ``scripts/smoke_notebooks.py``.

Run::

    uv run python scripts/export_paper_figures.py            # full paper scale
    uv run python scripts/export_paper_figures.py --quick    # fast local check
    uv run python scripts/export_paper_figures.py --out DIR
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aif_traffic.explainers import CHART_GUIDE  # noqa: E402
from aif_traffic.parameters import (  # noqa: E402
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    CohortSpec,
    FixedTimeControllerSpec,
    ObservationSignal,
    Params,
    PopulationParams,
    ReactiveControllerSpec,
    SignalType,
    SimParams,
)
from aif_traffic import plotting as pl  # noqa: E402
from aif_traffic.simulator import run_experiment  # noqa: E402

# --- experiment configuration ----------------------------------------------
SEED = 42
THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MEDIUM_THETA = 0.5
CONTROLLERS = {
    "fixed_time": FixedTimeControllerSpec(),
    "reactive": ReactiveControllerSpec(),
    "anticipatory": AnticipatoryControllerSpec(),
    "aif": AIFControllerSpec(),
}


def _cfg(quick: bool) -> dict:
    """Scale knobs: real paper scale, or a fast subset for local/CI smoke."""
    if quick:
        return {"days": 10, "n_agents": 80, "thetas": (0.0, 1.0)}
    return {"days": 90, "n_agents": 2000, "thetas": THETAS}


def _base(cfg: dict, *, controller=None, theta: float = 0.0,
          comm: SignalType = SignalType.NONE) -> Params:
    """A deterministic (noise-free, stationary) experiment at the given scale."""
    p = replace(
        Params.default(),
        sim=SimParams(days=cfg["days"], burn_in=0, seed=SEED),
        population=PopulationParams(
            cohorts=(CohortSpec(n_agents=cfg["n_agents"]),)),
        controller=controller if controller is not None else AIFControllerSpec(),
    ).with_noise_regime("off").with_stationary(True)
    if comm is not SignalType.NONE:
        p = p.with_comm(comm).with_compliance(1.0)
    return p.with_theta(theta)


# --- figure registry --------------------------------------------------------
# Each entry: (filename_stem, chart_id, render-thunk -> Figure). Grouped by the
# paper's experiment structure; the Section-5 figure slots use stems matching
# the paper's fig labels (within_day_profile_*, controller_comparison, ...).
# Built lazily so heavy sweeps run once.
def _build_registry(cfg: dict):
    reg: list[tuple[str, str, callable]] = []
    # The single representative day the paper's within-day figures inspect
    # ("e.g. day 80"); clamped so --quick still renders.
    profile_day = min(80, cfg["days"] - 1)

    # -- Experiment 1: single AIF run (medium theta, externality broadcast) ---
    p1 = _base(cfg, theta=MEDIUM_THETA, comm=SignalType.EXTERNALITY)
    r1 = run_experiment(p1, seeds=[SEED], snapshot_days=range(cfg["days"]))

    reg += [
        ("exp1_demand_profile", "plot_demand_profile",
         lambda: pl.plot_demand_profile(p1)),
        # Multi-day learning evolution (first/last + evenly spaced days).
        ("exp1_within_day_tt_vs_belief", "plot_within_day_tt_vs_belief",
         lambda: pl.plot_within_day_tt_vs_belief(r1.step, r1.snapshots, p1)),
        ("exp1_coupled_within_day", "plot_coupled_within_day",
         lambda: pl.plot_coupled_within_day(r1.step, p1)),
        # fig:within-day-profile -- panels (a)-(c) at the representative day.
        ("within_day_profile_a", "plot_coupled_within_day",
         lambda: pl.plot_coupled_within_day(r1.step, p1, days=[profile_day])),
        ("within_day_profile_b", "plot_within_day_tt_vs_belief",
         lambda: pl.plot_within_day_tt_vs_belief(
             r1.step, r1.snapshots, p1, days=[profile_day])),
        ("within_day_profile_c", "plot_belief_reality_queues",
         lambda: pl.plot_belief_reality_queues(
             r1.step, r1.snapshots, p1, day=profile_day)),
        # fig:across-day-profile -- heatmaps + daily profiles + cost & belief SD.
        ("across_day_profile", "plot_co_adaptation",
         lambda: pl.plot_co_adaptation(r1.step, r1.controller)),
        ("exp1_learning_uncertainty", "plot_learning_uncertainty",
         lambda: pl.plot_learning_uncertainty(r1.cohort, r1.controller)),
        ("exp1_signal_day", "plot_signal_day",
         lambda: pl.plot_signal_day(r1.step)),
        ("exp1_green_split_heatmap", "plot_green_split_heatmap",
         lambda: pl.plot_green_split_heatmap(r1.step)),
        ("exp1_daily_system_cost", "plot_daily_system_cost",
         lambda: pl.plot_daily_system_cost(r1.step)),
        ("exp1_route_share_over_days", "plot_route_share_over_days",
         lambda: pl.plot_route_share_over_days(r1.step)),
    ]

    # -- controller x theta grid (shared by Exp 1 summary + Exp 2 grid) -------
    grid = {
        name: {
            th: run_experiment(
                _base(cfg, controller=spec, theta=th,
                      comm=SignalType.EXTERNALITY), seeds=[SEED])
            for th in cfg["thetas"]
        }
        for name, spec in CONTROLLERS.items()
    }
    reg += [
        ("exp1_theta_summary", "plot_theta_summary",
         lambda: pl.plot_theta_summary(grid)),
        ("exp1_theta_route_choice", "plot_theta_route_choice",
         lambda: pl.plot_theta_route_choice(grid)),
        ("exp2_controller_theta_grid", "plot_controller_theta_grid",
         lambda: pl.plot_controller_theta_grid(grid)),
    ]

    # -- Experiment 2: controller benchmark (medium theta slice) --------------
    slice_theta = (MEDIUM_THETA if MEDIUM_THETA in cfg["thetas"]
                   else cfg["thetas"][len(cfg["thetas"]) // 2])
    by_ctrl = {name: grid[name][slice_theta] for name in CONTROLLERS}
    reg += [
        ("exp2_controller_metrics", "plot_controller_metrics",
         lambda: pl.plot_controller_metrics(by_ctrl)),
        # fig:controller-comparison -- cost + per-link queues (L2/L5/L6).
        ("controller_comparison", "plot_controller_queue_comparison",
         lambda: pl.plot_controller_queue_comparison(by_ctrl)),
        # fig:green-split-controllers.
        ("green_split_controllers", "plot_green_split_heatmaps_by_controller",
         lambda: pl.plot_green_split_heatmaps_by_controller(by_ctrl)),
        ("exp2_learned_obs_noise", "plot_learned_obs_noise",
         lambda: pl.plot_learned_obs_noise(by_ctrl["aif"].controller)),
    ]

    # -- Experiment 3: information communication (extra-observations) ---------
    base3 = _base(cfg)
    _CG, _SN = ObservationSignal.ROUTE_CONGESTION, ObservationSignal.SIGNAL_CONTROL
    settings = {
        "BL": base3.with_extra_observations(),
        "CG": base3.with_extra_observations(_CG),
        "SN": base3.with_extra_observations(_SN),
        "CG+SN": base3.with_extra_observations(_CG, _SN),
    }
    by_setting = {
        name: run_experiment(pp, seeds=[SEED], snapshot_days=range(cfg["days"]))
        for name, pp in settings.items()
    }
    reg += [
        # fig:across-day-communication -- system performance incl. green split.
        ("across_day_communication", "plot_sweep_metrics",
         lambda: pl.plot_sweep_metrics(
             by_setting, layout="grid",
             panels=("cost", "share", "peak_queue", "phi2"))),
        # Its companion uncertainty figure: belief SD on TT_a / TT_b / phi.
        ("belief_sd_communication", "plot_belief_sd_sweep",
         lambda: pl.plot_belief_sd_sweep(by_setting)),
        ("exp3_within_day_by_setting", "plot_within_day_by_setting",
         lambda: pl.plot_within_day_by_setting(by_setting, base3)),
        # fig:within-day-communication -- realised vs belief at the profile day.
        ("within_day_communication", "plot_within_day_communication",
         lambda: pl.plot_within_day_communication(
             by_setting, base3, day=profile_day)),
        # fig:vary_observation_info -- subfigures (a) P_alpha and (b) queue L2.
        ("vary_observation_info_a", "plot_route_choice_heatmaps",
         lambda: pl.plot_route_choice_heatmaps(by_setting)),
        ("vary_observation_info_b", "plot_route_choice_heatmaps",
         lambda: pl.plot_route_choice_heatmaps(by_setting, value="L2")),
    ]
    return reg


def export_all(out_dir: Path, *, quick: bool = False) -> list[str]:
    """Render every paper figure in the paper style into ``out_dir``.

    Returns the list of written PDF filenames. Also writes ``INDEX.md``.
    """
    pl.apply_style("paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(quick)
    print(f"[export] paper figures | config={cfg} | seed={SEED}")

    registry = _build_registry(cfg)
    index_rows: list[tuple[str, str]] = []
    written: list[str] = []
    for stem, chart_id, render in registry:
        fig = render()
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
        title = CHART_GUIDE.get(chart_id, {}).get("title", chart_id)
        index_rows.append((f"{stem}.pdf", f"{title}  (`{chart_id}`)"))
        written.append(pdf.name)
        print(f"  wrote {pdf.name}")

    index = out_dir / "INDEX.md"
    lines = ["# Paper figures", "",
             f"Rendered in the `paper` style (config: {cfg}, seed {SEED}).", "",
             "| file | figure |", "| --- | --- |"]
    lines += [f"| `{f}` | {d} |" for f, d in index_rows]
    index.write_text("\n".join(lines) + "\n")
    print(f"[export] {len(written)} figures + INDEX.md -> {out_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Export paper-styled figures.")
    ap.add_argument("--out", default="paper_figures", type=Path,
                    help="output directory (default: paper_figures/)")
    ap.add_argument("--quick", action="store_true",
                    help="fast reduced-scale render for a local/CI smoke check")
    args = ap.parse_args()
    export_all(args.out, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
