"""Render every paper figure in the publication (``"paper"``) style and write
them as vector PDFs to ``paper_figures/``.

This is the CI-exportable companion to the marimo notebooks: it applies the
Elsevier-`elsarticle`-3p paper style (IWAI-figure Arial/sans-serif with STIX
math, ~6.72 in text width, vector
PDF, colourblind/greyscale-safe palette) and re-renders the figures the paper
uses at the **notebook gold defaults** (medium measurement noise, stationary
continuous filtering), reproducible via the fixed seed, so the author can drop
the PDFs straight into the LaTeX. It reuses the same chart functions and
experiment builders as ``scripts/smoke_notebooks.py``.

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
    DemandParams,
    FixedTimeControllerSpec,
    ObservationSignal,
    Params,
    PopulationParams,
    ReactiveControllerSpec,
    SimParams,
)
from aif_traffic import plotting as pl  # noqa: E402
from aif_traffic.simulator import run_experiment  # noqa: E402

# --- experiment configuration ----------------------------------------------
SEED = 42
CONTROLLERS = {
    "fixed_time": FixedTimeControllerSpec(),
    "reactive": ReactiveControllerSpec(),
    "anticipatory": AnticipatoryControllerSpec(),
    "aif": AIFControllerSpec(),
}


def _cfg(quick: bool) -> dict:
    """Scale knobs: real paper scale, or a fast subset for local/CI smoke."""
    if quick:
        return {"days": 10, "n_agents": 80}
    return {"days": 90, "n_agents": 2000}


def _profile_day(cfg: dict) -> int:
    """The single day the paper's within-day figures inspect: the last recorded
    day, matching the notebooks' ``day_sel`` default (``max(days-1, 0)``)."""
    return max(cfg["days"] - 1, 0)


def _base(cfg: dict, *, controller=None) -> Params:
    """An experiment at the notebook gold defaults (medium measurement noise,
    stationary continuous filtering) at the given scale; reproducible via the
    fixed seed. The noise regime and toggles are inherited from ``Params.default``
    (which already encodes the notebook ``"medium"`` regime), so the exported
    figures match what the notebooks render with their defaults."""
    return replace(
        Params.default(),
        sim=SimParams(days=cfg["days"], burn_in=0, seed=SEED),
        population=PopulationParams(
            cohorts=(CohortSpec(n_agents=cfg["n_agents"]),)),
        controller=controller if controller is not None else AIFControllerSpec(),
    ).with_stationary(True)


def _demand_scaled(p: Params, scale: float) -> Params:
    """Scale the peak A--B and C--D demand by ``scale`` (Experiment 5), the way
    the notebooks apply the ``demand_scale`` control: reset to the default
    ``DemandParams`` then multiply the two maxima."""
    base = DemandParams()
    return replace(p, demand=replace(
        base, d_AB_max=base.d_AB_max * scale, d_CD_max=base.d_CD_max * scale))


# --- figure registry --------------------------------------------------------
# Each entry: (filename_stem, chart_id, render-thunk -> Figure). Grouped by the
# paper's experiment structure; the Section-5 figure slots use stems matching
# the paper's fig labels (within_day_profile_*, controller_comparison, ...).
# Built lazily so heavy sweeps run once.
def _build_registry(cfg: dict):
    reg: list[tuple[str, str, callable]] = []
    # LaTeX table fragments emitted alongside the figures: (stem, df-thunk,
    # column spec). Written as booktabs tabulars the paper can \input.
    tables: list[tuple[str, callable, dict]] = []
    # The single day the within-day figures inspect: the last recorded day, to
    # match the notebooks' day_sel default.
    profile_day = _profile_day(cfg)

    # -- Experiment 1: single AIF run (coordination mechanism) ---------------
    p1 = _base(cfg)
    r1 = run_experiment(p1, seeds=[SEED], snapshot_days=range(cfg["days"]))

    reg += [
        # fig:dynamic_demand: the paper's demand-profile figure.
        ("shifted_sine_demand_profiles", "plot_demand_profile",
         lambda: pl.plot_demand_profile(p1)),
        # Multi-day learning evolution (first/last + evenly spaced days).
        ("exp1_within_day_tt_vs_belief", "plot_within_day_tt_vs_belief",
         lambda: pl.plot_within_day_tt_vs_belief(r1.step, r1.snapshots, p1)),
        ("exp1_coupled_within_day", "plot_coupled_within_day",
         lambda: pl.plot_coupled_within_day(r1.step, p1)),
        # fig:within-day-profile: one combined 2x3 figure with columns (a)-(c)
        # at the representative day (single PDF, so the panels stay aligned).
        ("within_day_profile", "plot_within_day_profile",
         lambda: pl.plot_within_day_profile(
             r1.step, r1.snapshots, p1, day=profile_day)),
        # fig:across-day-profile: heatmaps + daily profiles + cost & belief SD.
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

    # -- Experiment 2: controller benchmark (each controller run once) --------
    by_ctrl = {
        name: run_experiment(_base(cfg, controller=spec), seeds=[SEED])
        for name, spec in CONTROLLERS.items()
    }
    reg += [
        ("exp2_controller_metrics", "plot_controller_metrics",
         lambda: pl.plot_controller_metrics(by_ctrl)),
        # fig:controller-TT_tot: (a) system cost, (b) total queue band.
        ("controller_comparison", "plot_controller_queue_comparison",
         lambda: pl.plot_controller_queue_comparison(by_ctrl)),
        # fig:within-day-queue-controller: within-day L2/L5/L6 per controller.
        ("within_day_queue_controllers", "plot_within_day_queue_by_controller",
         lambda: pl.plot_within_day_queue_by_controller(by_ctrl, day=profile_day)),
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
        # fig:across-day-communication: belief SD panels (traveller + controller).
        ("belief_sd_communication", "plot_belief_sd_sweep",
         lambda: pl.plot_belief_sd_sweep(by_setting)),
        # fig:communication-cost: daily cost trend + post-convergence mean/SD bars.
        ("communication_cost", "plot_communication_cost",
         lambda: pl.plot_communication_cost(by_setting)),
        ("exp3_within_day_by_setting", "plot_within_day_by_setting",
         lambda: pl.plot_within_day_by_setting(by_setting, base3)),
        # fig:within-day-communication: realised vs belief at the profile day.
        ("within_day_communication", "plot_within_day_communication",
         lambda: pl.plot_within_day_communication(
             by_setting, base3, day=profile_day)),
        # fig:vary_observation_info: subfigures (a) P_alpha and (b) queue L2.
        ("vary_observation_info_a", "plot_route_choice_heatmaps",
         lambda: pl.plot_route_choice_heatmaps(by_setting)),
        ("vary_observation_info_b", "plot_route_choice_heatmaps",
         lambda: pl.plot_route_choice_heatmaps(by_setting, value="L2")),
    ]
    # tab:communication-cost: the per-setting system-cost summary that
    # replaces the noisy day-by-day cost chart in the communication figure.
    tables += [
        ("communication_cost_table",
         lambda: pl.communication_cost_table(by_setting),
         {
             "caption": (
                 "Steady-state total system cost per information-communication "
                 "setting (mean, best, worst and standard deviation of the daily "
                 "system cost over the last recorded days). Signal-control "
                 "information (SN) gives the lowest cost."),
             "label": "tab:communication-cost",
             "colspec": "lrrrr",
             "header": ["Setting", "Mean", "Best", "Worst", "Std. dev."],
             "order": ["setting", "mean_SC", "best_SC", "worst_SC", "std_SC"],
             "int_cols": ["mean_SC", "best_SC", "worst_SC", "std_SC"],
         }),
    ]

    # -- Experiment 5: robustness to traffic demand (varying-demand analysis) --
    # Same coupled AIF base as Exp 1, re-run at several peak-demand scales; one
    # line per scale.
    demand_scales = (0.8, 1.0, 1.2, 1.4)
    p5 = _base(cfg)
    by_demand = {
        f"{s:g}x": run_experiment(_demand_scaled(p5, s), seeds=[SEED])
        for s in demand_scales
    }
    reg += [
        # fig:robust-within-day: within-day Q_alpha / Q_beta / phi_2 by demand.
        ("robustness_within_day_demand", "plot_within_day_by_demand",
         lambda: pl.plot_within_day_by_demand(by_demand, day=profile_day)),
        # fig:robust-across-day: daily P_alpha / phi_2 / cost (+ ctrl SD) by demand.
        ("robustness_across_day_demand", "plot_across_day_by_demand",
         lambda: pl.plot_across_day_by_demand(by_demand)),
    ]
    return reg, tables


def _comm_setting_label(key: str) -> str:
    """Full setting name (with abbreviation) for a table row, matching the
    paper's BL/CG/SN/CG+SN wording."""
    return {
        "BL": "Baseline (BL)",
        "CG": "Route congestion (CG)",
        "SN": "Signal control (SN)",
        "CG+SN": "Route cong. + signal (CG+SN)",
    }.get(str(key), str(key))


def _df_to_booktabs(df, spec: dict) -> str:
    """Render a DataFrame as a standalone booktabs ``table`` environment string.

    Integer columns are printed with a thousands separator; the ``setting``
    column is expanded to its full name. Emitted as a ``.tex`` fragment the paper
    ``\\input``s so the table stays reproducible from the simulation."""
    order = spec["order"]
    header = spec["header"]
    int_cols = set(spec.get("int_cols", []))

    def _fmt(col: str, val) -> str:
        if col == "setting":
            return _comm_setting_label(val)
        if col in int_cols:
            return f"{round(float(val)):,}"
        return f"{float(val):.3g}"

    body_rows = []
    for _i, row in df.iterrows():
        cells = [_fmt(col, row[col]) for col in order]
        body_rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"% Auto-generated by scripts/export_paper_figures.py; do not edit by hand.",
        r"\begin{table}[htp!]",
        r"    \centering",
        f"    \\caption{{{spec['caption']}}}",
        f"    \\label{{{spec['label']}}}",
        f"    \\begin{{tabular}}{{{spec['colspec']}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    lines += ["    " + r for r in body_rows]
    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def export_all(out_dir: Path, *, quick: bool = False) -> list[str]:
    """Render every paper figure in the paper style into ``out_dir``.

    Returns the list of written PDF filenames. Also writes ``INDEX.md``.
    """
    pl.apply_style("paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(quick)
    print(f"[export] paper figures | config={cfg} | seed={SEED}")

    registry, tables = _build_registry(cfg)
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

    # LaTeX table fragments (booktabs) the paper \inputs, kept reproducible from
    # the same runs as the figures.
    for stem, render_df, spec in tables:
        df = render_df()
        tex = out_dir / f"{stem}.tex"
        tex.write_text(_df_to_booktabs(df, spec))
        index_rows.append((f"{stem}.tex", f"{spec.get('label', stem)} (table)"))
        written.append(tex.name)
        print(f"  wrote {tex.name}")

    index = out_dir / "INDEX.md"
    lines = ["# Paper figures", "",
             f"Rendered in the `paper` style at the notebook gold defaults "
             f"(medium measurement noise, stationary continuous filtering; "
             f"config: {cfg}, seed {SEED}).", "",
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
