"""Quantitative summary tables (pure ``DataFrame``-returning helpers).

The charts are qualitative; these give the numbers behind them. Each table
aggregates the **steady state** (the last ``n_last`` recorded days) of results
the notebooks already computed: no new simulation. They reuse the per-day
metric helpers in :mod:`.comparison` and :mod:`.sweep`, and are rendered in the
notebooks via :func:`aif_traffic.notebook_io.table_block` (caption + table).
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .comparison import (
    _daily_cost,
    _daily_peak_total_queue,
    _daily_signal_variation,
)
from .palette import controller_label, ordered_controllers
from .sweep import _daily_belief_uncertainty, _daily_route_share


def _tail(series: pd.Series, n_last: int) -> pd.Series:
    return series.iloc[-int(n_last):]


def run_summary_table(res, *, n_last: int = 15) -> pd.DataFrame:
    """Steady-state summary of a **single run** as a tidy metric/mean/std table.

    Reports, over the last ``n_last`` recorded days: system cost, peak queue
    ``L_2+L_5+L_6``, intersection share ``P_alpha``, green-split variation
    ``sum|dphi_2|``, the travellers' belief SD on ``TT_alpha``, and (when the
    controller records it) its cost-belief SD. ``mean`` is the level, ``std`` the
    day-to-day variability.
    """
    step, cohort = res.step, res.cohort
    rows: list[dict] = []

    def add(metric: str, series: pd.Series) -> None:
        t = _tail(series, n_last)
        rows.append({"metric": metric, "mean": float(t.mean()),
                     "std": float(t.std())})

    add("system cost [veh-min]", _daily_cost(step))
    add("peak queue L2+L5+L6 [veh]", _daily_peak_total_queue(step))
    add("intersection share P_alpha", _daily_route_share(step))
    add("green-split variation sum|dphi2|", _daily_signal_variation(step))
    add("traveller belief SD TT_alpha [min]", _daily_belief_uncertainty(cohort))

    ctrl = getattr(res, "controller", None)
    if ctrl is not None and "SC_belief_sd" in getattr(ctrl, "columns", []) \
            and ctrl["SC_belief_sd"].notna().any():
        s = ctrl.sort_values("day").set_index("day")["SC_belief_sd"].dropna()
        add("controller cost-belief SD [veh-min]", s)
    return pd.DataFrame(rows)


def theta_summary_table(
    results_by_ctrl_theta: Mapping[str, Mapping[float, object]],
    *, n_last: int = 15,
) -> pd.DataFrame:
    """Steady-state metrics over (controller x theta), one row per pair.

    Columns: ``mean_SC, std_SC, mean_peak_queue, std_peak_queue, mean_P_alpha,
    std_P_alpha`` (the reviewer-requested set). Reading a controller's rows down
    theta shows whether social internalisation actually moves performance or the
    adaptive controller "absorbs" it.
    """
    ctrls = ordered_controllers(results_by_ctrl_theta)
    thetas = sorted({t for c in ctrls for t in results_by_ctrl_theta[c]})
    rows: list[dict] = []
    for c in ctrls:
        for t in thetas:
            res = results_by_ctrl_theta[c].get(t)
            if res is None:
                continue
            sc = _tail(_daily_cost(res.step), n_last)
            pk = _tail(_daily_peak_total_queue(res.step), n_last)
            pa = _tail(_daily_route_share(res.step), n_last)
            rows.append({
                "controller": controller_label(c, abbr=True),
                "theta": float(t),
                "mean_SC": float(sc.mean()), "std_SC": float(sc.std()),
                "mean_peak_queue": float(pk.mean()),
                "std_peak_queue": float(pk.std()),
                "mean_P_alpha": float(pa.mean()),
                "std_P_alpha": float(pa.std()),
            })
    return pd.DataFrame(rows)


def capacity_theta_summary(
    results_by_scale_theta: Mapping[str, Mapping[float, object]],
    *, n_last: int = 15,
) -> pd.DataFrame:
    """Steady-state summary per bypass-capacity scale over the theta sweep.

    One row per capacity scale: the ``theta=0`` and ``theta=1`` mean system cost
    and their change (%), the best ``theta`` in the sweep and its cost, and the
    day-to-day route-share oscillation (``P_alpha`` std) at ``theta=1``. A large
    ``dSC_pct`` with a large oscillation flags the advisory cobweb; smoothing the
    advisory (raising the window) shrinks both. Reads off whether internalisation
    helps at a given capacity and advisory-smoothing window.
    """
    rows: list[dict] = []
    for label, by_theta in results_by_scale_theta.items():
        thetas = sorted(t for t in by_theta if by_theta.get(t) is not None)
        if not thetas:
            continue
        costs = {t: float(_tail(_daily_cost(by_theta[t].step), n_last).mean())
                 for t in thetas}
        t0, t1 = thetas[0], thetas[-1]
        best_t = min(costs, key=costs.get)
        osc = float(_tail(_daily_route_share(by_theta[t1].step), n_last).std())
        rows.append({
            "bypass_scale": str(label),
            "SC_theta0": costs[t0],
            "SC_theta1": costs[t1],
            "dSC_pct": 100.0 * (costs[t1] - costs[t0]) / costs[t0],
            "best_theta": float(best_t),
            "best_SC": costs[best_t],
            "Palpha_std_theta1": osc,
        })
    return pd.DataFrame(rows)


def communication_cost_table(
    results_by_label: Mapping[str, object], *, n_last: int = 30,
) -> pd.DataFrame:
    """System-cost summary per information-communication setting.

    One row per setting (BL/CG/SN/CG+SN) with the **average**, **best** (lowest),
    **worst** (highest) and **standard deviation** of the daily system cost over
    the steady-state window (the last ``n_last`` recorded days, so the initial
    learning transient does not dominate the best/worst). Replaces the noisy
    day-by-day system-cost chart in the paper's communication figure with the
    numbers behind it (SN is the lowest-cost setting).
    """
    rows: list[dict] = []
    for label, res in results_by_label.items():
        sc = _tail(_daily_cost(res.step), n_last)
        rows.append({
            "setting": str(label),
            "mean_SC": float(sc.mean()),
            "best_SC": float(sc.min()),
            "worst_SC": float(sc.max()),
            "std_SC": float(sc.std()),
        })
    return pd.DataFrame(rows)


def communication_summary_table(
    results_by_label: Mapping[str, object], *, n_last: int = 15,
) -> pd.DataFrame:
    """Steady-state summary per information-communication setting.

    One row per setting (BL/CG/SN/CG+SN): mean system cost, its change vs the
    baseline (%), the travellers' belief SD on ``TT_alpha`` and ``TT_beta``
    (uncertainty), and the mean intersection share. Tabulates the experiment's
    claim (SN lowest cost, CG sharpest beliefs, CG+SN ~ redundant).
    """
    items = list(results_by_label.items())
    bl_key = "BL" if "BL" in results_by_label else (items[0][0] if items else None)
    bl_sc = (
        float(_tail(_daily_cost(results_by_label[bl_key].step), n_last).mean())
        if bl_key is not None else float("nan")
    )
    rows: list[dict] = []
    for label, res in items:
        sc = float(_tail(_daily_cost(res.step), n_last).mean())
        sd_a = float(_tail(_daily_belief_uncertainty(res.cohort), n_last).mean())
        sd_b = float(_tail(
            res.cohort.groupby("day")["sigma_beta_post"].mean(), n_last).mean())
        pa = float(_tail(_daily_route_share(res.step), n_last).mean())
        rows.append({
            "setting": str(label),
            "mean_SC": sc,
            "dSC_vs_BL_pct": (100.0 * (sc - bl_sc) / bl_sc
                              if bl_sc and bl_sc == bl_sc else float("nan")),
            "belief_SD_TT_alpha": sd_a,
            "belief_SD_TT_beta": sd_b,
            "mean_P_alpha": pa,
        })
    return pd.DataFrame(rows)
