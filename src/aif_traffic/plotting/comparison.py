"""Cross-controller comparison plots.

Scalar day-series metrics (system cost, peak queue, signal variation) are
overlaid on one chart, one coloured line per controller; the green-split policy
is shown as one heatmap column per controller. A small summary table collects
the headline numbers. Pure ``Figure`` / ``DataFrame``-returning helpers.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .network import _edges
from .primitives import TEXT_W, light_borders

# Display order, labels, and styling. Keys match each controller's
# ``snapshot()["name"]`` (and the keys the notebook uses).
_CTRL_ORDER = ("fixed_time", "reactive", "anticipatory", "aif")
_CTRL_LABELS = {
    "fixed_time": "Fixed-time",
    "reactive": "Reactive (SCOOT-like)",
    "anticipatory": "Anticipatory (predictive)",
    "aif": "AIF (proposed)",
}
# Distinct but unbiased styling: every controller gets the same line weight so
# no controller is visually favoured; colours only serve to tell them apart.
_CTRL_COLOURS = {
    "fixed_time": "#9e9e9e",   # grey
    "reactive": "#4393c3",     # blue
    "anticipatory": "#ff9800",  # amber
    "aif": "#1b5e20",          # green
}
_CTRL_LINEWIDTHS = {
    "fixed_time": 1.5,
    "reactive": 1.5,
    "anticipatory": 1.5,
    "aif": 1.5,
}


def _ordered(results_by_ctrl: Mapping[str, object]) -> list[tuple[str, object]]:
    """(name, result) pairs in canonical order, for the controllers present."""
    return [(k, results_by_ctrl[k]) for k in _CTRL_ORDER if k in results_by_ctrl]


# --- per-day metric series --------------------------------------------------
def _keys(step: pd.DataFrame) -> list[str]:
    return ["seed", "day"] if "seed" in step.columns else ["day"]


def _per_day(series_by_group: pd.Series, step: pd.DataFrame) -> pd.Series:
    """Collapse a (seed, day)-indexed series to a day-indexed one (mean over seeds)."""
    if "seed" in step.columns:
        return series_by_group.groupby("day").mean()
    return series_by_group


def _daily_cost(step: pd.DataFrame) -> pd.Series:
    return _per_day(step.groupby(_keys(step))["SC"].first(), step)


def _daily_peak_total_queue(step: pd.DataFrame) -> pd.Series:
    tmp = step.assign(_Ltot=step["L2"] + step["L6"])
    return _per_day(tmp.groupby(_keys(step))["_Ltot"].max(), step)


def _daily_signal_variation(step: pd.DataFrame) -> pd.Series:
    """Per-day total variation of the green split, ``sum |phi2(t) - phi2(t-1)|``.

    The split is piecewise-constant between control epochs, so this equals the
    sum of the absolute green-split changes the controller made that day. Lower
    means smoother, more stable signal operation.
    """
    def _tv(g: pd.DataFrame) -> float:
        return float(g.sort_values("tau")["phi2"].diff().abs().sum())

    per = step.groupby(_keys(step), group_keys=False).apply(_tv)
    return _per_day(per, step)


# --- figures ----------------------------------------------------------------
def plot_controller_metrics(results_by_ctrl: Mapping[str, object]):
    """Three stacked overlay panels (one line per controller): daily system
    cost, daily peak total queue, and daily green-split variation."""
    items = _ordered(results_by_ctrl)
    panels = [
        ("system cost [veh-min]", _daily_cost),
        ("peak queue $L_2+L_6$ [veh]", _daily_peak_total_queue),
        (r"green-split variation $\sum|\Delta\phi_2|$", _daily_signal_variation),
    ]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(TEXT_W, TEXT_W * 1.45), sharex=True,
    )
    for ax, (ylabel, fn) in zip(axes, panels):
        for name, res in items:
            s = fn(res.step)
            ax.plot(s.index.to_numpy(), s.to_numpy(),
                    color=_CTRL_COLOURS.get(name, "k"),
                    linewidth=_CTRL_LINEWIDTHS.get(name, 1.3))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("day")
    light_borders(axes)

    handles = [
        Line2D([0], [0], color=_CTRL_COLOURS[k],
               linewidth=_CTRL_LINEWIDTHS[k], label=_CTRL_LABELS[k])
        for k, _ in items
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_green_split_heatmaps_by_controller(
    results_by_ctrl: Mapping[str, object], value: str = "phi2",
):
    """One heatmap column per controller: ``value`` over (day x time-of-day),
    on a shared colour scale with a single colorbar."""
    items = _ordered(results_by_ctrl)
    n = len(items)
    fig, axes = plt.subplots(
        1, n, figsize=(TEXT_W, TEXT_W * 0.5), sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    pivots = []
    for name, res in items:
        step = res.step
        sd = step if "seed" not in step.columns else step[step["seed"] == step["seed"].min()]
        pivots.append(sd.pivot_table(index="tau", columns="day", values=value, aggfunc="mean"))
    vmax = float(max(p.values.max() for p in pivots))
    vmin = 0.0

    im = None
    for ax, (name, _), hm in zip(axes, items, pivots):
        taus = hm.index.to_numpy(dtype=float)
        days = hm.columns.to_numpy(dtype=float)
        im = ax.pcolormesh(_edges(days), _edges(taus), hm.values,
                           cmap="viridis", vmin=vmin, vmax=vmax, shading="flat")
        ax.set_title(_CTRL_LABELS.get(name, name), fontsize=7.5)
        ax.set_xlabel("day")
    axes[0].set_ylabel("time of day [min]")
    label = {"phi2": r"green split $\phi_2$", "L2": r"queue $L_2$",
             "L6": r"queue $L_6$"}.get(value, value)
    fig.colorbar(im, ax=list(axes), pad=0.02, label=label, fraction=0.046)
    return fig


def controller_summary(results_by_ctrl: Mapping[str, object]) -> pd.DataFrame:
    """One row per controller: mean cost, day-to-day cost stability (std),
    mean green-split variation, and mean daily peak total queue."""
    rows = []
    for name, res in _ordered(results_by_ctrl):
        cost = _daily_cost(res.step)
        rows.append({
            "controller": _CTRL_LABELS.get(name, name),
            "mean_SC": float(cost.mean()),
            "std_SC": float(cost.std()),
            "mean_signal_variation": float(_daily_signal_variation(res.step).mean()),
            "mean_peak_queue": float(_daily_peak_total_queue(res.step).mean()),
        })
    return pd.DataFrame(rows)
