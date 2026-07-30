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
from .palette import (
    CTRL_ORDER as _CTRL_ORDER,
    controller_colour,
    controller_label,
    controller_linestyle,
    ordered_controllers,
    route_colour,
    route_linestyle,
)
from .primitives import light_borders, panel_label, text_w
from .style import active_style


def _ordered(results_by_ctrl: Mapping[str, object]) -> list[tuple[str, object]]:
    """(name, result) pairs in canonical order, for the controllers present."""
    return [(k, results_by_ctrl[k]) for k in ordered_controllers(results_by_ctrl)]


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
    # Total network queue: both signalised movements plus the bypass (L5).
    tmp = step.assign(_Ltot=step["L2"] + step["L5"] + step["L6"])
    return _per_day(tmp.groupby(_keys(step))["_Ltot"].max(), step)


def _daily_total_queue_band(step: pd.DataFrame):
    """Per-day within-day mean / min / max of the total network queue
    ``L2+L5+L6`` (mean over seeds), as three day-indexed Series."""
    tmp = step.assign(_Ltot=step["L2"] + step["L5"] + step["L6"])
    g = tmp.groupby(_keys(step))["_Ltot"]
    return (_per_day(g.mean(), step), _per_day(g.min(), step),
            _per_day(g.max(), step))


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
        ("peak queue $L_2+L_5+L_6$ [veh]", _daily_peak_total_queue),
        (r"green-split variation $\sum|\Delta\phi_2|$", _daily_signal_variation),
    ]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(text_w(), text_w() * 1.45), sharex=True,
    )
    lw = active_style().line_main
    for ax, (ylabel, fn) in zip(axes, panels):
        for name, res in items:
            s = fn(res.step)
            ax.plot(s.index.to_numpy(), s.to_numpy(),
                    color=controller_colour(name), linewidth=lw,
                    linestyle=controller_linestyle(name))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("day")
    light_borders(axes)

    handles = [
        Line2D([0], [0], color=controller_colour(k), linewidth=lw,
               linestyle=controller_linestyle(k), label=controller_label(k))
        for k, _ in items
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_controller_queue_comparison(results_by_ctrl: Mapping[str, object]):
    """Two day-series panels, one coloured line per controller: (a) daily system
    cost and (b) the daily **total** network queue ``L_2+L_5+L_6``. The queue
    panel draws the within-day **mean** total queue as a solid line with the
    within-day min--max range as a shaded band, so both the level and the daily
    excursion of the total queue can be compared per controller.
    """
    items = _ordered(results_by_ctrl)
    st = active_style()
    lw, band_a = st.line_main, st.band_alpha

    fig, axes = plt.subplots(1, 2, figsize=(text_w(), text_w() * 0.42))
    for name, res in items:
        cost = _daily_cost(res.step)
        axes[0].plot(cost.index.to_numpy(), cost.to_numpy(),
                     color=controller_colour(name), linewidth=lw,
                     linestyle=controller_linestyle(name))
    axes[0].set_ylabel("system cost [veh-min]")
    panel_label(axes[0], "a")

    ax = axes[1]
    for name, res in items:
        mean, lo, hi = _daily_total_queue_band(res.step)
        days = mean.index.to_numpy()
        colour = controller_colour(name)
        ax.plot(days, mean.to_numpy(), color=colour, linewidth=lw,
                linestyle=controller_linestyle(name))
        ax.fill_between(days, lo.to_numpy(), hi.to_numpy(),
                        color=colour, alpha=band_a, linewidth=0)
    ax.set_ylabel(r"total queue $L_2{+}L_5{+}L_6$ [veh]")
    panel_label(ax, "b")

    for ax in axes:
        ax.set_xlabel("day")
        ax.grid(alpha=0.25)
    light_borders(axes)

    handles = [
        Line2D([0], [0], color=controller_colour(k), linewidth=lw,
               linestyle=controller_linestyle(k), label=controller_label(k))
        for k, _ in items
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_within_day_queue_by_controller(
    results_by_ctrl: Mapping[str, object], *, day: int | None = None,
    seed: int | None = None,
):
    """One square panel per controller: the within-day realised queue on the
    three critical links ``L_2`` (A--B intersection), ``L_5`` (A--B bypass) and
    ``L_6`` (C--D) at a representative day.

    A 1xN row (one column per controller in canonical order, FT/RF/AC/AIF) with
    a shared y-axis so the queue levels are directly comparable across strategies.
    ``day`` selects the inspected day (default the last recorded day); ``seed``
    picks the run when several are present (default the first).
    """
    items = _ordered(results_by_ctrl)
    n = max(len(items), 1)
    lw = active_style().line_main
    specs = [
        ("L2", route_colour("alpha"), route_linestyle("alpha"), r"$L_2$"),
        ("L5", route_colour("beta"), route_linestyle("beta"), r"$L_5$"),
        ("L6", route_colour("gamma"), route_linestyle("gamma"), r"$L_6$"),
    ]

    fig, axgrid = plt.subplots(
        1, n, figsize=(text_w(), text_w() * 0.40), sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axgrid[0]
    for ax, (name, res) in zip(axes, items):
        step = res.step
        if "seed" in step.columns:
            s = int(step["seed"].min()) if seed is None else int(seed)
            step = step[step["seed"] == s]
        d_use = int(step["day"].max()) if day is None else int(day)
        dd = step[step["day"] == d_use].sort_values("tau")
        tau = dd["tau"].to_numpy(dtype=float)
        for col, colour, ls, _lab in specs:
            ax.plot(tau, dd[col].to_numpy(), color=colour, linewidth=lw,
                    linestyle=ls)
        # Abbreviation-only titles (FT/RF/AC/AIF): the paper caption carries the
        # expansions, and the narrow per-controller columns cannot fit full names.
        ax.set_title(controller_label(name, abbr=True), fontsize=8)
        ax.set_xlabel("time of day [min]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("queue [veh]")
    light_borders(axgrid)

    handles = [
        Line2D([0], [0], color=colour, linewidth=lw, linestyle=ls, label=lab)
        for _col, colour, ls, lab in specs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_green_split_heatmaps_by_controller(
    results_by_ctrl: Mapping[str, object], value: str = "phi2",
):
    """One heatmap column per controller: ``value`` over (day x time-of-day),
    on a shared colour scale with a single colorbar."""
    items = _ordered(results_by_ctrl)
    n = len(items)
    # A short (about half-height) row of heatmaps, so the day x time-of-day green
    # split maps stay compact rather than the tall aspect used before.
    fig, axes = plt.subplots(
        1, n, figsize=(text_w(), text_w() * 0.30), sharex=True, sharey=True,
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
                           cmap="viridis", vmin=vmin, vmax=vmax, shading="flat",
                           edgecolors="face", linewidth=0.0,
                           rasterized=True)
        # Abbreviation-only titles (FT/RF/AC/AIF): the paper caption carries the
        # expansions, and the narrow heatmap columns cannot fit full names.
        ax.set_title(controller_label(name, abbr=True), fontsize=8)
        ax.set_xlabel("day")
    axes[0].set_ylabel("time of day [min]")
    label = {"phi2": r"green split $\phi_2$", "L2": r"queue $L_2$",
             "L6": r"queue $L_6$"}.get(value, value)
    fig.colorbar(im, ax=list(axes), pad=0.02, label=label, fraction=0.046)
    return fig


def controller_summary(results_by_ctrl: Mapping[str, object]) -> pd.DataFrame:
    """One row per controller: mean cost + day-to-day cost stability (std), the
    green-split variation (mean and its day-to-day std, the 'stable splits'
    claim), the mean daily peak total queue, and the mean daily peak on the C--D
    movement ``L_6`` (the queue the SCOOT-style controller mismanages)."""
    rows = []
    for name, res in _ordered(results_by_ctrl):
        cost = _daily_cost(res.step)
        sig_var = _daily_signal_variation(res.step)
        peak_l6 = _per_day(
            res.step.groupby(_keys(res.step))["L6"].max(), res.step)
        rows.append({
            "controller": controller_label(name),
            "mean_SC": float(cost.mean()),
            "std_SC": float(cost.std()),
            "mean_signal_variation": float(sig_var.mean()),
            "std_signal_variation": float(sig_var.std()),
            "mean_peak_queue": float(_daily_peak_total_queue(res.step).mean()),
            "mean_peak_L6": float(peak_l6.mean()),
        })
    return pd.DataFrame(rows)
