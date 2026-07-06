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
    ordered_controllers,
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
                    color=controller_colour(name), linewidth=lw)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("day")
    light_borders(axes)

    handles = [
        Line2D([0], [0], color=controller_colour(k),
               linewidth=lw, label=controller_label(k))
        for k, _ in items
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_controller_queue_comparison(results_by_ctrl: Mapping[str, object]):
    """One row of four day-series panels, one coloured line per controller:
    (a) daily system cost, then the daily queue on (b) ``L_2``, (c) ``L_5``,
    and (d) ``L_6``. The queue panels draw the within-day **mean** queue as a
    solid line with the within-day min--max range as a shaded band, so both the
    level and the daily excursion of each queue can be compared per controller.
    """
    items = _ordered(results_by_ctrl)
    st = active_style()
    lw, band_a = st.line_main, st.band_alpha

    fig, axes = plt.subplots(
        1, 4, figsize=(text_w(), text_w() * 0.32), sharex=True,
    )
    for name, res in items:
        cost = _daily_cost(res.step)
        axes[0].plot(cost.index.to_numpy(), cost.to_numpy(),
                     color=controller_colour(name), linewidth=lw)
    axes[0].set_ylabel("system cost [veh-min]")
    panel_label(axes[0], "a")

    for k, (ax, col) in enumerate(zip(axes[1:], ("L2", "L5", "L6"))):
        for name, res in items:
            step = res.step
            g = step.groupby(_keys(step))[col]
            mean = _per_day(g.mean(), step)
            lo = _per_day(g.min(), step)
            hi = _per_day(g.max(), step)
            days = mean.index.to_numpy()
            colour = controller_colour(name)
            ax.plot(days, mean.to_numpy(), color=colour, linewidth=lw)
            ax.fill_between(days, lo.to_numpy(), hi.to_numpy(),
                            color=colour, alpha=band_a, linewidth=0)
        ax.set_ylabel(f"queue $L_{col[1]}$ [veh]")
        # An (almost) always-empty queue (typical for the high-capacity bypass)
        # would otherwise get a +/-0.04 autoscale; pin a sane floor instead.
        if ax.get_ylim()[1] < 1.0:
            ax.set_ylim(-0.05, 1.0)
        panel_label(ax, "bcd"[k])
    for ax in axes:
        ax.set_xlabel("day")
        ax.grid(alpha=0.25)
    light_borders(axes)

    handles = [
        Line2D([0], [0], color=controller_colour(k),
               linewidth=lw, label=controller_label(k, abbr=True))
        for k, _ in items
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_green_split_heatmaps_by_controller(
    results_by_ctrl: Mapping[str, object], value: str = "phi2",
):
    """One heatmap column per controller: ``value`` over (day x time-of-day),
    on a shared colour scale with a single colorbar."""
    items = _ordered(results_by_ctrl)
    n = len(items)
    fig, axes = plt.subplots(
        1, n, figsize=(text_w(), text_w() * 0.5), sharex=True, sharey=True,
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
        # Abbreviation titles (FT/RF/AC/AIF) keep the heatmap text clean (Xue).
        ax.set_title(controller_label(name, abbr=True), fontsize=8)
        ax.set_xlabel("day")
    axes[0].set_ylabel("time of day [min]")
    label = {"phi2": r"green split $\phi_2$", "L2": r"queue $L_2$",
             "L6": r"queue $L_6$"}.get(value, value)
    fig.colorbar(im, ax=list(axes), pad=0.02, label=label, fraction=0.046)
    return fig


def plot_controller_theta_grid(
    results_by_ctrl_theta: Mapping[str, Mapping[float, object]],
    n_last: int = 15,
):
    """Heatmap of steady-state mean system cost over (theta x controller).

    ``results_by_ctrl_theta`` is a nested mapping
    ``{controller_name: {theta: ExperimentResult}}`` -- one full run per
    (controller, theta) cell. Rows are the social-internalisation values theta
    (ascending), columns the controllers in canonical order; each cell is the
    mean daily system cost over the last ``n_last`` recorded days (lower is
    better). A single shared colour scale makes both axes comparable, and the
    numeric value is annotated in every cell.
    """
    ctrls = [k for k in _CTRL_ORDER if k in results_by_ctrl_theta]
    thetas = sorted({t for c in ctrls for t in results_by_ctrl_theta[c]})

    grid = np.full((len(thetas), len(ctrls)), np.nan)
    for j, c in enumerate(ctrls):
        for t, res in results_by_ctrl_theta[c].items():
            i = thetas.index(t)
            cost = _daily_cost(res.step)
            grid[i, j] = float(cost.iloc[-n_last:].mean())

    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 0.62))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto", origin="lower")

    ax.set_xticks(range(len(ctrls)))
    ax.set_xticklabels([controller_label(c, abbr=True) for c in ctrls],
                       fontsize=8)
    ax.set_yticks(range(len(thetas)))
    ax.set_yticklabels([f"{t:g}" for t in thetas])
    ax.set_ylabel(r"social internalisation $\theta$")

    # Annotate each cell; pick a legible text colour against the cell shade.
    finite = grid[np.isfinite(grid)]
    mid = float(finite.mean()) if finite.size else 0.0
    for i in range(len(thetas)):
        for j in range(len(ctrls)):
            if not np.isfinite(grid[i, j]):
                continue
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if grid[i, j] > mid else "black")

    fig.colorbar(im, ax=ax, pad=0.02, label="system cost [veh-min]", fraction=0.046)
    fig.tight_layout()
    return fig


def _theta_axis(results_by_ctrl_theta):
    ctrls = [k for k in _CTRL_ORDER if k in results_by_ctrl_theta]
    thetas = sorted({t for c in ctrls for t in results_by_ctrl_theta[c]})
    return ctrls, thetas


def plot_theta_summary(
    results_by_ctrl_theta: Mapping[str, Mapping[float, object]],
    n_last: int = 15,
):
    """Steady-state performance vs social internalisation ``theta``, per
    controller (Xue's Experiment-1 theta Figure 1).

    Four panels -- mean and SD of the daily system cost, and mean and SD of the
    daily peak total queue ``L_2+L_5+L_6`` -- each over the last ``n_last`` days.
    x-axis is ``theta``; one line per controller (canonical colour). Reveals
    whether increasing ``theta`` moves performance and whether the adaptive
    controller "absorbs" that effect.
    """
    ctrls, thetas = _theta_axis(results_by_ctrl_theta)
    lw = active_style().line_main

    def _tail(series):
        return series.iloc[-n_last:]

    panels = [
        ("mean system cost [veh-min]", _daily_cost, "mean"),
        ("SD system cost [veh-min]", _daily_cost, "std"),
        ("mean peak queue [veh]", _daily_peak_total_queue, "mean"),
        ("SD peak queue [veh]", _daily_peak_total_queue, "std"),
    ]
    fig, axgrid = plt.subplots(2, 2, figsize=(text_w(), text_w() * 0.85))
    axes = axgrid.ravel()
    for ax, (ylabel, fn, stat) in zip(axes, panels):
        for c in ctrls:
            ys = []
            for t in thetas:
                res = results_by_ctrl_theta[c].get(t)
                if res is None:
                    ys.append(np.nan)
                    continue
                tail = _tail(fn(res.step))
                ys.append(float(tail.mean() if stat == "mean" else tail.std()))
            ax.plot(thetas, ys, color=controller_colour(c), linewidth=lw,
                    marker="o", markersize=3, label=controller_label(c, abbr=True))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[2:]:
        ax.set_xlabel(r"social internalisation $\theta$")
    light_borders(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02),
               fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_theta_route_choice(
    results_by_ctrl_theta: Mapping[str, Mapping[float, object]],
    n_last: int = 30,
):
    """Behavioural mechanism of ``theta``: the distribution of the daily
    intersection share ``P_alpha`` at each ``theta``, per controller (Xue's
    Experiment-1 theta Figure 2).

    A grouped box plot: x = ``theta``, one box per controller within each theta
    group (canonical colour), summarising the last ``n_last`` days' daily mean
    ``P_alpha``. Shows whether route choice actually responds to ``theta`` or the
    behavioural response is too small to matter.
    """
    ctrls, thetas = _theta_axis(results_by_ctrl_theta)
    n = max(len(ctrls), 1)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 0.5))
    for j, c in enumerate(ctrls):
        data, positions = [], []
        for i, t in enumerate(thetas):
            res = results_by_ctrl_theta[c].get(t)
            if res is None:
                continue
            daily = res.step.groupby("day")["P_alpha"].mean().iloc[-n_last:]
            data.append(daily.to_numpy())
            positions.append(i + (j - (n - 1) / 2) * width)
        if not data:
            continue
        colour = controller_colour(c)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=0.8))
        for box in bp["boxes"]:
            box.set(facecolor=colour, alpha=0.55, linewidth=0.6)
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set(color=colour, linewidth=0.6)
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([f"{t:g}" for t in thetas])
    ax.set_xlabel(r"social internalisation $\theta$")
    ax.set_ylabel(r"daily intersection share $P_\alpha$")
    ax.grid(alpha=0.25, axis="y")
    handles = [
        Line2D([0], [0], color=controller_colour(c), linewidth=6, alpha=0.55,
               label=controller_label(c, abbr=True))
        for c in ctrls
    ]
    ax.legend(handles=handles, ncol=len(handles), frameon=False, fontsize=7,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    light_borders([ax])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def controller_summary(results_by_ctrl: Mapping[str, object]) -> pd.DataFrame:
    """One row per controller: mean cost + day-to-day cost stability (std), the
    green-split variation (mean and its day-to-day std -- the 'stable splits'
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
