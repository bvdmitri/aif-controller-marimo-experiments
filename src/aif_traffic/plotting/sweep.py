"""Generic sweep-overlay plots: one coloured line per experiment variant.

Used by the information-communication sweep (Experiment 3, variants =
BL/CG/SN/CG+SN). Each helper takes an ordered mapping ``{label: ExperimentResult}``
and overlays a daily series, one line per label. Pure ``Figure``-returning.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .comparison import _daily_cost, _daily_peak_total_queue
from .network import _edges
from .palette import (
    COMM_ORDER,
    comm_colour,
    comm_label,
    comm_linestyle,
    sweep_linestyle,
)
from .primitives import light_borders, panel_label, text_w
from .style import active_style


def _daily_route_share(step) -> "object":
    """Day-indexed mean intersection-route share P_alpha (over tau and seeds)."""
    return step.groupby("day")["P_alpha"].mean()


def _daily_belief_uncertainty(cohort) -> "object":
    """Day-indexed mean traveller belief SD over the intersection-route travel
    time (the posterior predictive SD ``sigma_alpha_post``), averaged over
    cohorts and seeds. The direct 'value of information' readout."""
    return cohort.groupby("day")["sigma_alpha_post"].mean()


def _colours(n: int) -> list:
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.1, 0.9, max(n, 1))]


def _colours_for_labels(labels: list) -> list:
    """Colour per sweep label. Communication settings (BL/CG/SN/CG+SN) get
    their fixed palette colours so a setting keeps one colour across every
    figure; any other sweep (demand, compliance) falls back to the viridis
    ramp in insertion order."""
    keys = [str(x) for x in labels]
    if keys and all(k in COMM_ORDER for k in keys):
        return [comm_colour(k) for k in keys]
    return _colours(len(labels))


def _linestyles_for_labels(labels: list) -> list:
    """Dash pattern per sweep label, so lines stay distinct in greyscale.
    Communication settings get their fixed dash pattern; any other sweep
    (demand, compliance) falls back to the generic index cycle."""
    keys = [str(x) for x in labels]
    if keys and all(k in COMM_ORDER for k in keys):
        return [comm_linestyle(k) for k in keys]
    return [sweep_linestyle(i) for i in range(len(labels))]


def _is_comm_sweep(items) -> bool:
    """True when every sweep key is a communication setting (BL/CG/SN/CG+SN)."""
    return bool(items) and all(str(lab) in COMM_ORDER for lab, _ in items)


def _sweep_label(label) -> str:
    """Display label for a sweep line: full communication name (with the
    abbreviation in parentheses) for BL/CG/SN/CG+SN, else the label as-is."""
    return comm_label(str(label))


def _legend_ncol(items) -> int:
    """Legend columns: 2 for the long full communication names (so they do not
    overflow the figure width), else up to 5 across."""
    return 2 if _is_comm_sweep(items) else min(len(items), 5)


def _daily_green_split(step) -> "object":
    """Day-indexed mean applied green split ``phi_2`` (over tau and seeds)."""
    return step.groupby("day")["phi2"].mean()


# Named panels `plot_sweep_metrics` can compose (ylabel, series-from-result).
_SWEEP_PANELS: dict = {
    "cost": ("system cost [veh-min]", lambda r: _daily_cost(r.step)),
    "share": (r"intersection share $P_\alpha$",
              lambda r: _daily_route_share(r.step)),
    "peak_queue": ("peak queue $L_2+L_5+L_6$ [veh]",
                   lambda r: _daily_peak_total_queue(r.step)),
    "belief_sd": ("belief SD on $TT_\\alpha$ [min]",
                  lambda r: _daily_belief_uncertainty(r.cohort)),
    "phi2": (r"mean green split $\phi_2$",
             lambda r: _daily_green_split(r.step)),
}


def plot_sweep_metrics(
    results_by_label: Mapping[str, object], *, layout: str = "stacked",
    panels: tuple[str, ...] | None = None,
):
    """Overlay panels (one line per variant) of daily sweep metrics.

    The default four panels: daily system cost, daily mean route share, daily
    peak total queue, and daily traveller belief uncertainty over the
    intersection travel time. ``panels`` picks a different composition by name
    from ``{"cost", "share", "peak_queue", "belief_sd", "phi2"}`` (the paper's
    communication figure swaps ``belief_sd`` for the ``phi2`` green-split
    panel, showing how the controller responds to each setting).

    ``results_by_label`` is an ordered mapping ``{label: ExperimentResult}``;
    insertion order sets the line order (colour follows the palette: fixed
    per-setting colours for BL/CG/SN/CG+SN, else a viridis ramp).

    ``layout``: ``"stacked"`` (default) is the tall Nx1 column used by the
    demand / compliance sweeps; ``"grid"`` is the two-column arrangement Xue
    asks for in the information-communication experiment.
    """
    items = list(results_by_label.items())
    colours = _colours_for_labels([lab for lab, _ in items])
    styles = _linestyles_for_labels([lab for lab, _ in items])
    lw = active_style().line_main

    if panels is None:
        panels = ("cost", "share", "peak_queue", "belief_sd")
    unknown = [p for p in panels if p not in _SWEEP_PANELS]
    if unknown:
        raise ValueError(
            f"unknown sweep panels {unknown}; known: {sorted(_SWEEP_PANELS)}")
    chosen = [_SWEEP_PANELS[p] for p in panels]

    if layout == "grid":
        nrows = int(np.ceil(len(chosen) / 2))
        fig, axgrid = plt.subplots(
            nrows, 2, figsize=(text_w(), text_w() * 0.425 * nrows),
            squeeze=False,
        )
        axes = axgrid.ravel()
        for ax in axes[len(chosen):]:
            ax.set_visible(False)
        xlabel_axes = axes[max(len(chosen) - 2, 0):len(chosen)]  # bottom row
        rect_top = 0.90
    else:
        fig, axes = plt.subplots(
            len(chosen), 1, figsize=(text_w(), text_w() * 0.475 * len(chosen)),
            sharex=True, squeeze=False,
        )
        axes = axes.ravel()
        xlabel_axes = [axes[-1]]
        rect_top = 0.965

    for ax, (ylabel, fn) in zip(axes, chosen):
        for (label, res), colour, ls in zip(items, colours, styles):
            s = fn(res)
            ax.plot(s.index.to_numpy(), s.to_numpy(),
                    color=colour, linewidth=lw, linestyle=ls,
                    label=_sweep_label(label))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in xlabel_axes:
        ax.set_xlabel("day")
    light_borders(axes)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=_legend_ncol(items), frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, rect_top))
    return fig


def plot_route_choice_heatmaps(
    results_by_label: Mapping[str, object], value: str = "P_alpha",
):
    """One heatmap column per sweep variant: the intersection-route share
    ``P_alpha`` over (day x time-of-day), on a shared colour scale.

    Companion to :func:`plot_sweep_metrics` for the information-communication
    experiment (variants = BL/CG/SN/CG+SN): the day-series overlay shows *what*
    each setting converges to, this shows *when within the day* travellers pick
    the intersection and how that pattern settles across the learning days.
    ``results_by_label`` is an ordered mapping ``{label: ExperimentResult}``;
    insertion order sets the column order.
    """
    items = list(results_by_label.items())
    n = len(items)
    # A short (about half-height) row of heatmaps: the day x time-of-day maps are
    # legible without the tall aspect that wasted vertical space in the paper.
    fig, axes = plt.subplots(
        1, n, figsize=(text_w(), text_w() * 0.30), sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    pivots = []
    for _label, res in items:
        step = res.step
        sd = (
            step if "seed" not in step.columns
            else step[step["seed"] == step["seed"].min()]
        )
        pivots.append(
            sd.pivot_table(index="tau", columns="day", values=value, aggfunc="mean")
        )
    # P_alpha is a share in [0, 1]; pin the scale so columns are comparable.
    vmin, vmax = (0.0, 1.0) if value == "P_alpha" else (
        0.0, float(max(p.values.max() for p in pivots))
    )

    im = None
    for ax, (label, _res), hm in zip(axes, items, pivots):
        taus = hm.index.to_numpy(dtype=float)
        days = hm.columns.to_numpy(dtype=float)
        im = ax.pcolormesh(_edges(days), _edges(taus), hm.values,
                           cmap="magma", vmin=vmin, vmax=vmax, shading="flat",
                           edgecolors="face", linewidth=0.0,
                           rasterized=True)
        # Abbreviation-only titles (BL/CG/SN/CG+SN): the paper caption carries
        # the expansions, and the narrow heatmap columns cannot fit full names.
        ax.set_title(str(label), fontsize=8)
        ax.set_xlabel("day")
    axes[0].set_ylabel("time of day [min]")
    label_txt = {
        "P_alpha": r"intersection share $P_\alpha$",
        "L2": r"queue $L_2$ [veh]", "L6": r"queue $L_6$ [veh]",
    }.get(value, value)
    fig.colorbar(im, ax=list(axes), pad=0.02, label=label_txt, fraction=0.046)
    return fig


def plot_belief_sd_sweep(results_by_label: Mapping[str, object]):
    """Belief uncertainty per sweep variant, two stacked panels.

    * top: the **traveller** posterior SD on the intersection-route travel time
      ``TT_alpha`` (``sigma_alpha_post``), averaged over travellers;
    * bottom: the **controller** posterior SD on the total system travel time
      ``TT^tot``, read from its system-cost belief SD ``SC_belief_sd`` [veh-min]
      (the system cost is the total travel time in veh-min).

    One line per variant (e.g. BL/CG/SN/CG+SN, palette colours). The direct
    "value of information" readout of the communication experiment: route
    congestion relayed to travellers should shrink the top curve, while signal
    information should sharpen the controller's total-travel-time belief below.
    A variant whose result lacks a series (missing column, or a controller that
    records no cost belief) is skipped for that panel.
    """
    items = list(results_by_label.items())
    colours = _colours_for_labels([lab for lab, _ in items])
    styles = _linestyles_for_labels([lab for lab, _ in items])
    lw = active_style().line_main

    def _trav_alpha_sd(res):
        cohort = res.cohort
        if "sigma_alpha_post" not in cohort.columns:
            return None
        return cohort.groupby("day")["sigma_alpha_post"].mean()

    def _ctrl_tot_sd(res):
        ctrl = getattr(res, "controller", None)
        cols = getattr(ctrl, "columns", [])
        if ctrl is None or "SC_belief_sd" not in cols:
            return None
        if "seed" in cols:
            ctrl = ctrl[ctrl["seed"] == ctrl["seed"].min()]
        ctrl = ctrl.sort_values("day")
        if not ctrl["SC_belief_sd"].notna().any():
            return None
        return ctrl.set_index("day")["SC_belief_sd"]

    panels = [
        (r"traveller SD on $TT_\alpha$ [min]", _trav_alpha_sd),
        (r"controller SD on queue delay [veh-min]", _ctrl_tot_sd),
    ]
    # Side-by-side (1x2) so the two SD panels are roughly square rather than
    # two stacked full-width strips (paper Figure 6 redesign). Each panel has
    # its own "day" axis since they sit next to one another.
    fig, axes = plt.subplots(
        1, len(panels), figsize=(text_w(), text_w() * 0.44),
    )
    # Slightly enlarged fonts (Xue's review: the panel read a little flat).
    for ax, (ylabel, fn) in zip(axes, panels):
        for (label, res), colour, ls in zip(items, colours, styles):
            s = fn(res)
            if s is None:
                continue
            ax.plot(s.index.to_numpy(), s.to_numpy(),
                    color=colour, linewidth=lw, linestyle=ls,
                    label=_sweep_label(label))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("day", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    light_borders(axes)

    # Shared legend outside, above both panels (Xue's review: (a) and (b) use
    # the same variants, so one legend above the pair replaces the former
    # in-panel legend). Two columns keep the full communication names from
    # overflowing the figure width.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=_legend_ncol(items), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig


def plot_communication_cost(results_by_label: Mapping[str, object],
                            *, n_last: int = 30):
    """Total system cost per information-communication setting: trend + summary.

    Two panels, so the day-to-day behaviour and the steady-state numbers sit
    side by side (the paper's Figure 8, replacing the placeholder bar chart):

    * (a) the daily total system cost over the learning horizon, one line per
      setting (BL/CG/SN/CG+SN, palette colours + dash patterns); the
      steady-state window (the last ``n_last`` days that the summary averages) is
      shaded.
    * (b) a bar chart of the post-convergence mean daily cost per setting with a
      +/- 1 SD error bar, the mean and SD annotated on each bar; a dashed
      reference line marks the baseline (BL) mean so the settings that beat it
      (SN) and those that do not (CG) read at a glance.

    ``results_by_label`` is an ordered mapping ``{setting: ExperimentResult}``.
    """
    items = list(results_by_label.items())
    labels = [str(lab) for lab, _ in items]
    colours = _colours_for_labels(labels)
    styles = _linestyles_for_labels(labels)
    lw = active_style().line_main

    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(text_w(), text_w() * 0.42))

    # (a) daily cost trend, with the steady-state window shaded.
    last_day = first_day = None
    for (label, res), colour, ls in zip(items, colours, styles):
        cost = _daily_cost(res.step)
        ax_t.plot(cost.index.to_numpy(), cost.to_numpy(), color=colour,
                  linewidth=lw, linestyle=ls, label=_sweep_label(label))
        d_max, d_min = float(cost.index.max()), float(cost.index.min())
        last_day = d_max if last_day is None else max(last_day, d_max)
        first_day = d_min if first_day is None else min(first_day, d_min)
    if last_day is not None:
        # Clamp the window to the data so a short run does not extend the axis.
        span0 = max(first_day, last_day - n_last + 1)
        ax_t.axvspan(span0, last_day, color="0.5", alpha=0.10,
                     linewidth=0, zorder=0)
    ax_t.set_xlabel("day")
    ax_t.set_ylabel("system cost [veh-min]")
    ax_t.grid(alpha=0.25)
    panel_label(ax_t, "a")

    # (b) post-convergence mean +/- 1 SD per setting.
    means, sds = [], []
    for _label, res in items:
        sc = _daily_cost(res.step).iloc[-int(n_last):]
        means.append(float(sc.mean()))
        sds.append(float(sc.std()))
    x = np.arange(len(items))
    ax_b.bar(x, means, yerr=sds, color=colours, edgecolor="#333333",
             linewidth=0.5, width=0.66, capsize=3,
             error_kw=dict(elinewidth=0.9, ecolor="#333333"))
    # Baseline reference so "beats / does not beat BL" reads directly.
    if "BL" in labels:
        bl_mean = means[labels.index("BL")]
        ax_b.axhline(bl_mean, color="0.35", linewidth=0.8, linestyle="--",
                     zorder=1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("steady-state cost [veh-min]")
    ax_b.grid(alpha=0.25, axis="y")
    top = max(m + s for m, s in zip(means, sds)) if means else 1.0
    ax_b.set_ylim(0, top * 1.20)
    for xi, m, s in zip(x, means, sds):
        ax_b.annotate(f"{m:,.0f}\n(±{s:,.0f})", (xi, m + s), ha="center",
                      va="bottom", fontsize=6.6, xytext=(0, 2),
                      textcoords="offset points")
    panel_label(ax_b, "b")

    # Shared legend outside, above both panels, with the full communication
    # names (abbreviation in parentheses). Panel (b)'s x-axis keeps the short
    # abbreviations, so the legend is the only place the names are spelled out.
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=_legend_ncol(items), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return fig
