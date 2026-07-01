"""Generic sweep-overlay plots: one coloured line per experiment variant.

Used by the social-internalisation sweep (Experiment 1, variants = theta
values) and the information-communication sweep (Experiment 3, variants =
BL/CG/SN/CG+SN). Each helper takes an ordered mapping ``{label: ExperimentResult}``
and overlays a daily series, one line per label. Pure ``Figure``-returning.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .comparison import _daily_cost, _daily_peak_total_queue
from .network import _edges
from .palette import COMM_ORDER, comm_colour
from .primitives import TEXT_W, light_borders
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
    figure; any other sweep (theta, compliance) falls back to the viridis
    ramp in insertion order."""
    keys = [str(x) for x in labels]
    if keys and all(k in COMM_ORDER for k in keys):
        return [comm_colour(k) for k in keys]
    return _colours(len(labels))


def plot_sweep_metrics(
    results_by_label: Mapping[str, object], *, layout: str = "stacked",
):
    """Four overlay panels (one line per variant): daily system cost, daily
    mean route share, daily peak total queue, and daily traveller belief
    uncertainty over the intersection travel time.

    ``results_by_label`` is an ordered mapping ``{label: ExperimentResult}``;
    insertion order sets the line order (colour follows the palette: fixed
    per-setting colours for BL/CG/SN/CG+SN, else a viridis ramp).

    ``layout``: ``"stacked"`` (default) is the tall 4x1 column used by the
    theta / compliance sweeps; ``"grid"`` is the 2x2 arrangement Xue asks for
    in the information-communication experiment.
    """
    items = list(results_by_label.items())
    colours = _colours_for_labels([lab for lab, _ in items])
    lw = active_style().line_main

    panels = [
        ("system cost [veh-min]", lambda r: _daily_cost(r.step)),
        (r"intersection share $P_\alpha$", lambda r: _daily_route_share(r.step)),
        ("peak queue $L_2+L_6$ [veh]", lambda r: _daily_peak_total_queue(r.step)),
        ("belief SD on $TT_\\alpha$ [min]", lambda r: _daily_belief_uncertainty(r.cohort)),
    ]

    if layout == "grid":
        fig, axgrid = plt.subplots(2, 2, figsize=(TEXT_W, TEXT_W * 0.85))
        axes = axgrid.ravel()
        xlabel_axes = axes[2:]  # bottom row
        rect_top = 0.90
    else:
        fig, axes = plt.subplots(
            len(panels), 1, figsize=(TEXT_W, TEXT_W * 1.9), sharex=True,
        )
        xlabel_axes = [axes[-1]]
        rect_top = 0.965

    for ax, (ylabel, fn) in zip(axes, panels):
        for (label, res), colour in zip(items, colours):
            s = fn(res)
            ax.plot(s.index.to_numpy(), s.to_numpy(),
                    color=colour, linewidth=lw, label=str(label))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in xlabel_axes:
        ax.set_xlabel("day")
    light_borders(axes)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=min(len(items), 5), frameon=False,
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
    fig, axes = plt.subplots(
        1, n, figsize=(TEXT_W, TEXT_W * 0.5), sharex=True, sharey=True,
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
                           cmap="magma", vmin=vmin, vmax=vmax, shading="flat")
        ax.set_title(str(label), fontsize=7.5)
        ax.set_xlabel("day")
    axes[0].set_ylabel("time of day [min]")
    label_txt = {
        "P_alpha": r"intersection share $P_\alpha$",
        "L2": r"queue $L_2$ [veh]", "L6": r"queue $L_6$ [veh]",
    }.get(value, value)
    fig.colorbar(im, ax=list(axes), pad=0.02, label=label_txt, fraction=0.046)
    return fig
