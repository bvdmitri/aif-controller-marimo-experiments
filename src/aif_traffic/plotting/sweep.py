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
from .primitives import TEXT_W, light_borders


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


def plot_sweep_metrics(results_by_label: Mapping[str, object]):
    """Four stacked overlay panels (one line per variant): daily system cost,
    daily mean route share, daily peak total queue, and daily traveller belief
    uncertainty over the intersection travel time.

    ``results_by_label`` is an ordered mapping ``{label: ExperimentResult}``;
    insertion order sets the line order and colour ramp.
    """
    items = list(results_by_label.items())
    colours = _colours(len(items))

    panels = [
        ("system cost [veh-min]", lambda r: _daily_cost(r.step)),
        (r"intersection share $P_\alpha$", lambda r: _daily_route_share(r.step)),
        ("peak queue $L_2+L_6$ [veh]", lambda r: _daily_peak_total_queue(r.step)),
        ("belief SD on $TT_\\alpha$ [min]", lambda r: _daily_belief_uncertainty(r.cohort)),
    ]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(TEXT_W, TEXT_W * 1.9), sharex=True,
    )
    for ax, (ylabel, fn) in zip(axes, panels):
        for (label, res), colour in zip(items, colours):
            s = fn(res)
            ax.plot(s.index.to_numpy(), s.to_numpy(),
                    color=colour, linewidth=1.5, label=str(label))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("day")
    light_borders(axes)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=min(len(items), 5), frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig
