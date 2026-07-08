"""Robustness (varying-demand) overlays for Experiment 5.

The demand-robustness sweep runs the coupled AIF system at several traffic-demand
scales and overlays one coloured line per scale, so how travellers and the
controller re-coordinate under heavier / lighter load can be read within a day
and across the learning days. Each helper takes an ordered mapping
``{scale_label: ExperimentResult}``. Pure ``Figure``-returning.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .beliefs import _seed_slice
from .comparison import _daily_cost
from .primitives import light_borders, text_w
from .style import active_style


def _demand_colours(n: int) -> list:
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.1, 0.9, max(n, 1))]


def plot_within_day_by_demand(
    results_by_scale: Mapping[str, object], *, day: int | None = None,
    seed: int | None = None,
):
    """Within-day coordinated adaptation under different traffic demand.

    Three panels, one coloured line per demand scale (viridis ramp): (a) the
    intersection-route flow ``Q_alpha`` [veh/h], (b) the bypass-route flow
    ``Q_beta`` [veh/h], and (c) the controller's green split ``phi_2``, all at a
    representative day (``day``, default the last recorded day). Reads how, as the
    load grows, more travellers divert to the bypass at the peak while the
    controller re-allocates green time.
    """
    items = list(results_by_scale.items())
    colours = _demand_colours(len(items))
    lw = active_style().line_main
    panels = [
        (r"route flow $Q_\alpha$ [veh/h]", "Q_alpha"),
        (r"route flow $Q_\beta$ [veh/h]", "Q_beta"),
        (r"green split $\phi_2$", "phi2"),
    ]

    fig, axgrid = plt.subplots(1, 3, figsize=(text_w(), text_w() * 0.34),
                               squeeze=False)
    axes = axgrid[0]
    for (label, res), colour in zip(items, colours):
        day_step, _ = _seed_slice(res.step, seed)
        d_use = int(day_step["day"].max()) if day is None else int(day)
        dd = day_step[day_step["day"] == d_use].sort_values("tau")
        tau = dd["tau"].to_numpy(dtype=float)
        for ax, (_ylabel, col) in zip(axes, panels):
            ax.plot(tau, dd[col].to_numpy(), color=colour, linewidth=lw,
                    label=str(label))
    for ax, (ylabel, _col) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time of day [min]")
        ax.grid(alpha=0.25)
    axes[2].set_ylim(0, 1)
    light_borders(axgrid)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=max(len(labels), 1), frameon=False,
               bbox_to_anchor=(0.5, 1.05), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_across_day_by_demand(results_by_scale: Mapping[str, object]):
    """Across-day learning under different traffic demand.

    Three panels, one coloured line per demand scale (viridis ramp): (a) the
    daily intersection-route share ``P_alpha``, (b) the daily mean green split
    ``phi_2``, and (c) the daily total system cost on the left axis with the
    controller's cost-belief SD ``SC_belief_sd`` (dashed, matching colour) on a
    shared right axis. Reads whether the coupled system re-settles to a stable
    route split / signal policy at each load and how its cost and the
    controller's uncertainty co-evolve.
    """
    items = list(results_by_scale.items())
    colours = _demand_colours(len(items))
    lw = active_style().line_main

    fig, axgrid = plt.subplots(1, 3, figsize=(text_w(), text_w() * 0.34),
                               squeeze=False)
    ax_pa, ax_phi, ax_cost = axgrid[0]

    for (label, res), colour in zip(items, colours):
        step = res.step
        pa = step.groupby("day")["P_alpha"].mean()
        phi = step.groupby("day")["phi2"].mean()
        cost = _daily_cost(step)
        ax_pa.plot(pa.index.to_numpy(), pa.to_numpy(), color=colour,
                   linewidth=lw, label=str(label))
        ax_phi.plot(phi.index.to_numpy(), phi.to_numpy(), color=colour,
                    linewidth=lw)
        ax_cost.plot(cost.index.to_numpy(), cost.to_numpy(), color=colour,
                     linewidth=lw)

    ax_pa.set_ylabel(r"route share $P_\alpha$")
    ax_pa.set_ylim(0, 1)
    ax_phi.set_ylabel(r"green split $\phi_2$")
    ax_phi.set_ylim(0, 1)
    ax_cost.set_ylabel("system cost [veh-min]")

    # Controller uncertainty on a shared right axis of the cost panel: one dashed
    # line per demand scale (matching colour). Skipped for a scale whose result
    # records no cost belief (e.g. a non-AIF controller).
    ax_sd = ax_cost.twinx()
    drew_sd = False
    for (label, res), colour in zip(items, colours):
        ctrl = getattr(res, "controller", None)
        cols = getattr(ctrl, "columns", [])
        if ctrl is None or "SC_belief_sd" not in cols:
            continue
        c = ctrl[ctrl["seed"] == ctrl["seed"].min()] if "seed" in cols else ctrl
        c = c.sort_values("day")
        if not c["SC_belief_sd"].notna().any():
            continue
        ax_sd.plot(c["day"].to_numpy(), c["SC_belief_sd"].to_numpy(),
                   color=colour, linewidth=1.0, linestyle="--")
        drew_sd = True
    if drew_sd:
        ax_sd.set_ylabel("cost-belief SD [veh-min]", color="0.3")
        ax_sd.tick_params(axis="y", labelcolor="0.3")
    else:
        ax_sd.set_visible(False)

    for ax in (ax_pa, ax_phi, ax_cost):
        ax.set_xlabel("day")
        ax.grid(alpha=0.25)
    light_borders([ax_pa, ax_phi, ax_cost])

    handles, _labels = ax_pa.get_legend_handles_labels()
    if drew_sd:
        handles = handles + [
            Line2D([0], [0], color="0.3", linewidth=lw, label="system cost"),
            Line2D([0], [0], color="0.3", linewidth=1.0, linestyle="--",
                   label="controller SD"),
        ]
    labels = [h.get_label() for h in handles]
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=min(len(handles), 6), frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig
