"""Pure ``Figure``-returning visualisations.

Every plot function returns a ``matplotlib.figure.Figure`` and does NOT call
``plt.show`` or ``plt.savefig`` itself. Notebooks handle display and saving.

Kept intentionally small for now: a demand plot plus signalised-link / route
diagnostics. Plotting grows only as notebooks need it (no speculative code).
"""

from __future__ import annotations

import matplotlib as mpl

from .animation import animate_controller_comparison, animate_days
from .comparison import (
    controller_summary,
    plot_controller_metrics,
    plot_controller_theta_grid,
    plot_green_split_heatmaps_by_controller,
)
from .demand import plot_demand_profile
from .network import (
    plot_daily_system_cost,
    plot_green_split_heatmap,
    plot_network_state,
    plot_queue_belief_day,
    plot_route_flows,
    plot_route_share_over_days,
    plot_signal_day,
)
from .primitives import TEXT_W, TEXT_W_HALF, figure_placeholder
from .sweep import plot_route_choice_heatmaps, plot_sweep_metrics


def setup_style() -> None:
    """Publication-style matplotlib defaults (LLNCS ~4.8 in text width)."""
    mpl.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
        "legend.framealpha": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
    })


__all__ = [
    "setup_style",
    "TEXT_W",
    "TEXT_W_HALF",
    "figure_placeholder",
    "plot_demand_profile",
    "plot_signal_day",
    "plot_queue_belief_day",
    "plot_route_flows",
    "plot_network_state",
    "plot_route_share_over_days",
    "plot_green_split_heatmap",
    "plot_daily_system_cost",
    "animate_days",
    "plot_controller_metrics",
    "plot_green_split_heatmaps_by_controller",
    "plot_controller_theta_grid",
    "controller_summary",
    "animate_controller_comparison",
    "plot_sweep_metrics",
    "plot_route_choice_heatmaps",
]
