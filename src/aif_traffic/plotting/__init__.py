"""Pure ``Figure``-returning visualisations.

Every plot function returns a ``matplotlib.figure.Figure`` and does NOT call
``plt.show`` or ``plt.savefig`` itself. Notebooks handle display and saving.

Kept intentionally small for now: a demand plot plus signalised-link / route
diagnostics. Plotting grows only as notebooks need it (no speculative code).
"""

from __future__ import annotations

from .animation import (
    animate_controller_comparison,
    animate_days,
    animate_network_state,
    animate_route_flows,
)
from .beliefs import (
    plot_belief_reality_queues,
    plot_within_day_by_setting,
    plot_within_day_tt_vs_belief,
)
from .comparison import (
    controller_summary,
    plot_controller_metrics,
    plot_controller_theta_grid,
    plot_green_split_heatmaps_by_controller,
    plot_theta_route_choice,
    plot_theta_summary,
)
from .demand import plot_demand_profile
from .mechanism import (
    plot_co_adaptation,
    plot_coupled_within_day,
    plot_learning_uncertainty,
)
from .network import (
    plot_daily_system_cost,
    plot_day_overview_grid,
    plot_green_split_heatmap,
    plot_learned_obs_noise,
    plot_network_state,
    plot_queue_belief_day,
    plot_route_flows,
    plot_route_share_over_days,
    plot_signal_day,
)
from .primitives import TEXT_W, TEXT_W_HALF, figure_placeholder
from .style import active_style, apply_style
from .sweep import plot_route_choice_heatmaps, plot_sweep_metrics


def setup_style() -> None:
    """Activate the default (marimo) publication style.

    Thin wrapper over :func:`aif_traffic.plotting.style.apply_style` kept for
    the notebooks that call ``setup_style()``. The active style is the single
    switch point for a later marimo<->paper toggle and CI export (see
    :mod:`aif_traffic.plotting.style`)."""
    apply_style("marimo")


__all__ = [
    "setup_style",
    "apply_style",
    "active_style",
    "TEXT_W",
    "TEXT_W_HALF",
    "figure_placeholder",
    "plot_demand_profile",
    "plot_signal_day",
    "plot_day_overview_grid",
    "plot_queue_belief_day",
    "plot_learned_obs_noise",
    "plot_route_flows",
    "plot_network_state",
    "plot_route_share_over_days",
    "plot_green_split_heatmap",
    "plot_daily_system_cost",
    "animate_days",
    "animate_route_flows",
    "animate_network_state",
    "plot_controller_metrics",
    "plot_green_split_heatmaps_by_controller",
    "plot_controller_theta_grid",
    "plot_theta_summary",
    "plot_theta_route_choice",
    "controller_summary",
    "animate_controller_comparison",
    "plot_sweep_metrics",
    "plot_route_choice_heatmaps",
    "plot_within_day_tt_vs_belief",
    "plot_within_day_by_setting",
    "plot_belief_reality_queues",
    "plot_coupled_within_day",
    "plot_co_adaptation",
    "plot_learning_uncertainty",
]
