"""Network / control diagnostic plots: signalised-link queues, green split,
and the day-to-day route share. Pure ``Figure``-returning helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .primitives import TEXT_W


def plot_signal_day(step: pd.DataFrame, day: int | None = None):
    """Within-day queues on the two signalised links and the green split,
    for a single day (defaults to the last recorded day)."""
    if day is None:
        day = int(step["day"].max())
    d = step[step["day"] == day].sort_values("tau")

    fig, (ax_q, ax_phi) = plt.subplots(
        2, 1, figsize=(TEXT_W, TEXT_W * 0.9), sharex=True,
    )
    ax_q.plot(d["tau"], d["L2"], color="tab:blue", label=r"$L_2$ (A--B)")
    ax_q.plot(d["tau"], d["L6"], color="tab:orange", label=r"$L_6$ (C--D)")
    ax_q.set_ylabel("queue [veh]")
    ax_q.set_title(f"Signalised-link queues and green split (day {day})")
    ax_q.legend()
    ax_q.grid(alpha=0.25)

    ax_phi.plot(d["tau"], d["phi2"], color="tab:blue", label=r"$\phi_2$")
    ax_phi.plot(d["tau"], d["phi6"], color="tab:orange", label=r"$\phi_6$")
    ax_phi.set_xlabel("time of day [min]")
    ax_phi.set_ylabel("green fraction")
    ax_phi.legend()
    ax_phi.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_route_share_over_days(step: pd.DataFrame):
    """Demand-weighted daily share choosing the intersection route alpha."""
    g = step.groupby("day").apply(
        lambda x: (x["Q_alpha"].sum() / max(x["Q_alpha"].sum() + x["Q_beta"].sum(), 1e-9))
    )
    fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 3.5 / 5.0))
    ax.plot(g.index, g.values, color="tab:blue", marker="o", markersize=3)
    ax.set_xlabel("day")
    ax.set_ylabel(r"share on route $\alpha$")
    ax.set_ylim(0, 1)
    ax.set_title("Day-to-day route share (intersection route)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
