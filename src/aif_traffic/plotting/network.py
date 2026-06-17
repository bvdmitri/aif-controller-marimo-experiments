"""Network / control diagnostic plots: signalised-link queues, green split,
and the day-to-day route share. Pure ``Figure``-returning helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .primitives import TEXT_W, TEXT_W_HALF


def _edges(v: np.ndarray) -> np.ndarray:
    """Cell edges around the sample centres ``v`` for ``pcolormesh``."""
    v = np.asarray(v, dtype=float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    dd = np.diff(v)
    return np.concatenate([[v[0] - dd[0] / 2], v[:-1] + dd / 2, [v[-1] + dd[-1] / 2]])


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


def plot_route_flows(step: pd.DataFrame, day: int | None = None):
    """Within-day traveller flow on each route, for a single day.

    Shows the two A--B options -- the intersection route ``alpha`` (link 2) and
    the bypass ``beta`` (link 5) -- and the exogenous C--D stream ``gamma``
    (link 6), with the total A--B demand for reference. When travellers divert
    away from the congested intersection around the demand peak, ``Q_alpha``
    dips while ``Q_beta`` rises, which relieves the intersection queue.
    """
    if day is None:
        day = int(step["day"].max())
    d = step[step["day"] == day].sort_values("tau")
    tau = d["tau"]
    total_ab = d["Q_alpha"] + d["Q_beta"]

    fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 0.55))
    ax.plot(tau, total_ab, color="0.6", ls="--", label="A--B total demand")
    ax.plot(tau, d["Q_alpha"], color="tab:blue",
            label=r"A--B via intersection ($Q_\alpha$, link 2)")
    ax.plot(tau, d["Q_beta"], color="tab:green",
            label=r"A--B via bypass ($Q_\beta$, link 5)")
    ax.plot(tau, d["Q_gamma"], color="tab:orange",
            label=r"C--D ($Q_\gamma$, link 6)")
    ax.set_xlabel("time of day [min]")
    ax.set_ylabel("traveller flow [veh/h]")
    ax.set_title(f"Per-route traveller flow (day {day})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_green_split_heatmap(
    step: pd.DataFrame, value: str = "phi2", *, seed: int | None = None,
):
    """Heatmap of a within-day quantity over (day x time-of-day).

    ``value`` is any per-step column: ``phi2`` (green split, default), or a
    queue such as ``L2`` / ``L6``. The x-axis is the day, the y-axis the
    departure minute, so a column is one day's within-day profile.
    """
    sd = step if seed is None else step[step["seed"] == seed]
    hm = sd.pivot_table(index="tau", columns="day", values=value, aggfunc="mean")
    taus = hm.index.to_numpy(dtype=float)
    days = hm.columns.to_numpy(dtype=float)

    is_phi = value.startswith("phi")
    cmap = "viridis" if is_phi else "magma"
    vmin, vmax = (0.0, None)
    if is_phi:
        vmax = float(sd["phi2"].max() + sd["phi6"].max())  # = phi_sat

    fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 0.62))
    im = ax.pcolormesh(
        _edges(days), _edges(taus), hm.values,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
    )
    ax.set_xlabel("day")
    ax.set_ylabel("time of day [min]")
    label = {"phi2": r"green split $\phi_2$ (A--B)",
             "phi6": r"green split $\phi_6$ (C--D)",
             "L2": r"queue $L_2$ (A--B) [veh]",
             "L6": r"queue $L_6$ (C--D) [veh]"}.get(value, value)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(label)
    ax.set_title(f"Within-day {label} across days")
    fig.tight_layout()
    return fig


def plot_daily_system_cost(step: pd.DataFrame):
    """Daily total system cost (veh-min) over days."""
    g = step.groupby("day")["SC"].first()
    fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 3.5 / 5.0))
    ax.plot(g.index, g.values, color="tab:red", marker="o", markersize=3)
    ax.set_xlabel("day")
    ax.set_ylabel("system cost [veh-min]")
    ax.set_title("Day-to-day system cost")
    ax.grid(alpha=0.25)
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
