"""Animations of the simulation (saved as gif).

One frame per day: the day's within-day queues on the two signalised links and
the green split the controller chose, so the viewer watches the controller and
the junction co-evolve across days. Requires Pillow (``PillowWriter``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

from .network import _network_color_scale, _render_network_axes
from .primitives import TEXT_W, place_legend_above


def animate_days(
    step: pd.DataFrame,
    out_path: str | Path,
    *,
    seed: int | None = None,
    fps: int = 12,
) -> Path:
    """Write a gif with one frame per day (within-day queues + green split).

    Returns the path written. Axis limits are fixed across frames so the
    day-to-day evolution is visually comparable. ``fps`` sets the playback speed
    (frames per second); raise it for a quicker run-through of a long run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = step if seed is None else step[step["seed"] == seed]
    days = sorted(sd["day"].unique())

    tau_max = float(sd["tau"].max())
    q_max = float(max(sd["L2"].max(), sd["L6"].max(), 1.0)) * 1.05
    phi_max = float(sd["phi2"].max() + sd["phi6"].max())  # phi_sat

    fig, (ax_q, ax_phi) = plt.subplots(
        2, 1, figsize=(TEXT_W, TEXT_W * 0.9), sharex=True,
    )

    def draw(day: int) -> None:
        d = sd[sd["day"] == day].sort_values("tau")
        ax_q.clear()
        ax_phi.clear()
        ax_q.plot(d["tau"], d["L2"], color="tab:blue", label=r"$L_2$ (A--B)")
        ax_q.plot(d["tau"], d["L6"], color="tab:orange", label=r"$L_6$ (C--D)")
        ax_q.set_ylim(0, q_max)
        ax_q.set_ylabel("queue [veh]")
        ax_q.set_title(f"AIF controller -- day {day}")
        place_legend_above(ax_q)
        ax_q.grid(alpha=0.25)

        ax_phi.plot(d["tau"], d["phi2"], color="tab:blue", label=r"$\phi_2$")
        ax_phi.plot(d["tau"], d["phi6"], color="tab:orange", label=r"$\phi_6$")
        ax_phi.set_ylim(0, phi_max * 1.05)
        ax_phi.set_xlim(0, tau_max)
        ax_phi.set_xlabel("time of day [min]")
        ax_phi.set_ylabel("green fraction")
        place_legend_above(ax_phi)
        ax_phi.grid(alpha=0.25)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=days, interval=1000 / max(fps, 1))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def animate_route_flows(
    step: pd.DataFrame,
    out_path: str | Path,
    *,
    seed: int | None = None,
    fps: int = 12,
) -> Path:
    """Write a gif with one frame per day of the travellers' per-route flow.

    The day-by-day counterpart of :func:`animate_days` for the *traveller* layer:
    each frame is the within-day route flow of :func:`plot_route_flows` -- the
    A--B total demand and its split into the intersection route ``alpha`` and the
    bypass ``beta``, alongside the exogenous C--D stream ``gamma`` -- so the
    viewer watches the population redistribute between routes across learning
    days. Axis limits are fixed to the run-wide maximum across frames so heights
    are comparable. ``fps`` sets the playback speed. Returns the path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = step if seed is None else step[step["seed"] == seed]
    days = sorted(sd["day"].unique())

    tau_max = float(sd["tau"].max())
    total_all = (sd["Q_alpha"] + sd["Q_beta"]).to_numpy()
    f_max = max(float(np.nanmax([
        total_all,
        sd["Q_alpha"].to_numpy(),
        sd["Q_beta"].to_numpy(),
        sd["Q_gamma"].to_numpy(),
    ])), 1.0) * 1.05

    fig, ax = plt.subplots(figsize=(TEXT_W, TEXT_W * 0.6))

    def draw(day: int) -> None:
        d = sd[sd["day"] == day].sort_values("tau")
        ax.clear()
        ax.plot(d["tau"], d["Q_alpha"] + d["Q_beta"], color="0.6", ls="--",
                label=r"$Q_{AB}$")
        ax.plot(d["tau"], d["Q_alpha"], color="tab:blue", label=r"$Q_\alpha$")
        ax.plot(d["tau"], d["Q_beta"], color="tab:green", label=r"$Q_\beta$")
        ax.plot(d["tau"], d["Q_gamma"], color="tab:orange", label=r"$Q_{CD}$")
        ax.set_ylim(0, f_max)
        ax.set_xlim(0, tau_max)
        ax.set_xlabel("time of day [min]")
        ax.set_ylabel("traveller flow [veh/h]")
        ax.set_title(f"AIF travellers -- day {day}")
        place_legend_above(ax)
        ax.grid(alpha=0.25)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=days, interval=1000 / max(fps, 1))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def animate_network_state(
    step: pd.DataFrame,
    net,
    out_path: str | Path,
    *,
    day: int | None = None,
    color_by: str = "travellers",
    seed: int | None = None,
    shared_scale: bool = True,
    fps: int = 40,
) -> Path:
    """Write a gif of the network state, one frame per time of day within a day.

    Animates the node-edge network of :func:`plot_network_state` across the
    minutes of a single ``day`` (the day chosen by the notebook's inspect-day
    slider, or the last day when ``day is None``), so the viewer watches the
    traveller flow (or queue) wave build up and clear across the junction. The
    colour scale is fixed across frames -- to the run-wide maximum with
    ``shared_scale`` (default) so it is comparable with the other days, or the
    day's own maximum otherwise. ``fps`` sets the playback speed (frames per
    second); the default is brisk because a fine-grained day has many minutes.
    Returns the path written.
    """
    if tuple(net.link_ids) != (1, 2, 3, 4, 5, 6, 7):
        raise ValueError(
            "animate_network_state assumes the default 7-link intersection "
            f"network; got link ids {net.link_ids}."
        )
    if color_by not in ("travellers", "queue"):
        raise ValueError(f"color_by must be 'travellers' or 'queue', got {color_by!r}.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sd = step if (seed is None or "seed" not in step.columns) else step[step["seed"] == seed]
    if "seed" in sd.columns and seed is None:
        sd = sd[sd["seed"] == sd["seed"].min()]
    if day is None:
        day = int(sd["day"].max())
    day_df = sd[sd["day"] == day].sort_values("tau")
    taus = [int(t) for t in day_df["tau"].to_numpy()]

    scale_df = sd if shared_scale else day_df
    cmap, clabel, vmax = _network_color_scale(scale_df, net, color_by)
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    fig, ax = plt.subplots(figsize=(TEXT_W * 1.35, TEXT_W * 0.9))
    # Colour key as a slim horizontal strip below the map (created once, out of
    # the way of the link labels) rather than a cramped vertical side bar.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=clabel, orientation="horizontal",
                 fraction=0.05, pad=0.04)

    def draw(tau: int) -> None:
        row = day_df[day_df["tau"] == tau].iloc[0]
        ax.clear()
        _render_network_axes(ax, row, net, norm, cmap, color_by)
        phi2, phi6 = float(row["phi2"]), float(row["phi6"])
        ax.set_title(
            f"Network state — day {day}, t = {tau} min\n"
            f"green split $\\phi_2$={phi2:.2f}, $\\phi_6$={phi6:.2f}   "
            f"(colour: {color_by})",
            fontsize=8,
        )

    anim = FuncAnimation(fig, draw, frames=taus, interval=1000 / max(fps, 1))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def animate_controller_comparison(
    results_by_ctrl: Mapping[str, pd.DataFrame],
    out_path: str | Path,
    *,
    fps: int = 4,
) -> Path:
    """Write a gif comparing controllers, one frame per day.

    Layout is a 2 x N grid: top row the within-day queues (L2, L6), bottom row
    the green split, one column per controller. Axis limits are shared across
    frames and controllers so the comparison is fair. ``results_by_ctrl`` maps a
    controller label to its ``ExperimentResult`` (or its ``.step`` frame).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps = {
        name: (r.step if hasattr(r, "step") else r)
        for name, r in results_by_ctrl.items()
    }
    names = list(steps)
    n = len(names)

    all_step = pd.concat(steps.values())
    days = sorted(all_step["day"].unique())
    tau_max = float(all_step["tau"].max())
    q_max = float(max(all_step["L2"].max(), all_step["L6"].max(), 1.0)) * 1.05
    phi_max = float(all_step["phi2"].max() + all_step["phi6"].max())

    fig, axes = plt.subplots(
        2, n, figsize=(TEXT_W, TEXT_W * 0.55), sharex=True, squeeze=False,
    )

    def draw(day: int) -> None:
        for j, name in enumerate(names):
            d = steps[name][steps[name]["day"] == day].sort_values("tau")
            ax_q, ax_phi = axes[0][j], axes[1][j]
            ax_q.clear()
            ax_phi.clear()
            ax_q.plot(d["tau"], d["L2"], color="tab:blue", lw=1.0)
            ax_q.plot(d["tau"], d["L6"], color="tab:orange", lw=1.0)
            ax_q.set_ylim(0, q_max)
            ax_q.set_title(name, fontsize=7.5)
            ax_phi.plot(d["tau"], d["phi2"], color="tab:blue", lw=1.0)
            ax_phi.plot(d["tau"], d["phi6"], color="tab:orange", lw=1.0)
            ax_phi.set_ylim(0, phi_max * 1.05)
            ax_phi.set_xlim(0, tau_max)
            ax_phi.set_xlabel("time [min]")
            for ax in (ax_q, ax_phi):
                ax.grid(alpha=0.2)
        axes[0][0].set_ylabel("queue [veh]")
        axes[1][0].set_ylabel("green frac.")
        fig.suptitle(f"day {day}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

    anim = FuncAnimation(fig, draw, frames=days, interval=1000 / max(fps, 1))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
