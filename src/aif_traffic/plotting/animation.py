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

from .primitives import TEXT_W


def animate_days(
    step: pd.DataFrame,
    out_path: str | Path,
    *,
    seed: int | None = None,
    fps: int = 4,
) -> Path:
    """Write a gif with one frame per day (within-day queues + green split).

    Returns the path written. Axis limits are fixed across frames so the
    day-to-day evolution is visually comparable.
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
        ax_q.legend(loc="upper right")
        ax_q.grid(alpha=0.25)

        ax_phi.plot(d["tau"], d["phi2"], color="tab:blue", label=r"$\phi_2$")
        ax_phi.plot(d["tau"], d["phi6"], color="tab:orange", label=r"$\phi_6$")
        ax_phi.set_ylim(0, phi_max * 1.05)
        ax_phi.set_xlim(0, tau_max)
        ax_phi.set_xlabel("time of day [min]")
        ax_phi.set_ylabel("green fraction")
        ax_phi.legend(loc="upper right")
        ax_phi.grid(alpha=0.25)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=days, interval=1000 / max(fps, 1))
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
