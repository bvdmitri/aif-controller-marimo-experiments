"""Two-stream shifted-sine demand plot (A--B and C--D)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..demand import DemandProfile
from ..parameters import Params
from .primitives import text_w


def plot_demand_profile(params: Params):
    profile = DemandProfile.from_params(params.sim, params.demand)
    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 3.5 / 5.0))
    ax.plot(params.sim.time, profile.d_AB, label=r"$D_{AB}(t)$",
            linewidth=1.5, color="tab:blue")
    ax.plot(params.sim.time, profile.d_CD, label=r"$D_{CD}(t)$",
            linewidth=1.5, color="tab:orange")
    ax.set_xlabel("time of day [min]")
    ax.set_ylabel("demand [veh/h]")
    ax.set_title("Shifted-sine demand profiles")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
