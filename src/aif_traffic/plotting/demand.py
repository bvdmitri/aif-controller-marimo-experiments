"""Two-stream shifted-sine demand plot (A--B and C--D)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..demand import DemandProfile
from ..parameters import Params
from .palette import route_colour
from .primitives import text_w


def plot_demand_profile(params: Params):
    profile = DemandProfile.from_params(params.sim, params.demand)
    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 3.5 / 5.0))
    # Distinct colour AND dash per stream so the two demands stay apart in
    # greyscale print (A--B reuses the intersection-route hue, C--D the C--D hue).
    ax.plot(params.sim.time, profile.d_AB, label=r"$D_{AB}(t)$",
            linewidth=1.5, color=route_colour("alpha"), linestyle="-")
    ax.plot(params.sim.time, profile.d_CD, label=r"$D_{CD}(t)$",
            linewidth=1.5, color=route_colour("gamma"), linestyle="--")
    ax.set_xlabel("time of day [min]")
    ax.set_ylabel("demand [veh/h]")
    ax.set_title("Shifted-sine demand profiles")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
