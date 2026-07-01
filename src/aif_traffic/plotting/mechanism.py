"""Mechanism-analysis composites for Experiment 1 (single AIF run).

These stitch the atomic within-day / day-to-day series into the multi-panel
figures the paper's baseline-mechanism section calls for: the coupled
within-day decision process, the day-to-day co-adaptation summary, and the
learning-uncertainty appendix figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .beliefs import _pick_evolution_days, _seed_slice
from .comparison import _daily_cost
from .network import _edges
from .palette import route_colour
from .primitives import TEXT_W, light_borders
from .style import active_style


def plot_coupled_within_day(
    step_df: pd.DataFrame,
    params=None,
    *,
    n_days: int = 4,
    seed: int | None = None,
):
    """The controller half of the coupled within-day picture, one column per day.

    A grid with **one column per representative day** and two rows, so each day
    is read on its own axes rather than overlaid:

    * top -- per-route traveller flow: intersection ``alpha`` (blue) and bypass
      ``beta`` (green) [veh/h];
    * bottom -- the controller's green split ``phi_2``: **realised** (solid,
      what it applied reacting to the day) vs **planned/believed** (dots, what
      it would apply from its typical-day belief alone), when recorded.

    Read together with the traveller belief-vs-realised travel-time figure, this
    is the "coupled within-day decision process" of Xue's baseline figure.
    """
    day_step, _ = _seed_slice(step_df, seed)
    all_days = sorted(day_step["day"].unique())
    picked = _pick_evolution_days(all_days, n_days)
    lw = active_style().line_main
    has_plan = "phi2_plan" in day_step.columns and day_step["phi2_plan"].notna().any()
    c_alpha, c_beta = route_colour("alpha"), route_colour("beta")

    ncols = max(len(picked), 1)
    # One column per day: scale the width to fill the notebook content width
    # (~2 in per day) instead of the narrow paper text width.
    fig, axes = plt.subplots(
        2, ncols, figsize=(max(TEXT_W, 2.1 * ncols), 3.8), sharex=True,
        sharey="row", squeeze=False,
    )
    for col, d in enumerate(picked):
        dd = day_step[day_step["day"] == d].sort_values("tau")
        tau = dd["tau"].to_numpy(dtype=float)
        ax_q, ax_phi = axes[0][col], axes[1][col]
        ax_q.plot(tau, dd["Q_alpha"].to_numpy(), color=c_alpha, linewidth=lw)
        ax_q.plot(tau, dd["Q_beta"].to_numpy(), color=c_beta, linewidth=lw)
        ax_q.set_title(f"day {int(d)}", fontsize=8)
        ax_q.grid(alpha=0.25)
        ax_phi.plot(tau, dd["phi2"].to_numpy(), color="0.25", linewidth=lw,
                    zorder=4)
        if has_plan and dd["phi2_plan"].notna().any():
            ax_phi.plot(tau, dd["phi2_plan"].to_numpy(), linestyle="none",
                        marker="o", markersize=2.0, color="0.25", alpha=0.7,
                        zorder=2)
        ax_phi.set_ylim(0, 1)
        ax_phi.grid(alpha=0.25)
        ax_phi.set_xlabel("time [min]")
    axes[0][0].set_ylabel("route flow [veh/h]")
    axes[1][0].set_ylabel(r"green split $\phi_2$")
    fig.align_ylabels(axes[:, 0])

    handles = [
        Line2D([0], [0], color=c_alpha, lw=2, label=r"flow $\alpha$"),
        Line2D([0], [0], color=c_beta, lw=2, label=r"flow $\beta$"),
        Line2D([0], [0], color="0.25", lw=2, label=r"realised $\phi_2$"),
    ]
    if has_plan:
        handles.append(Line2D([0], [0], color="0.25", linestyle="none",
                              marker="o", markersize=3.5, label=r"believed $\phi_2$"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=7, columnspacing=1.4)
    light_borders(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_co_adaptation(step_df: pd.DataFrame, *, seed: int | None = None):
    """Day-to-day co-adaptation of route choice, signal control, and cost.

    Three stacked panels sharing the day axis:

    * top -- heatmap of the intersection share ``P_alpha(d, t)`` over
      (day x time-of-day);
    * middle -- heatmap of the green split ``phi_2(d, t)``;
    * bottom -- daily demand-weighted ``P_alpha`` and mean ``phi_2`` on the left
      axis and total system cost on the right axis.

    Shows how decentralised route choice and signal control co-evolve and how
    that drives system-level performance (Xue's Figure 3).
    """
    sd, _ = _seed_slice(step_df, seed)

    pa = sd.pivot_table(index="tau", columns="day", values="P_alpha", aggfunc="mean")
    ph = sd.pivot_table(index="tau", columns="day", values="phi2", aggfunc="mean")
    taus = pa.index.to_numpy(dtype=float)
    days = pa.columns.to_numpy(dtype=float)

    # Daily summaries.
    daily_pa = sd.groupby("day")["P_alpha"].mean()
    daily_phi = sd.groupby("day")["phi2"].mean()
    cost = _daily_cost(sd)

    fig, axes = plt.subplots(
        3, 1, figsize=(TEXT_W, TEXT_W * 1.35), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 2.2], "hspace": 0.22},
    )
    im0 = axes[0].pcolormesh(_edges(days), _edges(taus), pa.values,
                             cmap="magma", vmin=0.0, vmax=1.0, shading="flat")
    axes[0].set_ylabel("time of day")
    fig.colorbar(im0, ax=axes[0], pad=0.015, fraction=0.046,
                 label=r"$P_\alpha$")
    im1 = axes[1].pcolormesh(_edges(days), _edges(taus), ph.values,
                             cmap="viridis", vmin=0.0, vmax=1.0, shading="flat")
    axes[1].set_ylabel("time of day")
    fig.colorbar(im1, ax=axes[1], pad=0.015, fraction=0.046,
                 label=r"$\phi_2$")

    ax = axes[2]
    ax2 = ax.twinx()
    l1, = ax.plot(daily_pa.index.to_numpy(), daily_pa.to_numpy(),
                  color=route_colour("alpha"), linewidth=1.4, label=r"daily $P_\alpha$")
    l2, = ax.plot(daily_phi.index.to_numpy(), daily_phi.to_numpy(),
                  color="grey", linewidth=1.4, label=r"daily $\phi_2$")
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$P_\alpha$, $\phi_2$")
    ax.set_xlabel("day")
    ax.grid(alpha=0.25)
    l3, = ax2.plot(cost.index.to_numpy(), cost.to_numpy(),
                   color="tab:red", linewidth=1.4, label="system cost")
    ax2.set_ylabel("system cost [veh-min]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.legend(handles=[l1, l2, l3], labels=[h.get_label() for h in (l1, l2, l3)],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, fontsize=7)
    # A twin axis is present in the bottom panel, so avoid tight_layout's warning.
    fig.subplots_adjust(left=0.11, right=0.9, top=0.95, bottom=0.08)
    return fig


def plot_learning_uncertainty(
    cohort_df: pd.DataFrame,
    controller_df: pd.DataFrame | None = None,
    *,
    seed: int | None = None,
):
    """How the two agent layers' uncertainty shrinks over days (appendix).

    Two panels sharing the day axis:

    * top -- **traveller** posterior SD on the route travel time, ``TT_alpha``
      (blue) and ``TT_beta`` (green) [min];
    * bottom -- **controller** uncertainty: its learned queue observation-noise
      SD per movement (``sigma_obs`` L_2 / L_6) and, when recorded, its belief
      SD on the daily queue-delay (proxy for system cost) on a right axis.

    Supports the mechanism analysis without overloading the main text.
    """
    cd = cohort_df
    if "seed" in cd.columns:
        cd = cd[cd["seed"] == cd["seed"].min()]
    trav = cd.groupby("day")[["sigma_alpha_post", "sigma_beta_post"]].mean()

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W, TEXT_W * 0.9), sharex=True)
    lw = active_style().line_main
    axes[0].plot(trav.index.to_numpy(), trav["sigma_alpha_post"].to_numpy(),
                 color=route_colour("alpha"), linewidth=lw, label=r"$TT_\alpha$")
    axes[0].plot(trav.index.to_numpy(), trav["sigma_beta_post"].to_numpy(),
                 color=route_colour("beta"), linewidth=lw, label=r"$TT_\beta$")
    axes[0].set_ylabel("traveller belief SD [min]")
    axes[0].set_title("a. Traveller travel-time uncertainty", fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)

    ax = axes[1]
    used_twin = False
    if controller_df is not None and not controller_df.empty:
        ctrl = controller_df
        if "seed" in ctrl.columns:
            ctrl = ctrl[ctrl["seed"] == ctrl["seed"].min()]
        ctrl = ctrl.sort_values("day")
        drew = False
        if {"sigma_obs_l2", "sigma_obs_l6"}.issubset(ctrl.columns):
            ax.plot(ctrl["day"].to_numpy(), ctrl["sigma_obs_l2"].to_numpy(),
                    color=route_colour("alpha"), linewidth=lw,
                    label=r"$\sigma_{obs}\,L_2$")
            ax.plot(ctrl["day"].to_numpy(), ctrl["sigma_obs_l6"].to_numpy(),
                    color=route_colour("gamma"), linewidth=lw,
                    label=r"$\sigma_{obs}\,L_6$")
            drew = True
        ax.set_ylabel("controller obs-noise SD [veh]")
        if "SC_belief_sd" in ctrl.columns and ctrl["SC_belief_sd"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(ctrl["day"].to_numpy(), ctrl["SC_belief_sd"].to_numpy(),
                     color="tab:red", linewidth=lw, linestyle="--",
                     label="cost-belief SD")
            ax2.set_ylabel("cost-belief SD [veh-min]", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            used_twin = True
        if drew:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("b. Controller uncertainty", fontsize=8)
    ax.set_xlabel("day")
    ax.grid(alpha=0.25)
    light_borders(axes)
    # tight_layout warns with a twin axis present; fall back to a manual adjust.
    if used_twin:
        fig.subplots_adjust(hspace=0.28, left=0.13, right=0.87, top=0.93, bottom=0.12)
    else:
        fig.tight_layout()
    return fig
