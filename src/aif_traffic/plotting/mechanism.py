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
from matplotlib.ticker import MaxNLocator

from .beliefs import _pick_evolution_days, _seed_slice
from .comparison import _daily_cost
from .network import _edges
from .palette import route_colour
from .primitives import light_borders, panel_label, text_w, text_w_half
from .style import active_style


def plot_coupled_within_day(
    step_df: pd.DataFrame,
    params=None,
    *,
    n_days: int = 4,
    days=None,
    seed: int | None = None,
):
    """The controller half of the coupled within-day picture, one column per day.

    A grid with **one column per representative day** and two rows, so each day
    is read on its own axes rather than overlaid:

    * top: per-route traveller flow: intersection ``alpha`` (blue) and bypass
      ``beta`` (green) [veh/h];
    * bottom: the controller's green split ``phi_2``: **realised** (solid,
      what it applied reacting to the day) vs **planned/believed** (dots, what
      it would apply from its typical-day belief alone), when recorded.

    ``days`` (an explicit iterable of day numbers) overrides the automatic
    first/last/evenly-spaced pick of ``n_days`` days; the paper uses a single
    representative day (``days=[80]``).

    Read together with the traveller belief-vs-realised travel-time figure, this
    is the "coupled within-day decision process" of Xue's baseline figure.
    """
    day_step, _ = _seed_slice(step_df, seed)
    all_days = sorted(day_step["day"].unique())
    picked = ([d for d in days if d in all_days] if days is not None
              else _pick_evolution_days(all_days, n_days))
    lw = active_style().line_main
    has_plan = "phi2_plan" in day_step.columns and day_step["phi2_plan"].notna().any()
    c_alpha, c_beta = route_colour("alpha"), route_colour("beta")

    ncols = max(len(picked), 1)
    # One column per day: scale the width to fill the notebook content width
    # (~2 in per day) instead of the narrow paper text width. A single-day
    # render (the paper's profile figure) is authored at the half width so it
    # can sit beside its companion panel without shrinking the fonts.
    fig_w = max(text_w(), 2.1 * ncols) if ncols > 1 else text_w_half()
    fig, axes = plt.subplots(
        2, ncols, figsize=(fig_w, 3.8), sharex=True,
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


def plot_co_adaptation(
    step_df: pd.DataFrame,
    controller_df: pd.DataFrame | None = None,
    *,
    seed: int | None = None,
):
    """Day-to-day co-adaptation of route choice, signal control, and cost.

    A compact grid grouped as the paper's (a)-(c):

    * (a) side-by-side heatmaps of the intersection share ``P_alpha(d, t)`` and
      the green split ``phi_2(d, t)`` over (day x time-of-day);
    * (b) the daily profiles side by side: demand-weighted daily ``P_alpha``
      and the controller's daily mean ``phi_2``;
    * (c) total system cost (full width), with the **controller's cost-belief
      SD** (red dashed, right axis) overlaid when ``controller_df`` records it, so
      the controller's shrinking uncertainty can be read against the settling
      cost.

    Shows how decentralised route choice and signal control co-evolve and how
    that drives system-level performance (Xue's day-to-day adaptation figure).
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

    # Heatmaps (colourbars on top) in row 1, daily profiles in row 2, cost
    # full-width in row 3. Axes are shared so labels/scales are not repeated:
    # both heatmaps share "time of day", both daily panels share the 0-1 scale,
    # and the day axis is shared down each column (drawn once, on the daily row).
    fig = plt.figure(figsize=(text_w(), text_w() * 0.95), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.1, 1.05, 1.35])
    ax_pa = fig.add_subplot(gs[0, 0])
    ax_ph = fig.add_subplot(gs[0, 1], sharex=ax_pa, sharey=ax_pa)
    ax_dpa = fig.add_subplot(gs[1, 0], sharex=ax_pa)
    ax_dph = fig.add_subplot(gs[1, 1], sharex=ax_pa, sharey=ax_dpa)
    ax_cost = fig.add_subplot(gs[2, :], sharex=ax_pa)

    im0 = ax_pa.pcolormesh(_edges(days), _edges(taus), pa.values,
                           cmap="magma", vmin=0.0, vmax=1.0, shading="flat")
    ax_pa.set_ylabel("time of day")
    im1 = ax_ph.pcolormesh(_edges(days), _edges(taus), ph.values,
                           cmap="viridis", vmin=0.0, vmax=1.0, shading="flat")
    # Colourbars sit on top of each heatmap and span its full width: an inset
    # axis pinned to the axes x-extent (axes-fraction coords) makes the bar
    # exactly the chart width, unlike fig.colorbar's auto-stolen strip.
    for ax_hm, im, lab in ((ax_pa, im0, r"$P_\alpha$"), (ax_ph, im1, r"$\phi_2$")):
        cax = ax_hm.inset_axes([0.0, 1.03, 1.0, 0.05])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cb.set_label(lab)
    # More y-ticks so the "time of day" axis is readable (drives ax_ph via
    # shared y); default gave essentially one tick.
    ax_pa.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # Day axis is shared with the daily row below. Draw it there only.
    ax_pa.tick_params(labelbottom=False)
    ax_ph.tick_params(labelbottom=False, labelleft=False)
    panel_label(ax_pa, "a")

    ax_dpa.plot(daily_pa.index.to_numpy(), daily_pa.to_numpy(),
                color=route_colour("alpha"), linewidth=1.4)
    ax_dpa.set_ylim(0, 1)
    ax_dpa.set_ylabel("daily mean")
    ax_dpa.set_xlabel("day")
    # More y-ticks on the daily row (drives ax_dph via shared y).
    ax_dpa.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_dpa.grid(alpha=0.25)
    ax_dpa.text(0.04, 0.86, r"$P_\alpha$", transform=ax_dpa.transAxes, fontsize=9)
    panel_label(ax_dpa, "b")
    ax_dph.plot(daily_phi.index.to_numpy(), daily_phi.to_numpy(),
                color="0.25", linewidth=1.4)
    ax_dph.set_xlabel("day")
    ax_dph.grid(alpha=0.25)
    ax_dph.tick_params(labelleft=False)
    ax_dph.text(0.04, 0.86, r"$\phi_2$", transform=ax_dph.transAxes, fontsize=9)

    ax = ax_cost
    handles = []
    l_cost, = ax.plot(cost.index.to_numpy(), cost.to_numpy(),
                      color="k", linewidth=1.4, label="system cost")
    handles.append(l_cost)
    ax.set_ylabel("system cost\n[veh-min]")
    ax.set_xlabel("day")
    ax.grid(alpha=0.25)
    panel_label(ax, "c")
    ctrl = controller_df
    if ctrl is not None and "SC_belief_sd" in getattr(ctrl, "columns", []) \
            and ctrl["SC_belief_sd"].notna().any():
        if "seed" in ctrl.columns:
            ctrl = ctrl[ctrl["seed"] == ctrl["seed"].min()]
        ctrl = ctrl.sort_values("day")
        ax2 = ax.twinx()
        l_sd, = ax2.plot(ctrl["day"].to_numpy(), ctrl["SC_belief_sd"].to_numpy(),
                         color="tab:red", linewidth=1.2, linestyle="--",
                         label="controller cost-belief SD")
        ax2.set_ylabel("cost-belief SD\n[veh-min]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        handles.append(l_sd)
    ax.legend(handles=handles, labels=[h.get_label() for h in handles],
              loc="upper right", frameon=False, fontsize=7)
    return fig


def plot_learning_uncertainty(
    cohort_df: pd.DataFrame,
    controller_df: pd.DataFrame | None = None,
    *,
    seed: int | None = None,
):
    """How the two agent layers' uncertainty shrinks over days (appendix).

    Two panels sharing the day axis:

    * top: **traveller** posterior SD on the route travel time, ``TT_alpha``
      (blue) and ``TT_beta`` (green) [min];
    * bottom: **controller** uncertainty: its learned queue observation-noise
      SD per movement (``sigma_obs`` L_2 / L_6). (The controller's cost-belief
      SD now lives in the system-cost panel of :func:`plot_co_adaptation`.)

    Supports the mechanism analysis without overloading the main text.
    """
    cd = cohort_df
    if "seed" in cd.columns:
        cd = cd[cd["seed"] == cd["seed"].min()]
    trav = cd.groupby("day")[["sigma_alpha_post", "sigma_beta_post"]].mean()

    fig, axes = plt.subplots(2, 1, figsize=(text_w(), text_w() * 0.9), sharex=True)
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
    if controller_df is not None and not controller_df.empty:
        ctrl = controller_df
        if "seed" in ctrl.columns:
            ctrl = ctrl[ctrl["seed"] == ctrl["seed"].min()]
        ctrl = ctrl.sort_values("day")
        if {"sigma_obs_l2", "sigma_obs_l6"}.issubset(ctrl.columns):
            ax.plot(ctrl["day"].to_numpy(), ctrl["sigma_obs_l2"].to_numpy(),
                    color=route_colour("alpha"), linewidth=lw,
                    label=r"$\sigma_{obs}\,L_2$")
            ax.plot(ctrl["day"].to_numpy(), ctrl["sigma_obs_l6"].to_numpy(),
                    color=route_colour("gamma"), linewidth=lw,
                    label=r"$\sigma_{obs}\,L_6$")
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_ylabel("controller obs-noise SD [veh]")
    ax.set_title("b. Controller uncertainty", fontsize=8)
    ax.set_xlabel("day")
    ax.grid(alpha=0.25)
    light_borders(axes)
    fig.tight_layout()
    return fig


def plot_msc_tt_by_route(step_df: pd.DataFrame, *, seed: int | None = None):
    """Per-route cost decomposition over days: travel time, marginal social
    cost, and the congestion externality.

    Three stacked day-series panels, one line per traveller route (``alpha``
    intersection, ``beta`` bypass):

    * top: daily mean travel time ``TT_r``;
    * middle: daily mean marginal social cost ``MSC_r`` (the finite-difference
      cost of one extra vehicle, recorded while the externality advisory is
      broadcast);
    * bottom: the raw externality ``E_r = MSC_r - TT_r`` (may be negative
      off-peak; the broadcast clips it at zero, this shows the unclipped value).

    Where the two routes' curves coincide, user equilibrium and system optimum
    coincide too, and ``theta`` (which reweights ``E_r`` in the perceived cost)
    has no lever. Requires the ``MSC_alpha``/``MSC_beta`` step columns (present
    when the EXTERNALITY / MSC advisory is on).
    """
    sd, _ = _seed_slice(step_df, seed)
    if "MSC_alpha" not in sd.columns:
        raise ValueError(
            "step frame has no MSC columns; run with the EXTERNALITY or MSC "
            "advisory broadcast so the marginal social cost is computed."
        )
    daily = sd.groupby("day")[
        ["TT_alpha", "TT_beta", "MSC_alpha", "MSC_beta"]
    ].mean()
    days = daily.index.to_numpy()
    lw = active_style().line_main
    colours = {"alpha": route_colour("alpha"), "beta": route_colour("beta")}

    panels = [
        ("mean travel time [min]", {"alpha": "TT_alpha", "beta": "TT_beta"}),
        ("mean MSC [veh-min]", {"alpha": "MSC_alpha", "beta": "MSC_beta"}),
        ("mean externality $E_r$", None),  # derived below
    ]
    fig, axes = plt.subplots(3, 1, figsize=(text_w(), text_w() * 0.95), sharex=True)
    for ax, (ylabel, cols) in zip(axes, panels):
        for r in ("alpha", "beta"):
            if cols is not None:
                y = daily[cols[r]].to_numpy()
            else:
                y = (daily[f"MSC_{r}"] - daily[f"TT_{r}"]).to_numpy()
            ax.plot(days, y, color=colours[r], linewidth=lw,
                    label=rf"$\{r}$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[2].axhline(0.0, color="0.5", linewidth=0.6, zorder=0)
    axes[-1].set_xlabel("day")
    light_borders(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
