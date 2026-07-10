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

from .beliefs import (
    _ROUTE_MU,
    _ROUTE_TEX,
    _ROUTE_TT,
    _belief_profile_by_minute,
    _pick_evolution_days,
    _seed_slice,
)
from .comparison import _daily_cost
from .network import _edges
from .palette import route_colour, route_linestyle, signal_colour
from .primitives import (
    light_borders,
    panel_label,
    text_w,
    within_day_profile_size,
)
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
    c_phi = signal_colour()
    ls_alpha, ls_beta = route_linestyle("alpha"), route_linestyle("beta")

    ncols = max(len(picked), 1)
    # One column per day: multi-day (notebook) renders fill the content width
    # (~2 in per day); a single-day render (the paper's Figure 5 panel) is
    # authored at the shared within-day-profile size so all three panels of the
    # paper row match in aspect and keep their fonts legible.
    if ncols > 1:
        fig_w, fig_h = max(text_w(), 2.1 * ncols), 3.8
    else:
        fig_w, fig_h = within_day_profile_size()
    fig, axes = plt.subplots(
        2, ncols, figsize=(fig_w, fig_h), sharex=True,
        sharey="row", squeeze=False,
    )
    for col, d in enumerate(picked):
        dd = day_step[day_step["day"] == d].sort_values("tau")
        tau = dd["tau"].to_numpy(dtype=float)
        ax_q, ax_phi = axes[0][col], axes[1][col]
        ax_q.plot(tau, dd["Q_alpha"].to_numpy(), color=c_alpha, linewidth=lw,
                  linestyle=ls_alpha)
        ax_q.plot(tau, dd["Q_beta"].to_numpy(), color=c_beta, linewidth=lw,
                  linestyle=ls_beta)
        ax_q.set_title(f"day {int(d)}", fontsize=8)
        ax_q.grid(alpha=0.25)
        ax_phi.plot(tau, dd["phi2"].to_numpy(), color=c_phi, linewidth=lw,
                    zorder=4)
        if has_plan and dd["phi2_plan"].notna().any():
            ax_phi.plot(tau, dd["phi2_plan"].to_numpy(), linestyle="none",
                        marker="o", markersize=2.0, color=c_phi, alpha=0.7,
                        zorder=2)
        ax_phi.set_ylim(0, 1)
        ax_phi.grid(alpha=0.25)
        ax_phi.set_xlabel("time [min]")
    axes[0][0].set_ylabel("route flow [veh/h]")
    axes[1][0].set_ylabel(r"green split $\phi_2$")
    fig.align_ylabels(axes[:, 0])

    handles = [
        Line2D([0], [0], color=c_alpha, lw=2, linestyle=ls_alpha,
               label=r"flow $\alpha$"),
        Line2D([0], [0], color=c_beta, lw=2, linestyle=ls_beta,
               label=r"flow $\beta$"),
        Line2D([0], [0], color=c_phi, lw=2, label=r"realised $\phi_2$"),
    ]
    if has_plan:
        handles.append(Line2D([0], [0], color=c_phi, linestyle="none",
                              marker="o", markersize=3.5, label=r"believed $\phi_2$"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=7, columnspacing=1.4)
    light_borders(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_within_day_profile(
    step_df: pd.DataFrame,
    snapshots: dict,
    params,
    *,
    day: int | None = None,
    seed: int | None = None,
):
    """Combined within-day coupled-learning figure at one representative day.

    A single 2x3 panel (the paper's Figure 5), so the three columns stay aligned
    and equally sized rather than being three separate PDFs stitched by LaTeX:

    * column (a): traveller route flows $Q_\\alpha$ / $Q_\\beta$ (top) and the
      green split $\\phi_2$, realised (line) vs believed (dots) (bottom);
    * column (b): believed (dots) vs realised (line) travel time for route
      $\\alpha$ (top) and $\\beta$ (bottom);
    * column (c): realised queue vs the controller's queue belief ($\\pm$ 1 SD
      band) on the two signalised movements $L_2$ (top) and $L_6$ (bottom).

    Each column carries its own compact legend and an (a)/(b)/(c) label.
    ``day`` selects the inspected day (default the last recorded day); ``seed``
    picks the run when several are present.
    """
    day_step, sample_seed = _seed_slice(step_df, seed)
    all_days = sorted(int(x) for x in day_step["day"].unique())
    d = all_days[-1] if day is None else int(day)
    dd = day_step[day_step["day"] == d].sort_values("tau")
    tau = dd["tau"].to_numpy(dtype=float)
    dt_min = int(params.sim.dt_min)
    # Heavier weight for the main realised series so the (brighter) lines carry
    # the panel; the belief overlays stay a touch lighter to read as reference.
    lw = active_style().line_main + 0.4
    lw_ref = active_style().line_main
    c_alpha, c_beta, c_gamma = (route_colour("alpha"), route_colour("beta"),
                                route_colour("gamma"))
    c_phi = signal_colour()
    ls_alpha, ls_beta = route_linestyle("alpha"), route_linestyle("beta")

    fig, axes = plt.subplots(2, 3, figsize=(text_w(), text_w() * 0.58))

    # -- column (a): route flows (top) + green split realised/believed (bottom).
    ax_flow, ax_phi = axes[0][0], axes[1][0]
    ax_flow.plot(tau, dd["Q_alpha"].to_numpy(), color=c_alpha, linewidth=lw,
                 linestyle=ls_alpha)
    ax_flow.plot(tau, dd["Q_beta"].to_numpy(), color=c_beta, linewidth=lw,
                 linestyle=ls_beta)
    ax_flow.set_ylabel("route flow [veh/h]")
    has_plan = "phi2_plan" in dd.columns and dd["phi2_plan"].notna().any()
    ax_phi.plot(tau, dd["phi2"].to_numpy(), color=c_phi, linewidth=lw, zorder=4)
    if has_plan:
        ax_phi.plot(tau, dd["phi2_plan"].to_numpy(), linestyle="none", marker="o",
                    markersize=2.4, color=c_phi, alpha=0.75, zorder=2)
    ax_phi.set_ylim(0, 1)
    ax_phi.set_ylabel(r"green split $\phi_2$")
    # Column (a) carries two panels (flows / green split), so each gets its own
    # in-panel legend rather than one shared strip.
    handles_flow = [
        Line2D([0], [0], color=c_alpha, lw=lw, linestyle=ls_alpha,
               label=r"flow $\alpha$"),
        Line2D([0], [0], color=c_beta, lw=lw, linestyle=ls_beta,
               label=r"flow $\beta$"),
    ]
    handles_phi = [Line2D([0], [0], color=c_phi, lw=lw, label=r"realised $\phi_2$")]
    if has_plan:
        handles_phi.append(Line2D([0], [0], color=c_phi, linestyle="none",
                                  marker="o", markersize=3.5,
                                  label=r"believed $\phi_2$"))

    # -- column (b): believed (dots) vs realised (line) travel time.
    snap = (snapshots or {}).get((sample_seed, d))
    prof = _belief_profile_by_minute(snap, dt_min) if snap is not None else None
    for row, r in enumerate(("alpha", "beta")):
        axb = axes[row][1]
        rz = day_step.pivot_table(index="day", columns="tau",
                                  values=_ROUTE_TT[r], aggfunc="mean")
        taub = rz.columns.to_numpy(dtype=float)
        axb.plot(taub, rz.loc[d].to_numpy(), color=route_colour(r),
                 linewidth=lw_ref, alpha=0.9, zorder=4)
        if prof is not None:
            axb.plot(prof.index.to_numpy(), prof[_ROUTE_MU[r]].to_numpy(),
                     linestyle="none", marker="o", markersize=1.6,
                     color=route_colour(r), alpha=0.6, zorder=2)
        axb.set_ylabel(f"TT {_ROUTE_TEX[r]} [min]")
    handles_b = [
        Line2D([0], [0], color="grey", linewidth=1.8, label="realised"),
        Line2D([0], [0], color="grey", linestyle="none", marker="o",
               markersize=3.5, label="belief (pred. TT)"),
    ]

    # -- column (c): realised queue vs controller belief on L2 (top) / L6.
    has_ctrl = "L2_belief_mu" in dd.columns and dd["L2_belief_mu"].notna().any()
    for row, (col, colour, bmu, bsd, tex) in enumerate([
        ("L2", c_alpha, "L2_belief_mu", "L2_belief_sd", r"$L_2$"),
        ("L6", c_gamma, "L6_belief_mu", "L6_belief_sd", r"$L_6$"),
    ]):
        axc = axes[row][2]
        axc.plot(tau, dd[col].to_numpy(), color=colour, linewidth=lw_ref,
                 alpha=0.9, zorder=4)
        if has_ctrl and bmu in dd.columns:
            mu = dd[bmu].to_numpy()
            sdv = dd[bsd].to_numpy()
            axc.plot(tau, mu, color="k", linestyle="--", linewidth=1.0, zorder=3)
            axc.fill_between(tau, mu - sdv, mu + sdv, color="k", alpha=0.12,
                             linewidth=0, zorder=1)
        axc.set_ylabel(f"queue {tex} [veh]")
    handles_c = [
        Line2D([0], [0], color="grey", linewidth=1.8, label="realised"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.2,
               label="controller belief"),
    ]

    for row in range(2):
        for cc in range(3):
            axes[row][cc].grid(alpha=0.25)
    for cc in range(3):
        axes[1][cc].set_xlabel("time [min]")
    panel_label(axes[0][0], "a")
    panel_label(axes[0][1], "b")
    panel_label(axes[0][2], "c")
    light_borders(axes)
    # Legends live *inside* their panels (Xue's review) with a light translucent
    # frame so they sit over the data without hiding it. Placement is per-panel:
    # the panel label occupies the top-left corner, so legends avoid it.
    leg_kw = dict(frameon=True, framealpha=0.82, edgecolor="#cccccc",
                  fontsize=6.5, handlelength=1.6, handletextpad=0.4,
                  borderpad=0.35, labelspacing=0.25)
    ax_flow.legend(handles=handles_flow, loc="upper right", **leg_kw)
    ax_phi.legend(handles=handles_phi, loc="upper right", **leg_kw)
    axes[0][1].legend(handles=handles_b, loc="upper right", **leg_kw)
    axes[0][2].legend(handles=handles_c, loc="upper right", **leg_kw)
    fig.tight_layout()
    return fig


def plot_co_adaptation(
    step_df: pd.DataFrame,
    controller_df: pd.DataFrame | None = None,
    *,
    seed: int | None = None,
):
    """Day-to-day co-adaptation of route choice, signal control, and cost.

    A compact grid grouped as the paper's (a)-(c):

    * (a) heatmap of the intersection share ``P_alpha(d, t)`` and (b) heatmap of
      the green split ``phi_2(d, t)``, both over (day x time-of-day);
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

    # Daily summary.
    cost = _daily_cost(sd)

    # Heatmaps (colourbars on top) in row 1, cost full-width in row 2. Both
    # heatmaps share "time of day" on the y-axis, and each carries its own "day"
    # x-axis. The top row is kept short so it does not dominate the figure.
    fig = plt.figure(figsize=(text_w(), text_w() * 0.60), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.3])
    ax_pa = fig.add_subplot(gs[0, 0])
    ax_ph = fig.add_subplot(gs[0, 1], sharex=ax_pa, sharey=ax_pa)
    ax_cost = fig.add_subplot(gs[1, :], sharex=ax_pa)

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
    # Each top heatmap gets its own "day" x-axis (the shared y hides only the
    # right panel's y tick labels).
    ax_ph.tick_params(labelleft=False)
    ax_pa.set_xlabel("day")
    ax_ph.set_xlabel("day")
    panel_label(ax_pa, "a")
    panel_label(ax_ph, "b")

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
