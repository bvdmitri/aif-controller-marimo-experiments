"""Belief-vs-reality figures.

Adapted from the IWAI baseline-learning plots (routes A/B -> alpha/beta here):
within-day travel-time and queue beliefs overlaid on the realised profiles, so
one can see whether the agents' learned representation matches what actually
happens. These consume the per-agent ``snapshots`` recorded on ``snapshot_days``
plus the ``step`` DataFrame.

Route mapping for the queue figure: the A--B intersection route ``alpha``
traverses the signalised link ``L_2``; the A--B bypass ``beta`` traverses the
unsignalised link ``L_5``. The exogenous C--D stream (link ``L_6``) is *not* a
learning cohort, so there is no traveller queue belief for ``L_6``; only the
controller believes it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .palette import COMM_ORDER, comm_colour, comm_label, route_colour
from .primitives import light_borders, text_w, text_w_half, panel_label
from .style import active_style

# Realised link that stands in for each traveller route's queue.
_ROUTE_QUEUE_LINK = {"alpha": "L2", "beta": "L5"}
_ROUTE_TT = {"alpha": "TT_alpha", "beta": "TT_beta"}
_ROUTE_MU = {"alpha": "mu_alpha", "beta": "mu_beta"}
_ROUTE_TEX = {"alpha": r"$\alpha$", "beta": r"$\beta$"}
# ``last_choice`` code of each traveller route (alpha = intersection, beta = bypass).
_ROUTE_CHOICE = {"alpha": 0, "beta": 1}


def _seed_slice(step_df: pd.DataFrame, seed: int | None) -> tuple[pd.DataFrame, int]:
    if "seed" not in step_df.columns:
        return step_df, (seed if seed is not None else 0)
    s = int(sorted(step_df["seed"].unique())[0]) if seed is None else int(seed)
    return step_df[step_df["seed"] == s], s


def _pick_evolution_days(all_days: list, n_days: int) -> list:
    """First + last + evenly spaced learning days (deterministic)."""
    n_snap = min(int(n_days), len(all_days))
    sel = np.unique(
        np.round(np.linspace(0, len(all_days) - 1, num=n_snap)).astype(int)
    )
    return [all_days[k] for k in sel]


def _belief_profile_by_minute(snap: dict, dt_min: int) -> pd.DataFrame:
    """Cross-agent mean predictive-TT profile vs within-day departure minute.

    Each agent holds a posterior-predictive travel time (``mu_alpha``/
    ``mu_beta``) and a fixed ``departure_time`` (in time-step buckets).
    Averaging the agents that share a departure minute turns the per-agent
    beliefs on a day into a within-day *profile* comparable to the realised
    ``TT(tau)`` curve."""
    df = pd.DataFrame({
        "minute": np.asarray(snap["departure_time"]) * int(dt_min),
        "mu_alpha": np.asarray(snap["mu_alpha"]),
        "mu_beta": np.asarray(snap["mu_beta"]),
    })
    return (
        df.groupby("minute", as_index=True)[["mu_alpha", "mu_beta"]]
        .mean()
        .sort_index()
    )


def plot_within_day_tt_vs_belief(
    step_df: pd.DataFrame,
    snapshots: dict,
    params,
    *,
    n_days: int = 4,
    days=None,
    seed: int | None = None,
):
    """Realised within-day travel time vs the travellers' mean predictive-TT
    belief, both on the within-day departure-minute axis.

    A grid with **one column per representative day** and two rows (route alpha
    top, route beta bottom), so each day is read on its own axes rather than
    overlaid. In each panel the realised ``TT(tau)`` is drawn as-is (the raw,
    genuinely-stochastic within-day series, not averaged) and the belief is
    **dots**: the cross-agent mean posterior-predictive TT at each departure
    minute. Reading a row left-to-right shows the belief dots settling onto the
    realised as the agents learn.

    ``days`` (an explicit iterable of day numbers) overrides the automatic
    first/last/evenly-spaced pick of ``n_days`` days; the paper uses a single
    representative day (``days=[80]``).

    Requires per-agent ``snapshots`` on the plotted days (pass ``snapshot_days``
    to :func:`run_experiment`). Days without a snapshot show the realised line
    only.
    """
    day_step, sample_seed = _seed_slice(step_df, seed)
    all_days = sorted(day_step["day"].unique())
    picked_days = ([d for d in days if d in all_days] if days is not None
                   else _pick_evolution_days(all_days, n_days))
    dt_min = int(params.sim.dt_min)

    routes = ("alpha", "beta")
    colours = {r: route_colour(r) for r in routes}

    # Per route: the raw realised TT per (day, tau), shown as-is, NOT averaged
    # (the realised series is genuinely stochastic).
    realised = {}
    for r in routes:
        realised[r] = day_step.pivot_table(
            index="day", columns="tau", values=_ROUTE_TT[r], aggfunc="mean",
        ).sort_index()

    ncols = max(len(picked_days), 1)
    # One column per day: scale the width so the columns fill the notebook's
    # content width (~2 in per day) rather than being squeezed into the narrow
    # paper text width. A single-day render (the paper's profile figure) is
    # authored at the half width instead, so it can sit beside its companion
    # panel without shrinking the fonts.
    fig_w = max(text_w(), 2.1 * ncols) if ncols > 1 else text_w_half()
    fig, axes = plt.subplots(
        2, ncols, figsize=(fig_w, 3.8), sharex=True,
        sharey="row", squeeze=False,
    )
    for col, d in enumerate(picked_days):
        snap = snapshots.get((sample_seed, int(d))) if snapshots else None
        prof = _belief_profile_by_minute(snap, dt_min) if snap is not None else None
        for row, r in enumerate(routes):
            ax = axes[row][col]
            rz = realised[r]
            tau = rz.columns.to_numpy(dtype=float)
            # Raw realised at moderate weight: visible as a line, but light
            # enough that the belief dots still read against it (at the coarser
            # default time step there are far fewer points, so it is not busy).
            ax.plot(tau, rz.loc[d].to_numpy(), color=colours[r],
                    linewidth=0.95, alpha=0.8, zorder=4)
            if prof is not None:
                # Per-departure belief points: small dots reading as a cloud
                # settling onto the realised line.
                ax.plot(prof.index.to_numpy(), prof[_ROUTE_MU[r]].to_numpy(),
                        linestyle="none", marker="o", markersize=1.6,
                        color=colours[r], alpha=0.6, zorder=2)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f"day {int(d)}", fontsize=8)
    axes[0][0].set_ylabel(f"TT {_ROUTE_TEX['alpha']} [min]")
    axes[1][0].set_ylabel(f"TT {_ROUTE_TEX['beta']} [min]")
    for ax in axes[1]:
        ax.set_xlabel("time [min]")
    fig.align_ylabels(axes[:, 0])

    handles = [
        Line2D([0], [0], color="grey", linewidth=1.8, label="realised"),
        Line2D([0], [0], color="grey", linestyle="none", marker="o",
               markersize=3.5, label="belief (pred. TT)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=7, columnspacing=1.4)
    light_borders(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_within_day_by_setting(
    results_by_label,
    params,
    *,
    route: str = "alpha",
    n_days: int = 3,
    seed: int | None = None,
):
    """Within-day realised travel time vs belief, faceted by sweep setting.

    One panel per setting (e.g. BL / CG / SN / CG+SN), laid out 2xN. In each
    panel the realised ``TT(tau)`` for the given ``route`` is a line and the
    travellers' mean predictive-TT belief is dots, with ``n_days`` learning days
    overlaid as a shade gradient. This is Xue's "what the controller believes vs
    what actually happened" comparison across the communication settings.

    ``results_by_label`` is an ordered mapping ``{label: ExperimentResult}``;
    each result must carry per-agent ``snapshots`` (pass ``snapshot_days``).
    """
    items = list(results_by_label.items())
    n = len(items)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    dt_min = int(params.sim.dt_min)
    cmap = {"alpha": plt.cm.Blues, "beta": plt.cm.Greens}[route]
    tt_col, mu_col = _ROUTE_TT[route], _ROUTE_MU[route]

    fig, axgrid = plt.subplots(nrows, ncols, figsize=(text_w(), text_w() * 0.42 * nrows),
                               sharex=True, sharey=True, squeeze=False)
    axes = axgrid.ravel()
    picked_ref = None
    for ax, (label, res) in zip(axes, items):
        day_step, sample_seed = _seed_slice(res.step, seed)
        all_days = sorted(day_step["day"].unique())
        picked = _pick_evolution_days(all_days, n_days)
        picked_ref = picked
        shade = np.linspace(0.40, 1.0, len(picked))
        pivot = day_step.pivot_table(index="day", columns="tau", values=tt_col,
                                     aggfunc="mean").sort_index()
        tau = pivot.columns.to_numpy(dtype=float)
        snapshots = res.snapshots or {}
        for k, d in enumerate(picked):
            # Raw realised (not averaged over days), at moderate weight so the
            # overlaid days stay legible without forming a hairball.
            ax.plot(tau, pivot.loc[d].to_numpy(), color=cmap(shade[k]),
                    linewidth=1.0, alpha=0.85, zorder=4)
            snap = snapshots.get((sample_seed, int(d)))
            if snap is not None:
                prof = _belief_profile_by_minute(snap, dt_min)
                ax.plot(prof.index.to_numpy(), prof[mu_col].to_numpy(),
                        linestyle="none", marker="o", markersize=1.6,
                        color=cmap(shade[k]), alpha=0.6, zorder=2)
        ax.set_title(str(label), fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axgrid[-1]:
        ax.set_xlabel("within-day time [min]")
    for ax in axgrid[:, 0]:
        ax.set_ylabel("travel time [min]")

    handles = [
        Line2D([0], [0], color="grey", linewidth=1.8, label="realised"),
        Line2D([0], [0], color="grey", linestyle="none", marker="o",
               markersize=3.5, label="belief (pred. TT)"),
    ]
    if picked_ref is not None:
        handles += [
            Line2D([0], [0], color=cmap(np.linspace(0.4, 1.0, len(picked_ref))[k]),
                   linewidth=1.8, label=f"day {int(d)}")
            for k, d in enumerate(picked_ref)
        ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=6.5, columnspacing=1.2)
    light_borders(axgrid)
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _route_queue_belief_profile(
    snap: dict, route: str, dt_min: int, delay_min: int, tau_max: float,
):
    """Per-departure-minute profile of a traveller route's *queue* belief.

    Each A--B traveller holds a single scalar queue belief (``L_mean_alpha`` /
    ``L_mean_beta``) per route (the smoother latent is fixed within the day) and
    a fixed ``departure_time``. We keep only agents who actually *took* the
    given ``route``, their belief is first-hand for that route's queue link
    (alpha -> L_2, beta -> L_5), and place each at the within-day minute where
    they meet that queue: the **arrival** minute ``departure*dt_min +
    delay_min``. Averaging the agents in each minute bucket turns the population
    of per-agent scalars into a within-day *profile*, directly comparable to the
    realised queue curve, and its across-agent spread shows the heterogeneity
    between early- and peak-departers.

    Returns ``(mean_series, std_series)`` indexed by arrival minute, or ``None``
    if the snapshot lacks the fields / no agent took the route.
    """
    mean_key = f"L_mean_{route}"
    needed = {mean_key, "departure_time", "last_choice"}
    if not needed.issubset(snap):
        return None
    took = np.asarray(snap["last_choice"]) == _ROUTE_CHOICE[route]
    if not took.any():
        return None
    minute = (
        np.asarray(snap["departure_time"])[took] * dt_min + delay_min
    ).astype(float)
    minute = np.minimum(minute, tau_max)
    # Clamp each agent's queue belief at 0: the linear-Gaussian latent can go
    # slightly negative for near-empty (off-peak) queues, which is unphysical --
    # a negative belief just means "expects an empty queue".
    L = np.maximum(np.asarray(snap[mean_key])[took], 0.0)
    prof = pd.DataFrame({"minute": minute, "L": L})
    g = prof.groupby("minute")["L"]
    return g.mean(), g.std().fillna(0.0)


def _route_link_delay_min(params, route: str) -> int:
    """Minutes from departure until the traveller meets the route's queue link."""
    try:
        dt_min = int(params.sim.dt_min)
        link_id = int(_ROUTE_QUEUE_LINK[route][1:])
        return int(params.network.n_delay(dt_min)[link_id]) * dt_min
    except Exception:
        return 0


def plot_belief_reality_queues(
    step_df: pd.DataFrame,
    snapshots: dict,
    params,
    *,
    day: int | None = None,
    days: list[int] | None = None,
    links: tuple[str, ...] = ("L2", "L5", "L6"),
    show_traveller_belief: bool = True,
    seed: int | None = None,
):
    """Belief--reality consistency for the route-carrying queues, one column per
    inspected day so the day-to-day learning evolution is visible.

    Rows are the queues in ``links`` (default all three: ``L_2`` A--B
    intersection, ``L_5`` A--B bypass, ``L_6`` C--D); columns are the days in
    ``days`` (default: the first, middle, and last recorded day). Each panel
    shows the realised within-day queue (solid); the signalised movements
    ``L_2``/``L_6`` also show the **controller's** queue belief (dashed mean
    +/-1 sigma band). The controller holds no belief over the unsignalised
    bypass. On ``L_2``/``L_5`` the **traveller queue belief** of the route
    traversing the link is drawn as a per-departure-minute profile (dots +
    across-agent spread band), so early- vs peak-departers holding different
    beliefs stays visible. Rows share a y-axis (days directly comparable) and the
    x-axis is shared throughout.

    Answers "do both agent types learn a consistent representation of the network
    state, and how does it sharpen over days?". Requires per-agent ``snapshots``
    on each inspected day (pass ``snapshot_days`` to :func:`run_experiment`).
    ``day=`` (singular) renders a single column, for the notebooks' interactive
    day selector. The paper's within-day panel (c) passes ``links=("L2","L6")``
    and ``show_traveller_belief=False`` to show only the realised queue vs the
    controller's belief on the two signalised movements (the controller half of
    the coupled picture, complementing the traveller travel-time panel).
    """
    day_step, sample_seed = _seed_slice(step_df, seed)
    all_days = sorted(int(x) for x in day_step["day"].unique())
    if days is None:
        if day is not None:
            days = [int(day)]
        elif len(all_days) >= 3:
            days = [all_days[0], all_days[len(all_days) // 2], all_days[-1]]
        else:
            days = all_days
    days = [int(d) for d in days]
    ncol = max(len(days), 1)
    dt_min = int(params.sim.dt_min)
    tau_max = float(day_step["tau"].max())
    lw = active_style().line_main

    all_specs = [
        ("L2", r"$L_2$ (A--B int.)", route_colour("alpha"), "L2_belief_mu",
         "L2_belief_sd", "alpha"),
        ("L5", r"$L_5$ (A--B byp.)", route_colour("beta"), None, None, "beta"),
        ("L6", r"$L_6$ (C--D)", route_colour("gamma"), "L6_belief_mu",
         "L6_belief_sd", None),
    ]
    specs = [s for s in all_specs if s[0] in links]
    nrow = max(len(specs), 1)

    # Paper's within-day-profile panel (c): a single day and the two signalised
    # movements L2/L6 -> a tall, narrow 2-row figure sized like the companion
    # panels (a)/(b) so the three subfigures line up in one row.
    if ncol == 1 and nrow == 2:
        fig_w, height = text_w_half(), 3.8
    else:
        fig_w, height = text_w(), text_w() * (0.95 if ncol == 1 else 0.8)
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(fig_w, height),
                               sharex=True, sharey="row", squeeze=False)
    for ci, dsel in enumerate(days):
        d = day_step[day_step["day"] == dsel].sort_values("tau")
        tau = d["tau"].to_numpy(dtype=float)
        has_ctrl_belief = "L2_belief_mu" in d.columns and d["L2_belief_mu"].notna().any()
        trav: dict[str, tuple] = {}
        snap = (snapshots or {}).get((sample_seed, int(dsel)))
        if show_traveller_belief and snap is not None:
            for route in ("alpha", "beta"):
                prof = _route_queue_belief_profile(
                    snap, route, dt_min, _route_link_delay_min(params, route), tau_max,
                )
                if prof is not None:
                    trav[route] = prof
        for ri, (col, label, colour, bmu, bsd, route) in enumerate(specs):
            ax = axgrid[ri][ci]
            # The raw per-interval realised queue at moderate weight: a visible
            # line that still sits under the smoother beliefs.
            ax.plot(tau, d[col].to_numpy(), color=colour, linewidth=0.95,
                    alpha=0.8, label="realised", zorder=4)
            if has_ctrl_belief and bmu is not None and bmu in d.columns:
                mu = d[bmu].to_numpy()
                sdv = d[bsd].to_numpy()
                ax.plot(tau, mu, color="k", linestyle="--", linewidth=1.0,
                        label="controller belief", zorder=3)
                ax.fill_between(tau, mu - sdv, mu + sdv, color="k", alpha=0.12,
                                linewidth=0, zorder=1)
            if route is not None and route in trav:
                mu_prof, sd_prof = trav[route]
                x = mu_prof.index.to_numpy(dtype=float)
                ax.plot(x, mu_prof.to_numpy(), color=colour, linestyle=":",
                        linewidth=1.2, marker="o", markersize=2.0,
                        label="traveller belief (by departure)", zorder=5)
                lower = np.maximum((mu_prof - sd_prof).to_numpy(), 0.0)
                ax.fill_between(x, lower, (mu_prof + sd_prof).to_numpy(),
                                color=colour, alpha=0.15, linewidth=0, zorder=0)
            ax.grid(alpha=0.25)
            if ri == 0:
                ax.set_title(f"day {dsel}")
            if ci == 0:
                ax.set_ylabel(f"queue {label} [veh]")
    # An (almost) empty row (typical for the high-capacity bypass) would get a
    # 1e-9 offset scale; pin a sane floor (shared across the row via sharey).
    for ri in range(nrow):
        if axgrid[ri][0].get_ylim()[1] < 1.0:
            axgrid[ri][0].set_ylim(-0.05, 1.0)
    for ax in axgrid[-1]:
        ax.set_xlabel("within-day time [min]")

    # Legend labels repeat across panels; keep one handle per unique label.
    by_label: dict[str, object] = {}
    for ax in axgrid.ravel():
        hs, ls = ax.get_legend_handles_labels()
        for h, lab in zip(hs, ls):
            by_label.setdefault(lab, h)
    fig.legend(handles=list(by_label.values()), labels=list(by_label.keys()),
               loc="upper center", ncol=len(by_label), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    light_borders(axgrid)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_within_day_communication(
    results_by_label,
    params,
    *,
    day: int | None = None,
    seed: int | None = None,
):
    """Within-day realised-vs-belief profiles overlaid across the communication
    settings, for one inspected ``day`` (defaults to the last).

    A 4x2 grid, one line per setting (BL/CG/SN/CG+SN palette colours). The two
    columns are **realised** (left) and **belief** (right); the four rows are the
    quantities: route-``alpha`` travel time (via ``L_2``), route-``beta`` travel
    time (via ``L_5``), and the signalised queues ``L_2`` and ``L_6``. The travel-
    time belief is the travellers' mean predictive-TT per departure minute; the
    queue belief is the controller's queue-belief mean (travellers hold no belief
    over the exogenous C--D movement). Each row shares its y-axis between the
    realised and belief columns (and the x-axis is shared throughout), so the
    axes are not repeated between panels and the panels themselves are larger.
    Requires per-agent ``snapshots`` on the inspected day (pass ``snapshot_days``
    to :func:`run_experiment`).
    """
    items = list(results_by_label.items())
    labels = [str(lab) for lab, _ in items]
    if labels and all(lab in COMM_ORDER for lab in labels):
        colours = {lab: comm_colour(lab) for lab in labels}
    else:
        cmap = plt.get_cmap("viridis")
        xs = np.linspace(0.1, 0.9, max(len(labels), 1))
        colours = {lab: cmap(x) for lab, x in zip(labels, xs)}
    dt_min = int(params.sim.dt_min)
    lw = active_style().line_main

    first_step, _ = _seed_slice(items[0][1].step, seed)
    d_use = int(first_step["day"].max()) if day is None else int(day)

    # Rows: (realised column, belief accessor, is-TT-belief-a-per-minute-profile,
    # y-label). Columns: 0 = realised, 1 = belief.
    fig, axes = plt.subplots(
        4, 2, figsize=(text_w(), text_w() * 0.9), sharex=True, sharey="row",
    )
    # The raw per-minute realised series (~300 points) is dense and noisy, so
    # draw it thin and semi-transparent; it reads as a light noise cloud while
    # the smoother belief stays legible on top.
    raw_lw, raw_alpha = 0.9, 0.8
    row_specs = [
        ("TT_alpha", "mu_alpha", True, r"$TT_\alpha$ [min]"),
        ("TT_beta", "mu_beta", True, r"$TT_\beta$ [min]"),
        ("L2", "L2_belief_mu", False, r"queue $L_2$ [veh]"),
        ("L6", "L6_belief_mu", False, r"queue $L_6$ [veh]"),
    ]
    for lab, res in items:
        day_step, sample_seed = _seed_slice(res.step, seed)
        dd = day_step[day_step["day"] == d_use].sort_values("tau")
        if dd.empty:
            continue
        tau = dd["tau"].to_numpy(dtype=float)
        c = colours[str(lab)]
        snap = (res.snapshots or {}).get((sample_seed, d_use))
        prof = _belief_profile_by_minute(snap, dt_min) if snap is not None else None
        for r, (real_col, bel_col, from_prof, _ylab) in enumerate(row_specs):
            axes[r][0].plot(tau, dd[real_col].to_numpy(), color=c,
                            linewidth=raw_lw, alpha=raw_alpha,
                            label=comm_label(str(lab)))
            if from_prof:
                if prof is not None:
                    axes[r][1].plot(prof.index.to_numpy(),
                                    prof[bel_col].to_numpy(), color=c,
                                    linewidth=raw_lw, alpha=raw_alpha)
            elif bel_col in dd.columns and dd[bel_col].notna().any():
                axes[r][1].plot(tau, dd[bel_col].to_numpy(), color=c,
                                linewidth=raw_lw, alpha=raw_alpha)

    axes[0][0].set_title("realised", fontsize=8)
    axes[0][1].set_title("belief", fontsize=8)
    for r, (_, _, _, ylab) in enumerate(row_specs):
        axes[r][0].set_ylabel(ylab)
    for ax in axes[-1]:
        ax.set_xlabel("time [min]")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.align_ylabels(axes[:, 0])

    # Full-weight legend swatches (the plotted realised lines are deliberately
    # faint, which would otherwise make the legend hard to read).
    handles = [Line2D([0], [0], color=colours[str(lab)], linewidth=1.8,
                      label=comm_label(str(lab))) for lab, _ in items]
    fig.legend(handles=handles, loc="upper center",
               ncol=min(len(items), 2), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.suptitle(f"day {d_use}", fontsize=8, y=0.955)
    light_borders(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
