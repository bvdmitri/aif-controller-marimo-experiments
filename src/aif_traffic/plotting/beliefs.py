"""Belief-vs-reality figures.

Adapted from the IWAI baseline-learning plots (routes A/B -> alpha/beta here):
within-day travel-time and queue beliefs overlaid on the realised profiles, so
one can see whether the agents' learned representation matches what actually
happens. These consume the per-agent ``snapshots`` recorded on ``snapshot_days``
plus the ``step`` DataFrame.

Route mapping for the queue figure: the A--B intersection route ``alpha``
traverses the signalised link ``L_2``; the A--B bypass ``beta`` traverses the
unsignalised link ``L_5``. The exogenous C--D stream (link ``L_6``) is *not* a
learning cohort, so there is no traveller queue belief for ``L_6`` -- only the
controller believes it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .palette import route_colour
from .primitives import TEXT_W, light_borders, panel_label
from .style import active_style

# Realised link that stands in for each traveller route's queue.
_ROUTE_QUEUE_LINK = {"alpha": "L2", "beta": "L5"}
_ROUTE_TT = {"alpha": "TT_alpha", "beta": "TT_beta"}
_ROUTE_MU = {"alpha": "mu_alpha", "beta": "mu_beta"}
_ROUTE_TEX = {"alpha": r"$\alpha$", "beta": r"$\beta$"}


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


def _cohort_window(params) -> int:
    try:
        return int(params.population.cohorts[0].window_size)
    except (AttributeError, IndexError, TypeError):
        return 1


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
    seed: int | None = None,
    truth_smooth_window: int | None = None,
):
    """Realised within-day travel time vs the travellers' mean predictive-TT
    belief, both on the within-day departure-minute axis.

    Two stacked panels (route alpha, route beta). The realised ``TT(tau)`` is a
    line (a centered ``W``-day rolling mean, the windowed quantity the posterior
    actually estimates); the belief is **dots** -- the cross-agent mean
    posterior-predictive TT at each departure minute. ``n_days`` learning days
    are overlaid as a shade gradient (earliest dimmed, last saturated, always
    including the first and last day), so the belief cloud tightening onto the
    realised line reads as the quality of day-to-day learning.

    Requires per-agent ``snapshots`` on the plotted days (pass
    ``snapshot_days`` to :func:`run_experiment`). Days without a snapshot are
    silently skipped for the belief dots.
    """
    day_step, sample_seed = _seed_slice(step_df, seed)
    all_days = sorted(day_step["day"].unique())
    picked_days = _pick_evolution_days(all_days, n_days)
    dt_min = int(params.sim.dt_min)
    w_smooth = (
        int(truth_smooth_window) if truth_smooth_window is not None
        else _cohort_window(params)
    )

    routes = ("alpha", "beta")
    cmaps = {"alpha": plt.cm.Blues, "beta": plt.cm.Greens}
    shade = np.linspace(0.40, 1.0, len(picked_days))
    lw_lo, lw_hi = 1.0, active_style().line_main + 0.6

    # Per route: (day x tau) realised TT smoothed with a centered W-day window.
    smoothed = {}
    for r in routes:
        pivot = day_step.pivot_table(
            index="day", columns="tau", values=_ROUTE_TT[r], aggfunc="mean",
        ).sort_index()
        smoothed[r] = pivot.rolling(w_smooth, center=True, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W, 4.2), sharex=True)
    for row, r in enumerate(routes):
        ax = axes[row]
        sm = smoothed[r]
        tau = sm.columns.to_numpy(dtype=float)
        for k, d in enumerate(picked_days):
            lw = lw_lo + (lw_hi - lw_lo) * (shade[k] - shade[0]) / max(
                shade[-1] - shade[0], 1e-9)
            ax.plot(tau, sm.loc[d].to_numpy(), color=cmaps[r](shade[k]),
                    linewidth=lw, zorder=4)
            snap = snapshots.get((sample_seed, int(d)))
            if snap is not None:
                prof = _belief_profile_by_minute(snap, dt_min)
                ax.plot(prof.index.to_numpy(), prof[_ROUTE_MU[r]].to_numpy(),
                        linestyle="none", marker="o", markersize=1.8,
                        color=cmaps[r](shade[k]), alpha=0.6, zorder=2)
        ax.set_ylabel("travel time [min]")
        ax.set_title(f"{'ab'[row]}. Route {_ROUTE_TEX[r]}", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("within-day time [min]")
    fig.align_ylabels(axes)

    day_handles = [
        Line2D([0], [0], color=plt.cm.Greys(shade[k]), linewidth=1.8,
               label=f"day {int(d)}")
        for k, d in enumerate(picked_days)
    ]
    kind_handles = [
        Line2D([0], [0], color="grey", linewidth=1.8, label="realised"),
        Line2D([0], [0], color="grey", linestyle="none", marker="o",
               markersize=3.5, label="belief (pred. TT)"),
    ]
    fig.legend(handles=day_handles + kind_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(day_handles) + len(kind_handles),
               frameon=False, fontsize=6.5, columnspacing=1.2, handlelength=1.6)
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
    truth_smooth_window: int | None = None,
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

    fig, axgrid = plt.subplots(nrows, ncols, figsize=(TEXT_W, TEXT_W * 0.42 * nrows),
                               sharex=True, sharey=True, squeeze=False)
    axes = axgrid.ravel()
    picked_ref = None
    for ax, (label, res) in zip(axes, items):
        day_step, sample_seed = _seed_slice(res.step, seed)
        all_days = sorted(day_step["day"].unique())
        picked = _pick_evolution_days(all_days, n_days)
        picked_ref = picked
        shade = np.linspace(0.40, 1.0, len(picked))
        w = (int(truth_smooth_window) if truth_smooth_window is not None
             else _cohort_window(params))
        pivot = day_step.pivot_table(index="day", columns="tau", values=tt_col,
                                     aggfunc="mean").sort_index()
        sm = pivot.rolling(w, center=True, min_periods=1).mean()
        tau = sm.columns.to_numpy(dtype=float)
        snapshots = res.snapshots or {}
        for k, d in enumerate(picked):
            ax.plot(tau, sm.loc[d].to_numpy(), color=cmap(shade[k]),
                    linewidth=1.2, zorder=4)
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


def _alpha_queue_belief_profile(snap: dict, dt_min: int, delay_min: int, tau_max: float):
    """Per-departure-minute profile of the traveller route-alpha *queue* belief.

    Each A--B traveller holds a single scalar queue belief ``L_mean_alpha`` for
    route alpha (the smoother latent is fixed within the day) and a fixed
    ``departure_time``. We keep only agents who actually *took* route alpha
    (``last_choice == 0``) -- their belief is first-hand for the L_2 queue -- and
    place each at the within-day minute where they meet that queue: the
    **arrival** minute ``departure*dt_min + delay_min``. Averaging the agents in
    each minute bucket turns the population of per-agent scalars into a within-day
    *profile*, directly comparable to the realised L_2(tau) curve, and its
    across-agent spread shows the heterogeneity between early- and peak-departers.

    Returns ``(mean_series, std_series)`` indexed by arrival minute, or ``None``
    if the snapshot lacks the fields / no agent took route alpha.
    """
    needed = {"L_mean_alpha", "departure_time", "last_choice"}
    if not needed.issubset(snap):
        return None
    took_alpha = np.asarray(snap["last_choice"]) == 0
    if not took_alpha.any():
        return None
    minute = (
        np.asarray(snap["departure_time"])[took_alpha] * dt_min + delay_min
    ).astype(float)
    minute = np.minimum(minute, tau_max)
    # Clamp each agent's queue belief at 0: the linear-Gaussian latent can go
    # slightly negative for near-empty (off-peak) queues, which is unphysical --
    # a negative belief just means "expects an empty queue".
    L = np.maximum(np.asarray(snap["L_mean_alpha"])[took_alpha], 0.0)
    prof = pd.DataFrame({"minute": minute, "L": L})
    g = prof.groupby("minute")["L"]
    return g.mean(), g.std().fillna(0.0)


def plot_belief_reality_queues(
    step_df: pd.DataFrame,
    snapshots: dict,
    params,
    *,
    day: int | None = None,
    seed: int | None = None,
):
    """Belief--reality consistency for the two critical signalised queues.

    Two panels, ``L_2`` (A--B) and ``L_6`` (C--D). Each shows the realised
    within-day queue (solid) and the **controller's** queue belief (dashed mean
    + -/+1 sigma band, from its smoother posterior over the trajectory).

    On ``L_2`` the **traveller route-alpha queue belief** is drawn as a
    *per-departure-minute profile* (dots): each A--B traveller that took route
    alpha holds one scalar queue belief, placed at the minute it meets the L_2
    queue (its arrival minute), with agents in each minute bucket averaged; the
    faint band is the across-agent spread, so early- vs peak-departers holding
    different beliefs is visible rather than averaged into one number. ``L_6``
    (exogenous C--D) is not a learning cohort, so it shows the controller only.

    Answers Xue's "do both agent types learn a consistent representation of the
    network state?" for the single inspected ``day`` (defaults to the last).
    Requires per-agent ``snapshots`` on that day (pass ``snapshot_days`` to
    :func:`run_experiment`).
    """
    day_step, sample_seed = _seed_slice(step_df, seed)
    if day is None:
        day = int(day_step["day"].max())
    d = day_step[day_step["day"] == day].sort_values("tau")
    tau = d["tau"].to_numpy(dtype=float)
    tau_max = float(day_step["tau"].max())

    has_ctrl_belief = "L2_belief_mu" in d.columns and d["L2_belief_mu"].notna().any()

    # Traveller route-alpha queue-belief profile (dots) for the L_2 panel.
    trav = None
    snap = (snapshots or {}).get((sample_seed, int(day)))
    if snap is not None:
        dt_min = int(params.sim.dt_min)
        try:
            sig_ab = params.network.signalised_links[0]
            delay_min = int(params.network.n_delay(dt_min)[sig_ab]) * dt_min
        except Exception:
            delay_min = 0
        trav = _alpha_queue_belief_profile(snap, dt_min, delay_min, tau_max)

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W, TEXT_W * 0.92), sharex=True)
    specs = [
        ("L2", r"$L_2$ (A--B)", route_colour("alpha"), "L2_belief_mu",
         "L2_belief_sd", True),
        ("L6", r"$L_6$ (C--D)", route_colour("gamma"), "L6_belief_mu",
         "L6_belief_sd", False),
    ]
    for ax, (col, label, colour, bmu, bsd, is_alpha) in zip(axes, specs):
        ax.plot(tau, d[col].to_numpy(), color=colour, linewidth=active_style().line_main,
                label="realised", zorder=4)
        if has_ctrl_belief and bmu in d.columns:
            mu = d[bmu].to_numpy()
            sd = d[bsd].to_numpy()
            ax.plot(tau, mu, color="k", linestyle="--", linewidth=1.0,
                    label="controller belief", zorder=3)
            ax.fill_between(tau, mu - sd, mu + sd, color="k", alpha=0.12,
                            linewidth=0, zorder=1)
        if is_alpha and trav is not None:
            mu_prof, sd_prof = trav
            x = mu_prof.index.to_numpy(dtype=float)
            # A line + -/+1 sigma band (like the controller belief), with markers
            # kept so the sampled departure minutes -- and their sparsity at the
            # peak, where few travellers take the intersection -- stay visible.
            ax.plot(x, mu_prof.to_numpy(), color=colour, linestyle=":",
                    marker="o", markersize=2.6, linewidth=1.2,
                    label="traveller belief (by departure)", zorder=5)
            # Clip the band's lower edge at 0 -- a queue length is non-negative.
            lower = np.maximum((mu_prof - sd_prof).to_numpy(), 0.0)
            ax.fill_between(x, lower, (mu_prof + sd_prof).to_numpy(),
                            color=colour, alpha=0.15, linewidth=0, zorder=0)
        ax.set_ylabel(f"queue {label} [veh]")
        ax.grid(alpha=0.25)
    axes[0].set_title(f"Belief vs realised queue (day {day})")
    axes[-1].set_xlabel("within-day time [min]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="upper center",
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02),
               fontsize=7.5)
    light_borders(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
