"""Network / control diagnostic plots: signalised-link queues, green split,
and the day-to-day route share. Pure ``Figure``-returning helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from .primitives import text_w

# Fixed layout for the default 7-link intersection network (paper figure
# ``intersection_network_v2``): A--n1--nX--n3--B on a horizontal axis, with the
# C--D stream crossing vertically at the signalised junction nX.
_NODE_POS = {
    "A": (0.0, 0.0), "n1": (2.0, 0.0), "nX": (4.0, 0.0),
    "n3": (6.0, 0.0), "B": (8.0, 0.0), "C": (4.0, 2.4), "D": (4.0, -2.4),
}
_LINK_FROM_TO = {
    1: ("A", "n1"), 2: ("n1", "nX"), 3: ("nX", "n3"), 4: ("n3", "B"),
    5: ("n1", "n3"), 6: ("C", "nX"), 7: ("nX", "D"),
}
_BYPASS_LINK = 5  # drawn as an arc above the axis


def _edges(v: np.ndarray) -> np.ndarray:
    """Cell edges around the sample centres ``v`` for ``pcolormesh``."""
    v = np.asarray(v, dtype=float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    dd = np.diff(v)
    return np.concatenate([[v[0] - dd[0] / 2], v[:-1] + dd / 2, [v[-1] + dd[-1] / 2]])


def plot_signal_day(step: pd.DataFrame, day: int | None = None, *,
                    shared_ylim: bool = True):
    """Within-day queues on the two signalised links and the green split,
    for a single day (defaults to the last recorded day).

    With ``shared_ylim`` (default) the queue axis is fixed to the maximum queue
    over **all** recorded days, so heights are comparable as the day slider moves
    (a lower curve is a genuinely lower-queue day, not a rescaling); the green-
    fraction axis is fixed to ``[0, 1]``. Set ``shared_ylim=False`` for the old
    per-day autoscale."""
    if day is None:
        day = int(step["day"].max())
    d = step[step["day"] == day].sort_values("tau")

    fig, (ax_q, ax_phi) = plt.subplots(
        2, 1, figsize=(text_w(), text_w() * 0.9), sharex=True,
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

    if shared_ylim:
        qmax = float(np.nanmax(step[["L2", "L6"]].to_numpy()))
        ax_q.set_ylim(0, qmax * 1.05 if qmax > 0 else 1.0)
        ax_phi.set_ylim(0, 1.0)
    fig.tight_layout()
    return fig


def plot_queue_belief_day(step: pd.DataFrame, day: int | None = None, *,
                          shared_ylim: bool = True):
    """The controller's posterior belief over the within-day queue trajectory
    (mean +/- 1 sigma band) overlaid on the realised queue, for a single day
    (defaults to the last recorded day), on both signalised movements.

    The belief is the controller's rolling-window smoother posterior *after*
    folding day ``N`` into its window, i.e. what the controller now believes a
    typical day's queue looks like, compared against day ``N``'s single realised
    sample. The band narrows on later days as the window fills. Requires the
    ``*_belief_mu`` / ``*_belief_sd`` columns the simulator records for a
    controller that maintains a belief (the AIF controller); for a controller
    with no belief (baselines) only the realised queue is drawn.

    With ``shared_ylim`` (default) both panels are fixed to the maximum over
    **all** days of the realised queue and the belief envelope (mean + 1 sigma),
    so heights are comparable as the day slider moves."""
    if day is None:
        day = int(step["day"].max())
    d = step[step["day"] == day].sort_values("tau")
    has_belief = "L2_belief_mu" in d.columns and d["L2_belief_mu"].notna().any()

    fig, axes = plt.subplots(
        2, 1, figsize=(text_w(), text_w() * 0.9), sharex=True,
    )
    panels = [
        (axes[0], "L2", "tab:blue", r"$L_2$ (A--B)"),
        (axes[1], "L6", "tab:orange", r"$L_6$ (C--D)"),
    ]
    for ax, link, color, label in panels:
        ax.plot(d["tau"], d[link], color=color, label=f"{label} realised")
        if has_belief:
            mu = d[f"{link}_belief_mu"].to_numpy(dtype=float)
            sd = d[f"{link}_belief_sd"].to_numpy(dtype=float)
            tau = d["tau"].to_numpy(dtype=float)
            ax.plot(tau, mu, color="black", lw=1.0, ls="--", label="belief mean")
            ax.fill_between(
                tau, mu - sd, mu + sd, color="black", alpha=0.15,
                label=r"belief $\pm1\sigma$",
            )
        ax.set_ylabel("queue [veh]")
        ax.legend()
        ax.grid(alpha=0.25)

    if shared_ylim:
        ymax = float(np.nanmax(step[["L2", "L6"]].to_numpy()))
        for link in ("L2", "L6"):
            mucol, sdcol = f"{link}_belief_mu", f"{link}_belief_sd"
            if mucol in step.columns:
                env = (step[mucol].to_numpy(dtype=float)
                       + step[sdcol].to_numpy(dtype=float))
                if np.isfinite(env).any():
                    ymax = max(ymax, float(np.nanmax(env)))
        for ax in axes:
            ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)

    suffix = "" if has_belief else " (controller has no belief)"
    axes[0].set_title(f"Controller belief vs realised queue (day {day}){suffix}")
    axes[1].set_xlabel("time of day [min]")
    fig.tight_layout()
    return fig


def plot_learned_obs_noise(controller: pd.DataFrame):
    """The controller's *learned* queue observation-noise SD over days, per
    signalised movement (``sigma_obs_l2`` / ``sigma_obs_l6`` from the controller
    snapshot). Only meaningful when ``learn_obs_noise`` is on; with it off the
    series sit flat at the fixed ``sigma_obs`` default. A dashed line marks that
    fixed default for comparison."""
    c = controller if "seed" not in controller else (
        controller[controller["seed"] == controller["seed"].min()])
    c = c.sort_values("day")
    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 0.5))
    if "sigma_obs_l2" in c.columns:
        ax.plot(c["day"], c["sigma_obs_l2"], color="tab:blue",
                label=r"learned $\sigma_{obs}$, $L_2$ (A--B)")
        ax.plot(c["day"], c["sigma_obs_l6"], color="tab:orange",
                label=r"learned $\sigma_{obs}$, $L_6$ (C--D)")
    ax.set_xlabel("day")
    ax.set_ylabel(r"obs-noise SD [veh]")
    ax.set_title("Controller's learned queue observation noise")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_route_flows(step: pd.DataFrame, day: int | None = None, *,
                     shared_ylim: bool = True):
    """Within-day traveller flow on each route, for a single day.

    Shows the two A--B options, the intersection route ``alpha`` (link 2) and
    the bypass ``beta`` (link 5), and the exogenous C--D stream ``gamma``
    (link 6), with the total A--B demand for reference. ``alpha`` and ``beta``
    together make up the A--B demand, so when ``Q_alpha`` dips while ``Q_beta``
    rises travellers have shifted from the intersection to the bypass.

    With ``shared_ylim`` (default) the flow axis is fixed to the maximum over
    **all** days, so heights are comparable as the day slider moves."""
    if day is None:
        day = int(step["day"].max())
    d = step[step["day"] == day].sort_values("tau")
    tau = d["tau"]
    total_ab = d["Q_alpha"] + d["Q_beta"]

    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 0.55))
    ax.plot(tau, total_ab, color="0.6", ls="--", label=r"$Q_{AB}$")
    ax.plot(tau, d["Q_alpha"], color="tab:blue", label=r"$Q_\alpha$")
    ax.plot(tau, d["Q_beta"], color="tab:green", label=r"$Q_\beta$")
    ax.plot(tau, d["Q_gamma"], color="tab:orange", label=r"$Q_{CD}$")
    ax.set_xlabel("time of day [min]")
    ax.set_ylabel("traveller flow [veh/h]")
    ax.set_title(f"Per-route traveller flow (day {day})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    if shared_ylim:
        total_all = (step["Q_alpha"] + step["Q_beta"]).to_numpy()
        fmax = float(np.nanmax([
            total_all,
            step["Q_alpha"].to_numpy(),
            step["Q_beta"].to_numpy(),
            step["Q_gamma"].to_numpy(),
        ]))
        ax.set_ylim(0, fmax * 1.05 if fmax > 0 else 1.0)
    fig.tight_layout()
    return fig


def plot_day_overview_grid(step: pd.DataFrame, *, days=None):
    """A 4x3 at-a-glance grid comparing three representative days side by side.

    Columns are the first, middle and last recorded day (override with
    ``days=(a, b, c)``); rows are (1) queues ``L_2,L_6``, (2) green split
    ``phi_2,phi_6``, (3) belief-vs-realised for ``L_2`` and (4) for ``L_6``. The
    Y-axis is shared across each row (computed over all days), so the three
    columns are directly comparable, e.g. whether queues shrink from the first
    to the last day. If the belief columns are absent (a controller with no
    belief), rows 3-4 show only the realised queue."""
    all_days = sorted(int(x) for x in step["day"].unique())
    if days is None:
        days = (all_days[0], all_days[len(all_days) // 2], all_days[-1])
    days = [int(x) for x in days]

    has_belief = (
        "L2_belief_mu" in step.columns and step["L2_belief_mu"].notna().any()
    )
    qmax = float(np.nanmax(step[["L2", "L6"]].to_numpy()))
    bmax = qmax
    for link in ("L2", "L6"):
        mucol, sdcol = f"{link}_belief_mu", f"{link}_belief_sd"
        if mucol in step.columns:
            env = step[mucol].to_numpy(dtype=float) + step[sdcol].to_numpy(dtype=float)
            if np.isfinite(env).any():
                bmax = max(bmax, float(np.nanmax(env)))

    fig, axes = plt.subplots(
        4, 3, figsize=(text_w() * 1.7, text_w() * 1.75), sharex=True,
    )
    row_ylabels = [
        "queue [veh]", "green fraction",
        r"$L_2$ belief vs real", r"$L_6$ belief vs real",
    ]
    for j, day in enumerate(days):
        d = step[step["day"] == day].sort_values("tau")
        tau = d["tau"].to_numpy(dtype=float)

        ax = axes[0][j]
        ax.plot(tau, d["L2"], color="tab:blue", label=r"$L_2$ (A--B)")
        ax.plot(tau, d["L6"], color="tab:orange", label=r"$L_6$ (C--D)")
        ax.set_ylim(0, qmax * 1.05 if qmax > 0 else 1.0)
        ax.set_title(f"day {day}", fontsize=9)

        ax = axes[1][j]
        ax.plot(tau, d["phi2"], color="tab:blue", label=r"$\phi_2$")
        ax.plot(tau, d["phi6"], color="tab:orange", label=r"$\phi_6$")
        ax.set_ylim(0, 1.0)

        for r, link, color in ((2, "L2", "tab:blue"), (3, "L6", "tab:orange")):
            ax = axes[r][j]
            ax.plot(tau, d[link], color=color, label="realised")
            mucol = f"{link}_belief_mu"
            if has_belief and mucol in d.columns:
                mu = d[mucol].to_numpy(dtype=float)
                sdv = d[f"{link}_belief_sd"].to_numpy(dtype=float)
                ax.plot(tau, mu, color="black", lw=1.0, ls="--", label="belief")
                ax.fill_between(tau, mu - sdv, mu + sdv, color="black", alpha=0.15)
            ax.set_ylim(0, bmax * 1.05 if bmax > 0 else 1.0)

        for r in range(4):
            axes[r][j].grid(alpha=0.25)

    for r, ylabel in enumerate(row_ylabels):
        axes[r][0].set_ylabel(ylabel, fontsize=8)
        axes[r][0].legend(fontsize=6, loc="upper left")
    for j in range(3):
        axes[3][j].set_xlabel("time of day [min]")

    fig.suptitle(
        "Multi-day overview: first / middle / last day (Y shared per row)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def _link_flows(row, net) -> dict[int, float]:
    """Per-link traveller flow [veh/h] at one (day, tau) row, from the route
    flows via the route->link incidence (a link carries the sum of the route
    flows that traverse it)."""
    route_flow = {"alpha": float(row["Q_alpha"]),
                  "beta": float(row["Q_beta"]),
                  "gamma": float(row["Q_gamma"])}
    flows = {}
    for lid in net.link_ids:
        flows[lid] = sum(route_flow[r] for r in net.routes
                         if lid in net.route_links[r])
    return flows


def _network_color_scale(scale_df: pd.DataFrame, net, color_by: str):
    """Colour map, colourbar label and value maximum for a network-state frame.

    ``scale_df`` is the slice the colour scale is normalised over (one day, or
    all days for a run-wide shared scale). Shared between the static
    :func:`plot_network_state` and the per-frame animation so the colour scale is
    computed identically in both.
    """
    if color_by == "queue":
        # Traffic-signal palette: green = empty, red = congested. Unlike YlOrRd
        # (pale yellow at 0, invisible on the white background) an empty link is
        # a clearly visible green here and only the mid-range is pale.
        cmap, clabel = plt.get_cmap("RdYlGn_r"), "queue length [veh]"
        vmax = float(np.nanmax([scale_df[f"L{lid}"].max() for lid in net.link_ids]))
    else:
        # Max link flow = max over links of the route-flow sums.
        flow_scale = {lid: np.zeros(len(scale_df)) for lid in net.link_ids}
        for r in net.routes:
            qr = scale_df[f"Q_{r}"].to_numpy()
            for lid in net.route_links[r]:
                flow_scale[lid] = flow_scale[lid] + qr
        cmap, clabel = plt.get_cmap("viridis"), "traveller flow [veh/h]"
        vmax = float(np.nanmax([flow_scale[lid].max() for lid in net.link_ids]))
    return cmap, clabel, vmax


def _render_network_axes(ax, row, net, norm, cmap, color_by: str) -> None:
    """Draw one network-state frame (links + labels + nodes) onto ``ax``.

    Pure drawing of the graph for a single ``(day, tau)`` ``row``: link arrows
    coloured by the selected metric, per-link labels (flow / queue / green
    split) and the OD/junction nodes. Sets the aspect, hides the axis and pins
    the limits. Shared by the static plot and the animation so the layout is
    defined once.
    """
    flows = _link_flows(row, net)
    queues = {lid: float(row[f"L{lid}"]) for lid in net.link_ids}
    phi = {2: float(row["phi2"]), 6: float(row["phi6"])}
    values = queues if color_by == "queue" else flows

    ax.set_aspect("equal")
    ax.axis("off")

    for lid, (a, b) in _LINK_FROM_TO.items():
        (x0, y0), (x1, y1) = _NODE_POS[a], _NODE_POS[b]
        colour = cmap(norm(values[lid]))
        # Bypass arcs above the axis (matching the paper figure); rad<0 bows up.
        arc = -0.45 if lid == _BYPASS_LINK else 0.0
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={arc}",
            arrowstyle="-|>", mutation_scale=14,
            linewidth=3.0, color=colour, shrinkA=12, shrinkB=12, zorder=1,
        ))
        # Label position: left part of the arc for the bypass (clear of the
        # vertical C--D links), perpendicular offset otherwise.
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length  # unit normal
        if lid == _BYPASS_LINK:
            mx, my, off = mx - 1.0, my + 0.95, (0.0, 0.0)
        else:
            off = (nx * 0.34, ny * 0.34)
        label = f"L{lid}\n{flows[lid]:.0f} veh/h\nq={queues[lid]:.0f}"
        if lid in phi:
            label += f"\n$\\phi$={phi[lid]:.2f}"
        ax.text(mx + off[0], my + off[1], label, ha="center", va="center",
                fontsize=6.5, zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc",
                          alpha=0.85, lw=0.5))

    for name, (x, y) in _NODE_POS.items():
        is_od = name in ("A", "B", "C", "D")
        ax.scatter([x], [y], s=380 if is_od else 200,
                   c="#1f4e79" if is_od else "#888888", zorder=2)
        ax.text(x, y, name, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=4)

    ax.set_xlim(-1.0, 9.0)
    ax.set_ylim(-3.2, 3.2)


def plot_network_state(
    step: pd.DataFrame,
    net,
    day: int | None = None,
    tau: int | None = None,
    color_by: str = "travellers",
    *,
    seed: int | None = None,
    shared_scale: bool = True,
):
    """Node-edge diagram of the network at one day and time of day.

    Each link is coloured and labelled by either the traveller flow
    (``color_by="travellers"``, veh/h) or the queue length
    (``color_by="queue"``, veh); the label always shows both, and the two
    signalised links (2, 6) additionally show the current green split. With
    ``shared_scale`` (default) the colour scale is fixed to the global maximum of
    the selected metric across **all** days, so snapshots are comparable both
    across the time-of-day slider and across days; set ``shared_scale=False`` to
    normalise to the selected day's maximum instead.
    """
    if tuple(net.link_ids) != (1, 2, 3, 4, 5, 6, 7):
        raise ValueError(
            "plot_network_state assumes the default 7-link intersection "
            f"network; got link ids {net.link_ids}."
        )
    if color_by not in ("travellers", "queue"):
        raise ValueError(f"color_by must be 'travellers' or 'queue', got {color_by!r}.")

    sd = step if (seed is None or "seed" not in step.columns) else step[step["seed"] == seed]
    if "seed" in sd.columns and seed is None:
        sd = sd[sd["seed"] == sd["seed"].min()]
    if day is None:
        day = int(sd["day"].max())
    day_df = sd[sd["day"] == day].sort_values("tau")
    taus = day_df["tau"].to_numpy()
    if tau is None:
        tau = int(taus[len(taus) // 2])
    tau = int(taus[np.argmin(np.abs(taus - tau))])  # snap to an available tau
    row = day_df[day_df["tau"] == tau].iloc[0]

    phi = {2: float(row["phi2"]), 6: float(row["phi6"])}

    # Normalise the colour scale to the day's maximum, or (default) the global
    # maximum across all recorded days so snapshots are comparable across days.
    scale_df = sd if shared_scale else day_df
    cmap, clabel, vmax = _network_color_scale(scale_df, net, color_by)
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    fig, ax = plt.subplots(figsize=(text_w() * 1.35, text_w() * 0.85))
    _render_network_axes(ax, row, net, norm, cmap, color_by)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=clabel, fraction=0.04, pad=0.02)

    ax.set_title(
        f"Network state, day {day}, t = {tau} min\n"
        f"green split $\\phi_2$={phi[2]:.2f}, $\\phi_6$={phi[6]:.2f}   "
        f"(colour: {color_by})",
        fontsize=8,
    )
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

    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 0.62))
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
    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 3.5 / 5.0))
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
    fig, ax = plt.subplots(figsize=(text_w(), text_w() * 3.5 / 5.0))
    ax.plot(g.index, g.values, color="tab:blue", marker="o", markersize=3)
    ax.set_xlabel("day")
    ax.set_ylabel(r"share on route $\alpha$")
    ax.set_ylim(0, 1)
    ax.set_title("Day-to-day route share (intersection route)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
