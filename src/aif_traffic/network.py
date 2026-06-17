"""Link-level intersection network: incidence, queue dynamics, travel times.

The new paper's network is **link-keyed** (links 1-7) rather than the IWAI
two-route corridor. Route inflows are mapped to link inflows through a
route-link incidence matrix; congestion is a per-link store-and-forward
queue. The two signalised links (2 = A--B, 6 = C--D) take their effective
discharge capacity from the controller's green-time split each control
interval -- this is the single point at which the macro-layer controller
drives the physics.

This module is **controller-agnostic**: :func:`run_within_day` accepts a
plain callable ``green_split_fn(queue_obs, k) -> (phi2, phi6)`` so any
controller object (or a closure over one) can drive the signals without the
network importing the ``control`` package.

Numerics mirror the IWAI store-and-forward model:

* queue update uses the *delayed* inflow ``Q_l(k - N_l)`` (vehicles that
  entered earlier and now reach the downstream stop line),
* travel time looks *forward* over the free-flow lag,
  ``TT_l(k) = F_l + 60 L_l(k + N_l) / C_l(k + N_l)``.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .parameters import NetworkParams, SignalParams, SimParams


def incidence_matrix(net: NetworkParams) -> np.ndarray:
    """``A[l, r] = 1`` iff route ``r`` uses link ``l``. Shape ``(n_links, n_routes)``."""
    link_ids = net.link_ids
    li = {lid: i for i, lid in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(net.routes)), dtype=float)
    for r, route in enumerate(net.routes):
        for lid in net.route_links[route]:
            A[li[lid], r] = 1.0
    return A


def link_inflows(
    inflow_by_route: Mapping[str, np.ndarray],
    net: NetworkParams,
) -> dict[int, np.ndarray]:
    """Map per-route inflows ``Q_r(t)`` to per-link inflows ``Q_l(t)``."""
    out: dict[int, np.ndarray] = {}
    for lid in net.link_ids:
        total = None
        for route in net.routes:
            if lid in net.route_links[route]:
                q = np.asarray(inflow_by_route[route], dtype=float)
                total = q if total is None else total + q
        out[lid] = total if total is not None else np.zeros_like(
            next(iter(inflow_by_route.values())), dtype=float
        )
    return out


def effective_capacities(
    phi2: np.ndarray,
    phi6: np.ndarray,
    net: NetworkParams,
) -> dict[int, np.ndarray]:
    """Per-link, per-minute discharge capacity (veh/h).

    Ordinary links use their nominal ``cbar``; the two signalised links take
    a green-time fraction of it: ``C_2 = phi2 * cbar_2``, ``C_6 = phi6 * cbar_6``.
    """
    sig_ab, sig_cd = net.signalised_links
    caps: dict[int, np.ndarray] = {}
    for lid in net.link_ids:
        cbar = net.cbar(lid)
        if lid == sig_ab:
            caps[lid] = np.asarray(phi2, dtype=float) * cbar
        elif lid == sig_cd:
            caps[lid] = np.asarray(phi6, dtype=float) * cbar
        else:
            caps[lid] = np.full_like(np.asarray(phi2, dtype=float), cbar)
    return caps


def link_and_route_travel_times(
    queues: Mapping[int, np.ndarray],
    caps: Mapping[int, np.ndarray],
    net: NetworkParams,
    sim: SimParams,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Forward-look link travel times and their route sums.

    ``TT_l(k) = F_l + 60 * L_l(k+N_l) / C_l(k+N_l)`` and
    ``TT_r(k) = sum_l A_{lr} TT_l(k)``.
    """
    K = sim.K
    n_delay = net.n_delay(sim.dt_min)
    tt_link: dict[int, np.ndarray] = {}
    for lid in net.link_ids:
        F_l = net.link(lid).F_min
        N_l = n_delay[lid]
        k_arr = np.minimum(np.arange(K) + N_l, K - 1)
        cap_arr = np.maximum(np.asarray(caps[lid])[k_arr], 1e-6)
        tt_link[lid] = F_l + 60.0 * np.asarray(queues[lid])[k_arr] / cap_arr

    tt_route: dict[str, np.ndarray] = {}
    for route in net.routes:
        tt = np.zeros(K)
        for lid in net.route_links[route]:
            tt = tt + tt_link[lid]
        tt_route[route] = tt
    return tt_link, tt_route


def route_arrival_queues(
    queues: Mapping[int, np.ndarray],
    net: NetworkParams,
    sim: SimParams,
) -> dict[str, np.ndarray]:
    """Total vehicles queued along a route at the traveller's arrival, per
    departure minute: ``L_r(k) = sum_l A_{lr} L_l(k + N_l)``.

    Used as the route-level queue observation the AIF travellers sense.
    """
    K = sim.K
    n_delay = net.n_delay(sim.dt_min)
    out: dict[str, np.ndarray] = {}
    for route in net.routes:
        agg = np.zeros(K)
        for lid in net.route_links[route]:
            k_arr = np.minimum(np.arange(K) + n_delay[lid], K - 1)
            agg = agg + np.asarray(queues[lid])[k_arr]
        out[route] = agg
    return out


def _integrate_queues(
    Q_link: Mapping[int, np.ndarray],
    caps: Mapping[int, np.ndarray],
    net: NetworkParams,
    sim: SimParams,
) -> dict[int, np.ndarray]:
    """Store-and-forward queue integration given full per-minute capacities."""
    K = sim.K
    dt_h = sim.dt_h
    n_delay = net.n_delay(sim.dt_min)
    queues = {lid: np.zeros(K) for lid in net.link_ids}
    for lid in net.link_ids:
        N_l = n_delay[lid]
        q_in = np.asarray(Q_link[lid], dtype=float)
        cap = np.asarray(caps[lid], dtype=float)
        L = queues[lid]
        for k in range(K - 1):
            arr = q_in[k - N_l] if k - N_l >= 0 else 0.0
            L[k + 1] = max(0.0, L[k] + dt_h * (arr - cap[k]))
    return queues


def simulate_link_queues_const_phi(
    inflow_by_route: Mapping[str, np.ndarray],
    phi2_value: float,
    net: NetworkParams,
    sim: SimParams,
    signal: SignalParams,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Fixed-split full-day queue sim (used for predictive rollouts and tests).

    Holds ``phi2 = phi2_value`` (and ``phi6 = phi_sat - phi2``) all day.
    Returns ``(queues_by_link, tt_by_route)``.
    """
    K = sim.K
    phi2 = np.full(K, float(phi2_value))
    phi6 = np.full(K, signal.phi_sat - float(phi2_value))
    caps = effective_capacities(phi2, phi6, net)
    Q_link = link_inflows(inflow_by_route, net)
    queues = _integrate_queues(Q_link, caps, net, sim)
    _, tt_route = link_and_route_travel_times(queues, caps, net, sim)
    return queues, tt_route


def run_within_day(
    inflow_by_route: Mapping[str, np.ndarray],
    green_split_fn: Callable[[Mapping[int, float], int], tuple[float, float]],
    control_interval: int,
    net: NetworkParams,
    sim: SimParams,
    signal: SignalParams,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Integrate the day while the controller sets the green split online.

    Every ``control_interval`` minutes ``green_split_fn(queue_obs, k)`` is
    called with the *current* link queues and returns ``(phi2, phi6)``, held
    until the next control epoch. Returns
    ``(queues_by_link, tt_by_route, phi2_arr, phi6_arr)``.
    """
    K = sim.K
    dt_h = sim.dt_h
    n_delay = net.n_delay(sim.dt_min)
    sig_ab, sig_cd = net.signalised_links
    Q_link = link_inflows(inflow_by_route, net)

    queues = {lid: np.zeros(K) for lid in net.link_ids}
    phi2_arr = np.empty(K)
    phi6_arr = np.empty(K)

    phi2_cur, phi6_cur = signal.phi_sat / 2.0, signal.phi_sat / 2.0
    # Single interleaved pass: decide the split on the *current* queues at
    # each control epoch, hold it, then integrate one step forward.
    for k in range(K):
        if k % control_interval == 0:
            queue_obs = {lid: float(queues[lid][k]) for lid in net.link_ids}
            phi2_cur, phi6_cur = green_split_fn(queue_obs, k)
        phi2_arr[k] = phi2_cur
        phi6_arr[k] = phi6_cur

        if k < K - 1:
            for lid in net.link_ids:
                if lid == sig_ab:
                    cap_k = phi2_cur * net.cbar(lid)
                elif lid == sig_cd:
                    cap_k = phi6_cur * net.cbar(lid)
                else:
                    cap_k = net.cbar(lid)
                N_l = n_delay[lid]
                arr = Q_link[lid][k - N_l] if k - N_l >= 0 else 0.0
                L = queues[lid]
                L[k + 1] = max(0.0, L[k] + dt_h * (arr - cap_k))

    caps = effective_capacities(phi2_arr, phi6_arr, net)
    _, tt_route = link_and_route_travel_times(queues, caps, net, sim)
    return queues, tt_route, phi2_arr, phi6_arr
