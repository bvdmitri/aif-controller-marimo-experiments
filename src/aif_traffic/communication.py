"""Inter-layer communication: the controller's broadcast to travellers.

The controller has a network-wide view and may broadcast an information signal
that travellers fold into their perceived route cost
``zeta_r = TT_r + theta * E_r`` (paper Eq. for the perceived cost). This module
defines the broadcast payload and assembles it from the realised day. The
candidate signal types (travel time, congestion, externality, marginal social
cost) are all reduced to a common per-route *advisory* the traveller treats as
an externality-like offset, scaled per agent by ``theta`` and compliance in
:mod:`inference.population`.

The ``TRAVEL_TIME`` and ``CONGESTION`` signals are direct readings of the
realised day (the controller simply relays what it measured). The
``EXTERNALITY`` and ``MSC`` signals are the paper-faithful quantities: the
marginal social cost ``MSC_r`` is computed by finite-difference re-rolling of
the store-and-forward queue model (insert one extra vehicle on route ``r`` in
interval ``t``, re-integrate the day under the *realised* green splits, and
measure the increase in system cost), and the externality is
``E_r = MSC_r - TT_r``. This is performance-heavy -- it re-rolls the network
once per (traveller route, minute) -- so it only runs when one of those two
signals is actually broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .network import (
    effective_capacities,
    link_and_route_travel_times,
    link_inflows,
    route_arrival_queues,
)
from .parameters import CommunicationSpec, NetworkParams, SignalType, SimParams
from .utils import daily_system_cost


@dataclass(frozen=True)
class Broadcast:
    """Per-traveller-route advisory over departure minutes.

    ``value[route]`` is a length-``K`` array; a traveller departing at minute
    ``t`` on route ``r`` reads ``value[r][t]``. Higher values discourage the
    route once folded into the perceived cost. ``signal_type`` records which
    information was broadcast.
    """

    signal_type: SignalType
    value: Mapping[str, np.ndarray]


def empty_broadcast(net: NetworkParams, sim: SimParams) -> Broadcast:
    """A no-information broadcast (all zeros): travellers use private cost only."""
    K = sim.K
    return Broadcast(
        signal_type=SignalType.NONE,
        value={r: np.zeros(K) for r in net.traveller_routes},
    )


def _marginal_social_cost(
    inflow_by_route: Mapping[str, np.ndarray],
    phi2: np.ndarray,
    phi6: np.ndarray,
    net: NetworkParams,
    sim: SimParams,
) -> dict[str, np.ndarray]:
    """Per-route, per-minute marginal social cost by finite difference.

    ``MSC_r(t) = SC(d; Q_r(t) + dQ) - SC(d)`` where ``dQ = 60/dt`` is one extra
    vehicle in the interval. The day is re-integrated under the *realised* green
    splits ``(phi2, phi6)`` for each perturbation, so the measured cost increase
    reflects both the extra vehicle and the congestion it imposes on others.

    Cost: one full-day re-roll per (traveller route, minute). Computed only for
    the EXTERNALITY / MSC signals.
    """
    from .network import _integrate_queues  # local: heavy path only

    K = sim.K
    dt_h = sim.dt_h
    dQ = 60.0 / sim.dt_min  # one vehicle within a dt-minute interval, in veh/h
    caps = effective_capacities(phi2, phi6, net)

    def _system_cost(infl: Mapping[str, np.ndarray]) -> float:
        Q_link = link_inflows(infl, net)
        queues = _integrate_queues(Q_link, caps, net, sim)
        _, tt_route = link_and_route_travel_times(queues, caps, net, sim)
        return daily_system_cost(infl, tt_route, dt_h)

    base_infl = {r: np.asarray(q, dtype=float) for r, q in inflow_by_route.items()}
    sc_base = _system_cost(base_infl)

    msc: dict[str, np.ndarray] = {}
    for r in net.traveller_routes:
        out = np.zeros(K)
        for t in range(K):
            perturbed = dict(base_infl)
            qr = base_infl[r].copy()
            qr[t] += dQ
            perturbed[r] = qr
            out[t] = _system_cost(perturbed) - sc_base
        msc[r] = out
    return msc


def build_broadcast(
    comm: CommunicationSpec,
    tt_by_route: Mapping[str, np.ndarray],
    queues_by_link: Mapping[int, np.ndarray],
    net: NetworkParams,
    sim: SimParams,
    inflow_by_route: Mapping[str, np.ndarray] | None = None,
    phi2: np.ndarray | None = None,
    phi6: np.ndarray | None = None,
) -> Broadcast:
    """Assemble the broadcast for the *next* day from the realised day.

    Each signal type maps to a per-route advisory (length ``K``):

    * ``NONE``         -> zeros (no information shared);
    * ``TRAVEL_TIME``  -> the route travel time ``TT_r`` (direct reading);
    * ``CONGESTION``   -> the total queued vehicles along the route (direct);
    * ``EXTERNALITY``  -> ``E_r = MSC_r - TT_r`` (finite-difference);
    * ``MSC``          -> the marginal social cost ``MSC_r`` (finite-difference).

    The EXTERNALITY / MSC signals require the realised route inflows and green
    splits (``inflow_by_route``, ``phi2``, ``phi6``) to re-roll the queue model.
    All advisories are clipped to be non-negative (higher discourages a route).
    """
    st = comm.signal_type
    if st is SignalType.NONE:
        return empty_broadcast(net, sim)

    value: dict[str, np.ndarray] = {}

    if st in (SignalType.EXTERNALITY, SignalType.MSC):
        if inflow_by_route is None or phi2 is None or phi6 is None:
            raise ValueError(
                f"Signal {st!r} needs inflow_by_route, phi2, phi6 to compute the "
                "finite-difference marginal social cost."
            )
        msc = _marginal_social_cost(inflow_by_route, phi2, phi6, net, sim)
        for r in net.traveller_routes:
            tt = np.asarray(tt_by_route[r], dtype=float)
            if st is SignalType.MSC:
                value[r] = np.maximum(msc[r], 0.0)
            else:  # EXTERNALITY: E_r = MSC_r - TT_r
                value[r] = np.maximum(msc[r] - tt, 0.0)
        return Broadcast(signal_type=st, value=value)

    route_queue = route_arrival_queues(queues_by_link, net, sim)
    for r in net.traveller_routes:
        if st is SignalType.TRAVEL_TIME:
            v = np.asarray(tt_by_route[r], dtype=float)
        elif st is SignalType.CONGESTION:
            v = np.maximum(route_queue[r], 0.0)
        else:  # pragma: no cover - exhaustive over SignalType
            raise ValueError(f"Unknown signal type {st!r}.")
        value[r] = v
    return Broadcast(signal_type=st, value=value)
