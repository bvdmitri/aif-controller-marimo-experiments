"""Inter-layer communication: the controller's broadcast to travellers.

The controller has a network-wide view and may broadcast an information signal
that travellers fold into their perceived route cost
``zeta_r = TT_r + theta * E_r`` (paper Eq. for the perceived cost). This module
defines the broadcast payload and assembles it from the realised day. The
candidate signal types (travel time, congestion, externality, marginal social
cost) are all reduced to a common per-route *advisory* the traveller treats as
an externality-like offset, scaled per agent by ``theta`` and compliance in
:mod:`inference.population`.

Scope note (deliberate): the advisories here are cheap proxies computed from
the realised day. The paper-faithful definitions, in particular the marginal
social cost by finite-difference re-rolling of the queue model (Eq. MSC_r) and
the externality ``E_r = MSC_r - TT_r`` (Eq. E_r), are a deferred extension --
they are performance-heavy and tied to the controller methodology we have not
yet settled. The *mechanism* (broadcast -> perceived cost -> EFE choice -> the
compliance switch) is concrete; the exact signal definition is an open knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .network import route_arrival_queues
from .parameters import CommunicationSpec, NetworkParams, SignalType, SimParams


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


def build_broadcast(
    comm: CommunicationSpec,
    tt_by_route: Mapping[str, np.ndarray],
    queues_by_link: Mapping[int, np.ndarray],
    net: NetworkParams,
    sim: SimParams,
) -> Broadcast:
    """Assemble the broadcast for the *next* day from the realised day.

    Each signal type maps to a per-route advisory (length ``K``):

    * ``NONE``         -> zeros (no information shared);
    * ``TRAVEL_TIME``  -> the route travel time ``TT_r``;
    * ``CONGESTION``   -> the total queued vehicles along the route;
    * ``EXTERNALITY``  -> the route delay ``TT_r - F_r`` (proxy for ``E_r``);
    * ``MSC``          -> ``1.5 * (TT_r - F_r)`` (crude marginal-cost proxy).

    The EXTERNALITY / MSC proxies stand in for the finite-difference marginal
    social cost (deferred). All are clipped to be non-negative.
    """
    st = comm.signal_type
    if st is SignalType.NONE:
        return empty_broadcast(net, sim)

    route_queue = route_arrival_queues(queues_by_link, net, sim)
    value: dict[str, np.ndarray] = {}
    for r in net.traveller_routes:
        tt = np.asarray(tt_by_route[r], dtype=float)
        delay = np.maximum(tt - net.route_free_flow(r), 0.0)
        if st is SignalType.TRAVEL_TIME:
            v = tt
        elif st is SignalType.CONGESTION:
            v = np.maximum(route_queue[r], 0.0)
        elif st is SignalType.EXTERNALITY:
            v = delay
        elif st is SignalType.MSC:
            v = 1.5 * delay
        else:  # pragma: no cover - exhaustive over SignalType
            raise ValueError(f"Unknown signal type {st!r}.")
        value[r] = v
    return Broadcast(signal_type=st, value=value)
