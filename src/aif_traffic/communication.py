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

Separately from this cost-offset advisory, the controller can also broadcast
**belief-informing** signals (:class:`BeliefBroadcast`, paper Experiment 3:
BL/CG/SN/CG+SN). Unlike the ``Broadcast`` above -- which only shifts the
traveller's *perceived cost* -- a ``BeliefBroadcast`` is fed straight into the
traveller's smoother as *observations* of routes the traveller did not take
(see :mod:`inference.population` / :mod:`inference.filter`). It carries the raw
route queue ``L_hat_r`` (CG) and/or the route green split ``phi_hat_r`` (SN);
these are NOT clipped (a Kalman observation, not a cost advisory).
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
from .parameters import (
    BeliefSignal,
    CommunicationSpec,
    NetworkParams,
    SignalType,
    SimParams,
)
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


# ---------------------------------------------------------------------------
# Controller-belief broadcasts for decision-time fusion (Experiment 3/4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeliefBroadcast:
    """The controller's belief shared with travellers, for decision-time fusion.

    Carries the controller's forward-predicted belief about the **intersection
    route** for the upcoming day, arrival-aligned to the traveller's departure
    minute (so a traveller departing at minute ``t`` reads index ``t``):

    * ``mu_L`` / ``var_L`` -- the controller's predicted queue belief
      ``N(mu_L[t], var_L[t])`` (the QUEUE_BELIEF signal), or ``None`` when not
      shared.
    * ``phi`` / ``var_phi`` -- the controller's planned green split and its
      variance (the SPLIT_PLAN signal), or ``None`` when not shared.

    Both ``None`` is the baseline (BL) case. Only the signalised intersection
    route is informed (the controller has no belief about the uncongested
    bypass). Compliant travellers fuse these Gaussians into their own posterior
    over the intersection-route latent ``(L, phi)`` *before* choosing -- a
    transient, decision-time fusion that never enters the smoother (see
    :mod:`inference.population`).
    """

    mu_L: np.ndarray | None
    var_L: np.ndarray | None
    phi: np.ndarray | None
    var_phi: float | None

    def is_empty(self) -> bool:
        return self.mu_L is None and self.phi is None


def empty_belief_broadcast() -> BeliefBroadcast:
    """The baseline (BL) belief broadcast: nothing shared."""
    return BeliefBroadcast(mu_L=None, var_L=None, phi=None, var_phi=None)


def build_belief_broadcast(
    comm: CommunicationSpec,
    forecast,
    net: NetworkParams,
    sim: SimParams,
) -> BeliefBroadcast:
    """Assemble the controller-belief broadcast from a :class:`QueueForecast`.

    The controller's ``forecast`` carries its forward-predicted ``L_2`` belief
    (mean+variance) and planned split per within-day minute. Here we select the
    requested signals and apply the traveller's **arrival alignment** ``k+N_2``
    (the queue/split a traveller departing at minute ``k`` will actually meet at
    the signalised link), matching the chosen-route observation the simulator
    builds first-hand:

    * ``BeliefSignal.QUEUE_BELIEF`` (QB) -> ``(mu_L, var_L)`` -- the controller's
      predicted intersection queue belief. (``L_2`` is the controlling component
      of the intersection route's queue; the free-flow approach links rarely
      queue, so ``L_alpha ~= L_2``.)
    * ``BeliefSignal.SPLIT_PLAN`` (SP) -> ``(phi, var_phi)`` -- the controller's
      planned green split, which a traveller cannot otherwise anticipate.

    Empty ``belief_signals`` or a ``None`` forecast returns the baseline
    (BL) ``empty_belief_broadcast()``.
    """
    signals = comm.belief_signals
    if not signals or forecast is None:
        return empty_belief_broadcast()

    sig_ab, _sig_cd = net.signalised_links
    N_ab = net.n_delay(sim.dt_min)[sig_ab]
    k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)

    mu_L = var_L = phi = None
    var_phi = None
    if BeliefSignal.QUEUE_BELIEF in signals:
        mu_L = np.asarray(forecast.mu_L, dtype=float)[k_arr]
        var_L = np.asarray(forecast.var_L, dtype=float)[k_arr]
    if BeliefSignal.SPLIT_PLAN in signals:
        phi = np.asarray(forecast.phi2, dtype=float)[k_arr]
        var_phi = float(forecast.var_phi)

    return BeliefBroadcast(mu_L=mu_L, var_L=var_L, phi=phi, var_phi=var_phi)
