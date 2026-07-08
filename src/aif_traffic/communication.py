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
``E_r = MSC_r - TT_r``. This is performance-heavy, it re-rolls the network
once per (traveller route, minute), so it only runs when one of those two
signals is actually broadcast.

Two further controller -> traveller channels are defined here, both orthogonal
to the cost-offset advisory above (which only shifts the *perceived cost*):

* **Extra observations** (:class:`ObservationBroadcast`, paper Experiment 3
  default, BL/CG/SN/CG+SN). Travellers natively observe only the route they
  took; this channel relays the **realised** route queue ``L_r`` (CG) and/or
  green split ``phi_r`` (SN) of the routes they did *not* take, fed straight into
  the traveller's smoother as *observations* (see :mod:`inference.population` /
  :mod:`inference.filter`). It reaches all travellers and works with any
  controller. The values are raw readings (not clipped, unlike the cost
  advisory): the smoother treats them as noisy readings of the latent ``L``/``phi``.
* **Belief sharing** (:class:`BeliefBroadcast`, paper Experiment 3 optional,
  BL/QB/SP/QB+SP). The AIF controller shares its own forward-predicted belief
  (queue belief QB, planned split SP) *before* travellers choose; a compliant
  traveller fuses it transiently into a copy of its posterior at decision time
  (never written back to the smoother).
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
    ObservationSignal,
    SignalType,
    SimParams,
)
from .utils import daily_system_cost


@dataclass(frozen=True)
class Broadcast:
    """Per-traveller-route advisory over departure minutes.

    ``value[route]`` is a length-``K`` array for the direct / finite-difference
    signals; a traveller departing at minute ``t`` on route ``r`` reads
    ``value[r][t]``. For the sequential signals (``EXTERNALITY_SEQUENTIAL`` /
    ``MSC_SEQUENTIAL``) it is instead a ``(K, M)`` schedule: a per-departure-minute
    marginal-social-cost curve over ``M`` increment bins, and the traveller reads
    ``value[r][t, bin]`` at its stable within-minute rank bin (see
    :meth:`inference.population.Population._broadcast_cost_offset`). Higher values
    discourage the route once folded into the perceived cost. ``signal_type``
    records which information was broadcast. :func:`average_broadcasts` is
    shape-agnostic and handles both.
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


def average_broadcasts(broadcasts: "list[Broadcast]") -> Broadcast:
    """Elementwise mean of a trailing window of cost-offset advisories.

    Used to smooth the advisory over the last few days (damping the one-day
    cobweb): a traveller reads the mean of ``value[r]`` over the window rather
    than only yesterday's. ``signal_type`` is taken from the most recent
    broadcast. A single-element window returns that broadcast's values
    unchanged (a no-op, so smoothing over one day is bit-identical)."""
    latest = broadcasts[-1]
    if len(broadcasts) == 1:
        return latest
    routes = latest.value.keys()
    value = {
        r: np.mean([np.asarray(b.value[r], dtype=float) for b in broadcasts], axis=0)
        for r in routes
    }
    return Broadcast(signal_type=latest.signal_type, value=value)


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


def _sequential_marginal_social_cost(
    inflow_by_route: Mapping[str, np.ndarray],
    phi2: np.ndarray,
    phi6: np.ndarray,
    net: NetworkParams,
    sim: SimParams,
    M: int,
) -> dict[str, np.ndarray]:
    """Per-route, per-minute, per-increment marginal social cost schedule.

    The whole day's A--B demand is redistributed **from empty**, jointly across
    all departure minutes, in ``M`` equal increments. The AB routes (``alpha`` /
    ``beta``) start at zero inflow at every minute; ``gamma`` (the exogenous C--D
    demand) and the realised green splits ``(phi2, phi6)`` are held fixed. At each
    increment, every minute ``t`` is probed for the per-vehicle marginal social
    cost of adding one chunk to each route against the shared running assignment,
    the pair is recorded, and each minute's chunk is assigned to its argmin route
    (so a route's cost rises as it fills). The per-minute chunk size is
    ``d_ab(t) / M`` where ``d_ab(t) = alpha(t) + beta(t)`` is the realised total
    AB demand at ``t``.

    Filling from empty is what makes the schedule a stable day-to-day fixed point:
    the total AB demand ``d_ab(t)`` is the *split-independent* exogenous demand, so
    the schedule depends only on that demand, ``gamma`` and the green splits, not
    on yesterday's realised route split. (An earlier per-minute variant that held
    the other minutes at yesterday's realised inflow inherited yesterday's herd
    and failed to damp the cobweb.)

    Returns ``{route: (K, M)}`` raw (unclipped) marginal social cost. Cost is
    ``M * (2K + 1)`` full-day re-rolls.
    """
    from .network import _integrate_queues  # local: heavy path only

    K = sim.K
    dt_h = sim.dt_h
    caps = effective_capacities(phi2, phi6, net)

    def _system_cost(infl: Mapping[str, np.ndarray]) -> float:
        Q_link = link_inflows(infl, net)
        queues = _integrate_queues(Q_link, caps, net, sim)
        _, tt_route = link_and_route_travel_times(queues, caps, net, sim)
        return daily_system_cost(infl, tt_route, dt_h)

    base_infl = {r: np.asarray(q, dtype=float) for r, q in inflow_by_route.items()}
    routes = list(net.traveller_routes)  # AB routes, e.g. ["alpha", "beta"]
    M = max(1, int(M))
    sched = {r: np.zeros((K, M)) for r in routes}

    # Total AB demand per minute (split-independent) and the per-increment chunk.
    d_ab = np.zeros(K)
    for r in routes:
        d_ab = d_ab + base_infl[r]
    chunk_flow = d_ab / M            # (K,) veh/h added per increment per minute
    chunk_veh = chunk_flow * dt_h    # (K,) vehicles that flow represents

    # Start from empty AB everywhere; gamma / non-AB routes stay at the realised
    # base, green splits fixed via caps.
    running = {rr: base_infl[rr].copy() for rr in base_infl}
    for r in routes:
        running[r][:] = 0.0
    sc_now = _system_cost(running)

    for m in range(M):
        # Probe each (minute, route) against the shared start-of-increment state.
        marginals = {r: np.zeros(K) for r in routes}
        for t in range(K):
            if chunk_veh[t] <= 0.0:
                continue
            for r in routes:
                probe = {rr: running[rr].copy() for rr in running}
                probe[r][t] += chunk_flow[t]
                marginals[r][t] = (_system_cost(probe) - sc_now) / chunk_veh[t]
                sched[r][t, m] = marginals[r][t]
        # Assign each minute's chunk to its argmin route, then re-roll once for
        # the next increment's baseline.
        for t in range(K):
            if chunk_veh[t] <= 0.0:
                continue
            winner = min(routes, key=lambda rr: marginals[rr][t])
            running[winner][t] += chunk_flow[t]
        sc_now = _system_cost(running)

    return sched


def build_broadcast(
    comm: CommunicationSpec,
    tt_by_route: Mapping[str, np.ndarray],
    queues_by_link: Mapping[int, np.ndarray],
    net: NetworkParams,
    sim: SimParams,
    inflow_by_route: Mapping[str, np.ndarray] | None = None,
    phi2: np.ndarray | None = None,
    phi6: np.ndarray | None = None,
    out_diagnostics: dict | None = None,
) -> Broadcast:
    """Assemble the broadcast for the *next* day from the realised day.

    Each signal type maps to a per-route advisory (length ``K``):

    * ``NONE``         -> zeros (no information shared);
    * ``TRAVEL_TIME``  -> the route travel time ``TT_r`` (direct reading);
    * ``CONGESTION``   -> the total queued vehicles along the route (direct);
    * ``EXTERNALITY``  -> ``E_r = MSC_r - TT_r`` (finite-difference);
    * ``MSC``          -> the marginal social cost ``MSC_r`` (finite-difference);
    * ``EXTERNALITY_SEQUENTIAL`` / ``MSC_SEQUENTIAL`` -> a per-departure-minute
      ``(K, M)`` schedule (incremental marginal social cost over ``M`` rank bins;
      see :func:`_sequential_marginal_social_cost`), delivered per traveller by
      rank in :mod:`inference.population`.

    The EXTERNALITY / MSC (and their sequential variants) require the realised
    route inflows and green splits (``inflow_by_route``, ``phi2``, ``phi6``) to
    re-roll the queue model. All advisories are clipped to be non-negative
    (higher discourages a route).

    ``out_diagnostics``, when given, receives the **raw** (unclipped) marginal
    social cost under ``"msc"`` (``{route: length-K array}``) whenever it is
    computed, so the simulator can record it without re-rolling the day. For the
    sequential signals the recorded value is the schedule averaged over the ``M``
    bins (a length-K summary), so the existing MSC step columns and charts stay
    unchanged. It is left untouched for the direct (non-MSC) signals.
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
        if out_diagnostics is not None:
            out_diagnostics["msc"] = {r: v.copy() for r, v in msc.items()}
        for r in net.traveller_routes:
            tt = np.asarray(tt_by_route[r], dtype=float)
            if st is SignalType.MSC:
                value[r] = np.maximum(msc[r], 0.0)
            else:  # EXTERNALITY: E_r = MSC_r - TT_r
                value[r] = np.maximum(msc[r] - tt, 0.0)
        return Broadcast(signal_type=st, value=value)

    if st in (SignalType.EXTERNALITY_SEQUENTIAL, SignalType.MSC_SEQUENTIAL):
        if inflow_by_route is None or phi2 is None or phi6 is None:
            raise ValueError(
                f"Signal {st!r} needs inflow_by_route, phi2, phi6 to compute the "
                "sequential marginal social cost schedule."
            )
        sched = _sequential_marginal_social_cost(
            inflow_by_route, phi2, phi6, net, sim, comm.sequential_increments,
        )
        if out_diagnostics is not None:
            # Length-K summary (mean over increment bins) so the recorded MSC
            # columns / route-cost chart keep the same shape as the raw signals.
            out_diagnostics["msc"] = {r: v.mean(axis=1) for r, v in sched.items()}
        for r in net.traveller_routes:
            tt = np.asarray(tt_by_route[r], dtype=float)
            if st is SignalType.MSC_SEQUENTIAL:
                value[r] = np.maximum(sched[r], 0.0)
            else:  # EXTERNALITY_SEQUENTIAL: E_r = MSC_r - TT_r per bin
                value[r] = np.maximum(sched[r] - tt[:, None], 0.0)
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
# Extra-observation broadcasts folded into the traveller smoother (Experiment 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObservationBroadcast:
    """Extra observations of the non-chosen routes relayed to travellers.

    ``L`` maps each traveller route to a length-``K`` realised route queue
    ``L_r(k)`` (the CG signal); ``phi`` maps the *signalised* traveller route to
    its arrival-aligned realised green split ``phi_r(k)`` (the SN signal). Either
    may be ``None`` when that signal is not relayed; both ``None`` is the
    baseline (BL) case. A traveller departing at minute ``t`` on a route it did
    *not* take reads ``L[r][t]`` / ``phi[r][t]`` and folds it into its Gaussian
    belief over that route's latent ``(L, phi)`` at the end of the day (see
    :mod:`inference.population`).

    These are raw observations (not clipped, unlike the cost-advisory
    :class:`Broadcast`): the smoother treats them as noisy readings of the latent
    ``L`` and ``phi``. They are the **realised** (noisy) values of the day, so the
    relay simply lifts the traveller's partial observation to a fuller one.
    """

    L: Mapping[str, np.ndarray] | None
    phi: Mapping[str, np.ndarray] | None

    def is_empty(self) -> bool:
        return self.L is None and self.phi is None


def empty_observation_broadcast() -> ObservationBroadcast:
    """The baseline (BL) extra-observation broadcast: nothing relayed."""
    return ObservationBroadcast(L=None, phi=None)


def build_observation_broadcast(
    comm: CommunicationSpec,
    queues_by_link: Mapping[int, np.ndarray],
    phi2: np.ndarray,
    phi6: np.ndarray,
    net: NetworkParams,
    sim: SimParams,
) -> ObservationBroadcast:
    """Assemble the extra-observation broadcast from the *realised* day.

    * ``ObservationSignal.ROUTE_CONGESTION`` (CG) -> ``L_r``: the arrival-aligned
      route queue ``route_arrival_queues``, the very quantity a traveller senses
      first-hand when it *does* take the route, relayed for every traveller
      route.
    * ``ObservationSignal.SIGNAL_CONTROL`` (SN) -> ``phi``: the arrival-aligned
      realised intersection green split ``phi2`` for the signalised traveller
      route only (``phi`` is inert on the bypass), mirroring the chosen-route
      green-split observation the simulator builds first-hand.

    Empty ``obs_signals`` returns ``empty_observation_broadcast()`` so the
    baseline (BL) path is bit-identical to relaying no observations. ``phi6`` is
    accepted for signature symmetry with the realised day; the bypass carries no
    green split.
    """
    del phi6  # the signalised traveller route carries only the phi2 split
    signals = comm.obs_signals
    if not signals:
        return empty_observation_broadcast()

    L_payload: dict[str, np.ndarray] | None = None
    phi_payload: dict[str, np.ndarray] | None = None

    if ObservationSignal.ROUTE_CONGESTION in signals:
        route_queue = route_arrival_queues(queues_by_link, net, sim)
        L_payload = {
            r: np.asarray(route_queue[r], dtype=float) for r in net.traveller_routes
        }

    if ObservationSignal.SIGNAL_CONTROL in signals:
        # Arrival-aligned intersection split, identical to the chosen-route
        # green-split observation (simulator: k + N_l forward look). Only the
        # signalised traveller route (alpha) carries phi.
        sig_ab, _sig_cd = net.signalised_links
        N_ab = net.n_delay(sim.dt_min)[sig_ab]
        k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)
        phi_alpha = np.asarray(phi2, dtype=float)[k_arr]
        phi_payload = {net.traveller_routes[0]: phi_alpha}

    return ObservationBroadcast(L=L_payload, phi=phi_payload)


# ---------------------------------------------------------------------------
# Controller-belief broadcasts for decision-time fusion (Experiment 3/4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeliefBroadcast:
    """The controller's belief shared with travellers, for decision-time fusion.

    Carries the controller's forward-predicted belief about the **intersection
    route** for the upcoming day, arrival-aligned to the traveller's departure
    minute (so a traveller departing at minute ``t`` reads index ``t``):

    * ``mu_L`` / ``var_L``: the controller's predicted queue belief
      ``N(mu_L[t], var_L[t])`` (the QUEUE_BELIEF signal), or ``None`` when not
      shared.
    * ``phi`` / ``var_phi``: the controller's planned green split and its
      variance (the SPLIT_PLAN signal), or ``None`` when not shared.

    Both ``None`` is the baseline (BL) case. Only the signalised intersection
    route is informed (the controller has no belief about the uncongested
    bypass). Compliant travellers fuse these Gaussians into their own posterior
    over the intersection-route latent ``(L, phi)`` *before* choosing, a
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

    * ``BeliefSignal.QUEUE_BELIEF`` (QB) -> ``(mu_L, var_L)``: the controller's
      predicted intersection queue belief. (``L_2`` is the controlling component
      of the intersection route's queue; the free-flow approach links rarely
      queue, so ``L_alpha ~= L_2``.)
    * ``BeliefSignal.SPLIT_PLAN`` (SP) -> ``(phi, var_phi)``: the controller's
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
