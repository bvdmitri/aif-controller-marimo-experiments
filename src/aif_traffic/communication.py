"""Inter-layer communication: the controller's channels to travellers.

The controller has a network-wide view and offers two orthogonal
controller -> traveller channels, both assembled here from the realised day:

* **Extra observations** (:class:`ObservationBroadcast`, paper Experiment 3
  default, BL/CG/SN/CG+SN). Travellers natively observe only the route they
  took; this channel relays the **realised** route queue ``L_r`` (CG) and/or
  green split ``phi_r`` (SN) of the routes they did *not* take, fed straight into
  the traveller's smoother as *observations* (see :mod:`inference.population` /
  :mod:`inference.filter`). It reaches all travellers and works with any
  controller. The values are raw readings: the smoother treats them as noisy
  readings of the latent ``L``/``phi``.
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

from .network import route_arrival_queues
from .parameters import (
    BeliefSignal,
    CommunicationSpec,
    NetworkParams,
    ObservationSignal,
    SimParams,
)


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

    These are raw observations: the smoother treats them as noisy readings of the
    latent ``L`` and ``phi``. They are the **realised** (noisy) values of the day,
    so the relay simply lifts the traveller's partial observation to a fuller one.
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
# Controller-belief broadcasts for decision-time fusion (the parked
# belief-sharing channel, optional in the Experiment 3 dropdown)
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
