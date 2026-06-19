"""Communication + compliance mechanism.

These pin the *mechanism* (broadcast -> perceived-cost offset -> choice, gated
by theta and compliance). The exact signal definitions are provisional and not
asserted here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aif_traffic.communication import build_belief_broadcast, build_broadcast
from aif_traffic.parameters import (
    BeliefSignal,
    CohortSpec,
    CommunicationSpec,
    FixedTimeControllerSpec,
    NetworkParams,
    Params,
    PopulationParams,
    SignalType,
    SimParams,
)
from aif_traffic.simulator import run_experiment


def _params(signal_type: SignalType, theta: float, compliance: float) -> Params:
    cohort = CohortSpec(n_agents=80, window_size=2, theta=theta,
                        compliance_fraction=compliance)
    return replace(
        Params.default(),
        sim=SimParams(days=3, h_min=20, dt_min=1, burn_in=0, seed=11,
                      selected_days=(0, 1, 2)),
        population=PopulationParams(cohorts=(cohort,)),
        controller=FixedTimeControllerSpec(),
        comm=CommunicationSpec(signal_type=signal_type),
    )


def test_zero_compliance_matches_no_broadcast():
    """No-one reads the broadcast -> choices identical to the no-info case."""
    res_signal = run_experiment(_params(SignalType.EXTERNALITY, theta=0.5, compliance=0.0))
    res_none = run_experiment(_params(SignalType.NONE, theta=0.5, compliance=0.0))
    assert np.allclose(res_signal.step["P_alpha"], res_none.step["P_alpha"])


def test_zero_theta_neutralises_broadcast():
    """theta = 0 -> the externality offset is zero -> identical to no-info."""
    res_signal = run_experiment(_params(SignalType.EXTERNALITY, theta=0.0, compliance=1.0))
    res_none = run_experiment(_params(SignalType.NONE, theta=0.5, compliance=1.0))
    assert np.allclose(res_signal.step["P_alpha"], res_none.step["P_alpha"])


def _congested_alpha_scenario():
    """A day where the signalised A--B movement (alpha) is oversaturated and the
    bypass (beta) is free-flowing, with a constant balanced split."""
    from aif_traffic.network import simulate_link_queues_const_phi
    from aif_traffic.parameters import SignalParams

    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    signal = SignalParams()
    K = sim.K
    inflow = {
        "alpha": np.full(K, 1600.0),   # over the signalised link-2 capacity
        "beta": np.full(K, 150.0),     # on the high-capacity bypass
        "gamma": np.full(K, 300.0),
    }
    phi2_val = signal.phi_sat / 2.0
    queues, tt_route = simulate_link_queues_const_phi(inflow, phi2_val, net, sim, signal)
    phi2 = np.full(K, phi2_val)
    phi6 = np.full(K, signal.phi_sat - phi2_val)
    return net, sim, inflow, queues, tt_route, phi2, phi6


def test_build_broadcast_msc_is_finite_difference_and_ordered():
    net, sim, inflow, queues, tt_route, phi2, phi6 = _congested_alpha_scenario()
    bc = build_broadcast(
        CommunicationSpec(signal_type=SignalType.MSC),
        tt_route, queues, net, sim, inflow_by_route=inflow, phi2=phi2, phi6=phi6,
    )
    assert bc.signal_type is SignalType.MSC
    assert np.all(bc.value["alpha"] >= 0.0) and np.all(bc.value["beta"] >= 0.0)
    # Adding a vehicle to the oversaturated A--B approach imposes far more
    # marginal social cost than adding one to the free-flowing bypass.
    assert bc.value["alpha"].mean() > bc.value["beta"].mean()


def test_build_broadcast_externality_nonneg_and_ordered():
    net, sim, inflow, queues, tt_route, phi2, phi6 = _congested_alpha_scenario()
    bc = build_broadcast(
        CommunicationSpec(signal_type=SignalType.EXTERNALITY),
        tt_route, queues, net, sim, inflow_by_route=inflow, phi2=phi2, phi6=phi6,
    )
    assert bc.signal_type is SignalType.EXTERNALITY
    assert np.all(bc.value["alpha"] >= 0.0)
    assert np.all(bc.value["beta"] >= 0.0)
    # The congested route carries the larger externality.
    assert bc.value["alpha"].mean() > bc.value["beta"].mean()


def test_none_broadcast_is_zero():
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    K = sim.K
    tt_route = {r: np.full(K, 5.0) for r in net.routes}
    queues = {lid: np.zeros(K) for lid in net.link_ids}
    bc = build_broadcast(
        CommunicationSpec(signal_type=SignalType.NONE), tt_route, queues, net, sim,
    )
    assert np.allclose(bc.value["alpha"], 0.0)
    assert np.allclose(bc.value["beta"], 0.0)


# --------------------------------------------------------------------------
# Belief-informing broadcasts (paper Experiment 3: BL / CG / SN / CG+SN)
# --------------------------------------------------------------------------
def test_belief_broadcast_baseline_is_empty():
    """BL (no belief signals) -> both payloads None: no information shared."""
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    bb = build_belief_broadcast(CommunicationSpec(), queues, phi2, phi6, net, sim)
    assert bb.L is None and bb.phi is None


def test_belief_broadcast_cg_matches_route_arrival_queues():
    """CG broadcasts L_hat_r equal to the arrival-aligned route queue -- the
    same quantity a traveller senses first-hand on the route it takes."""
    from aif_traffic.network import route_arrival_queues

    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset({BeliefSignal.CONGESTION})),
        queues, phi2, phi6, net, sim,
    )
    assert bb.phi is None  # CG only
    route_q = route_arrival_queues(queues, net, sim)
    for r in net.traveller_routes:
        assert np.allclose(bb.L[r], route_q[r])
    # The oversaturated intersection route carries the larger queue.
    assert bb.L["alpha"].mean() > bb.L["beta"].mean()


def test_belief_broadcast_sn_is_alpha_only_and_arrival_aligned():
    """SN broadcasts phi_hat for the signalised route only, arrival-aligned."""
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset({BeliefSignal.GREEN_SPLIT})),
        queues, phi2, phi6, net, sim,
    )
    assert bb.L is None  # SN only
    assert set(bb.phi.keys()) == {"alpha"}  # phi inert on the bypass
    sig_ab, _ = net.signalised_links
    N_ab = net.n_delay(sim.dt_min)[sig_ab]
    k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)
    assert np.allclose(bb.phi["alpha"], np.asarray(phi2)[k_arr])


def test_belief_broadcast_cg_sn_carries_both():
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset(
            {BeliefSignal.CONGESTION, BeliefSignal.GREEN_SPLIT})),
        queues, phi2, phi6, net, sim,
    )
    assert bb.L is not None and bb.phi is not None


def test_baseline_belief_signals_match_no_information():
    """An empty belief-signal set is bit-identical to the default (no belief
    channel): the BL path draws no randomness and folds no observations."""
    base = _params(SignalType.NONE, theta=0.5, compliance=1.0)
    res_bl = run_experiment(replace(base, comm=CommunicationSpec()))
    res_default = run_experiment(base)
    assert np.allclose(res_bl.step["P_alpha"], res_default.step["P_alpha"])
    assert np.allclose(
        res_bl.cohort["sigma_beta_post"], res_default.cohort["sigma_beta_post"]
    )
