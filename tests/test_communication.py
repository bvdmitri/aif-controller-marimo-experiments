"""Communication + compliance mechanism.

These pin the *mechanism* (broadcast -> perceived-cost offset -> choice, gated
by theta and compliance). The exact signal definitions are provisional and not
asserted here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aif_traffic.communication import build_broadcast
from aif_traffic.parameters import (
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
