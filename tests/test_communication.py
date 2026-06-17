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


def test_build_broadcast_externality_nonneg_and_ordered():
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    K = sim.K
    # alpha is more delayed than beta.
    tt_route = {
        "alpha": np.full(K, net.route_free_flow("alpha") + 5.0),
        "beta": np.full(K, net.route_free_flow("beta") + 1.0),
        "gamma": np.full(K, net.route_free_flow("gamma")),
    }
    queues = {lid: np.zeros(K) for lid in net.link_ids}
    bc = build_broadcast(
        CommunicationSpec(signal_type=SignalType.EXTERNALITY),
        tt_route, queues, net, sim,
    )
    assert bc.signal_type is SignalType.EXTERNALITY
    assert np.all(bc.value["alpha"] >= 0.0)
    assert np.all(bc.value["beta"] >= 0.0)
    # The more-delayed route carries the larger advisory.
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
