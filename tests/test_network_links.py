"""Link-level network: incidence, signal capacities, travel times."""

from __future__ import annotations

import numpy as np

from aif_traffic.network import (
    effective_capacities,
    incidence_matrix,
    link_and_route_travel_times,
    link_inflows,
)
from aif_traffic.parameters import NetworkParams, SimParams


def _link_row(net: NetworkParams, link_id: int) -> int:
    return net.link_ids.index(link_id)


def test_incidence_matrix_routes():
    net = NetworkParams()
    A = incidence_matrix(net)
    assert A.shape == (7, 3)
    ai = net.routes.index("alpha")
    bi = net.routes.index("beta")
    gi = net.routes.index("gamma")
    # alpha uses links 1,2,3,4; beta uses 1,5,4; gamma uses 6,7.
    assert A[_link_row(net, 2), ai] == 1.0
    assert A[_link_row(net, 2), bi] == 0.0
    assert A[_link_row(net, 5), bi] == 1.0
    assert A[_link_row(net, 5), ai] == 0.0
    assert A[_link_row(net, 6), gi] == 1.0
    assert A[_link_row(net, 1), ai] == 1.0 and A[_link_row(net, 1), bi] == 1.0


def test_link_inflows_accumulate_shared_links():
    net = NetworkParams()
    K = 5
    Q = {
        "alpha": np.full(K, 100.0),
        "beta": np.full(K, 40.0),
        "gamma": np.full(K, 70.0),
    }
    Ql = link_inflows(Q, net)
    # Link 1 is shared by alpha and beta.
    assert np.allclose(Ql[1], 140.0)
    assert np.allclose(Ql[2], 100.0)   # alpha only
    assert np.allclose(Ql[5], 40.0)    # beta only
    assert np.allclose(Ql[6], 70.0)    # gamma only


def test_signal_capacity_is_green_fraction():
    net = NetworkParams()
    K = 4
    phi2 = np.full(K, 0.45)
    phi6 = np.full(K, 0.45)
    caps = effective_capacities(phi2, phi6, net)
    assert np.allclose(caps[2], 0.45 * net.cbar(2))
    assert np.allclose(caps[6], 0.45 * net.cbar(6))
    # Ordinary link keeps nominal capacity.
    assert np.allclose(caps[1], net.cbar(1))


def test_zero_queue_route_tt_is_sum_of_free_flows():
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    K = sim.K
    queues = {lid: np.zeros(K) for lid in net.link_ids}
    caps = effective_capacities(np.full(K, 0.45), np.full(K, 0.45), net)
    _, tt_route = link_and_route_travel_times(queues, caps, net, sim)
    assert np.allclose(tt_route["alpha"], net.route_free_flow("alpha"))
    assert np.allclose(tt_route["beta"], net.route_free_flow("beta"))
