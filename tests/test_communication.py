"""Communication + compliance mechanism.

These pin the *mechanism* (broadcast -> perceived-cost offset -> choice, gated
by theta and compliance). The exact signal definitions are provisional and not
asserted here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aif_traffic.communication import (
    build_belief_broadcast,
    build_broadcast,
    build_observation_broadcast,
)
from aif_traffic.control.interface import QueueForecast
from aif_traffic.network import route_arrival_queues
from aif_traffic.parameters import (
    BeliefSignal,
    CohortSpec,
    CommunicationSpec,
    FixedTimeControllerSpec,
    NetworkParams,
    ObservationSignal,
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


def test_build_broadcast_records_raw_msc_diagnostics():
    """The out_diagnostics hook captures the raw (unclipped) per-route MSC."""
    net, sim, inflow, queues, tt_route, phi2, phi6 = _congested_alpha_scenario()
    diag: dict = {}
    bc = build_broadcast(
        CommunicationSpec(signal_type=SignalType.EXTERNALITY),
        tt_route, queues, net, sim, inflow_by_route=inflow, phi2=phi2, phi6=phi6,
        out_diagnostics=diag,
    )
    assert "msc" in diag and set(diag["msc"]) == set(net.traveller_routes)
    for r in net.traveller_routes:
        assert diag["msc"][r].shape == (sim.K,)
        assert np.all(np.isfinite(diag["msc"][r]))
    # The advisory is the clipped externality of the recorded raw MSC.
    assert np.allclose(
        bc.value["alpha"],
        np.maximum(diag["msc"]["alpha"] - tt_route["alpha"], 0.0),
    )
    # A vehicle's marginal social cost includes at least its own travel time,
    # so the raw MSC is strictly positive on both routes.
    assert diag["msc"]["alpha"].min() > 0.0
    assert diag["msc"]["beta"].min() > 0.0


def test_msc_columns_recorded_only_with_msc_signal():
    """MSC_alpha/MSC_beta step columns exist iff the advisory computes MSC."""
    res_ext = run_experiment(_params(SignalType.EXTERNALITY, theta=0.5, compliance=1.0))
    assert {"MSC_alpha", "MSC_beta"} <= set(res_ext.step.columns)
    assert np.isfinite(res_ext.step["MSC_alpha"]).all()
    assert np.isfinite(res_ext.step["MSC_beta"]).all()

    res_none = run_experiment(_params(SignalType.NONE, theta=0.5, compliance=1.0))
    assert "MSC_alpha" not in res_none.step.columns
    assert "MSC_beta" not in res_none.step.columns


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
# Controller-belief broadcasts for decision-time fusion (Exp 3: BL/QB/SP/QB+SP)
# --------------------------------------------------------------------------
def _toy_forecast(K: int) -> QueueForecast:
    """A synthetic controller forecast: a ramping queue belief with growing
    variance and a sweeping planned split."""
    return QueueForecast(
        mu_L=np.linspace(0.0, 100.0, K),
        var_L=np.linspace(25.0, 400.0, K),
        phi2=np.linspace(0.3, 0.5, K),
        var_phi=0.0004,
    )


def test_belief_broadcast_baseline_is_empty():
    """BL (no belief signals) -> empty payload, even given a forecast."""
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    bb = build_belief_broadcast(CommunicationSpec(), _toy_forecast(sim.K), net, sim)
    assert bb.is_empty()


def test_belief_broadcast_none_forecast_is_empty():
    """A controller that forecasts nothing (e.g. a baseline) yields BL."""
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset({BeliefSignal.QUEUE_BELIEF})),
        None, net, sim,
    )
    assert bb.is_empty()


def test_belief_broadcast_qb_is_arrival_aligned_queue_belief():
    """QB carries the controller's predicted queue belief (mean+variance),
    arrival-aligned (k + N_2); the split is not shared."""
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    fc = _toy_forecast(sim.K)
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset({BeliefSignal.QUEUE_BELIEF})),
        fc, net, sim,
    )
    assert bb.phi is None and bb.var_phi is None  # QB only
    sig_ab, _ = net.signalised_links
    N_ab = net.n_delay(sim.dt_min)[sig_ab]
    k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)
    assert np.allclose(bb.mu_L, fc.mu_L[k_arr])
    assert np.allclose(bb.var_L, fc.var_L[k_arr])


def test_belief_broadcast_sp_is_arrival_aligned_split_plan():
    """SP carries the planned split (and its variance), arrival-aligned; the
    queue belief is not shared."""
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    fc = _toy_forecast(sim.K)
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset({BeliefSignal.SPLIT_PLAN})),
        fc, net, sim,
    )
    assert bb.mu_L is None and bb.var_L is None  # SP only
    sig_ab, _ = net.signalised_links
    N_ab = net.n_delay(sim.dt_min)[sig_ab]
    k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)
    assert np.allclose(bb.phi, fc.phi2[k_arr])
    assert bb.var_phi == pytest.approx(fc.var_phi)


def test_belief_broadcast_qb_sp_carries_both():
    net = NetworkParams()
    sim = SimParams(h_min=10, dt_min=1)
    bb = build_belief_broadcast(
        CommunicationSpec(belief_signals=frozenset(
            {BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN})),
        _toy_forecast(sim.K), net, sim,
    )
    assert bb.mu_L is not None and bb.phi is not None


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


# --------------------------------------------------------------------------
# Extra-observation broadcasts folded into the smoother (Exp 3: BL/CG/SN/CG+SN)
# --------------------------------------------------------------------------
def test_observation_broadcast_baseline_is_empty():
    """BL (no obs signals) -> empty payload, even given a realised day."""
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    ob = build_observation_broadcast(CommunicationSpec(), queues, phi2, phi6, net, sim)
    assert ob.is_empty()


def test_observation_broadcast_cg_is_arrival_aligned_route_queue():
    """CG carries the true realised route queue for every traveller route; the
    split is not relayed."""
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    ob = build_observation_broadcast(
        CommunicationSpec(obs_signals=frozenset({ObservationSignal.ROUTE_CONGESTION})),
        queues, phi2, phi6, net, sim,
    )
    assert ob.phi is None  # CG only
    route_q = route_arrival_queues(queues, net, sim)
    for r in net.traveller_routes:
        assert np.allclose(ob.L[r], route_q[r])
    # The oversaturated A--B (alpha) route queue dwarfs the free-flowing bypass.
    assert ob.L["alpha"].mean() > ob.L["beta"].mean()


def test_observation_broadcast_sn_is_arrival_aligned_split():
    """SN carries the true realised green split for the signalised route only,
    arrival-aligned (k + N_2); the route queue is not relayed."""
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    ob = build_observation_broadcast(
        CommunicationSpec(obs_signals=frozenset({ObservationSignal.SIGNAL_CONTROL})),
        queues, phi2, phi6, net, sim,
    )
    assert ob.L is None  # SN only
    assert set(ob.phi) == {net.traveller_routes[0]}  # signalised route only
    sig_ab, _ = net.signalised_links
    N_ab = net.n_delay(sim.dt_min)[sig_ab]
    k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)
    assert np.allclose(ob.phi[net.traveller_routes[0]], np.asarray(phi2)[k_arr])


def test_observation_broadcast_cg_sn_carries_both():
    net, sim, _inflow, queues, _tt, phi2, phi6 = _congested_alpha_scenario()
    ob = build_observation_broadcast(
        CommunicationSpec(obs_signals=frozenset(
            {ObservationSignal.ROUTE_CONGESTION, ObservationSignal.SIGNAL_CONTROL})),
        queues, phi2, phi6, net, sim,
    )
    assert ob.L is not None and ob.phi is not None


def test_baseline_obs_signals_match_no_information():
    """An empty obs-signal set is bit-identical to the default (no extra-obs
    channel): the BL fold has all-zero masks (an exact no-op in the smoother)."""
    base = _params(SignalType.NONE, theta=0.5, compliance=1.0)
    res_bl = run_experiment(
        replace(base, comm=CommunicationSpec(obs_signals=frozenset()))
    )
    res_default = run_experiment(base)
    assert np.allclose(res_bl.step["P_alpha"], res_default.step["P_alpha"])
    assert np.allclose(
        res_bl.cohort["sigma_beta_post"], res_default.cohort["sigma_beta_post"]
    )


def test_extra_observations_change_belief_about_non_chosen_route():
    """CG+SN relayed to all travellers shifts the smoother posterior vs BL:
    folding the true non-chosen-route queue/split is *not* a no-op (unlike the
    masked-off baseline). Asserts the channel actually enters the belief update."""
    base = _params(SignalType.NONE, theta=0.0, compliance=0.0)
    res_bl = run_experiment(replace(base, comm=CommunicationSpec()))
    res_eo = run_experiment(replace(base, comm=CommunicationSpec(
        obs_signals=frozenset(
            {ObservationSignal.ROUTE_CONGESTION, ObservationSignal.SIGNAL_CONTROL}))))
    # Compliance is 0, so this difference cannot come from any decision-time
    # fusion -- only the extra-observation belief fold can move it.
    assert not np.allclose(
        res_bl.cohort["sigma_beta_post"], res_eo.cohort["sigma_beta_post"]
    )
