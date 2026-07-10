"""Behavioural characterization of the controller-belief communication channel.

Paper Experiment 3 (BL / QB / SP / QB+SP), redesigned. The controller
forward-predicts the upcoming day's queue belief (QB) and its planned green
split (SP) and broadcasts them *before* travellers choose. A **compliant**
traveller fuses the controller's Gaussian into its own posterior at decision
time; a non-compliant traveller ignores it. Crucially the fusion is
**transient**: it informs the route choice but never enters the smoother, so
the traveller's first-hand belief is untouched.

WHAT THESE PIN DOWN
    1. Sharing the controller's queue belief (QB) measurably shifts route
       choice vs. the baseline (the channel is not inert).
    2. Sharing the planned split (SP) likewise shifts route choice.
    3. Non-compliant travellers ignore the broadcast -> identical to baseline.
    4. The fusion is transient: ``begin_day`` does not mutate ``self.state``
       (the smoother stays first-hand-only).

HOW TO READ THE OUTPUT
    Run with ``-s`` to see the narration:

        uv run --extra dev pytest tests/test_belief_informing.py -s --runslow
"""

from __future__ import annotations

import numpy as np
import pytest

from aif_traffic.communication import BeliefBroadcast
from aif_traffic.demand import DemandProfile
from aif_traffic.inference.population import build_population
from aif_traffic.parameters import BeliefSignal, Params
from aif_traffic.simulator import run_experiment

# Full-scale characterization (~6 x 90-day runs): heavy, so gated behind
# --runslow (off per-push CI; run on demand / nightly). See tests/conftest.py
# and .github/workflows/heavy-tests.yml.
pytestmark = pytest.mark.slow

SEED = 7
DAYS = 90  # the full default experiment (past burn-in + window)


def _narrate(title, lines):
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


@pytest.fixture(scope="module")
def runs():
    """BL / QB / SP / QB+SP experiments sharing one base config (full
    compliance, fixed AIF controller)."""
    base = Params.default().with_days(DAYS).with_seed(SEED).with_compliance(1.0)
    out = {
        "BL": run_experiment(base.with_belief_signals(), seeds=[SEED]),
        "QB": run_experiment(
            base.with_belief_signals(BeliefSignal.QUEUE_BELIEF), seeds=[SEED]),
        "SP": run_experiment(
            base.with_belief_signals(BeliefSignal.SPLIT_PLAN), seeds=[SEED]),
        "QBSP": run_experiment(
            base.with_belief_signals(
                BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN), seeds=[SEED]),
    }
    return out


def _max_abs_dP(a, b) -> float:
    return float(np.max(np.abs(a.step["P_alpha"].values - b.step["P_alpha"].values)))


def test_qb_shifts_route_choice(runs):
    """Broadcasting the controller's predicted queue belief (QB) measurably
    changes compliant travellers' route choice vs. the baseline."""
    d = _max_abs_dP(runs["QB"], runs["BL"])
    _narrate(
        "QB (queue belief) shifts route choice vs baseline",
        [
            f"max |P_alpha(QB) - P_alpha(BL)| = {d:.4f}",
            "Expectation: clearly non-zero; compliant travellers fold the",
            "controller's predicted intersection queue into their decision.",
            f"Verdict: {'PASS' if d > 1e-3 else 'FAIL'}.",
        ],
    )
    assert d > 1e-3


def test_sp_shifts_route_choice(runs):
    """Broadcasting the planned green split (SP) measurably changes route
    choice vs. the baseline; travellers anticipate the intersection's
    effective capacity they could not otherwise observe."""
    d = _max_abs_dP(runs["SP"], runs["BL"])
    _narrate(
        "SP (planned split) shifts route choice vs baseline",
        [
            f"max |P_alpha(SP) - P_alpha(BL)| = {d:.4f}",
            "Expectation: clearly non-zero.",
            f"Verdict: {'PASS' if d > 1e-3 else 'FAIL'}.",
        ],
    )
    assert d > 1e-3


def test_non_compliant_population_recovers_baseline():
    """With zero compliance, QB+SP must be bit-identical to BL: nobody fuses
    the broadcast, so it is an exact no-op."""
    base = Params.default().with_days(DAYS).with_seed(SEED).with_compliance(0.0)
    qbsp_nc = run_experiment(
        base.with_belief_signals(
            BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN), seeds=[SEED])
    bl_nc = run_experiment(base.with_belief_signals(), seeds=[SEED])
    max_dP = _max_abs_dP(qbsp_nc, bl_nc)
    max_dsig = float(np.max(np.abs(
        qbsp_nc.cohort["sigma_alpha_post"].values
        - bl_nc.cohort["sigma_alpha_post"].values)))
    _narrate(
        "Zero compliance -> QB+SP collapses onto the baseline",
        [
            f"max |dP_alpha|       = {max_dP:.2e}",
            f"max |d sigma_alpha|  = {max_dsig:.2e}",
            "Expectation: both ~0 (non-compliant travellers ignore the broadcast).",
            f"Verdict: {'PASS' if max_dP < 1e-9 and max_dsig < 1e-9 else 'FAIL'}.",
        ],
    )
    assert max_dP < 1e-9
    assert max_dsig < 1e-9


def test_fusion_is_transient_smoother_state_untouched():
    """The decision-time fusion must NOT mutate the traveller's persistent
    posterior: ``begin_day`` with a belief broadcast leaves ``self.state``
    bit-identical (the smoother stays first-hand-only)."""
    p = Params.default()
    sim = p.sim
    demand = DemandProfile.from_params(sim, p.demand)
    pop = build_population(
        p.population, sim, demand, np.random.default_rng(0),
        route_names=p.network.traveller_routes, signal=p.signal,
    )
    pop.complies[:] = True
    before_mu = np.asarray(pop.state.mu).copy()
    before_tril = np.asarray(pop.state.scale_tril).copy()

    K = sim.K
    bb = BeliefBroadcast(
        mu_L=np.full(K, 60.0), var_L=np.full(K, 9.0),
        phi=np.full(K, 0.3), var_phi=0.0004,
    )
    pop.begin_day(p.efe, np.random.default_rng(1), belief_broadcast=bb)

    same_mu = np.array_equal(before_mu, np.asarray(pop.state.mu))
    same_tril = np.array_equal(before_tril, np.asarray(pop.state.scale_tril))
    chose_each = int((pop.last_choice == 0).sum()), int((pop.last_choice == 1).sum())
    _narrate(
        "Decision-time fusion is transient (smoother state untouched)",
        [
            f"self.state.mu unchanged:        {same_mu}",
            f"self.state.scale_tril unchanged: {same_tril}",
            f"agents choosing (alpha, beta):   {chose_each}",
            "Expectation: the fusion informs the choice but never persists.",
            f"Verdict: {'PASS' if same_mu and same_tril else 'FAIL'}.",
        ],
    )
    assert same_mu and same_tril
    # Sanity: the broadcast actually drove a non-degenerate split of choices.
    assert chose_each[0] > 0 and chose_each[1] > 0
