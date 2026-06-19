"""Behavioural characterization of the belief-informing communication channel.

Paper Experiment 3 (BL / CG / SN / CG+SN). Unlike the cost-offset advisory
(which only nudges route *choice* via theta), these broadcasts feed the
traveller's *belief update*: a compliant traveller folds the controller's
broadcast of route queue (CG) and/or green split (SN) into its Gaussian belief
about routes it did NOT take that day.

WHAT THESE PIN DOWN
    1. CG lowers a traveller's uncertainty about the route it usually avoids
       (it now hears that route's queue without driving it).
    2. SN lowers uncertainty about the intersection green split for travellers
       who habitually take the bypass.
    3. Non-compliant travellers ignore the broadcast -> identical to baseline.
    4. The broadcast never overrides first-hand experience: on the route a
       traveller actually took, the broadcast observation is suppressed (no
       double counting). This is the key correctness invariant.

HOW TO READ THE OUTPUT
    Run with ``-s`` to see the narration:

        uv run --extra dev pytest tests/test_belief_informing.py -s

    Each test prints what it expected, the observed numbers, and a verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from aif_traffic.communication import BeliefBroadcast
from aif_traffic.demand import DemandProfile
from aif_traffic.inference.population import build_population
from aif_traffic.parameters import BeliefSignal, Params
from aif_traffic.simulator import run_experiment

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
    """BL / CG / SN / CG+SN experiments sharing one base config (full
    compliance), snapshotting the last day for belief-uncertainty readouts."""
    base = Params.default().with_days(DAYS).with_seed(SEED).with_compliance(1.0)
    last = DAYS - 1
    out = {}
    for name, params in {
        "BL": base.with_belief_signals(),
        "CG": base.with_belief_signals(BeliefSignal.CONGESTION),
        "SN": base.with_belief_signals(BeliefSignal.GREEN_SPLIT),
        "CGSN": base.with_belief_signals(
            BeliefSignal.CONGESTION, BeliefSignal.GREEN_SPLIT
        ),
    }.items():
        out[name] = run_experiment(params, seeds=[SEED], snapshot_days=[last])
    return out, last


def _last_day_mean(res, col, day):
    return float(res.cohort.loc[res.cohort["day"] == day, col].mean())


def test_cg_lowers_uncertainty_about_the_avoided_route(runs):
    """CG broadcasts route queues. The intersection (alpha) is congested and
    many travellers divert to the bypass, so alpha is the route they seldom
    take -- and therefore the route they are most uncertain about. Hearing its
    queue should shrink that uncertainty relative to baseline."""
    out, last = runs
    bl = _last_day_mean(out["BL"], "sigma_alpha_post", last)
    cg = _last_day_mean(out["CG"], "sigma_alpha_post", last)
    _narrate(
        "CG reduces uncertainty about the usually-avoided intersection route",
        [
            f"alpha predictive SD on day {last}:  BL = {bl:.3f}   CG = {cg:.3f}",
            "Expectation: CG < BL (the broadcast queue informs the route most",
            "travellers do not drive, so their alpha belief sharpens).",
            f"Verdict: {'PASS' if cg < bl else 'FAIL'} "
            f"(reduction = {bl - cg:.3f}).",
        ],
    )
    assert cg < bl


def test_sn_lowers_green_split_uncertainty_for_bypass_users(runs):
    """SN broadcasts the intersection green split. Travellers who habitually
    take the bypass never observe the split first-hand, so their phi belief is
    diffuse under baseline; the broadcast should sharpen it."""
    out, last = runs
    snap_bl = out["BL"].snapshots[(SEED, last)]
    snap_sn = out["SN"].snapshots[(SEED, last)]
    # Restrict to travellers who took the bypass (alpha is unchosen for them).
    bypass_bl = snap_bl["last_choice"] == 1
    bypass_sn = snap_sn["last_choice"] == 1
    sd_bl = float(snap_bl["phi_sd_alpha"][bypass_bl].mean())
    sd_sn = float(snap_sn["phi_sd_alpha"][bypass_sn].mean())
    _narrate(
        "SN sharpens the green-split belief of habitual bypass users",
        [
            f"alpha green-split belief SD (bypass users, day {last}):",
            f"    BL = {sd_bl:.4f}   SN = {sd_sn:.4f}",
            "Expectation: SN < BL (they hear the split they would otherwise",
            "only learn by taking the intersection).",
            f"Verdict: {'PASS' if sd_sn < sd_bl else 'FAIL'} "
            f"(reduction = {sd_bl - sd_sn:.4f}).",
        ],
    )
    assert sd_sn < sd_bl


def test_non_compliant_population_recovers_baseline(runs):
    """If no-one reads the broadcast, CG must be bit-identical to BL."""
    out, _last = runs
    base = Params.default().with_days(DAYS).with_seed(SEED).with_compliance(0.0)
    cg_nc = run_experiment(
        base.with_belief_signals(BeliefSignal.CONGESTION), seeds=[SEED]
    )
    bl_nc = run_experiment(base.with_belief_signals(), seeds=[SEED])
    max_dP = float(
        np.max(np.abs(cg_nc.step["P_alpha"].values - bl_nc.step["P_alpha"].values))
    )
    max_dsig = float(
        np.max(
            np.abs(
                cg_nc.cohort["sigma_alpha_post"].values
                - bl_nc.cohort["sigma_alpha_post"].values
            )
        )
    )
    _narrate(
        "Zero compliance -> CG collapses onto the baseline",
        [
            f"max |dP_alpha|        = {max_dP:.2e}",
            f"max |d sigma_alpha|   = {max_dsig:.2e}",
            "Expectation: both ~0 (non-compliant travellers ignore the signal).",
            f"Verdict: {'PASS' if max_dP < 1e-9 and max_dsig < 1e-9 else 'FAIL'}.",
        ],
    )
    assert max_dP < 1e-9
    assert max_dsig < 1e-9


def test_broadcast_for_chosen_route_is_suppressed():
    """Double-count guard: a compliant traveller's broadcast observation is
    masked OFF on the route it actually took (first-hand experience wins) and
    ON for the route it did not. Verified directly on the gate."""
    p = Params.default()
    sim = p.sim
    demand = DemandProfile.from_params(sim, p.demand)
    rng = np.random.default_rng(0)
    pop = build_population(
        p.population, sim, demand, rng,
        route_names=p.network.traveller_routes, signal=p.signal,
    )
    pop.complies[:] = True
    pop.last_choice = np.zeros(pop.N, dtype=int)  # everyone took alpha (route 0)
    K = sim.K
    bb = BeliefBroadcast(
        L={"alpha": np.full(K, 9999.0), "beta": np.full(K, 10.0)}, phi=None,
    )
    pop._append_belief_broadcast(bb, pop.departure_time, rng=None)
    mask_alpha = pop._belief_mask_L[:, 0, -1]  # chosen route
    mask_beta = pop._belief_mask_L[:, 1, -1]   # unchosen route
    _narrate(
        "Broadcast is suppressed on the route the traveller actually took",
        [
            "Everyone chose alpha; controller broadcasts queues for both routes.",
            f"belief mask on alpha (chosen)   = {mask_alpha.mean():.2f} (want 0)",
            f"belief mask on beta  (unchosen) = {mask_beta.mean():.2f} (want 1)",
            "Expectation: chosen-route broadcast masked off (no double counting);",
            "unchosen-route broadcast active for compliant travellers.",
            f"Verdict: {'PASS' if mask_alpha.max() == 0 and mask_beta.min() == 1 else 'FAIL'}.",
        ],
    )
    assert np.all(mask_alpha == 0.0)
    assert np.all(mask_beta == 1.0)
