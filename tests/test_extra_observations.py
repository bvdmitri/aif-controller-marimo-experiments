"""Behavioural characterization of the *extra observations* channel (Exp 3).

Travellers natively have a **partial view** of the network -- each observes only
the route it actually took that day. The extra-observation channel (CG/SN)
relays the *true realised* route congestion / green split of the routes a
traveller did *not* take into its end-of-day belief update, lifting it toward a
fuller view.

WHAT THESE PIN (and print, so they can be audited -- run with ``-s``):
  1. Relaying CG+SN *reduces* travellers' route-belief uncertainty vs the BL
     baseline -- the channel actually enters the smoother and sharpens beliefs.
  2. The effect is present at **zero compliance**: extra observations reach
     *every* traveller (they are sensor data, not a recommendation to follow),
     unlike the belief-sharing (QB/SP) channel, which only compliant travellers
     fuse. This is the key distinction motivating the channel.

These run the full default population; the BL-vs-CG+SN pair is a multi-run
comparison, so it is marked ``slow`` and skipped unless ``--runslow`` is passed
(see ``tests/conftest.py``). The fast unit/communication tests cover the
mechanism itself.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aif_traffic.parameters import ObservationSignal, Params, SimParams
from aif_traffic.simulator import run_experiment

SEED = 42
DAYS = 90  # the full default experiment (reaches the learned equilibrium)


def _base_params(compliance: float) -> Params:
    """Default-scale AIF experiment at a given compliance (extra obs ignore it)."""
    return replace(
        Params(), sim=replace(SimParams(), days=DAYS, seed=SEED),
    ).with_compliance(compliance)


@pytest.fixture(scope="module")
def runs():
    """BL vs CG+SN at full compliance, and CG+SN at *zero* compliance."""
    eo = (ObservationSignal.ROUTE_CONGESTION, ObservationSignal.SIGNAL_CONTROL)
    base = _base_params(1.0)
    bl = run_experiment(base.with_extra_observations(), seeds=[SEED])
    cgsn = run_experiment(base.with_extra_observations(*eo), seeds=[SEED])
    cgsn_no_comply = run_experiment(
        _base_params(0.0).with_extra_observations(*eo), seeds=[SEED]
    )
    return bl, cgsn, cgsn_no_comply


def _narrate(title, lines):
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


def _steady_mean_uncertainty(res) -> float:
    """Mean route-belief SD (averaged over both routes) over the last 15 days."""
    co = res.cohort
    last = sorted(co["day"].unique())[-15:]
    d = co[co["day"].isin(last)]
    return float(0.5 * (d["sigma_alpha_post"].mean() + d["sigma_beta_post"].mean()))


@pytest.mark.slow
def test_extra_observations_reduce_belief_uncertainty(runs):
    """CG+SN relayed to travellers narrows their route-belief uncertainty: each
    now also sees the route it did not take, so the smoother is no longer left at
    the inflated prior for that route."""
    bl, cgsn, _ = runs
    u_bl = _steady_mean_uncertainty(bl)
    u_eo = _steady_mean_uncertainty(cgsn)
    _narrate(
        "Extra observations (CG+SN) reduce route-belief uncertainty",
        [
            "EXPECT: relaying the true non-chosen-route queue/split lowers the",
            "        population-mean route-belief SD vs the BL baseline.",
            f"OBSERVED: mean belief SD  BL = {u_bl:.3f}   CG+SN = {u_eo:.3f}",
            f"VERDICT: {'reduced (consistent)' if u_eo < u_bl else 'NOT reduced (MISMATCH)'}",
        ],
    )
    assert u_eo < u_bl


@pytest.mark.slow
def test_extra_observations_are_ungated_by_compliance(runs):
    """The CG+SN effect survives at compliance = 0: extra observations reach
    every traveller (sensor data, not a recommendation), unlike belief sharing.
    The zero-compliance CG+SN belief is far closer to the full-compliance CG+SN
    belief than to BL."""
    bl, cgsn, cgsn_nc = runs
    u_bl = _steady_mean_uncertainty(bl)
    u_eo = _steady_mean_uncertainty(cgsn)
    u_eo_nc = _steady_mean_uncertainty(cgsn_nc)
    gap_to_eo = abs(u_eo_nc - u_eo)
    gap_to_bl = abs(u_eo_nc - u_bl)
    _narrate(
        "Extra observations are NOT gated by compliance",
        [
            "EXPECT: with nobody compliant, CG+SN still sharpens beliefs (it is",
            "        ungated) -- so its uncertainty tracks full-compliance CG+SN,",
            "        not BL.",
            f"OBSERVED: SD  BL = {u_bl:.3f}  CG+SN(c=1) = {u_eo:.3f}  "
            f"CG+SN(c=0) = {u_eo_nc:.3f}",
            f"          |c=0 - c=1| = {gap_to_eo:.3f}   |c=0 - BL| = {gap_to_bl:.3f}",
            f"VERDICT: {'ungated (consistent)' if gap_to_eo < gap_to_bl else 'looks gated (MISMATCH)'}",
        ],
    )
    assert u_eo_nc < u_bl            # still sharper than the no-information baseline
    assert gap_to_eo < gap_to_bl     # and close to the full-compliance CG+SN run
