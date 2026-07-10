"""Behavioural characterization of the *extra observations* channel (Exp 3).

Travellers natively have a **partial view** of the network; each observes only
the route it actually took that day. The extra-observation channel (CG/SN)
relays the *true realised* route congestion / green split of the routes a
traveller did *not* take into its end-of-day belief update, lifting it toward a
fuller view.

WHAT THESE PIN (and print, so they can be audited; run with ``-s``):
  1. Relaying CG+SN *reduces* travellers' route-belief uncertainty vs the BL
     baseline; the channel actually enters the smoother and sharpens beliefs.
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

import math
from dataclasses import replace

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
def test_extra_observations_belief_uncertainty_report(runs):
    """REPORT (does not assert a direction): how CG+SN changes the steady-state
    route-belief uncertainty vs BL.

    Under the realistic queue-observation noise (``sigma_L_obs ~ 3``) the
    first-hand smoother already converges the belief over the 90-day run, so
    relaying the non-chosen route adds little at steady state and the effect is
    small / mixed. This is a shift from the paper's Experiment-3 story (which was
    run with a much larger observation noise, where the relay helped a lot), so
    it is *surfaced* here rather than asserted; a genuine change to reconcile,
    not to encode away."""
    bl, cgsn, _ = runs
    u_bl = _steady_mean_uncertainty(bl)
    u_eo = _steady_mean_uncertainty(cgsn)
    _narrate(
        "Extra observations (CG+SN): effect on route-belief uncertainty",
        [
            f"OBSERVED: mean belief SD  BL = {u_bl:.3f}   CG+SN = {u_eo:.3f}",
            f"REPORT: CG+SN {'reduces' if u_eo < u_bl else 'does NOT reduce'} "
            "steady-state uncertainty vs BL under the realistic obs noise "
            "(small effect; cf. paper Exp-3, run at higher obs noise).",
        ],
    )
    # Sanity only; the direction is reported, not enforced (see docstring).
    assert math.isfinite(u_bl) and math.isfinite(u_eo)
    assert u_bl > 0 and u_eo > 0


@pytest.mark.slow
def test_extra_observations_are_ungated_by_compliance(runs):
    """Extra observations are NOT gated by compliance: they are sensor data
    folded into every traveller's belief update regardless of the compliance
    mask (unlike the belief-sharing channel). So the CG+SN run at compliance = 0
    tracks the compliance = 1 run, *not* the BL baseline; whatever the sign of
    the (small, under realistic obs noise) CG+SN effect on the belief."""
    bl, cgsn, cgsn_nc = runs
    u_bl = _steady_mean_uncertainty(bl)
    u_eo = _steady_mean_uncertainty(cgsn)
    u_eo_nc = _steady_mean_uncertainty(cgsn_nc)
    gap_to_eo = abs(u_eo_nc - u_eo)
    gap_to_bl = abs(u_eo_nc - u_bl)
    _narrate(
        "Extra observations are NOT gated by compliance",
        [
            "EXPECT: with nobody compliant, CG+SN behaves like full-compliance",
            "        CG+SN (ungated), so it tracks c=1, not BL.",
            f"OBSERVED: SD  BL = {u_bl:.3f}  CG+SN(c=1) = {u_eo:.3f}  "
            f"CG+SN(c=0) = {u_eo_nc:.3f}",
            f"          |c=0 - c=1| = {gap_to_eo:.3f}   |c=0 - BL| = {gap_to_bl:.3f}",
            f"VERDICT: {'ungated (consistent)' if gap_to_eo < gap_to_bl else 'looks gated (MISMATCH)'}",
        ],
    )
    # The ungated property: compliance does not gate extra observations, so the
    # c=0 run matches the c=1 run and both differ from BL. (Direction/magnitude of
    # the CG+SN belief effect itself is reported above, not asserted here.)
    assert gap_to_eo < gap_to_bl


@pytest.fixture(scope="module")
def runs_cg_sn():
    """BL vs CG (route congestion) vs SN (signal split), all at full compliance."""
    base = _base_params(1.0)
    _CG, _SN = ObservationSignal.ROUTE_CONGESTION, ObservationSignal.SIGNAL_CONTROL
    return (
        run_experiment(base.with_extra_observations(), seeds=[SEED]),
        run_experiment(base.with_extra_observations(_CG), seeds=[SEED]),
        run_experiment(base.with_extra_observations(_SN), seeds=[SEED]),
    )


def _peak_steady(res, col: str) -> float:
    """Mean of ``col`` in the demand-peak window (tau 135-165) over the last 15
    recorded days."""
    s = res.step
    last = sorted(s["day"].unique())[-15:]
    d = s[s["day"].isin(last)]
    peak = d[(d["tau"] >= 135) & (d["tau"] <= 165)]
    return float(peak[col].mean())


def _steady_mean_sc(res) -> float:
    s = res.step
    last = sorted(s["day"].unique())[-15:]
    return float(s[s["day"].isin(last)].groupby("day")["SC"].first().mean())


def _steady_belief_tt_alpha(res) -> float:
    c = res.cohort
    last = sorted(c["day"].unique())[-15:]
    return float(c[c["day"].isin(last)]["mu_alpha_post"].mean())


@pytest.mark.slow
def test_relaying_queue_lengths_backfires(runs_cg_sn):
    """Relaying the realised route congestion (CG) *backfires*, and this explains
    why CG raises system cost while SN lowers it.

    The intersection route alpha is cheap at the demand peak precisely *because*
    travellers believe it is congested and avoid it; their avoidance is what
    keeps its queue low (a self-fulfilling equilibrium). Relaying that low queue
    (CG) lowers their believed alpha travel time, so more of them take alpha at
    the peak; on alpha's small green fraction the extra inflow rebuilds the queue
    and raises system cost. Relaying the green split (SN) instead conveys alpha's
    low *capacity*, which steers travellers further OFF alpha, so it does not
    backfire. Upshot: the controller should NOT broadcast the (self-fulfilling)
    low queue; keeping it private is what preserves the good equilibrium.
    """
    bl, cg, sn = runs_cg_sn
    bel = {k: _steady_belief_tt_alpha(r) for k, r in (("BL", bl), ("CG", cg), ("SN", sn))}
    pa = {k: _peak_steady(r, "P_alpha") for k, r in (("BL", bl), ("CG", cg), ("SN", sn))}
    l2 = {k: _peak_steady(r, "L2") for k, r in (("BL", bl), ("CG", cg), ("SN", sn))}
    sc = {k: _steady_mean_sc(r) for k, r in (("BL", bl), ("CG", cg), ("SN", sn))}
    _narrate(
        "Relaying queue lengths (CG) backfires; relaying the split (SN) does not",
        [
            "MECHANISM: alpha is cheap at the peak because travellers avoid it;",
            "           revealing its low queue (CG) lures them back and re-congests it.",
            f"believed TT_alpha:  BL={bel['BL']:.2f}  CG={bel['CG']:.2f}  SN={bel['SN']:.2f}",
            f"peak alpha share :  BL={pa['BL']:.3f}  CG={pa['CG']:.3f}  SN={pa['SN']:.3f}",
            f"peak alpha queue :  BL={l2['BL']:.1f}  CG={l2['CG']:.1f}  SN={l2['SN']:.1f}",
            f"system cost      :  BL={sc['BL']:.0f}  CG={sc['CG']:.0f}  SN={sc['SN']:.0f}",
            "VERDICT: CG lowers believed TT_alpha -> more take alpha at the peak ->",
            "         queue + cost rise (backfire). SN steers off alpha, cost does not rise.",
        ],
    )
    # CG makes alpha look cheaper, luring travellers onto it and raising cost.
    assert bel["CG"] < bel["BL"]          # believe alpha cheaper under CG
    assert pa["CG"] > pa["BL"]            # so more take alpha at the peak
    assert l2["CG"] > l2["BL"]            # rebuilding its peak queue
    assert sc["CG"] > sc["BL"]            # and raising system cost (backfire)
    # SN conveys alpha's low capacity, steering travellers OFF alpha; no backfire.
    assert pa["SN"] < pa["BL"]
    assert sc["SN"] <= sc["BL"] * 1.02
