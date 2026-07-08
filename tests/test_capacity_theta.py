"""Social internalisation theta vs bypass capacity, and the advisory cobweb.

WHY THIS EXISTS
    Sweeping theta 0->1 barely moves system cost at the default (uncongestable)
    bypass: diverting off the intersection is free, so the externality wedge that
    theta acts on is tiny. Throttling the bypass (link 5) should give theta
    something real to redistribute. Investigating that surfaced a sharper,
    non-obvious behaviour we want to pin down:

      1. With the bypass throttled, the *un-smoothed* externality advisory makes
         theta BACKFIRE: it is built from yesterday's state and acted on today, a
         one-day-stale feedback that drives a violent day-to-day route-choice
         cobweb (P_alpha flips ~0.24<->0.78 every day), so alternating overloads
         send total system cost far above the theta=0 baseline.
      2. Averaging the advisory over a multi-day window damps the cobweb
         monotonically, and past a threshold (~20-25 days here) theta stops
         backfiring and slightly HELPS even at the throttled scale -- i.e. the
         internalisation lever works once the signal is temporally stable.
      3. At the default full bypass capacity theta is nearly inert (the known
         result), because the bypass never congests.
      4. The SEQUENTIAL externality advisory is the other fix: instead of one
         value broadcast to everyone, it hands each traveller a rank-indexed
         marginal cost from an incremental assignment, so the population splits
         instead of herding. It collapses the same cobweb even at W=1 (no
         smoothing), so theta stops backfiring at the throttled scale. Both
         seedings work (fill from empty, or reassign from the controller's
         belief); the belief seed is at least as stable and reaches a lower cost.

HOW TO READ THE OUTPUT
    Run with ``-s`` to see the narration:

        uv run --extra dev pytest tests/test_capacity_theta.py --runslow -s

    The report prints steady-state total system cost and the day-to-day P_alpha
    oscillation for each corner, then a verdict. Assertions use generous margins
    on robust, monotone facts; if one fails, re-investigate rather than loosen.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aif_traffic.parameters import (
    AIFControllerSpec,
    Params,
    SignalType,
    SimParams,
)
from aif_traffic.simulator import run_experiment

# Full-scale, deterministic, multi-run: heavy, so gated behind --runslow.
pytestmark = pytest.mark.slow

SEED = 42
DAYS = 90
N_LAST = 15
# Smoothing window above the empirical cobweb-stability threshold (~20-25 days
# at scale 0.25); enough to damp the oscillation and let theta help again.
SMOOTH_W = 25


def _steady(theta: float, scale: float, smoothing: int, *,
            signal: SignalType = SignalType.EXTERNALITY, increments: int = 12,
            seed: str = "empty"):
    """Steady-state total system cost and day-to-day P_alpha std for one cell.

    AIF controller, externality advisory on at full compliance (required for
    theta to act), deterministic, throttling link 5 (bypass) by ``scale`` and
    averaging the advisory over ``smoothing`` days. ``signal`` selects the
    advisory mechanism (raw ``EXTERNALITY`` or per-traveller
    ``EXTERNALITY_SEQUENTIAL`` with ``increments`` bins, ``seed`` = 'empty' or
    'belief')."""
    p = (
        replace(Params(), sim=replace(SimParams(), days=DAYS, seed=SEED),
                controller=AIFControllerSpec())
        .with_noise_free(True)
        .with_comm(signal)
        .with_compliance(1.0)
        .with_bypass_capacity_scale(scale)
        .with_theta(theta)
        .with_advisory_smoothing(smoothing)
    )
    if signal in (SignalType.EXTERNALITY_SEQUENTIAL, SignalType.MSC_SEQUENTIAL):
        p = p.with_sequential_increments(increments).with_sequential_seed(seed)
    step = run_experiment(p, seeds=[SEED]).step
    daily_cost = step.groupby("day")["SC"].first()
    daily_share = step.groupby("day")["P_alpha"].mean()
    sc = float(daily_cost.iloc[-N_LAST:].mean())
    pa_std = float(daily_share.iloc[-N_LAST:].std())
    return sc, pa_std


@pytest.fixture(scope="module")
def corners():
    """The corner runs that tell the story (deterministic, shared)."""
    return {
        "full_off": _steady(0.0, 1.0, 1),    # full bypass, theta off (reference)
        "full_on": _steady(1.0, 1.0, 1),     # full bypass, theta on (benign)
        "thr_off": _steady(0.0, 0.25, 1),    # throttled, theta off (baseline)
        "thr_raw": _steady(1.0, 0.25, 1),    # throttled, theta on, stale advisory
        "thr_smooth": _steady(1.0, 0.25, SMOOTH_W),  # throttled, theta on, smoothed
        # throttled, theta on, per-traveller sequential advisory, NO smoothing
        "thr_seq": _steady(1.0, 0.25, 1, signal=SignalType.EXTERNALITY_SEQUENTIAL),
        # ... same but seeded from the controller's belief (posterior-as-prior)
        "thr_seq_belief": _steady(1.0, 0.25, 1,
                                  signal=SignalType.EXTERNALITY_SEQUENTIAL,
                                  seed="belief"),
    }


def _narrate(title, lines):
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


def test_throttling_bypass_raises_baseline_and_theta_is_inert_at_full_capacity(corners):
    """Sanity: throttling the bypass congests the network (higher theta=0 cost),
    and at full (uncongestable) bypass capacity theta barely moves cost."""
    full_off_sc, _ = corners["full_off"]
    full_on_sc, _ = corners["full_on"]
    thr_off_sc, _ = corners["thr_off"]

    _narrate(
        "BEHAVIOUR: throttling raises baseline cost; theta inert at full capacity",
        [
            f"theta=0 cost, full bypass (x1.0):   {full_off_sc:10.0f} veh-min",
            f"theta=0 cost, throttled (x0.25):    {thr_off_sc:10.0f} veh-min "
            f"({thr_off_sc / full_off_sc:.1f}x higher)",
            f"theta 0->1 at full capacity:        "
            f"{100 * (full_on_sc - full_off_sc) / full_off_sc:+.1f}% "
            "(near-inert: the bypass never congests, so the externality wedge is tiny)",
            "VERDICT: throttling makes the bypass a real bottleneck; at full "
            "capacity theta has almost nothing to redistribute.",
        ],
    )

    assert np.isfinite([full_off_sc, full_on_sc, thr_off_sc]).all()
    assert (np.array([full_off_sc, full_on_sc, thr_off_sc]) > 0).all()
    # Throttling the bypass raises the theta=0 baseline cost (real congestion).
    assert thr_off_sc > 1.5 * full_off_sc, (thr_off_sc, full_off_sc)
    # At full capacity theta 0->1 moves cost only marginally (|.| < 10%).
    assert abs(full_on_sc - full_off_sc) < 0.10 * full_off_sc, (full_on_sc, full_off_sc)


def test_stale_advisory_makes_theta_backfire_via_a_cobweb(corners):
    """With the bypass throttled, the un-smoothed (act-on-yesterday) advisory
    makes theta=1 oscillate route choice day-to-day and drives cost far above
    the theta=0 baseline."""
    thr_off_sc, thr_off_std = corners["thr_off"]
    thr_raw_sc, thr_raw_std = corners["thr_raw"]

    _narrate(
        "BEHAVIOUR: a stale externality advisory makes theta backfire (cobweb)",
        [
            f"throttled (x0.25), stale advisory (W=1):",
            f"   theta=0 cost:  {thr_off_sc:10.0f} veh-min   P_alpha std/day: {thr_off_std:.3f}",
            f"   theta=1 cost:  {thr_raw_sc:10.0f} veh-min   P_alpha std/day: {thr_raw_std:.3f}",
            f"   theta 0->1:    {100 * (thr_raw_sc - thr_off_sc) / thr_off_sc:+.1f}%   <- blows up",
            "VERDICT: theta=1 with a one-day-stale advisory drives a large "
            "day-to-day route-choice oscillation (cobweb); the alternating "
            "overloads push total cost far above the theta=0 baseline. The "
            "advisory backfires, it does not help.",
        ],
    )

    # theta=1 with the stale advisory costs much more than theta=0 (backfire).
    assert thr_raw_sc > 1.5 * thr_off_sc, (thr_raw_sc, thr_off_sc)
    # The signature is a day-to-day oscillation: theta=1 is far more volatile.
    assert thr_raw_std > 0.10, thr_raw_std
    assert thr_raw_std > 5.0 * max(thr_off_std, 1e-3), (thr_raw_std, thr_off_std)


def test_smoothing_the_advisory_damps_the_cobweb_and_theta_helps(corners):
    """Averaging the advisory over a multi-day window damps the oscillation and,
    past the threshold, lets theta help even at the throttled scale."""
    thr_off_sc, _ = corners["thr_off"]
    thr_raw_sc, thr_raw_std = corners["thr_raw"]
    thr_smooth_sc, thr_smooth_std = corners["thr_smooth"]

    _narrate(
        "BEHAVIOUR: smoothing the advisory rescues the theta lever",
        [
            f"throttled (x0.25), theta=1:",
            f"   stale advisory  (W=1):   cost {thr_raw_sc:10.0f}   P_alpha std: {thr_raw_std:.3f}",
            f"   smoothed (W={SMOOTH_W:>2}):        cost {thr_smooth_sc:10.0f}   P_alpha std: {thr_smooth_std:.3f}",
            f"   theta=0 baseline:        cost {thr_off_sc:10.0f}",
            f"   smoothed theta 0->1:     {100 * (thr_smooth_sc - thr_off_sc) / thr_off_sc:+.1f}%",
            "VERDICT: averaging the advisory over ~25 days kills the cobweb "
            "(oscillation collapses) and theta stops backfiring, reaching roughly "
            "the theta=0 baseline or slightly below. The internalisation lever "
            "works once the signal is temporally stable.",
        ],
    )

    # Smoothing collapses the oscillation.
    assert thr_smooth_std < 0.05, thr_smooth_std
    assert thr_smooth_std < 0.5 * thr_raw_std, (thr_smooth_std, thr_raw_std)
    # Smoothing removes most of the cost blow-up.
    assert thr_smooth_sc < 0.5 * thr_raw_sc, (thr_smooth_sc, thr_raw_sc)
    # With the advisory stable, theta no longer backfires: cost is back to about
    # the theta=0 baseline (within 10%), i.e. neutral-or-better rather than +200%.
    assert thr_smooth_sc < 1.10 * thr_off_sc, (thr_smooth_sc, thr_off_sc)


def test_sequential_advisory_damps_the_cobweb_without_smoothing(corners):
    """The per-traveller SEQUENTIAL advisory breaks the herd directly: with the
    bypass throttled and NO smoothing (W=1), theta=1 no longer oscillates or
    backfires, unlike the raw single-value advisory at the same W."""
    thr_off_sc, thr_off_std = corners["thr_off"]
    thr_raw_sc, thr_raw_std = corners["thr_raw"]
    thr_seq_sc, thr_seq_std = corners["thr_seq"]
    thr_belief_sc, thr_belief_std = corners["thr_seq_belief"]

    _narrate(
        "BEHAVIOUR: the sequential (per-traveller) advisory kills the cobweb at W=1",
        [
            f"throttled (x0.25), theta=1, no smoothing (W=1):",
            f"   raw advisory:             cost {thr_raw_sc:10.0f}   P_alpha std: {thr_raw_std:.3f}  <- cobweb",
            f"   sequential (from empty):  cost {thr_seq_sc:10.0f}   P_alpha std: {thr_seq_std:.3f}",
            f"   sequential (from belief): cost {thr_belief_sc:10.0f}   P_alpha std: {thr_belief_std:.3f}",
            f"   theta=0 baseline:         cost {thr_off_sc:10.0f}",
            f"   seq(empty)  theta 0->1: {100 * (thr_seq_sc - thr_off_sc) / thr_off_sc:+.1f}%",
            f"   seq(belief) theta 0->1: {100 * (thr_belief_sc - thr_off_sc) / thr_off_sc:+.1f}%",
            "VERDICT: handing each traveller a rank-indexed marginal cost splits "
            "the population instead of herding it, so the day-to-day oscillation "
            "collapses and theta stops backfiring, all without any temporal "
            "smoothing of the advisory. Seeding from the controller's belief "
            "(posterior-as-prior) is at least as stable and reaches a lower cost.",
        ],
    )

    for sc, std in ((thr_seq_sc, thr_seq_std), (thr_belief_sc, thr_belief_std)):
        assert np.isfinite([sc, std]).all()
        # The oscillation is gone even at W=1 (matches the smoothed-corner threshold).
        assert std < 0.05, std
        assert std < 0.5 * thr_raw_std, (std, thr_raw_std)
        # theta no longer backfires: cost near the theta=0 baseline, not +200%.
        assert sc < 1.10 * thr_off_sc, (sc, thr_off_sc)
        assert sc < 0.5 * thr_raw_sc, (sc, thr_raw_sc)
