"""Behavioural characterization of traveller compliance (paper Experiment 4).

The controller shares its full belief (QB+SP -- its forward-predicted queue
belief and its planned green split) before travellers choose. A compliant
traveller fuses that belief into its decision; a non-compliant one ignores it.
This module varies the **compliance fraction** to characterise how the
coordination effect of shared anticipation changes as fewer travellers listen.

Report-style (see CLAUDE.md): it runs the real sweep and PRINTS whether the
"degrades gracefully" claim holds (cost changing smoothly, no cliff), asserting
only sanity -- the direction of the value-of-information effect is deliberately
surfaced, not enforced (the belief channel carries no social term, so it need
not lower system cost; see tests/test_belief_informing.py for the mechanism).

The guaranteed invariant "compliance 0 == baseline" is covered (fast) by
``test_belief_informing.test_non_compliant_population_recovers_baseline``.

This is a full-scale multi-run sweep, so it is slow and opt-in:

    uv run --extra dev pytest tests/test_compliance.py --runslow -s
"""

from __future__ import annotations

import math

import pytest

from aif_traffic.parameters import BeliefSignal, Params
from aif_traffic.simulator import run_experiment

# Full-scale multi-run sweep: skipped unless --runslow.
pytestmark = pytest.mark.slow

SEED = 42
N_LAST = 15  # steady-state window: mean over the last N recorded days
FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _narrate(title: str, lines: list[str]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


def _verdict(claim: str, observed: str, holds: bool) -> list[str]:
    mark = (
        "consistent with the paper"
        if holds
        else "MISMATCH -- worth checking the paper text"
    )
    return ["", f"PAPER CLAIMS: {claim}", f"OBSERVED:     {observed}",
            f"VERDICT:      {mark}"]


def _steady_cost(res, n_last: int = N_LAST) -> float:
    daily = res.step.groupby("day")["SC"].first()
    return float(daily.iloc[-n_last:].mean())


def _base() -> Params:
    """Full default experiment, AIF controller, controller belief shared (QB+SP)."""
    return (
        Params.default()
        .with_seed(SEED)
        .with_belief_signals(BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN)
    )


@pytest.fixture(scope="module")
def sweep():
    """Steady-state system cost across the compliance sweep (shared)."""
    base = _base()
    return {f: _steady_cost(run_experiment(base.with_compliance(f), seeds=[SEED]))
            for f in FRACTIONS}


def test_report_compliance_graceful_degradation(sweep):
    """Does the coordination effect change GRACEFULLY as compliance varies --
    smoothly, with no cliff -- as Experiment 4 / the Conclusion claims?
    Reported, not asserted (direction-wise)."""
    cost = sweep
    seq = [cost[f] for f in FRACTIONS]  # ordered low -> high compliance

    cost_none = cost[0.0]   # nobody fuses: the baseline
    cost_full = cost[1.0]   # everybody fuses the controller's belief
    rel = (cost_none - cost_full) / cost_none if cost_none else float("nan")
    monotone = all(b <= a * 1.02 for a, b in zip(seq, seq[1:]))
    steps = [abs(a - b) for a, b in zip(seq, seq[1:])]
    span = max(seq) - min(seq)
    biggest_step_frac = (max(steps) / span) if span > 1e-9 else float("nan")

    lines = [
        "Controller belief shared: QB+SP (queue belief + planned split).",
        "Steady-state system cost by compliance fraction:",
        *[f"   compliance={f:<4}  cost={cost[f]:9.1f} veh-min" for f in FRACTIONS],
        "",
        f"Baseline (compliance=0):  {cost_none:9.1f}",
        f"Full compliance (=1):     {cost_full:9.1f}",
        f"Relative change full vs baseline: {rel:+.1%}",
        f"Monotone non-increasing in compliance: {monotone}",
        f"Largest single step as a share of the total span: {biggest_step_frac:.0%} "
        f"(a 'cliff' would be ~100%)",
        *_verdict(
            "the coordination effect degrades gracefully with compliance "
            "(smooth; no cliff)",
            f"full-vs-baseline {rel:+.1%}; monotone={monotone}; "
            f"biggest step={biggest_step_frac:.0%}",
            holds=(cost_full < cost_none and monotone),
        ),
    ]
    _narrate("REPORT: compliance and graceful degradation", lines)

    # Sanity only -- direction not asserted.
    assert all(math.isfinite(c) for c in cost.values()), cost
    assert all(c > 0 for c in cost.values()), cost
