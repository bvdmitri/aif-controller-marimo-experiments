"""Report-style characterization tests: do the paper's claims actually hold?

These are the *non-directional* sibling of the behavioural tests in
``test_behaviour.py`` / ``test_belief_informing.py``. Those pin a qualitative
fact and assert its direction. These, instead, run each experiment, **print**
whether the paper's claim holds, and assert only *sanity* (every number finite,
in range, not NaN). The direction is deliberately NOT asserted -- the point is
to surface a discrepancy, not to bake one in.

This is the classic *characterization / golden-master* testing idea (Michael
Feathers): describe what the code actually does so a human or another agent can
read it back. Here the "golden master" is the paper text, and each test reports
``PAPER CLAIMS ... / OBSERVED ... / consistent | MISMATCH`` so a reader can say
"the text says theta lowers cost, but it does not here -- worth checking".

Run with ``-s`` to read the narration:

    uv run --extra dev pytest tests/test_narrative_reports.py -s

If a verdict reads MISMATCH, that is a FINDING to report against the paper, not
a test to tighten -- see the module docstring of ``test_behaviour.py``.
"""

from __future__ import annotations

import math

import pytest

from aif_traffic.parameters import (
    AnticipatoryControllerSpec,
    BeliefSignal,
    FixedTimeControllerSpec,
    Params,
    ReactiveControllerSpec,
    SignalType,
)
from aif_traffic.plotting import controller_summary
from aif_traffic.simulator import run_experiment

# Full-scale and slow on purpose: skipped unless `pytest --runslow` is given.
pytestmark = pytest.mark.slow

SEED = 42
N_LAST = 15      # steady-state window: mean over the last N recorded days


def _full_base() -> Params:
    """The FULL default experiment: 90 recorded days, 30 burn-in, the default
    2000-traveller population, and the full 300-min within-day horizon. These
    reports run the real experiment, not a reduced stand-in -- they are slow on
    purpose, so the verdicts reflect the model as actually configured."""
    return Params.default().with_seed(SEED)


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
    return [
        "",
        f"PAPER CLAIMS: {claim}",
        f"OBSERVED:     {observed}",
        f"VERDICT:      {mark}",
    ]


def _steady_cost(res, n_last: int = N_LAST) -> float:
    """Mean daily system cost over the last ``n_last`` recorded days."""
    daily = res.step.groupby("day")["SC"].first()
    return float(daily.iloc[-n_last:].mean())


def _steady_belief_sd(res, n_last: int = N_LAST) -> float:
    """Mean traveller belief SD over the intersection travel time
    (``sigma_alpha_post``) over the last ``n_last`` recorded days."""
    last = sorted(res.cohort["day"].unique())[-n_last:]
    return float(res.cohort[res.cohort["day"].isin(last)]["sigma_alpha_post"].mean())


def _steady_mean_share(res, n_last: int = N_LAST) -> float:
    """Mean intersection-route share over the last ``n_last`` recorded days
    (all within-day intervals) -- a proxy for how the load is split."""
    step = res.step
    last = sorted(step["day"].unique())[-n_last:]
    return float(step[step["day"].isin(last)]["P_alpha"].mean())


def _finite(*xs) -> bool:
    return all(math.isfinite(x) for x in xs)


# ---------------------------------------------------------------------------
# Report 1 -- Experiment 1: traveller social internalisation (theta)
# ---------------------------------------------------------------------------
def test_report_theta_effect_on_system_cost():
    """Does higher social internalisation theta lower total system cost (and
    even out the load), as Experiment 1 claims? Reported, not asserted.

    Note on faithfulness: theta enters the perceived cost as
    ``zeta_r = TT_r + theta * E_r``, so it can only bite if the externality
    ``E_r`` is actually communicated. We therefore broadcast EXTERNALITY with
    full compliance here -- the paper's "no broadcast" wording for Experiment 1
    is itself worth a look, since without an E_r channel theta is inert in the
    code. The externality re-rolls the day per (route, minute) on the full
    300-min horizon, which is why this report is slow (~minutes).
    """
    base = (
        _full_base()
        .with_comm(SignalType.EXTERNALITY)
        .with_compliance(1.0)
    )
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    cost, share = {}, {}
    for th in thetas:
        res = run_experiment(base.with_theta(th), seeds=[SEED])
        cost[th] = _steady_cost(res)
        share[th] = _steady_mean_share(res)

    cost_ue, cost_so = cost[0.0], cost[1.0]
    rel = (cost_ue - cost_so) / cost_ue if cost_ue else float("nan")
    # "Monotone non-increasing" with a small tolerance for emergent noise.
    seq = [cost[t] for t in thetas]
    monotone = all(b <= a * 1.02 for a, b in zip(seq, seq[1:]))
    theta_does_anything = abs(cost_so - cost_ue) > 1e-6

    lines = [
        "Cost-offset channel: EXTERNALITY (so theta has an E_r to act on).",
        "Steady-state system cost and mean intersection share by theta:",
        *[
            f"   theta={t:<4}  cost={cost[t]:9.1f} veh-min   mean P_alpha={share[t]:.2f}"
            for t in thetas
        ],
        "",
        f"User equilibrium (theta=0) cost:  {cost_ue:9.1f}",
        f"System optimum  (theta=1) cost:  {cost_so:9.1f}",
        f"Relative change UE -> SO:         {rel:+.1%}",
        f"Monotone non-increasing in theta: {monotone}",
        f"theta changes the outcome at all:  {theta_does_anything}",
        *_verdict(
            "higher theta spreads demand and lowers total system cost",
            f"theta=1 cost is {rel:+.1%} vs theta=0; monotone={monotone}; "
            f"theta-has-effect={theta_does_anything}",
            holds=(cost_so < cost_ue),
        ),
    ]
    _narrate("REPORT: effect of social internalisation theta", lines)

    # Sanity only -- no direction asserted.
    assert _finite(*cost.values()), cost
    assert all(c > 0 for c in cost.values()), cost
    assert all(0.0 <= s <= 1.0 for s in share.values()), share


# ---------------------------------------------------------------------------
# Report 2 -- Experiment 3: value of information communication
# ---------------------------------------------------------------------------
def test_report_communication_value():
    """Does sharing the controller's belief (QB / SP / QB+SP) lower system cost
    and belief uncertainty vs the baseline, with QB+SP best, as Experiment 3
    claims?"""
    base = _full_base().with_compliance(1.0)
    settings = {
        "BL": base.with_belief_signals(),
        "QB": base.with_belief_signals(BeliefSignal.QUEUE_BELIEF),
        "SP": base.with_belief_signals(BeliefSignal.SPLIT_PLAN),
        "QB+SP": base.with_belief_signals(
            BeliefSignal.QUEUE_BELIEF, BeliefSignal.SPLIT_PLAN
        ),
    }
    cost, sd = {}, {}
    for name, params in settings.items():
        res = run_experiment(params, seeds=[SEED])
        cost[name] = _steady_cost(res)
        sd[name] = _steady_belief_sd(res)

    best_cost = min(cost, key=cost.get)
    best_sd = min(sd, key=sd.get)
    any_helps = any(cost[k] < cost["BL"] for k in ("QB", "SP", "QB+SP"))

    lines = [
        "Steady-state system cost and intersection belief SD by setting:",
        *[
            f"   {name:<6} cost={cost[name]:9.1f} veh-min   belief SD={sd[name]:6.3f}"
            for name in settings
        ],
        "",
        f"Lowest system cost:      {best_cost}",
        f"Lowest belief uncertainty: {best_sd}",
        f"Any information beats BL on cost: {any_helps}",
        *_verdict(
            "richer shared belief lowers cost and uncertainty; QB+SP is best",
            f"cheapest={best_cost}, least-uncertain={best_sd}, any-helps={any_helps}",
            holds=(best_cost == "QB+SP"),
        ),
    ]
    _narrate("REPORT: value of information communication", lines)

    assert _finite(*cost.values(), *sd.values()), (cost, sd)
    assert all(c > 0 for c in cost.values()), cost
    assert all(s >= 0 for s in sd.values()), sd


# ---------------------------------------------------------------------------
# Report 3 -- Experiment 2: controller benchmark
# ---------------------------------------------------------------------------
def test_report_controller_benchmark():
    """Does the AIF controller achieve the lowest system cost against the
    fixed-time, reactive, and anticipatory baselines, as Experiment 2 claims?"""
    base = _full_base()
    controllers = {
        "fixed_time": FixedTimeControllerSpec(),
        "reactive": ReactiveControllerSpec(),
        "anticipatory": AnticipatoryControllerSpec(),
        "aif": base.controller,  # the default AIFControllerSpec
    }
    results = {
        name: run_experiment(base.with_controller(spec), seeds=[SEED])
        for name, spec in controllers.items()
    }
    summary = controller_summary(results)  # one row per controller

    ranked = summary.sort_values("mean_SC").reset_index(drop=True)
    cheapest = ranked.loc[0, "controller"]
    aif_is_cheapest = "AIF" in str(cheapest)

    lines = [
        "Controller benchmark (mean over recorded days):",
        f"{'controller':<26}{'mean_SC':>12}{'peak_queue':>12}{'sig_var':>10}",
        *[
            f"{row.controller:<26}{row.mean_SC:>12.1f}{row.mean_peak_queue:>12.1f}"
            f"{row.mean_signal_variation:>10.3f}"
            for row in summary.itertuples()
        ],
        "",
        f"Cheapest controller (lowest mean system cost): {cheapest}",
        *_verdict(
            "the AIF controller outperforms the fixed-time, reactive, and "
            "anticipatory baselines on system cost",
            f"cheapest controller = {cheapest}",
            holds=aif_is_cheapest,
        ),
    ]
    _narrate("REPORT: controller benchmark", lines)

    assert not summary.empty
    assert _finite(*summary["mean_SC"].tolist()), summary["mean_SC"].tolist()
    assert (summary["mean_SC"] > 0).all(), summary
    assert (summary["mean_peak_queue"] >= 0).all(), summary
