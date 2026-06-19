"""Behavioural characterization tests for the coupled two-layer simulation.

These are higher-level than the unit tests: they run the default AIF experiment
and assert *emergent* facts about how the travellers and controller interact,
each accompanied by printed reasoning so the behaviour can be audited, not just
pass/fail-checked.

WHY THESE EXIST
    They pin down two non-obvious behaviours we investigated and want future
    changes to preserve (or consciously revise):
      1. The signalised-link queues *dip* at the demand peak (tau=150) -- this
         is route diversion, not a bug.
      2. Travellers learn the green split only by taking the intersection; the
         belief tracks the realised split for users and stays uncertain for
         habitual bypass users.

HOW TO READ THE OUTPUT
    Run with ``-s`` to see the narration on success:

        uv run --extra dev pytest tests/test_behaviour.py -s

    Each test prints what it expected, the observed numbers, and a verdict.
    If an assertion fails, pytest shows the same narration plus the failing
    line -- so a future agent can confirm or disconfirm the documented
    understanding directly from the output.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aif_traffic.demand import DemandProfile
from aif_traffic.parameters import Params, SimParams
from aif_traffic.simulator import run_experiment

SEED = 42
DAYS = 90  # the full default experiment (reaches the learned equilibrium)


@pytest.fixture(scope="module")
def run():
    """One default-AIF experiment, shared by the behavioural tests."""
    params = replace(Params(), sim=replace(SimParams(), days=DAYS, seed=SEED))
    last_day = params.sim.days - 1
    res = run_experiment(params, seeds=[SEED], snapshot_days=[last_day])
    return params, res, last_day


def _steady_profile(step, columns):
    """Mean within-day profile over the last 15 recorded days, indexed by tau."""
    last = sorted(step["day"].unique())[-15:]
    prof = step[step["day"].isin(last)].groupby("tau")[columns].mean()
    return prof


def _narrate(title, lines):
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


def _window_mean(prof, col, lo, hi):
    sl = prof.loc[(prof.index >= lo) & (prof.index <= hi), col]
    return float(sl.mean())


def test_peak_demand_queue_dip_is_route_diversion(run):
    """The intersection queue DROPS at the demand peak because travellers divert
    to the bypass there; queues instead peak on the shoulders.

    Expectations:
      * demand peaks at tau=150 (sanity);
      * the share choosing the intersection collapses mid-day (diversion);
      * the intersection inflow Q_alpha has a trough at the peak;
      * the intersection queue L2 at the peak is below the post-peak shoulder,
        and the daily L2 maximum sits well away from tau=150.
    """
    params, res, _ = run
    prof = _steady_profile(res.step, ["P_alpha", "Q_alpha", "Q_beta", "L2"])
    dem = DemandProfile.from_params(params.sim, params.demand)
    peak_tau = int(np.argmax(dem.d_AB))

    p_edge = 0.5 * (_window_mean(prof, "P_alpha", 30, 75)
                    + _window_mean(prof, "P_alpha", 225, 270))
    p_peak = _window_mean(prof, "P_alpha", 135, 165)
    qa_edge = max(_window_mean(prof, "Q_alpha", 100, 120),
                  _window_mean(prof, "Q_alpha", 180, 200))
    qa_peak = _window_mean(prof, "Q_alpha", 140, 160)
    l2_peak = _window_mean(prof, "L2", 145, 155)
    l2_post = _window_mean(prof, "L2", 175, 185)
    l2_argmax = int(prof["L2"].idxmax())

    _narrate(
        "BEHAVIOUR: queue dip at the demand peak is ROUTE DIVERSION",
        [
            f"Demand peaks at tau={peak_tau} (expected 150).",
            "",
            "Share choosing the intersection (route alpha):",
            f"   day edges (low demand):  P_alpha ~ {p_edge:.2f}",
            f"   demand peak  (tau~150):  P_alpha ~ {p_peak:.2f}   <- collapses",
            "",
            "Intersection inflow Q_alpha [veh/h]:",
            f"   shoulders: ~ {qa_edge:.0f}      peak: ~ {qa_peak:.0f}   <- trough",
            "",
            "Intersection queue L2 [veh]:",
            f"   at the demand peak (tau~150):     {l2_peak:.1f}",
            f"   at the post-peak shoulder (~180): {l2_post:.1f}   <- higher",
            f"   daily L2 maximum is at tau={l2_argmax} (away from 150)",
            "",
            "VERDICT: at the peak the intersection is least attractive, so "
            "travellers shed onto the bypass; the intersection queue is relieved "
            "at the centre and peaks on the shoulders. Sensible, not a bug.",
        ],
    )

    assert peak_tau == 150
    # Diversion: mid-day intersection share is far below the day-edge share.
    assert p_edge - p_peak > 0.15, (p_edge, p_peak)
    # Intersection inflow troughs at the peak.
    assert qa_peak < qa_edge, (qa_peak, qa_edge)
    # The queue dips at the centre relative to the post-peak shoulder ...
    assert l2_peak < l2_post, (l2_peak, l2_post)
    # ... and the queue maximum is on a shoulder, not at the demand peak.
    assert abs(l2_argmax - 150) > 15, l2_argmax


def test_green_split_belief_tracks_realised_split(run):
    """Travellers learn the green split only by taking the intersection.

    Expectations:
      * agents who took the intersection on the last day have a belief that
        tracks the realised split closely and is sharp (small SD);
      * habitual bypass users are much more uncertain (larger SD) -- the green
        split is only observed on the chosen, signalised route.
    """
    params, res, last_day = run
    snap = res.snapshots[(SEED, last_day)]
    phi_mean = snap["phi_mean_alpha"]
    phi_sd = snap["phi_sd_alpha"]
    chose_alpha = snap["last_choice"] == 0
    dep = snap["departure_time"]

    prof = _steady_profile(res.step, ["phi2"])
    realised_by_dep = prof["phi2"].reindex(range(params.sim.K)).to_numpy()

    err = phi_mean[chose_alpha] - realised_by_dep[dep[chose_alpha]]
    bias = float(np.nanmean(err))
    abs_med = float(np.nanmedian(np.abs(err)))
    sd_users = float(phi_sd[chose_alpha].mean())
    sd_bypass = float(phi_sd[~chose_alpha].mean())

    _narrate(
        "BEHAVIOUR: green-split belief is learned only by intersection users",
        [
            f"Agents choosing the intersection on the last day: {chose_alpha.mean():.0%}",
            "",
            "Intersection users -- believed vs realised green split:",
            f"   bias (believed - realised): {bias:+.3f}",
            f"   |error| median:             {abs_med:.3f}   <- should be small",
            "",
            "Belief sharpness (posterior SD of phi):",
            f"   intersection users: {sd_users:.3f}   <- sharp (they observe it)",
            f"   bypass users:       {sd_bypass:.3f}   <- uncertain (never observe it)",
            "",
            "VERDICT: the belief tracks the realised split for those who "
            "experience the intersection and stays wide for those who do not -- "
            "the partial observability is working as intended.",
        ],
    )

    assert chose_alpha.sum() > 20, "too few intersection users to assess"
    assert abs(bias) < 0.12, bias
    assert abs_med < 0.10, abs_med
    assert sd_users < sd_bypass, (sd_users, sd_bypass)
