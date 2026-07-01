"""The "assume stationary environment" mode: continuous filtering vs the
rolling window.

Two flavours (see CLAUDE.md):

- a fast **plumbing** unit test that ``Params.with_stationary`` flips the flag
  on every cohort and the AIF controller;
- a slow, **narrated behavioural** comparison (``--runslow``) that runs the same
  experiment stationary vs windowed and checks the emergent facts the mode is
  meant to deliver: posteriors tighten more (lower late-day SD), and the
  day-to-day system-cost churn around day == window is smaller (no "forgetting
  spike" when the rolling window first drops its oldest day).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aif_traffic.parameters import (
    AIFControllerSpec,
    CohortSpec,
    NoiseParams,
    Params,
    PopulationParams,
    SimParams,
)
from aif_traffic.simulator import run_experiment


def test_with_stationary_sets_flag_on_all_layers():
    """`with_stationary` toggles both layers; default Params is stationary."""
    base = Params.default()
    assert all(c.stationary for c in base.population.cohorts)
    assert base.controller.stationary  # AIF default

    off = base.with_stationary(False)
    assert all(c.stationary is False for c in off.population.cohorts)
    assert off.controller.stationary is False

    on = off.with_stationary(True)
    assert all(c.stationary is True for c in on.population.cohorts)
    assert on.controller.stationary is True


def _cmp_params(stationary: bool, *, days: int, window: int) -> Params:
    return replace(
        Params.default(),
        sim=SimParams(days=days, h_min=120, dt_min=1, burn_in=0, seed=11),
        population=PopulationParams(
            cohorts=(CohortSpec(n_agents=300, window_size=window),)),
        noise=NoiseParams(obs_noise_sd=0.0),
    ).with_stationary(stationary)


@pytest.mark.slow
def test_stationary_tightens_and_reduces_window_spike():
    """Stationary continuous filtering vs the rolling window, same experiment.

    Expectations:
      * travellers' late-day posterior SD over the intersection travel time is
        LOWER under stationary (evidence accumulates instead of being forgotten);
      * the controller's late-day queue-belief SD is LOWER under stationary;
      * the day-to-day system-cost churn around day == window (where the rolling
        window first drops its oldest, cold-start day) is SMALLER under
        stationary -- the "forgetting spike" the mode is designed to remove.
    """
    W = 20
    days = 60
    st = run_experiment(_cmp_params(True, days=days, window=W), seeds=[11])
    wd = run_experiment(_cmp_params(False, days=days, window=W), seeds=[11])

    def late_sigma_alpha(res):
        s = res.cohort.groupby("day")["sigma_alpha_post"].mean()
        return float(s.iloc[-15:].mean())

    def late_ctrl_sd(res):
        # mean marginal belief SD over the run's final day (lower = tighter)
        c = res.controller
        return float(np.sqrt(c[c["day"] == c["day"].max()]["belief_var_mean"].mean()))

    def churn_near_window(res):
        # mean |Delta system cost| in a band straddling day == W
        sc = res.step.groupby("day")["SC"].first()
        band = sc.loc[(sc.index >= W - 3) & (sc.index <= W + 6)]
        return float(band.diff().abs().mean())

    st_sig, wd_sig = late_sigma_alpha(st), late_sigma_alpha(wd)
    st_csd, wd_csd = late_ctrl_sd(st), late_ctrl_sd(wd)
    st_churn, wd_churn = churn_near_window(st), churn_near_window(wd)

    print("\n" + "=" * 72)
    print("BEHAVIOUR: stationary continuous filtering vs the rolling window")
    print("-" * 72)
    print(f"(window W={W} days, {days}-day run)")
    print("")
    print("Traveller posterior SD on TT_alpha (late days, lower = tighter):")
    print(f"   stationary: {st_sig:.3f}      windowed: {wd_sig:.3f}")
    print("Controller queue-belief SD (final day, lower = tighter):")
    print(f"   stationary: {st_csd:.2f}       windowed: {wd_csd:.2f}")
    print(f"Day-to-day |dSC| churn around day=={W} (lower = less forgetting spike):")
    print(f"   stationary: {st_churn:.1f}     windowed: {wd_churn:.1f}")
    print("")
    print("VERDICT: continuous filtering accumulates evidence, so both layers' "
          "posteriors are tighter and the window-boundary cost churn is reduced.")
    print("=" * 72)

    assert st_sig < wd_sig, (st_sig, wd_sig)
    assert st_csd <= wd_csd, (st_csd, wd_csd)
    assert st_churn < wd_churn, (st_churn, wd_churn)
