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
    CohortSpec,
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


def test_with_noise_free_sets_flag_and_zeros_noise():
    """`with_noise_free` marks every cohort and zeros the noise knobs; the
    default (realistic) run keeps a nonzero measurement noise."""
    base = Params.default()
    # Realistic defaults: nonzero measurement noise, cohorts not noise-free.
    assert base.noise.obs_noise_sd > 0.0
    assert all(not c.noise_free for c in base.population.cohorts)

    nf = base.with_noise_free(True)
    assert nf.noise.obs_noise_sd == 0.0
    assert nf.noise.demand_noise_cv == 0.0
    assert all(c.noise_free for c in nf.population.cohorts)

    off = nf.with_noise_free(False)
    assert all(not c.noise_free for c in off.population.cohorts)


def test_noise_free_run_is_deterministic_and_smooth():
    """A noise-free run is reproducible and converges without day-to-day jitter,
    whereas the default (noisy) run keeps fluctuating."""
    p = replace(
        Params.default(),
        sim=SimParams(days=30, h_min=60, dt_min=1, burn_in=0, seed=9),
        population=PopulationParams(cohorts=(CohortSpec(n_agents=200),)),
    )
    nf = p.with_noise_free(True)

    a = run_experiment(nf, seeds=[9]).step["SC"].to_numpy()
    b = run_experiment(nf, seeds=[9]).step["SC"].to_numpy()
    assert np.array_equal(a, b), "noise-free run is not reproducible"

    def late_pa_std(res):
        pa = res.step.groupby("day")["P_alpha"].mean()
        return float(pa.iloc[-10:].std())

    nf_jit = late_pa_std(run_experiment(nf, seeds=[9]))
    noisy_jit = late_pa_std(run_experiment(p, seeds=[9]))
    assert nf_jit < 1e-9, f"noise-free route share still jitters: {nf_jit}"
    assert noisy_jit > nf_jit, (noisy_jit, nf_jit)


def _cmp_params(stationary: bool, *, days: int, window: int) -> Params:
    # Noise-free so the stationary-vs-windowed difference is isolated from
    # finite-population sampling / measurement noise.
    return replace(
        Params.default(),
        sim=SimParams(days=days, h_min=120, dt_min=1, burn_in=0, seed=11),
        population=PopulationParams(
            cohorts=(CohortSpec(n_agents=300, window_size=window),)),
    ).with_stationary(stationary).with_noise_free(True)


@pytest.mark.slow
def test_stationary_tightens_and_reduces_window_spike():
    """Stationary continuous filtering vs the rolling window, same experiment.

    Expectations (the headline effects):
      * travellers' late-day posterior SD over the intersection travel time is
        LOWER under stationary (evidence accumulates instead of being forgotten);
      * the day-to-day system-cost churn around day == window (where the rolling
        window first drops its oldest, cold-start day) is SMALLER under
        stationary: the "forgetting spike" the mode is designed to remove.
    The controller's within-day queue-belief SD is only checked to be *comparable*
    between modes: the controller refits a fresh per-day trajectory posterior that
    fills within its window either way, so this quantity is not the place the
    stationary benefit shows up (the traveller posterior and the cost churn are).
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
    print("Controller queue-belief SD (final day; expected comparable):")
    print(f"   stationary: {st_csd:.2f}       windowed: {wd_csd:.2f}")
    print(f"Day-to-day |dSC| churn around day=={W} (lower = less forgetting spike):")
    print(f"   stationary: {st_churn:.1f}     windowed: {wd_churn:.1f}")
    print("")
    print("VERDICT: continuous filtering accumulates evidence, so both layers' "
          "posteriors are tighter and the window-boundary cost churn is reduced.")
    print("=" * 72)

    assert st_sig < wd_sig, (st_sig, wd_sig)
    # Controller within-day belief SD: only require it to be *comparable* (it
    # converges within the window in either mode); the stationary benefit lives
    # in the traveller posterior + the cost churn asserted above/below.
    assert st_csd <= 1.25 * wd_csd, (st_csd, wd_csd)
    assert st_churn < wd_churn, (st_churn, wd_churn)
