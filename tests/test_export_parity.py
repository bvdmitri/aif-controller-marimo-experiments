"""Guard that the CI paper-figure export uses the notebook gold defaults.

The figures rendered by ``scripts/export_paper_figures.py`` (on CI) must be
generated with exactly the same parameters the marimo notebooks use by default,
so what the paper shows matches what a user sees running the notebooks. This test
pins ``export_paper_figures._base`` to the canonical defaults in
``aif_traffic.notebook_controls`` (the single source of truth, itself guarded by
``test_notebook_controls``), so either side drifting -- e.g. re-introducing a
``.with_noise_regime("off")`` override, or changing a control default -- breaks
CI instead of silently producing inconsistent figures.
"""

from __future__ import annotations

import pytest

import scripts.export_paper_figures as exp
from aif_traffic import notebook_controls as nc
from aif_traffic.parameters import CohortSpec, _NOISE_REGIMES


def test_export_base_matches_notebook_gold_defaults():
    p = exp._base(exp._cfg(quick=False))
    cohorts = p.population.cohorts
    ctrl = p.controller

    # -- Noise: the notebook default regime (medium), NOT off / noise-free. ----
    regime = nc.noise_regime().value
    assert regime == "medium"
    obs_sd, sig_L, sig_phi = _NOISE_REGIMES[regime]
    assert p.noise.obs_noise_sd == pytest.approx(obs_sd)
    assert all(not c.noise_free for c in cohorts), "export must not be noise-free"
    assert all(c.sigma_L_obs == pytest.approx(sig_L) for c in cohorts)
    assert all(c.sigma_phi_obs == pytest.approx(sig_phi) for c in cohorts)

    # -- Toggles inherited from the gold defaults (both layers). ---------------
    stationary = nc.stationary().value
    learn = nc.learn_noise().value
    assert all(c.stationary == stationary for c in cohorts)
    assert ctrl.stationary == stationary
    assert all(c.learn_obs_noise == learn for c in cohorts)
    assert ctrl.learn_obs_noise == learn

    # -- Controller knobs match the notebook control defaults. -----------------
    assert ctrl.gamma == pytest.approx(nc.gamma().value)
    assert ctrl.omega == pytest.approx(nc.omega().value)
    assert ctrl.sigma_pref == pytest.approx(nc.sigma_pref().value)
    assert ctrl.phi_grid_size == int(nc.phi_grid().value)
    assert ctrl.control_interval_min == int(nc.control_interval().value)
    assert ctrl.horizon_min == int(nc.control_interval().value)
    assert ctrl.controller_window_size == int(nc.controller_window().value)

    # -- Sim / population. -----------------------------------------------------
    assert p.sim.seed == int(nc.seed().value)
    assert p.sim.burn_in == int(nc.warmup().value)
    assert p.sim.days == int(nc.days().value)
    assert all(c.n_agents == CohortSpec().n_agents for c in cohorts)
