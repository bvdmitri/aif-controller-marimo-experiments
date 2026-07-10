"""Traveller belief over the signal green split (4th latent ``phi``).

Covers the three properties of the extension: travel-time coupling on the
signalised route, partial observability (learned only when the intersection is
chosen, otherwise reverting/inflating), and the epistemic pull from green-split
uncertainty.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aif_traffic.inference.efe import _predictive_moments, efe_route_probabilities
from aif_traffic.inference.filter import (
    PHI_IDX,
    CohortPriors,
    init_variational_state,
    predicted_tt,
    window_step,
)

SIG = jnp.asarray([1.0, 0.0])  # alpha (route 0) signalised, beta bypass
PHI_LO, PHI_HI = 0.1, 0.9


def _priors(N=4, phi_alpha=0.45, phi_sigma=0.2, C_alpha=2000.0, L_alpha=120.0):
    return CohortPriors(
        F_mu=jnp.full((N, 2), 5.0),
        F_sigma=jnp.full((N, 2), 1.0),
        C_mu=jnp.array([[C_alpha, 4000.0]] * N),
        C_sigma=jnp.full((N, 2), 300.0),
        L_mu=jnp.array([[L_alpha, 20.0]] * N),
        L_sigma=jnp.full((N, 2), 100.0),
        phi_mu=jnp.array([[phi_alpha, 0.45]] * N),
        phi_sigma=jnp.full((N, 2), phi_sigma),
    )


def test_forward_map_couples_phi_on_signalised_route_only():
    """Lower believed green split raises predicted TT on the intersection, and
    the bypass is unaffected by phi."""
    F = jnp.array([[5.0, 5.0]])
    C = jnp.array([[2000.0, 2000.0]])
    L = jnp.array([[120.0, 120.0]])
    tt_hi = predicted_tt(F, C, L, jnp.array([[0.6, 0.6]]), SIG, PHI_LO, PHI_HI)
    tt_lo = predicted_tt(F, C, L, jnp.array([[0.2, 0.2]]), SIG, PHI_LO, PHI_HI)
    # Intersection (route 0): smaller phi -> smaller effective capacity -> larger TT.
    assert float(tt_lo[0, 0]) > float(tt_hi[0, 0])
    # Bypass (route 1): phi inert -> TT identical across phi values.
    assert float(tt_lo[0, 1]) == pytest.approx(float(tt_hi[0, 1]))


def test_phi_uncertainty_inflates_intersection_predictive_variance():
    """A wider green-split belief raises predictive TT variance (hence the
    epistemic info-gain) on the signalised route but not the bypass."""
    state_tight = init_variational_state(_priors(phi_sigma=0.02))
    state_wide = init_variational_state(_priors(phi_sigma=0.30))
    _, var_tight = _predictive_moments(state_tight, SIG, PHI_LO, PHI_HI)
    _, var_wide = _predictive_moments(state_wide, SIG, PHI_LO, PHI_HI)
    # Intersection variance grows with phi uncertainty.
    assert float(var_wide[0, 0]) > float(var_tight[0, 0])
    # Bypass variance is unchanged (phi does not enter its forward map).
    assert float(var_wide[0, 1]) == pytest.approx(float(var_tight[0, 1]), rel=1e-5)


def _window(N, W, route_idx, y_phi, *, tt=8.0, L=120.0):
    route = jnp.full((N, W), route_idx, dtype=jnp.int32)
    y_tt = jnp.full((N, W), tt)
    sigma_tt = jnp.full((N, W), 5.0)
    y_L = jnp.full((N, W), L)
    sigma_L = jnp.full((N, W), 30.0)
    y_phi_w = jnp.full((N, W), y_phi)
    sigma_phi = jnp.full((N, W), 0.05)
    mask = jnp.ones((N, W))
    return route, y_tt, sigma_tt, y_L, sigma_L, y_phi_w, sigma_phi, mask


def _run(state, priors, route, y_tt, sigma_tt, y_L, sigma_L, y_phi_w, sigma_phi, mask,
         **kw):
    return window_step(
        state, priors,
        route_chosen_window=route, y_tt_window=y_tt, sigma_tt_window=sigma_tt,
        y_L_window=y_L, sigma_L_window=sigma_L,
        y_phi_window=y_phi_w, sigma_phi_window=sigma_phi,
        obs_mask=mask, signalised=SIG, W=mask.shape[1],
        phi_lo=PHI_LO, phi_hi=PHI_HI, **kw,
    )


def test_phi_learned_when_intersection_chosen():
    """Choosing the intersection pulls the alpha green-split belief toward the
    observed split (here 0.25, away from the 0.45 prior)."""
    N, W = 3, 6
    priors = _priors(N=N, phi_alpha=0.45)
    state = init_variational_state(priors)
    win = _window(N, W, route_idx=0, y_phi=0.25)
    state = _run(state, priors, *win, n_laplace_iters=3)
    phi_alpha = float(state.mu[0, 0, PHI_IDX])
    assert 0.22 < phi_alpha < 0.40, phi_alpha  # moved from 0.45 toward 0.25


def test_phi_not_updated_when_bypass_chosen():
    """Always choosing the bypass leaves the alpha green-split belief at its
    (mean-reverted) prior; the green split is observed only on the intersection."""
    N, W = 3, 6
    priors = _priors(N=N, phi_alpha=0.45)
    state = init_variational_state(priors)
    # Bypass chosen every day; the y_phi value is irrelevant (gated off for beta).
    win = _window(N, W, route_idx=1, y_phi=0.25)
    state = _run(state, priors, *win, n_laplace_iters=3)
    # Alpha phi stays at the prior (never observed on the bypass).
    assert float(state.mu[0, 0, PHI_IDX]) == pytest.approx(0.45, abs=1e-3)


def test_low_expected_green_pushes_choice_to_bypass():
    """If travellers expect a poor green split on the intersection, the higher
    predicted intersection TT shifts route choice toward the bypass."""
    N = 32
    hi = init_variational_state(_priors(N=N, phi_alpha=0.7))
    lo = init_variational_state(_priors(N=N, phi_alpha=0.15))
    kw = dict(sigma_obs=jnp.full(N, 3.0), sigma_pref=jnp.full(N, 3.0),
              gamma=jnp.full(N, 2.0), risk_weight=1.0, info_gain_weight=1.0,
              signalised=SIG, phi_lo=PHI_LO, phi_hi=PHI_HI)
    P_hi = efe_route_probabilities(state=hi, **kw)
    P_lo = efe_route_probabilities(state=lo, **kw)
    # Expecting less green on the intersection lowers its share.
    assert float(jnp.mean(P_lo[:, 0])) < float(jnp.mean(P_hi[:, 0]))
