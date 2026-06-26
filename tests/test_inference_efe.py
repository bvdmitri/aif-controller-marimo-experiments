"""Closed-form EFE: symmetry, limiting cases, and the broadcast cost offset."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aif_traffic.inference.efe import (
    efe_route_probabilities,
    efe_route_probabilities_jit,
)
from aif_traffic.inference.filter import CohortPriors, init_variational_state

# These EFE-mechanics tests treat both routes as non-signalised so the
# green-split latent is inert (the dedicated phi tests use signalised=[1, 0]).
SIGNALISED = jnp.asarray([0.0, 0.0])


def _state(N=10, F_A=16.0, F_B=16.0, C_A=4000.0, C_B=4000.0,
            L_A=0.0, L_B=0.0, phi_A=0.45, phi_B=0.45, sigma_scale=1.0,
            phi_sigma=0.2):
    priors = CohortPriors(
        F_mu=jnp.array([[F_A, F_B]] * N),
        F_sigma=jnp.full((N, 2), sigma_scale * 0.5),
        C_mu=jnp.array([[C_A, C_B]] * N),
        C_sigma=jnp.full((N, 2), sigma_scale * 200.0),
        L_mu=jnp.array([[L_A, L_B]] * N),
        L_sigma=jnp.full((N, 2), sigma_scale * 50.0),
        phi_mu=jnp.array([[phi_A, phi_B]] * N),
        phi_sigma=jnp.full((N, 2), phi_sigma),
    )
    return init_variational_state(cohort_priors=priors)


def test_softmax_returns_per_route_distribution():
    N = 8
    state = _state(N=N)
    P = efe_route_probabilities(
        state=state,
        sigma_obs=jnp.full(N, 5.0), sigma_pref=jnp.full(N, 4.0),
        gamma=jnp.full(N, 1.0), risk_weight=1.0, info_gain_weight=1.0,
        signalised=SIGNALISED,
    )
    assert P.shape == (N, 2)
    assert jnp.allclose(P.sum(axis=-1), 1.0, atol=1e-6)


def test_jit_matches_eager_efe():
    """The JIT-compiled choice step computes the same probabilities as the eager
    function (the speedup does not change the math), with and without a broadcast
    cost offset. Tight tolerance -- only XLA float reassociation may differ."""
    import numpy as np

    N = 16
    state = _state(N=N, L_A=20.0, L_B=5.0, phi_A=0.4, phi_B=0.5)
    sig = jnp.asarray([1.0, 0.0])  # route 0 signalised (exercises the phi path)
    common = dict(
        state=state, sigma_obs=jnp.full(N, 5.0), sigma_pref=jnp.full(N, 4.0),
        gamma=jnp.full(N, 3.0), risk_weight=1.0, info_gain_weight=1.0,
        signalised=sig,
    )
    # No cost offset.
    eager = efe_route_probabilities(**common)
    jit = efe_route_probabilities_jit(**common)
    assert np.allclose(np.asarray(eager), np.asarray(jit), rtol=1e-4, atol=1e-6)
    # With a cost offset (the other pytree structure of cost_offset).
    offset = jnp.asarray([[2.0, 0.0]] * N)
    eager_o = efe_route_probabilities(cost_offset=offset, **common)
    jit_o = efe_route_probabilities_jit(cost_offset=offset, **common)
    assert np.allclose(np.asarray(eager_o), np.asarray(jit_o), rtol=1e-4, atol=1e-6)


def test_symmetric_priors_give_p_a_half():
    N = 64
    state = _state(N=N)
    P = efe_route_probabilities(
        state=state,
        sigma_obs=jnp.full(N, 5.0), sigma_pref=jnp.full(N, 4.0),
        gamma=jnp.full(N, 1.0), risk_weight=1.0, info_gain_weight=1.0,
        signalised=SIGNALISED,
    )
    assert float(jnp.mean(P[:, 0])) == pytest.approx(0.5, abs=1e-5)


def test_congested_route_penalised():
    N = 32
    state = _state(N=N, F_A=16.0, F_B=18.0, L_A=0.0, L_B=500.0)
    P = efe_route_probabilities(
        state=state,
        sigma_obs=jnp.full(N, 3.0), sigma_pref=jnp.full(N, 3.0),
        gamma=jnp.full(N, 2.0), risk_weight=1.0, info_gain_weight=1.0,
        signalised=SIGNALISED,
    )
    assert float(jnp.mean(P[:, 0])) > 0.95


def test_cost_offset_none_reproduces_baseline():
    """cost_offset=None is identical to an explicit all-zero offset."""
    N = 16
    state = _state(N=N)
    kw = dict(sigma_obs=jnp.full(N, 5.0), sigma_pref=jnp.full(N, 4.0),
              gamma=jnp.full(N, 1.0), risk_weight=1.0, info_gain_weight=1.0,
              signalised=SIGNALISED)
    P_none = efe_route_probabilities(state=state, cost_offset=None, **kw)
    P_zero = efe_route_probabilities(state=state, cost_offset=jnp.zeros((N, 2)), **kw)
    assert jnp.allclose(P_none, P_zero, atol=1e-7)


def test_positive_externality_offset_on_beta_shifts_to_alpha():
    """A positive perceived-cost offset on route beta (index 1) raises P_alpha."""
    N = 32
    state = _state(N=N)  # symmetric -> P_alpha = 0.5 without offset
    kw = dict(sigma_obs=jnp.full(N, 3.0), sigma_pref=jnp.full(N, 3.0),
              gamma=jnp.full(N, 2.0), risk_weight=1.0, info_gain_weight=1.0,
              signalised=SIGNALISED)
    offset = jnp.stack([jnp.zeros(N), jnp.full(N, 5.0)], axis=-1)  # penalise beta
    P = efe_route_probabilities(state=state, cost_offset=offset, **kw)
    assert float(jnp.mean(P[:, 0])) > 0.5
