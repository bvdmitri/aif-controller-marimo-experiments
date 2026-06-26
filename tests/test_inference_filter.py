"""Closed-form rolling-window Gaussian smoother: prior, window step,
F/C stability, independence."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aif_traffic.inference.filter import (
    C_IDX,
    CohortPriors,
    F_IDX,
    L_IDX,
    PHI_IDX,
    _build_prior,
    _inflate_sigma,
    init_variational_state,
)
from aif_traffic.inference.filter import window_step as _window_step_impl


W_DEFAULT = 5  # smaller W in tests for speed
# Existing (F, C, L)-oriented tests treat both routes as non-signalised so the
# green-split latent is inert and the forward map reduces to F + 60 L/C. The
# dedicated phi tests below pass signalised=[1, 0] explicitly.
SIGNALISED = jnp.asarray([0.0, 0.0])


def window_step(state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask, **kw):
    """Back-compat wrapper: supply default green-split obs + signalised mask
    so the existing (F, C, L)-oriented tests need no per-call changes."""
    N, W = mask.shape
    y_phi = kw.pop("y_phi", jnp.full((N, W), 0.45))
    sigma_phi = kw.pop("sigma_phi", jnp.full((N, W), 0.05))
    signalised = kw.pop("signalised", SIGNALISED)
    return _window_step_impl(
        state, priors,
        route_chosen_window=route, y_tt_window=y_tt, sigma_tt_window=sigma_tt,
        y_L_window=y_L, sigma_L_window=sigma_L,
        y_phi_window=y_phi, sigma_phi_window=sigma_phi,
        obs_mask=mask, signalised=signalised, **kw,
    )


def _cohort_priors(N=4, F_mu=16.0, C_mu=4000.0, L_mu=50.0,
                    F_sigma=2.0, C_sigma=1500.0, L_sigma=200.0,
                    phi_mu=0.45, phi_sigma=0.2):
    return CohortPriors(
        F_mu=jnp.full((N, 2), F_mu),
        F_sigma=jnp.full((N, 2), F_sigma),
        C_mu=jnp.full((N, 2), C_mu),
        C_sigma=jnp.full((N, 2), C_sigma),
        L_mu=jnp.full((N, 2), L_mu),
        L_sigma=jnp.full((N, 2), L_sigma),
        phi_mu=jnp.full((N, 2), phi_mu),
        phi_sigma=jnp.full((N, 2), phi_sigma),
    )


def _init_state(N=4, W=W_DEFAULT, **prior_kwargs):
    priors = _cohort_priors(N=N, **prior_kwargs)
    return init_variational_state(cohort_priors=priors), priors


def _marginal_var(state, slot):
    """Marginal variance of one slot per (agent, route)."""
    return jnp.sum(state.scale_tril[..., slot, :] ** 2, axis=-1)


def _empty_window(N, W):
    """Empty observation buffers for the window: route + TT/L observations + mask.

    Returns ``(route, y_tt, sigma_tt, y_L, sigma_L, mask)``.
    """
    route = jnp.zeros((N, W), dtype=jnp.int32)
    y_tt = jnp.zeros((N, W))
    sigma_tt = jnp.ones((N, W))
    y_L = jnp.zeros((N, W))
    sigma_L = jnp.full((N, W), 30.0)
    mask = jnp.zeros((N, W))
    return route, y_tt, sigma_tt, y_L, sigma_L, mask


def _fill_window(route, y_tt, sigma_tt, y_L, sigma_L, mask, slot,
                  route_idx, tt_val, L_val, sigma_tt_val=1.0, sigma_L_val=30.0):
    """Write today's TT + L observations into slot ``slot``."""
    return (
        route.at[:, slot].set(route_idx),
        y_tt.at[:, slot].set(tt_val),
        sigma_tt.at[:, slot].set(sigma_tt_val),
        y_L.at[:, slot].set(L_val),
        sigma_L.at[:, slot].set(sigma_L_val),
        mask.at[:, slot].set(1.0),
    )


def test_vb_recovers_per_agent_L_observation_noise():
    """With observation-noise learning on, the per-agent Gamma posterior over the
    queue-channel precision recovers the SD the data were generated with -- moving
    from the prior centre (30) toward the injected 15, despite only W per-agent
    observations (the shared weakly-informative prior gives shrinkage)."""
    import numpy as np
    N, W = 8, 30
    priors = _cohort_priors(N=N, L_sigma=200.0)        # diffuse L state prior
    state = init_variational_state(cohort_priors=priors)
    rng = np.random.default_rng(0)
    true_L, sd_L = 50.0, 15.0
    route = jnp.zeros((N, W), dtype=jnp.int32)         # all take route 0
    y_L = jnp.asarray(true_L + rng.normal(0, sd_L, size=(N, W)))
    # Make TT uninformative (huge SD) so the learned L-noise is what we test.
    y_tt = jnp.full((N, W), 16.0 + 60.0 * true_L / 4000.0)
    sigma_tt = jnp.full((N, W), 1e4)
    sigma_L = jnp.full((N, W), 30.0)                   # prior centre for L noise
    y_phi = jnp.full((N, W), 0.45)
    sigma_phi = jnp.full((N, W), 0.05)
    mask = jnp.ones((N, W))
    _, obs = _window_step_impl(
        state, priors, route_chosen_window=route, y_tt_window=y_tt,
        sigma_tt_window=sigma_tt, y_L_window=y_L, sigma_L_window=sigma_L,
        y_phi_window=y_phi, sigma_phi_window=sigma_phi, obs_mask=mask,
        signalised=SIGNALISED, W=W, learn_obs_noise=True, obs_noise_a0=1.0,
        obs_noise_vb_iters=10, return_obs_noise=True)
    learned = float(jnp.sqrt(obs.b_L / (obs.a_L - 1.0)).mean())
    print(f"\nInjected L obs SD = {sd_L}; learned (cohort mean) = {learned:.2f}"
          f"  (prior centre 30) -> {'PASS' if 8 < learned < 20 else 'FAIL'}")
    assert 8.0 < learned < 20.0                        # moved toward the truth
    assert abs(learned - sd_L) < abs(learned - 30.0)   # closer to truth than prior


def test_learn_obs_noise_off_is_identical_to_default():
    """With ``learn_obs_noise=False`` the smoother is bit-identical to not passing
    the flag at all -- the opt-in path never perturbs the default model."""
    N, W = 4, W_DEFAULT
    state, priors = _init_state(N=N)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _fill_window(
        route, y_tt, sigma_tt, y_L, sigma_L, mask, 0, 0, 18.0, 60.0)
    base = window_step(state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask, W=W)
    off = window_step(state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
                      W=W, learn_obs_noise=False)
    assert jnp.allclose(base.mu, off.mu)
    assert jnp.allclose(base.scale_tril, off.scale_tril)


def test_extra_obs_zero_mask_is_identical_to_omitting():
    """All-zero extra-observation masks fold no information: the smoother is
    bit-identical to not passing the extra-obs windows at all (the BL no-op)."""
    N, W = 4, W_DEFAULT
    state, priors = _init_state(N=N)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _fill_window(
        route, y_tt, sigma_tt, y_L, sigma_L, mask, W - 1, 0, 25.0, 120.0)
    base = window_step(state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask, W=W)
    z = jnp.zeros((N, 2, W))
    ones = jnp.ones((N, 2, W))
    with_extra = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask, W=W,
        y_extra_L_window=z, sigma_extra_L_window=ones, mask_extra_L_window=z,
        y_extra_phi_window=z, sigma_extra_phi_window=ones, mask_extra_phi_window=z)
    assert jnp.allclose(base.mu, with_extra.mu)
    assert jnp.allclose(base.scale_tril, with_extra.scale_tril)


def test_extra_obs_informs_the_unchosen_route():
    """An extra-observation relay of the route the agent did NOT take pulls that
    route's queue belief toward the relayed value and shrinks its variance --
    the documented departure from the chosen-route-only smoother. The chosen
    route (which routes are independent) is untouched by the relay."""
    N, W = 4, W_DEFAULT
    state, priors = _init_state(N=N, L_mu=50.0, L_sigma=200.0)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    # All agents take route 0 on the last day; route 1 is never chosen.
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _fill_window(
        route, y_tt, sigma_tt, y_L, sigma_L, mask, W - 1, 0, 25.0, 120.0)
    base = window_step(state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask, W=W)

    relay_val = 200.0
    y_xL = jnp.zeros((N, 2, W)).at[:, 1, W - 1].set(relay_val)
    sig_xL = jnp.full((N, 2, W), 10.0)
    m_xL = jnp.zeros((N, 2, W)).at[:, 1, W - 1].set(1.0)
    z = jnp.zeros((N, 2, W))
    ones = jnp.ones((N, 2, W))
    relayed = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask, W=W,
        y_extra_L_window=y_xL, sigma_extra_L_window=sig_xL, mask_extra_L_window=m_xL,
        y_extra_phi_window=z, sigma_extra_phi_window=ones, mask_extra_phi_window=z)

    L1_base = base.mu[:, 1, L_IDX]
    L1_relayed = relayed.mu[:, 1, L_IDX]
    var1_base = _marginal_var(base, L_IDX)[:, 1]
    var1_relayed = _marginal_var(relayed, L_IDX)[:, 1]
    assert jnp.all(L1_relayed > L1_base + 20.0)       # pulled toward 200
    assert jnp.all(var1_relayed < var1_base)          # and made more certain
    # Routes are independent: the route-1 relay leaves route 0 unchanged.
    assert jnp.allclose(base.mu[:, 0, L_IDX], relayed.mu[:, 0, L_IDX])


def test_init_variational_state_cohort_priors():
    """Initial state stores natural-space (F, C, L) means at the right slots."""
    state, priors = _init_state(N=2, F_mu=16.0, C_mu=4000.0, C_sigma=1500.0)
    F_mu = state.mu[..., F_IDX]
    C_mu = state.mu[..., C_IDX]
    L_mu = state.mu[..., L_IDX]
    assert float(F_mu[0, 0]) == pytest.approx(16.0)
    assert float(C_mu[0, 0]) == pytest.approx(4000.0)
    assert float(L_mu[0, 0]) == pytest.approx(50.0)


def test_init_state_has_diagonal_cholesky():
    """At init the Cholesky is diagonal: F, C, L are independent in the prior."""
    state, _ = _init_state(N=1)
    L = state.scale_tril[0, 0]  # one (agent, route), shape (3, 3)
    # Off-diagonal entries must all be zero.
    assert jnp.allclose(L[0, 1:], 0.0)
    assert jnp.allclose(L[1, 2:], 0.0)
    assert jnp.allclose(L[1:, 0], 0.0)
    assert jnp.allclose(L[2:, 1], 0.0)


def test_latent_state_shape_is_4():
    """Latent dim per (agent, route) is 4: F, C, L, phi."""
    state, _ = _init_state(N=1, C_sigma=1500.0)
    assert state.mu.shape == (1, 2, 4)
    assert state.scale_tril.shape == (1, 2, 4, 4)
    var_C = float(_marginal_var(state, C_IDX)[0, 0])
    assert var_C == pytest.approx(1500.0 ** 2, rel=1e-4)
    assert float(state.mu[0, 0, PHI_IDX]) == pytest.approx(0.45)


def test_window_step_pulls_predictive_toward_observation():
    """Closed-form Laplace must align the chosen route's predicted TT
    with the observation. With ``y_L`` anchoring L, the W observations
    jointly identify F and C in one window_step."""
    N = 4
    W = W_DEFAULT
    state, priors = _init_state(N=N, C_mu=3000.0, C_sigma=1200.0,
                                  L_mu=50.0, L_sigma=200.0)
    target_tt = 18.5
    # Pick an L observation consistent with target_tt and prior F=16, C=3000:
    #   target_tt = F + 60*L/C  ->  L = (target_tt - F)*C/60 = 2.5*3000/60 = 125
    target_L = 125.0

    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    for k in range(W):
        route, y_tt, sigma_tt, y_L, sigma_L, mask = _fill_window(
            route, y_tt, sigma_tt, y_L, sigma_L, mask, k,
            route_idx=0, tt_val=target_tt, L_val=target_L,
        )

    state = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_laplace_iters=3,
    )

    F = state.mu[0, 0, F_IDX]
    C = state.mu[0, 0, C_IDX]
    L = state.mu[0, 0, L_IDX]
    pred_A = F + 60.0 * jnp.maximum(L, 0.0) / jnp.maximum(C, 100.0)
    # Closed-form: tight tolerance (no SVI noise).
    assert float(pred_A) == pytest.approx(target_tt, abs=0.5)


def test_window_step_carries_forward_F_and_C_means_only_not_sigma():
    """At the boundary the new prior reuses previous-posterior means for F
    and C but resets their SDs to the cohort default."""
    N = 2
    W = W_DEFAULT
    state, priors = _init_state(N=N)

    # Hand-craft a state where F and C posterior means are shifted and
    # SDs are very tight.
    new_F_mean = 22.0
    new_C_mean = 1234.0
    mu = state.mu.at[..., F_IDX].set(new_F_mean)
    mu = mu.at[..., C_IDX].set(new_C_mean)
    scale_tril = state.scale_tril.at[..., F_IDX, F_IDX].set(0.1)
    scale_tril = scale_tril.at[..., C_IDX, C_IDX].set(1.0)
    state = state._replace(mu=mu, scale_tril=scale_tril)

    # The prior builder uses carry-forward means + cohort SDs.
    F_carry = state.mu[..., F_IDX]
    C_carry = state.mu[..., C_IDX]
    L_carry = state.mu[..., L_IDX]
    fresh_prior = _build_prior(
        F_mu=F_carry, F_sigma=priors.F_sigma,
        C_mu=C_carry, C_sigma=priors.C_sigma,
        L_mu=L_carry, L_sigma=priors.L_sigma,
        phi_mu=state.mu[..., PHI_IDX], phi_sigma=priors.phi_sigma,
    )
    # Means are carried forward.
    assert float(fresh_prior.mu[0, 0, F_IDX]) == pytest.approx(new_F_mean)
    assert float(fresh_prior.mu[0, 0, C_IDX]) == pytest.approx(new_C_mean)
    # SDs are the cohort defaults, NOT the tight previous-posterior SDs.
    assert float(fresh_prior.scale_tril[0, 0, F_IDX, F_IDX]) == pytest.approx(2.0)
    assert float(fresh_prior.scale_tril[0, 0, C_IDX, C_IDX]) == pytest.approx(1500.0)


def test_Sigma_does_not_collapse_over_many_windows():
    """After many window-steps with constant observations, F's marginal SD
    must stay bounded away from zero (Σ-reset prevents posterior collapse)."""
    N = 3
    W = W_DEFAULT
    state, priors = _init_state(N=N)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    for k in range(W):
        route, y_tt, sigma_tt, y_L, sigma_L, mask = _fill_window(
            route, y_tt, sigma_tt, y_L, sigma_L, mask, k,
            route_idx=0, tt_val=16.5, L_val=30.0,
        )

    F_sd_initial = jnp.sqrt(_marginal_var(state, F_IDX))[0, 0]
    for _ in range(20):
        state = window_step(
            state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
            W=W, n_laplace_iters=3,
        )
    F_sd_final = float(jnp.sqrt(_marginal_var(state, F_IDX))[0, 0])
    assert F_sd_final < float(F_sd_initial)
    assert F_sd_final > 0.05, f"F SD collapsed to {F_sd_final}"


def test_inflate_sigma_matches_random_walk_variance():
    """``σ_eff² = σ_cohort² + n_stale · σ_drift²`` element-wise."""
    sigma_cohort = jnp.array([[2.0, 3.0], [1.0, 5.0]])
    n_stale = jnp.array([[0, 10], [20, 5]])
    sigma_drift = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    out = _inflate_sigma(sigma_cohort, n_stale, sigma_drift)
    expected = jnp.sqrt(sigma_cohort ** 2 + n_stale.astype(jnp.float32) * sigma_drift ** 2)
    assert jnp.allclose(out, expected)
    # n_stale = 0 reproduces the cohort default exactly.
    out0 = _inflate_sigma(sigma_cohort, jnp.zeros_like(sigma_cohort, dtype=jnp.int32),
                          sigma_drift)
    assert jnp.allclose(out0, sigma_cohort)


def test_window_step_inflates_unchosen_route_prior():
    """When ``n_stale`` is high for route B and zero for route A, the
    rebuilt prior at the start of the window has inflated σ on B and the
    cohort default on A. Verified by checking the prior via _build_prior."""
    N = 2
    W = W_DEFAULT
    _, priors = _init_state(N=N, F_sigma=1.0, C_sigma=1500.0, L_sigma=200.0)
    # n_stale = 0 for A, n_stale = 20 for B.
    n_stale = jnp.array([[0, 20]] * N, dtype=jnp.int32)
    sigma_F_drift = jnp.full((N, 2), 0.2)
    sigma_C_drift = jnp.full((N, 2), 300.0)
    sigma_L_drift = jnp.full((N, 2), 50.0)

    F_sigma_eff = _inflate_sigma(priors.F_sigma, n_stale, sigma_F_drift)
    C_sigma_eff = _inflate_sigma(priors.C_sigma, n_stale, sigma_C_drift)
    L_sigma_eff = _inflate_sigma(priors.L_sigma, n_stale, sigma_L_drift)

    # Route A (n_stale = 0): σ unchanged.
    assert float(F_sigma_eff[0, 0]) == pytest.approx(1.0)
    assert float(C_sigma_eff[0, 0]) == pytest.approx(1500.0)
    assert float(L_sigma_eff[0, 0]) == pytest.approx(200.0)
    # Route B (n_stale = 20): σ inflated.
    assert float(F_sigma_eff[0, 1]) == pytest.approx((1.0 + 20 * 0.04) ** 0.5)
    assert float(C_sigma_eff[0, 1]) == pytest.approx((1500.0 ** 2 + 20 * 300.0 ** 2) ** 0.5)
    assert float(L_sigma_eff[0, 1]) == pytest.approx((200.0 ** 2 + 20 * 50.0 ** 2) ** 0.5)


def test_window_step_zero_obs_window_returns_inflated_prior_exactly():
    """When the window has zero observations (all-mask-zero), the closed-form
    update has nothing to apply: posterior == (inflated) prior bit-for-bit."""
    N = 3
    W = W_DEFAULT
    state, priors = _init_state(N=N)

    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    # All mask == 0, observations are zero → no Kalman updates should fire.

    n_stale = jnp.full((N, 2), 5, dtype=jnp.int32)
    sigma_F_drift = jnp.full((N, 2), 0.2)
    sigma_C_drift = jnp.full((N, 2), 300.0)
    sigma_L_drift = jnp.full((N, 2), 50.0)
    expected = _build_prior(
        F_mu=state.mu[..., F_IDX],
        F_sigma=_inflate_sigma(priors.F_sigma, n_stale, sigma_F_drift),
        C_mu=state.mu[..., C_IDX],
        C_sigma=_inflate_sigma(priors.C_sigma, n_stale, sigma_C_drift),
        L_mu=state.mu[..., L_IDX],
        L_sigma=_inflate_sigma(priors.L_sigma, n_stale, sigma_L_drift),
        phi_mu=state.mu[..., PHI_IDX],
        phi_sigma=_inflate_sigma(priors.phi_sigma, n_stale, jnp.zeros_like(sigma_F_drift)),
    )

    state_new = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W,
        n_stale_days=n_stale,
        sigma_F_drift=sigma_F_drift,
        sigma_C_drift=sigma_C_drift,
        sigma_L_drift=sigma_L_drift,
        n_laplace_iters=3,
    )

    assert jnp.allclose(state_new.mu, expected.mu, atol=1e-5)
    # Reconstruct covariances and compare; Cholesky representation can
    # differ slightly even for matching covariances, so compare Σ itself.
    Sigma_new = state_new.scale_tril @ jnp.swapaxes(state_new.scale_tril, -1, -2)
    Sigma_exp = expected.scale_tril @ jnp.swapaxes(expected.scale_tril, -1, -2)
    assert jnp.allclose(Sigma_new, Sigma_exp, atol=1e-4)


def test_iterated_laplace_converges():
    """The Laplace iteration converges quickly: posterior at n_iters=5
    matches posterior at n_iters=3 within a tight tolerance."""
    N = 4
    W = W_DEFAULT
    state, priors = _init_state(N=N, C_mu=3000.0, C_sigma=1200.0,
                                  L_mu=50.0, L_sigma=200.0)
    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    for k in range(W):
        route, y_tt, sigma_tt, y_L, sigma_L, mask = _fill_window(
            route, y_tt, sigma_tt, y_L, sigma_L, mask, k,
            route_idx=0, tt_val=18.5, L_val=125.0,
        )

    state_3 = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_laplace_iters=3,
    )
    state_5 = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_laplace_iters=5,
    )
    assert jnp.allclose(state_3.mu, state_5.mu, atol=1e-3)
    assert jnp.allclose(state_3.scale_tril, state_5.scale_tril, atol=1e-2)


# ---------------------------- per-agent independence --------------------------


def test_per_agent_independence_under_different_observations():
    """Two agents start identical; feed them different (TT, L) -- posteriors diverge."""
    N = 2
    W = W_DEFAULT
    state, priors = _init_state(N=N)

    route = jnp.zeros((N, W), dtype=jnp.int32)
    sigma_tt = jnp.ones((N, W))
    sigma_L = jnp.full((N, W), 30.0)
    mask = jnp.ones((N, W))
    # Agent 0: short TT (free-flowing). Agent 1: long TT (queued).
    y_tt = jnp.zeros((N, W)).at[0, :].set(12.0).at[1, :].set(30.0)
    y_L = jnp.zeros((N, W)).at[0, :].set(0.0).at[1, :].set(500.0)

    state = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_laplace_iters=3,
    )

    def pred(agent_idx):
        F = state.mu[agent_idx, 0, F_IDX]
        C = state.mu[agent_idx, 0, C_IDX]
        L = state.mu[agent_idx, 0, L_IDX]
        return float(F + 60.0 * jnp.maximum(L, 0.0) / jnp.maximum(C, 100.0))

    p0, p1 = pred(0), pred(1)
    assert abs(p0 - 12.0) < 3.5, p0
    assert abs(p1 - 30.0) < 3.5, p1
    assert abs(p0 - p1) > 10.0


def test_mean_reversion_pulls_toward_prior():
    """Quadratic (back-loaded) mean reversion: at n_stale == mean_revert_days
    the carry-forward mean is exactly at the cohort prior (f=t²=1); at half
    the days f=0.25 (back-loaded: most reversion happens late)."""
    N = 4
    W = W_DEFAULT
    state, priors = _init_state(N=N, F_mu=16.0)
    mu = state.mu.at[:, 1, F_IDX].set(30.0)
    state = state._replace(mu=mu)

    route, y_tt, sigma_tt, y_L, sigma_L, mask = _empty_window(N, W)
    forget_days = 20

    # At n_stale = forget_days: t=1, f=1 → prior exactly = 16.
    n_stale_full = jnp.zeros((N, 2), dtype=jnp.int32).at[:, 1].set(forget_days)
    state_full = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_stale_days=n_stale_full, n_laplace_iters=1,
        mean_revert_days=float(forget_days),
    )
    assert float(state_full.mu[0, 1, F_IDX]) == pytest.approx(16.0, abs=0.1)

    # At n_stale = forget_days/2: t=0.5, f=0.25 → 0.75*30 + 0.25*16 = 26.5.
    n_stale_half = jnp.zeros((N, 2), dtype=jnp.int32).at[:, 1].set(forget_days // 2)
    state_half = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_stale_days=n_stale_half, n_laplace_iters=1,
        mean_revert_days=float(forget_days),
    )
    assert float(state_half.mu[0, 1, F_IDX]) == pytest.approx(26.5, abs=0.1)

    # mean_revert_days=0: carry-forward preserved exactly.
    state_off = window_step(
        state, priors, route, y_tt, sigma_tt, y_L, sigma_L, mask,
        W=W, n_stale_days=n_stale_full, n_laplace_iters=1,
        mean_revert_days=0.0,
    )
    assert float(state_off.mu[0, 1, F_IDX]) == pytest.approx(30.0, abs=0.1)

    # Route A (n_stale=0) unaffected.
    assert float(state_full.mu[0, 0, F_IDX]) == pytest.approx(
        float(state_off.mu[0, 0, F_IDX]), abs=0.01,
    )


def test_perturbing_one_agent_does_not_affect_others():
    """Run the SAME window twice; on the second run, change agent 0's
    observations only. Agents 1..N-1 must end with bit-identical params."""
    N = 6
    W = W_DEFAULT
    base_state, priors = _init_state(N=N)
    route = jnp.zeros((N, W), dtype=jnp.int32)
    sigma_tt = jnp.ones((N, W))
    sigma_L = jnp.full((N, W), 30.0)
    mask = jnp.ones((N, W))
    y_L = jnp.full((N, W), 50.0)

    y_tt_a = jnp.full((N, W), 16.0)
    y_tt_b = y_tt_a.at[0, :].set(30.0)  # only agent 0 differs

    state_a = window_step(
        base_state, priors, route, y_tt_a, sigma_tt, y_L, sigma_L, mask,
        W=W, n_laplace_iters=3,
    )
    state_b = window_step(
        base_state, priors, route, y_tt_b, sigma_tt, y_L, sigma_L, mask,
        W=W, n_laplace_iters=3,
    )

    assert not jnp.allclose(state_a.mu[0], state_b.mu[0], atol=1e-3)
    assert jnp.allclose(state_a.mu[1:], state_b.mu[1:], atol=1e-5)
    assert jnp.allclose(
        state_a.scale_tril[1:], state_b.scale_tril[1:], atol=1e-5,
    )

# NOTE: the smoother is first-hand-only (IWAI-verbatim) -- it folds in
# observations of the chosen route alone. The controller's belief is fused
# transiently at route-choice time (see tests/test_belief_informing.py and
# inference/population.py), never into this smoother, so there are no
# belief-broadcast folds to test here.
