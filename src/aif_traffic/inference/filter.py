"""Closed-form rolling-window Gaussian smoother (linearised Kalman / Laplace).

The smoother carries a multivariate Gaussian posterior over the latent
vector::

    z = (F, C, L, phi)  in R^4

per (agent, route). All four latents are treated as fixed within the
W-day window -- the same L value explains all W queue observations.

Extension beyond the verbatim IWAI model: ``phi`` is the traveller's belief
over the **green-split fraction** allocated to the route's signalised movement.
On a signalised route the effective capacity decomposes as ``C_eff = phi * C``
(so ``C`` is the saturation flow), giving ``TT = F + 60*L/(phi*C)``; on a
non-signalised route ``phi`` is inert and ``TT = F + 60*L/C`` as before. The
green split is observed *directly* only when the route is chosen (it is what
disentangles ``phi`` from the saturation flow ``C``), and decays/inflates
between observations exactly like the other latents. A per-route ``signalised``
mask selects where ``phi`` couples. Each day:

1. The observation buffer is shifted left by one slot and today's
   ``(route, y_TT, y_L)`` is written into slot ``W-1``. Cold-start slots
   (before day W) stay masked off.
2. A fresh prior is built using the carry-forward **means** (previous
   window's posterior means for $F$, $C$, and $L$) and the
   **cohort-default covariance** (Σ-reset), inflated by ``n_stale_days``
   per route to widen the prior on routes the agent has been neglecting.
3. The W likelihoods are folded in via **closed-form linearised Gaussian
   updates** (iterated extended Kalman / Laplace). The TT forward map
   ``h(F, C, L) = F + 60·L/C`` is the only non-linearity; we linearise
   it around the current posterior mean and apply rank-1 Kalman updates
   sequentially over the W days. A small number of relinearisation
   iterations (default 3) converges the Laplace approximation.

There is no SVI, no stochastic gradient, no PRNG plumbing -- the
posterior is deterministic given the inputs. For an unobserved route
the Kalman updates apply no information and the posterior equals the
(inflated) prior exactly; no "clamp" needed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# Index conventions for the 4-dim latent vector (F, C, L, phi) per (agent, route).
F_IDX = 0
C_IDX = 1
L_IDX = 2
PHI_IDX = 3

# Default green-split guard bounds (overridable per call via phi_lo/phi_hi).
PHI_LO_DEFAULT = 0.1
PHI_HI_DEFAULT = 0.9


def _green(
    phi: jnp.ndarray, signalised: jnp.ndarray, phi_lo: float, phi_hi: float,
) -> jnp.ndarray:
    """Effective green multiplier per (agent, route): clamp ``phi`` to
    ``[phi_lo, phi_hi]`` on signalised routes, ``1`` elsewhere."""
    phi_safe = jnp.clip(phi, phi_lo, phi_hi)
    return jnp.where(signalised > 0.5, phi_safe, 1.0)


def predicted_tt(
    F: jnp.ndarray, C: jnp.ndarray, L: jnp.ndarray, phi: jnp.ndarray,
    signalised: jnp.ndarray,
    phi_lo: float = PHI_LO_DEFAULT, phi_hi: float = PHI_HI_DEFAULT,
) -> jnp.ndarray:
    """Deterministic forward map :math:`TT = F + 60\\,L/(\\phi\\,C)`.

    On a signalised route the effective capacity is ``phi * C`` (``C`` is the
    saturation flow); on a non-signalised route ``phi`` is ignored (``green=1``)
    and the map reduces to the IWAI form ``TT = F + 60 L / C``.

    Numerical guards:

    * Capacity floored at 100 veh/h (keeps ``1/C`` bounded for negative samples).
    * Queue length floored at 0 (physically can't be negative).
    * Green split clamped to ``[phi_lo, phi_hi]`` (keeps ``1/phi`` bounded).
    """
    C_safe = jnp.maximum(C, 100.0)
    L_safe = jnp.maximum(L, 0.0)
    green = _green(phi, signalised, phi_lo, phi_hi)
    return F + 60.0 * L_safe / (C_safe * green)


class CohortPriors(NamedTuple):
    """Per-agent cohort-default priors used to rebuild Σ every window.

    All fields are JAX arrays of shape ``(N, 2)`` (per agent, per route).
    These priors are NOT updated across days -- they are the fresh
    "reset" applied at every window boundary (then inflated by
    ``n_stale_days``).
    """

    F_mu: jnp.ndarray
    F_sigma: jnp.ndarray
    C_mu: jnp.ndarray
    C_sigma: jnp.ndarray
    L_mu: jnp.ndarray
    L_sigma: jnp.ndarray
    phi_mu: jnp.ndarray
    phi_sigma: jnp.ndarray


class VariationalState(NamedTuple):
    """Per-agent, per-route multivariate-Gaussian posterior.

    Attributes:
        mu: ``(N, 2, 4)`` per-agent, per-route mean of $(F, C, L, \\phi)$.
        scale_tril: ``(N, 2, 4, 4)`` lower-triangular Cholesky factor.
    """

    mu: jnp.ndarray
    scale_tril: jnp.ndarray


def _inflate_sigma(
    sigma_cohort: jnp.ndarray,
    n_stale: jnp.ndarray,
    sigma_drift: jnp.ndarray,
) -> jnp.ndarray:
    """Inflate the cohort-default σ by ``n_stale`` days of drift.

    ``σ_eff² = σ_cohort² + n_stale · σ_drift²``. This is the variance the
    latent would have under a Gaussian random walk with per-day SD
    ``σ_drift`` propagating undisturbed for ``n_stale`` days since the
    last observation. All inputs broadcast to ``(N, 2)``.
    """
    n_stale_f = n_stale.astype(sigma_cohort.dtype)
    return jnp.sqrt(sigma_cohort ** 2 + n_stale_f * sigma_drift ** 2)


def _build_prior(
    F_mu: jnp.ndarray, F_sigma: jnp.ndarray,
    C_mu: jnp.ndarray, C_sigma: jnp.ndarray,
    L_mu: jnp.ndarray, L_sigma: jnp.ndarray,
    phi_mu: jnp.ndarray, phi_sigma: jnp.ndarray,
) -> VariationalState:
    """Assemble the joint Gaussian prior per (agent, route).

    $F$, $C$, $L$, and $\\phi$ are independent at the prior level; the Cholesky
    factor is diagonal. Returns a :class:`VariationalState` (mu + scale_tril).
    """
    # Mean: (F, C, L, phi) stacked along last axis → (N, 2, 4).
    mu = jnp.stack([F_mu, C_mu, L_mu, phi_mu], axis=-1)

    # Diagonal Cholesky: (N, 2, 4, 4).
    scale_tril = jnp.zeros((*F_mu.shape, 4, 4))
    scale_tril = scale_tril.at[..., F_IDX, F_IDX].set(F_sigma)
    scale_tril = scale_tril.at[..., C_IDX, C_IDX].set(C_sigma)
    scale_tril = scale_tril.at[..., L_IDX, L_IDX].set(L_sigma)
    scale_tril = scale_tril.at[..., PHI_IDX, PHI_IDX].set(phi_sigma)
    return VariationalState(mu=mu, scale_tril=scale_tril)


def init_variational_state(
    cohort_priors: CohortPriors,
) -> VariationalState:
    """Day-0 state = the cohort-default joint prior.

    No observations yet; the closed-form smoother starts from this and
    rebuilds a fresh prior with mean carry-forward + Σ-reset (and stale-
    route inflation) on every window-step.
    """
    return _build_prior(
        F_mu=cohort_priors.F_mu, F_sigma=cohort_priors.F_sigma,
        C_mu=cohort_priors.C_mu, C_sigma=cohort_priors.C_sigma,
        L_mu=cohort_priors.L_mu, L_sigma=cohort_priors.L_sigma,
        phi_mu=cohort_priors.phi_mu, phi_sigma=cohort_priors.phi_sigma,
    )


# --------------------------------------------------------------------------
# Closed-form linearised Kalman update
# --------------------------------------------------------------------------


def _kalman_one_obs(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    H: jnp.ndarray,
    innovation: jnp.ndarray,
    R_var: jnp.ndarray,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One rank-1 Kalman update, gated by ``active``.

    Standard innovation-form update:

    * ``S = H Σ H^T + R``,
    * ``K = Σ H^T / S``,
    * ``μ ← μ + K · innovation``,
    * ``Σ ← Σ - K · (H Σ)``.

    Where ``active = 0`` the update is a no-op (innovation gated to zero
    and Σ left untouched via ``jnp.where``).

    Args:
        mu:         ``(N, 2, D)``
        Sigma:      ``(N, 2, D, D)``
        H:          ``(N, 2, D)`` row of the observation Jacobian
        innovation: ``(N, 2)`` residual ``y - H·μ - b``
        R_var:      ``(N, 2)`` observation noise variance
        active:     ``(N, 2)`` 0/1 gate per (agent, route)
    """
    H_Sigma = jnp.einsum("...d,...de->...e", H, Sigma)  # (N, 2, D)
    S = jnp.einsum("...d,...d->...", H_Sigma, H) + R_var  # (N, 2)
    K = H_Sigma / S[..., None]                            # (N, 2, D)

    gated_innov = active * innovation                     # (N, 2)
    mu_new = mu + K * gated_innov[..., None]

    Sigma_update = jnp.einsum("...i,...j->...ij", K, H_Sigma)
    Sigma_new = jnp.where(
        active[..., None, None] > 0.5,
        Sigma - Sigma_update,
        Sigma,
    )
    return mu_new, Sigma_new


def _laplace_iter_step(
    mu_lin: jnp.ndarray,
    prior_mu: jnp.ndarray,
    prior_Sigma: jnp.ndarray,
    route_chosen_window: jnp.ndarray,
    y_tt_window: jnp.ndarray,
    sigma_tt_window: jnp.ndarray,
    y_L_window: jnp.ndarray,
    sigma_L_window: jnp.ndarray,
    y_phi_window: jnp.ndarray,
    sigma_phi_window: jnp.ndarray,
    obs_mask: jnp.ndarray,
    signalised: jnp.ndarray,
    phi_lo: float,
    phi_hi: float,
    W: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One iterated-Laplace step.

    Linearises the TT forward map around ``mu_lin`` and folds in all W
    chosen-route observations (TT, L, and -- on signalised routes -- the green
    split ``phi``) starting from the prior.
    """
    N, R, _ = prior_mu.shape
    F_lin = mu_lin[..., F_IDX]
    C_lin = mu_lin[..., C_IDX]
    L_lin = mu_lin[..., L_IDX]
    phi_lin = mu_lin[..., PHI_IDX]
    C_safe = jnp.maximum(C_lin, 100.0)
    phi_clip = jnp.clip(phi_lin, phi_lo, phi_hi)
    sig = signalised[None, :]                             # (1, R)
    green = jnp.where(sig > 0.5, phi_clip, 1.0)           # (N, R)
    denom = C_safe * green

    chosen = (
        route_chosen_window[:, None, :] == jnp.arange(R)[None, :, None]
    ).astype(prior_mu.dtype)  # (N, R, W)
    active_all = chosen * obs_mask[:, None, :]  # (N, R, W)

    mu = prior_mu
    Sigma = prior_Sigma

    for d in range(W):
        active_d = active_all[..., d]                     # (N, R)

        # --- y_TT,d: linearise h(F,C,L,phi) at mu_lin ------------------
        # h = F + 60 L / (phi*C) on signalised routes (green=phi), else
        # F + 60 L / C (green=1). Jacobian rows:
        #   ∂h/∂F = 1,  ∂h/∂C = -60 L /(C²·green),  ∂h/∂L = 60/(C·green),
        #   ∂h/∂phi = -60 L /(C·phi²)  on signalised routes, else 0.
        H_tt = jnp.zeros((N, R, 4), dtype=mu.dtype)
        H_tt = H_tt.at[..., F_IDX].set(1.0)
        H_tt = H_tt.at[..., C_IDX].set(-60.0 * L_lin / (C_safe ** 2 * green))
        H_tt = H_tt.at[..., L_IDX].set(60.0 / denom)
        H_tt = H_tt.at[..., PHI_IDX].set(
            jnp.where(sig > 0.5, -60.0 * L_lin / (C_safe * phi_clip ** 2), 0.0)
        )
        # Predicted TT at linearisation point and standard-form innovation
        # ``y - H·μ - b`` with ``b = h(mu_lin) - H·mu_lin``. After the
        # first observation in this iter ``mu`` may have moved off
        # ``mu_lin``; the next Laplace iter re-linearises around the
        # updated mean so the local Taylor stays valid.
        h_pred = F_lin + 60.0 * L_lin / denom
        H_dot_mu = jnp.einsum("...d,...d->...", H_tt, mu)
        H_dot_mu_lin = jnp.einsum("...d,...d->...", H_tt, mu_lin)
        innovation = y_tt_window[:, d][:, None] - h_pred - (H_dot_mu - H_dot_mu_lin)
        R_var = (sigma_tt_window[:, d][:, None]) ** 2     # (N, R)

        mu, Sigma = _kalman_one_obs(mu, Sigma, H_tt, innovation, R_var, active_d)

        # --- y_L,d: linear; H_L is one-hot at L_IDX ---------------------
        H_L = jnp.zeros((N, R, 4), dtype=mu.dtype)
        H_L = H_L.at[..., L_IDX].set(1.0)
        innov_L = y_L_window[:, d][:, None] - mu[..., L_IDX]
        R_var_L = (sigma_L_window[:, d][:, None]) ** 2

        mu, Sigma = _kalman_one_obs(mu, Sigma, H_L, innov_L, R_var_L, active_d)

        # --- y_phi,d: direct green-split obs, signalised routes only ----
        # H_phi one-hot at PHI_IDX; gated to chosen AND signalised routes.
        # This is what identifies phi separately from the saturation flow C.
        H_phi = jnp.zeros((N, R, 4), dtype=mu.dtype)
        H_phi = H_phi.at[..., PHI_IDX].set(1.0)
        innov_phi = y_phi_window[:, d][:, None] - mu[..., PHI_IDX]
        R_var_phi = (sigma_phi_window[:, d][:, None]) ** 2
        active_phi = active_d * sig                       # (N, R)

        mu, Sigma = _kalman_one_obs(mu, Sigma, H_phi, innov_phi, R_var_phi, active_phi)

    # Symmetrise Σ for numerical safety (rank-1 updates preserve symmetry
    # in exact arithmetic; float rounding can drift).
    Sigma = 0.5 * (Sigma + jnp.swapaxes(Sigma, -1, -2))
    return mu, Sigma


def window_step(
    state: VariationalState,
    cohort_priors: CohortPriors,
    route_chosen_window: jnp.ndarray,
    y_tt_window: jnp.ndarray,
    sigma_tt_window: jnp.ndarray,
    y_L_window: jnp.ndarray,
    sigma_L_window: jnp.ndarray,
    y_phi_window: jnp.ndarray,
    sigma_phi_window: jnp.ndarray,
    obs_mask: jnp.ndarray,
    signalised: jnp.ndarray,
    W: int,
    n_stale_days: jnp.ndarray | None = None,
    sigma_F_drift: jnp.ndarray | float = 0.0,
    sigma_C_drift: jnp.ndarray | float = 0.0,
    sigma_L_drift: jnp.ndarray | float = 0.0,
    sigma_phi_drift: jnp.ndarray | float = 0.0,
    n_laplace_iters: int = 3,
    mean_revert_days: jnp.ndarray | float = 0.0,
    phi_lo: float = PHI_LO_DEFAULT,
    phi_hi: float = PHI_HI_DEFAULT,
) -> VariationalState:
    """One smoother step: rebuild prior (with carry-forward + Σ inflation),
    then iterated linearised Kalman over the W-day window.

    For an unobserved route (no `obs_mask=1` & `route_chosen==r` slot in
    the buffer), the Kalman updates have ``active = 0`` everywhere and
    the posterior equals the (inflated) prior exactly. No SVI noise, no
    "clamp" needed.

    Inputs:
        ``y_tt_window``:    realised TT obs per (agent, day), shape ``(N, W)``.
        ``sigma_tt_window``: TT noise SD,                       ``(N, W)``.
        ``y_L_window``:     noisy queue-length obs,             ``(N, W)``.
        ``sigma_L_window``: L noise SD,                         ``(N, W)``.
        ``n_stale_days``:   ``(N, 2)`` days since each (agent, route) was
                            last observed; ``None`` ⇒ all-zeros (no inflation).
        ``sigma_F_drift``, ``sigma_C_drift``, ``sigma_L_drift``: per-day
            drift SDs. Each scalar or ``(N, 2)``. Defaults of ``0`` ⇒
            Σ-reset to cohort default with no inflation.
        ``n_laplace_iters``: relinearisation count. ``3`` is plenty for
            convergence at default knobs; ``1`` ≡ basic EKF.
        ``mean_revert_days``: number of stale days after which the carry-forward
            mean for an unobserved route is fully reverted to the cohort prior
            mean. Quadratic (back-loaded) schedule: ``t = min(n_stale/days, 1)``,
            ``f = t²``. The mean stays near its carry-forward value for most of
            the window, then accelerates toward the prior, reaching it exactly at
            ``N`` days. ``0`` = disabled. Per-agent ``(N,)``, ``(N, 2)``, scalar.
    """
    F_carry = state.mu[..., F_IDX]                       # (N, 2)
    C_carry = state.mu[..., C_IDX]                       # (N, 2)
    L_carry = state.mu[..., L_IDX]                       # (N, 2)
    phi_carry = state.mu[..., PHI_IDX]                   # (N, 2)

    if n_stale_days is None:
        n_stale_days = jnp.zeros_like(F_carry, dtype=jnp.int32)
    n_stale_b = jnp.broadcast_to(jnp.asarray(n_stale_days), F_carry.shape)

    md = jnp.asarray(mean_revert_days, dtype=F_carry.dtype)
    if md.ndim == 1:
        md = md[:, None]   # (N,) → (N, 1) to broadcast over routes
    md_b = jnp.broadcast_to(md, F_carry.shape)
    # Quadratic (back-loaded): f = min(n_stale / days, 1)^2. md_b == 0
    # means disabled for that (agent, route); the where-mask below selects
    # f = 0 in that case. The 1.0 floor inside the divide is purely to keep
    # the value finite for md_b == 0 entries (the where then discards it).
    md_safe = jnp.where(md_b > 0, md_b, 1.0)
    t = jnp.minimum(n_stale_b.astype(F_carry.dtype) / md_safe, 1.0)
    f = jnp.where(md_b > 0, t ** 2, 0.0)
    F_carry = (1.0 - f) * F_carry + f * cohort_priors.F_mu
    C_carry = (1.0 - f) * C_carry + f * cohort_priors.C_mu
    L_carry = (1.0 - f) * L_carry + f * cohort_priors.L_mu
    phi_carry = (1.0 - f) * phi_carry + f * cohort_priors.phi_mu
    sigma_F_drift_b = jnp.broadcast_to(
        jnp.asarray(sigma_F_drift, dtype=F_carry.dtype), F_carry.shape,
    )
    sigma_C_drift_b = jnp.broadcast_to(
        jnp.asarray(sigma_C_drift, dtype=F_carry.dtype), F_carry.shape,
    )
    sigma_L_drift_b = jnp.broadcast_to(
        jnp.asarray(sigma_L_drift, dtype=F_carry.dtype), F_carry.shape,
    )
    sigma_phi_drift_b = jnp.broadcast_to(
        jnp.asarray(sigma_phi_drift, dtype=F_carry.dtype), F_carry.shape,
    )

    F_sigma_eff = _inflate_sigma(cohort_priors.F_sigma, n_stale_b, sigma_F_drift_b)
    C_sigma_eff = _inflate_sigma(cohort_priors.C_sigma, n_stale_b, sigma_C_drift_b)
    L_sigma_eff = _inflate_sigma(cohort_priors.L_sigma, n_stale_b, sigma_L_drift_b)
    phi_sigma_eff = _inflate_sigma(cohort_priors.phi_sigma, n_stale_b, sigma_phi_drift_b)

    prior = _build_prior(
        F_mu=F_carry, F_sigma=F_sigma_eff,
        C_mu=C_carry, C_sigma=C_sigma_eff,
        L_mu=L_carry, L_sigma=L_sigma_eff,
        phi_mu=phi_carry, phi_sigma=phi_sigma_eff,
    )

    prior_Sigma = prior.scale_tril @ jnp.swapaxes(prior.scale_tril, -1, -2)

    mu = prior.mu
    Sigma = prior_Sigma
    for _ in range(n_laplace_iters):
        mu, Sigma = _laplace_iter_step(
            mu_lin=mu,
            prior_mu=prior.mu,
            prior_Sigma=prior_Sigma,
            route_chosen_window=route_chosen_window,
            y_tt_window=y_tt_window,
            sigma_tt_window=sigma_tt_window,
            y_L_window=y_L_window,
            sigma_L_window=sigma_L_window,
            y_phi_window=y_phi_window,
            sigma_phi_window=sigma_phi_window,
            obs_mask=obs_mask,
            signalised=signalised,
            phi_lo=phi_lo,
            phi_hi=phi_hi,
            W=W,
        )

    scale_tril = jnp.linalg.cholesky(Sigma)
    return VariationalState(mu=mu, scale_tril=scale_tril)
