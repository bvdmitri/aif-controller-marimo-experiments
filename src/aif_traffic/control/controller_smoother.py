"""Rolling-window Gaussian smoother for the controller's queue-trajectory belief.

The Active-Inference signal controller is **one big agent** whose latent is the
entire within-day queue trajectory of a signalised movement, ``x = (L(t))_{t=1..M}``
on a grid of ``M`` nodes (per within-day minute, or per control epoch). It is the
macro analogue of the travellers' rolling-window smoother (:mod:`inference.filter`):
each day it observes the realised trajectory, and it estimates the *typical*
trajectory over a window of the last ``W`` days. Two movements (links 2 and 6)
are independent given the (known) split, so each is smoothed by a separate call.

Generative model (linear-Gaussian state-space over the trajectory):

* **Dynamics prior** -- a random walk with drift, ``L(t+1) = L(t) + u(t) + w(t)``,
  ``w ~ N(0, q)`` (``q`` the per-step process variance), anchored at the start by
  ``L(0) ~ N(0, sigma0^2)``. This is what makes the posterior covariance **full**
  (dense, with temporal correlations ``Cov(L(t), L(s)) ∝ min(t, s)``); the
  precision of this prior is **tridiagonal**. The drift only sets the prior
  *mean* ``mu_prior`` (the deterministic store-and-forward rollout); the precision
  depends only on ``q`` and ``sigma0``.
* **Observation** -- linear, identity, per node: ``o(t) = L(t) + v(t)``,
  ``v ~ N(0, R(t))`` with a split-dependent precision (more green => sharper).
  Linear observations need no linearisation -- this is an exact Gaussian solve.

Folding ``W`` days of observations (the same typical trajectory explains all ``W``
days, exactly as a traveller's ``L`` explains all ``W`` of its days) keeps the
posterior precision tridiagonal::

    Lambda_post = Lambda_prior + diag( sum_d mask_d / R_d )      (banded, bandwidth 1)
    b_post      = Lambda_prior @ mu_prior + sum_d mask_d * o_d / R_d
    mu_post     = Lambda_post^{-1} b_post

solved by an ``O(M)`` LDL^T factorisation; the per-node marginal variances
``diag(Lambda_post^{-1})`` come from an ``O(M)`` backward recursion. So the big
state is genuinely big yet cheap. :func:`dense_reference` recomputes the same
posterior with dense linear algebra and is used only to validate the banded path.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tridiagonal (symmetric, SPD) linear algebra: LDL^T, solve, inverse-diagonal.
# A symmetric tridiagonal matrix is stored as (diag, off) with
# ``off[i] = M[i, i+1] = M[i+1, i]``.
# ---------------------------------------------------------------------------
def _ldl_tridiag(diag: np.ndarray, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``M = L D L^T`` for symmetric tridiagonal SPD ``M``. Returns ``(d, l)``
    with ``d`` the diagonal of ``D`` (length ``K``) and ``l`` the unit-lower
    subdiagonal ``L[i+1, i]`` (length ``K-1``). Stable for SPD ``M`` (``d>0``)."""
    K = diag.shape[0]
    d = np.empty(K)
    l = np.empty(max(K - 1, 0))
    d[0] = diag[0]
    for i in range(1, K):
        l[i - 1] = off[i - 1] / d[i - 1]
        d[i] = diag[i] - l[i - 1] * off[i - 1]
    return d, l


def _solve_ldl(d: np.ndarray, l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``M x = b`` given the ``(d, l)`` LDL^T factors of tridiagonal ``M``."""
    K = d.shape[0]
    z = np.empty(K)
    z[0] = b[0]
    for i in range(1, K):
        z[i] = b[i] - l[i - 1] * z[i - 1]      # forward: L z = b
    w = z / d                                   # diagonal: D w = z
    x = np.empty(K)
    x[K - 1] = w[K - 1]
    for i in range(K - 2, -1, -1):
        x[i] = w[i] - l[i] * x[i + 1]           # backward: L^T x = w
    return x


def _inv_diag_tridiag(d: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Diagonal of ``M^{-1}`` (the per-node marginal variances) from the ``(d, l)``
    LDL^T factors, via the stable ``O(K)`` backward recursion
    ``S[K-1] = 1/d[K-1]``; ``S[i] = 1/d[i] + l[i]^2 S[i+1]``."""
    K = d.shape[0]
    S = np.empty(K)
    S[K - 1] = 1.0 / d[K - 1]
    for i in range(K - 2, -1, -1):
        S[i] = 1.0 / d[i] + l[i] ** 2 * S[i + 1]
    return S


def _tridiag_matvec(diag: np.ndarray, off: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``M x`` for symmetric tridiagonal ``M`` stored as ``(diag, off)``."""
    y = diag * x
    y[:-1] += off * x[1:]
    y[1:] += off * x[:-1]
    return y


# ---------------------------------------------------------------------------
# Random-walk trajectory prior
# ---------------------------------------------------------------------------
def rw_prior_precision(M: int, q: float, sigma0: float) -> tuple[np.ndarray, np.ndarray]:
    """Tridiagonal precision of the anchored random-walk prior on ``M`` nodes.

    Increment penalty ``(1/q) * sum_t (L(t+1) - L(t) - u(t))^2`` (the drift ``u``
    only shifts the mean, not the precision) plus an anchor ``L(0) ~ N(0, sigma0^2)``
    making the prior proper. Returns ``(diag, off)``: interior diag ``2/q``,
    endpoints ``1/q`` (``+1/sigma0^2`` at node 0), off-diagonal ``-1/q``.
    """
    q = float(q)
    diag = np.full(M, 2.0 / q)
    diag[0] = 1.0 / q
    diag[-1] = 1.0 / q
    diag[0] += 1.0 / (sigma0 ** 2)
    off = np.full(max(M - 1, 0), -1.0 / q)
    return diag, off


def _accumulate_data(
    obs_traj: np.ndarray, obs_var: np.ndarray, obs_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum the ``W`` days' linear-identity observations into the information form.

    Returns ``(prec_data, info_data)`` of shape ``(M,)``:
    ``prec_data[t] = sum_d mask_d / R_d[t]`` and
    ``info_data[t] = sum_d mask_d * o_d[t] / R_d[t]``.
    """
    inv_R = obs_mask[:, None] / np.maximum(obs_var, 1e-9)   # (W, M)
    prec_data = inv_R.sum(axis=0)                            # (M,)
    info_data = (inv_R * obs_traj).sum(axis=0)               # (M,)
    return prec_data, info_data


def window_smoother(
    prior_mean: np.ndarray,
    q: float,
    sigma0: float,
    obs_traj: np.ndarray,
    obs_var: np.ndarray,
    obs_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean + per-node marginal variance of the trajectory belief.

    Args:
        prior_mean: ``(M,)`` prior-mean trajectory (the store-and-forward rollout).
        q:          per-step process variance of the random-walk prior.
        sigma0:     start-of-day anchor SD.
        obs_traj:   ``(W, M)`` realised trajectories over the window.
        obs_var:    ``(W, M)`` per-day, per-node observation variance ``R``.
        obs_mask:   ``(W,)`` 0/1 window-slot activity (cold-start days = 0).

    Returns ``(mu_post, var_post)`` each ``(M,)``. With no active days the
    posterior equals the prior (mean = ``prior_mean``, var = ``diag(Lambda_prior^{-1})``).
    """
    M = prior_mean.shape[0]
    diag, off = rw_prior_precision(M, q, sigma0)
    b = _tridiag_matvec(diag, off, prior_mean)               # b_prior = Lambda_prior mu_prior

    prec_data, info_data = _accumulate_data(obs_traj, obs_var, obs_mask)
    diag = diag + prec_data                                  # Lambda_post (still tridiagonal)
    b = b + info_data

    d, l = _ldl_tridiag(diag, off)
    mu_post = _solve_ldl(d, l, b)
    var_post = _inv_diag_tridiag(d, l)
    return mu_post, var_post


def window_smoother_vb(
    prior_mean: np.ndarray,
    q: float,
    sigma0: float,
    obs_traj: np.ndarray,
    obs_weight: np.ndarray,
    obs_mask: np.ndarray,
    a0: float,
    b0: float,
    n_iters: int = 8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Like :func:`window_smoother`, but **learns the observation noise** by
    mean-field coordinate-ascent variational Bayes.

    The observation precision is factored as ``prec_d[t] = tau * w_d[t]`` where
    ``w_d[t] = obs_weight[d, t]`` is the **known** structural weight (the
    split-dependent ``phi/phi_ref``: more green -> sharper) and ``tau = 1/sigma_obs^2``
    is a single unknown precision *scale* shared across the window, with a
    conjugate prior ``tau ~ Gamma(a0, b0)``. We seek the mean-field posterior
    ``q(L_traj) q(tau)`` with ``q(tau) = Gamma(a_post, b_post)``.

    Coordinate ascent (closed-form, deterministic):

    * **state step** -- fix ``E[tau] = a/b``; the per-node observation variance is
      ``R_d[t] = 1/(E[tau] * w_d[t])`` and the trajectory posterior is the banded
      :func:`window_smoother` solve (reused verbatim);
    * **noise step** -- fix ``q(L)``; the conjugate Gamma update over the active
      observations is ``a_post = a0 + 1/2 * N_active`` and
      ``b_post = b0 + 1/2 * sum_{d,t active} w_d[t] * E[(o_d[t] - L(t))^2]`` with
      ``E[(o - L)^2] = (o - mu_post[t])^2 + var_post[t]`` (the residual expectation
      includes the state posterior variance).

    Args mirror :func:`window_smoother`, with ``obs_weight`` (``(W, M)`` known
    weights) replacing ``obs_var``, plus the Gamma prior ``(a0, b0)`` and the
    iteration count. Returns ``(mu_post, var_post, a_post, b_post)``; the learned
    noise scale is ``E[tau] = a_post / b_post`` and ``E[sigma_obs^2] = b_post /
    (a_post - 1)`` (for ``a_post > 1``).
    """
    M = prior_mean.shape[0]
    w = np.maximum(np.asarray(obs_weight, dtype=float), 1e-12)   # (W, M)
    n_active = float(obs_mask.sum()) * M                          # scalar obs count
    a_post = a0 + 0.5 * n_active

    e_tau = a0 / b0
    mu_post = prior_mean
    var_post = np.zeros(M)
    b_post = b0
    for _ in range(max(1, int(n_iters))):
        obs_var = 1.0 / (e_tau * w)
        mu_post, var_post = window_smoother(prior_mean, q, sigma0, obs_traj,
                                            obs_var, obs_mask)
        # Weighted residual sum of squares, with the state posterior variance.
        resid2 = (obs_traj - mu_post[None, :]) ** 2 + var_post[None, :]   # (W, M)
        ss = float((obs_mask[:, None] * w * resid2).sum())
        b_post = b0 + 0.5 * ss
        e_tau = a_post / b_post
    return mu_post, var_post, a_post, b_post


def dense_reference(
    prior_mean: np.ndarray,
    q: float,
    sigma0: float,
    obs_traj: np.ndarray,
    obs_var: np.ndarray,
    obs_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense recomputation of :func:`window_smoother` (for tests only)."""
    M = prior_mean.shape[0]
    diag, off = rw_prior_precision(M, q, sigma0)
    Lam = np.diag(diag).astype(float)
    for i in range(M - 1):
        Lam[i, i + 1] = Lam[i + 1, i] = off[i]
    b = Lam @ prior_mean
    prec_data, info_data = _accumulate_data(obs_traj, obs_var, obs_mask)
    Lam = Lam + np.diag(prec_data)
    b = b + info_data
    cov = np.linalg.inv(Lam)
    mu_post = cov @ b
    return mu_post, np.diag(cov).copy()


def cross_covariance(
    q: float, sigma0: float, obs_var: np.ndarray, obs_mask: np.ndarray,
) -> np.ndarray:
    """Full posterior covariance matrix ``Lambda_post^{-1}`` (dense, ``M x M``).

    Provided for diagnostics/tests that inspect the temporal correlation
    structure; not used on the hot path (the hot path needs only the marginal
    variances from :func:`window_smoother`).
    """
    M = obs_var.shape[1]
    diag, off = rw_prior_precision(M, q, sigma0)
    Lam = np.diag(diag).astype(float)
    for i in range(M - 1):
        Lam[i, i + 1] = Lam[i + 1, i] = off[i]
    prec_data, _ = _accumulate_data(obs_traj=np.zeros_like(obs_var), obs_var=obs_var,
                                    obs_mask=obs_mask)
    Lam = Lam + np.diag(prec_data)
    return np.linalg.inv(Lam)


def expand_to_minutes(values: np.ndarray, control_interval: int, K: int) -> np.ndarray:
    """Zero-order-hold expand an epoch-resolution array (length ``M``) to the
    per-minute grid (length ``K``): minute ``k`` takes epoch ``k // control_interval``
    (the split is held constant over a control interval, matching the within-day
    physics). For minute-resolution arrays (``len==K``) this is the identity."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] == K:
        return values
    idx = np.minimum(np.arange(K) // max(1, control_interval), values.shape[0] - 1)
    return values[idx]
