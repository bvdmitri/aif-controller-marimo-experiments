"""Closed-form Expected Free Energy from the smoother posterior.

Reused from the IWAI route-choice model. Given the per-(agent, route) joint
MVN posterior over ``z = (F, C, L)``, the predictive TT moments are computed
by a first-order Taylor expansion of ``h(F,C,L) = F + 60 L/C`` around the
posterior mean, and the EFE components are closed-form:

* ``risk(a)      = KL[N(mu_y, sigma_y^2 + sigma_obs^2) || N(mu_F_r, sigma_pref^2)]``
* ``info_gain(a) = 1/2 log(1 + sigma_y^2 / sigma_obs^2)``

with ``G(a) = w_R*risk - w_I*info_gain`` and route probability a softmax over
``-gamma*G``.

**Macro coupling (this repo).** An optional ``cost_offset`` of shape
``(N, n_routes)`` shifts the predicted perceived cost used in the risk term:
``mu_y -> mu_y + cost_offset``. This is how the controller broadcast enters
route choice -- the offset is ``theta * E_r`` per agent (zero for
non-compliant agents or no broadcast). ``cost_offset=None`` reproduces the
IWAI behaviour exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .filter import C_IDX, F_IDX, L_IDX, VariationalState, predicted_tt


def _predictive_moments(
    state: VariationalState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Closed-form tomorrow's TT predictive moments per (agent, route).

    Returns ``(mu_y, var_y)`` each shape ``(N, n_routes)``. ``var_y`` is the
    posterior-uncertainty variance only; the caller adds ``sigma_obs^2``.
    """
    Sigma = state.scale_tril @ jnp.swapaxes(state.scale_tril, -1, -2)

    mu_F = state.mu[..., F_IDX]
    mu_C = state.mu[..., C_IDX]
    mu_L = state.mu[..., L_IDX]

    mu_y = predicted_tt(mu_F, mu_C, mu_L)

    C_safe = jnp.maximum(mu_C, 100.0)
    L_safe = jnp.maximum(mu_L, 0.0)
    grad = jnp.stack(
        [
            jnp.ones_like(mu_F),
            -60.0 * L_safe / (C_safe ** 2),
            60.0 / C_safe,
        ],
        axis=-1,
    )

    var_y = jnp.einsum("...i,...ij,...j->...", grad, Sigma, grad)
    return mu_y, var_y


def _gaussian_kl(
    mu_q: jnp.ndarray, var_q: jnp.ndarray,
    mu_p: jnp.ndarray, var_p: jnp.ndarray,
) -> jnp.ndarray:
    """KL[N(mu_q, var_q) || N(mu_p, var_p)] elementwise."""
    return 0.5 * (
        jnp.log(var_p / var_q)
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )


def efe_route_probabilities(
    state: VariationalState,
    sigma_obs: jnp.ndarray,
    sigma_pref: jnp.ndarray,
    gamma: jnp.ndarray,
    risk_weight: float,
    info_gain_weight: float,
    cost_offset: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return ``p(a | i)`` of shape ``(N, n_routes)`` for each agent.

    ``cost_offset`` (``(N, n_routes)`` or ``None``) shifts the predicted
    perceived cost in the risk term, implementing the perceived route cost
    ``zeta_r = TT_r + theta * E_r`` when the controller broadcast is active.
    """
    mu_y, var_y = _predictive_moments(state)
    sigma_obs_sq = sigma_obs ** 2
    var_pred = var_y + sigma_obs_sq[:, None]

    if cost_offset is not None:
        mu_y = mu_y + cost_offset

    preferred_mean = state.mu[..., F_IDX]

    risk = _gaussian_kl(
        mu_q=mu_y, var_q=var_pred,
        mu_p=preferred_mean, var_p=(sigma_pref[:, None]) ** 2,
    )
    info_gain = 0.5 * jnp.log1p(var_y / sigma_obs_sq[:, None])

    G = risk_weight * risk - info_gain_weight * info_gain
    return jax.nn.softmax(-gamma[:, None] * G, axis=-1)
