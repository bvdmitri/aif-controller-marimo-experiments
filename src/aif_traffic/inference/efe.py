"""Closed-form Expected Free Energy from the smoother posterior.

Reused from the IWAI route-choice model. Given the per-(agent, route) joint
MVN posterior over ``z = (F, C, L)``, the predictive travel-time moments are
computed by a first-order Taylor expansion of ``h(F,C,L) = F + 60 L/C`` around
the posterior mean.

**The agent's outcome is the perceived generalized cost.** In active-inference
terms the outcome the traveller has preferences about is
``zeta_r = TT_r + theta * E_r`` -- its predicted travel time plus the share
``theta`` of the congestion externality ``E_r`` it internalises. This is the
textbook-clean way to encode a *social* preference: ``theta`` scales a goal /
preference term, it is **not** a distortion of the agent's belief about its own
private trip. Because ``E_r`` is relayed as a *known* offset
``cost_offset = theta * E_r`` (per agent; zero for non-compliant agents or with
no broadcast), it shifts the predictive *mean* of the outcome
``mu_zeta = mu_TT + cost_offset`` but carries no belief uncertainty, so it does
**not** touch the predictive variance or the epistemic term. The preference is a
prior over this outcome, ``N(mu_F_r, sigma_pref^2)`` ("prefer a generalized cost
near the free-flow ideal ``F``"). The closed-form EFE components are then:

* ``risk(a)      = KL[N(mu_zeta, sigma_y^2 + sigma_obs^2) || N(mu_F_r, sigma_pref^2)]``
* ``info_gain(a) = 1/2 log(1 + sigma_y^2 / sigma_obs^2)``   (unaffected by theta)

with ``G(a) = w_R*risk - w_I*info_gain`` and route probability a softmax over
``-gamma*G``. ``cost_offset=None`` gives ``zeta_r = TT_r`` -- the purely selfish
(user-equilibrium) IWAI behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .filter import (
    C_IDX,
    F_IDX,
    L_IDX,
    PHI_HI_DEFAULT,
    PHI_IDX,
    PHI_LO_DEFAULT,
    VariationalState,
    predicted_tt,
)


def _predictive_moments(
    state: VariationalState,
    signalised: jnp.ndarray,
    phi_lo: float = PHI_LO_DEFAULT,
    phi_hi: float = PHI_HI_DEFAULT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Closed-form tomorrow's TT predictive moments per (agent, route).

    Returns ``(mu_y, var_y)`` each shape ``(N, n_routes)``. ``var_y`` is the
    posterior-uncertainty variance only; the caller adds ``sigma_obs^2``.
    On signalised routes the forward map is ``F + 60 L/(phi*C)`` so an uncertain
    green split ``phi`` inflates ``var_y`` (and hence the epistemic info gain).
    """
    Sigma = state.scale_tril @ jnp.swapaxes(state.scale_tril, -1, -2)

    mu_F = state.mu[..., F_IDX]
    mu_C = state.mu[..., C_IDX]
    mu_L = state.mu[..., L_IDX]
    mu_phi = state.mu[..., PHI_IDX]

    mu_y = predicted_tt(mu_F, mu_C, mu_L, mu_phi, signalised, phi_lo, phi_hi)

    C_safe = jnp.maximum(mu_C, 100.0)
    L_safe = jnp.maximum(mu_L, 0.0)
    phi_clip = jnp.clip(mu_phi, phi_lo, phi_hi)
    sig = signalised[None, :]                            # (1, R) broadcast
    green = jnp.where(sig > 0.5, phi_clip, 1.0)
    grad = jnp.stack(
        [
            jnp.ones_like(mu_F),                              # ∂/∂F
            -60.0 * L_safe / (C_safe ** 2 * green),           # ∂/∂C
            60.0 / (C_safe * green),                          # ∂/∂L
            jnp.where(                                        # ∂/∂phi
                sig > 0.5, -60.0 * L_safe / (C_safe * phi_clip ** 2), 0.0,
            ),
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
    signalised: jnp.ndarray,
    cost_offset: jnp.ndarray | None = None,
    phi_lo: float = PHI_LO_DEFAULT,
    phi_hi: float = PHI_HI_DEFAULT,
) -> jnp.ndarray:
    """Return ``p(a | i)`` of shape ``(N, n_routes)`` for each agent.

    The outcome the agent has preferences about is the **perceived generalized
    cost** ``zeta_r = TT_r + theta * E_r``. ``cost_offset`` (``(N, n_routes)`` or
    ``None``) is the externality term ``theta * E_r`` folded into the *mean* of
    that outcome (a known offset, so it does not change the predictive variance
    or the epistemic term). With ``cost_offset=None`` the outcome is just the
    private travel time ``TT_r``. ``signalised`` (``(n_routes,)`` 0/1) marks
    routes where the green-split latent couples into the predicted travel time.
    """
    mu_tt, var_y = _predictive_moments(state, signalised, phi_lo, phi_hi)
    sigma_obs_sq = sigma_obs ** 2
    var_pred = var_y + sigma_obs_sq[:, None]

    # Predictive mean of the outcome = the perceived generalized cost zeta_r:
    # the private travel time plus the internalised externality (a known offset).
    mu_zeta = mu_tt if cost_offset is None else mu_tt + cost_offset

    # Preference: a prior over the perceived generalized cost, centred on the
    # free-flow ideal F (empty road, no externality).
    preferred_mean = state.mu[..., F_IDX]

    # Pragmatic value (risk): divergence of the predicted generalized cost from
    # the preferred one. Epistemic value (info_gain): expected reduction in
    # state uncertainty -- independent of the preference, hence of theta.
    risk = _gaussian_kl(
        mu_q=mu_zeta, var_q=var_pred,
        mu_p=preferred_mean, var_p=(sigma_pref[:, None]) ** 2,
    )
    info_gain = 0.5 * jnp.log1p(var_y / sigma_obs_sq[:, None])

    G = risk_weight * risk - info_gain_weight * info_gain
    return jax.nn.softmax(-gamma[:, None] * G, axis=-1)


# JIT-compiled per-day choice/predictive steps. Same eager math, compiled once
# and reused (constant shapes within a run). ``phi_lo``/``phi_hi``/``risk_weight``/
# ``info_gain_weight`` stay traced (no coercion to Python scalars inside), so
# their values may change without recompiling. ``cost_offset`` is consistently
# ``None`` or an array within a run, so its pytree structure is stable.
_predictive_moments_jit = jax.jit(_predictive_moments)
efe_route_probabilities_jit = jax.jit(efe_route_probabilities)
