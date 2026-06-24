"""The controller's rolling-window trajectory smoother (the big AIF agent).

The signal controller is one large AIF agent whose latent is the within-day
queue trajectory of each signalised movement, estimated from the per-interval
observations by a windowed Gaussian smoother (the macro analogue of the
travellers' window smoother). These tests pin:

  1. the banded O(M) solver equals a dense reference (mean and variance);
  2. the posterior covariance is FULL (non-diagonal) with temporal correlation
     that decays with lag -- the correlations the design requires;
  3. the posterior variance shrinks as more days are observed / the window grows;
  4. identity observations with tiny noise recover the observed trajectory;
  5. the controller's broadcast forecast IS the smoother posterior (not a
     prior-predictive rollout), and falls back to a flat prior before any day.

Run the narrated ones with ``-s``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aif_traffic.control import controller_smoother as cs
from aif_traffic.control.aif_controller import AIFController
from aif_traffic.parameters import AIFControllerSpec, SignalParams, NetworkParams


def _narrate(title, lines):
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)
    for line in lines:
        print(line)
    print("=" * 72)


def _setup(M=40, W=6, seed=0):
    rng = np.random.default_rng(seed)
    true = np.clip(np.linspace(0, 100, M) + 12 * np.sin(np.linspace(0, 6, M)), 0, None)
    obs_traj = true[None, :] + rng.normal(0, 5, size=(W, M))
    obs_var = np.full((W, M), 25.0)
    obs_mask = np.ones(W)
    prior_mean = np.zeros(M)
    return prior_mean, obs_traj, obs_var, obs_mask, true


def test_vb_recovers_injected_observation_noise():
    """The variational smoother learns the observation-noise scale: with data
    generated at a known SD, the Gamma posterior mean ``E[sigma_obs]`` recovers
    it (and tightens as the window grows). Tested across a range and under a
    split-dependent observation weight (the known structure VB factors out)."""
    M = 60
    t = np.arange(M)
    true_L = 40.0 * np.exp(-((t - 30) ** 2) / (2 * 12.0 ** 2))
    a0, sigma_guess = 1.0, 5.0           # weakly-informative prior, centred at 5
    b0 = a0 * sigma_guess ** 2
    lines, ok = [], True
    for W, true_sigma in [(30, 2.0), (30, 8.0), (30, 20.0), (60, 8.0)]:
        rng = np.random.default_rng(int(true_sigma) * 100 + W)
        w = np.ones((W, M))
        obs = true_L[None, :] + rng.normal(0, true_sigma, size=(W, M))
        _, _, a_p, b_p = cs.window_smoother_vb(
            np.zeros(M), 4.0, 50.0, obs, w, np.ones(W), a0=a0, b0=b0, n_iters=10)
        learned = float(np.sqrt(b_p / (a_p - 1.0)))
        rel = abs(learned - true_sigma) / true_sigma
        ok = ok and rel < 0.12
        lines.append(f"W={W:3d} true sigma={true_sigma:5.1f} -> learned {learned:6.2f}"
                     f"  (rel err {rel:.1%})")

    # Split-dependent weight: per-node SD = scale / sqrt(w); recover the scale.
    W = 40
    rng = np.random.default_rng(7)
    w = np.tile(np.linspace(0.4, 1.0, M), (W, 1))
    obs = true_L[None, :] + rng.normal(0, 1, size=(W, M)) * (6.0 / np.sqrt(w))
    _, _, a_p, b_p = cs.window_smoother_vb(
        np.zeros(M), 4.0, 50.0, obs, w, np.ones(W), a0=a0, b0=b0, n_iters=10)
    learned_scale = float(np.sqrt(b_p / (a_p - 1.0)))
    ok = ok and abs(learned_scale - 6.0) / 6.0 < 0.12
    lines.append(f"split-weighted: true scale 6.0 -> learned {learned_scale:.2f}")
    _narrate("Controller VB recovers the injected observation-noise scale", lines
             + [f"Verdict: {'PASS' if ok else 'FAIL'}."])
    assert ok


def test_vb_reduces_to_fixed_smoother_when_prior_is_dominant():
    """With a very strong prior (huge shape) pinning the precision, the VB
    posterior mean equals the fixed-noise smoother at that sigma_obs."""
    prior_mean, obs_traj, _, obs_mask, _ = _setup(M=40, W=6)
    q, sigma0, sigma_fixed = 4.0, 5.0, 5.0
    W, M = obs_traj.shape
    w = np.ones((W, M))                              # unit weight
    obs_var = np.full((W, M), sigma_fixed ** 2)      # matching fixed variance
    mu_fix, _ = cs.window_smoother(prior_mean, q, sigma0, obs_traj, obs_var, obs_mask)
    a0 = 1e8                                          # prior pins E[tau]=1/25
    b0 = a0 * sigma_fixed ** 2
    mu_vb, _, _, _ = cs.window_smoother_vb(
        prior_mean, q, sigma0, obs_traj, w, obs_mask, a0=a0, b0=b0, n_iters=5)
    assert np.allclose(mu_fix, mu_vb, atol=1e-6), float(np.max(np.abs(mu_fix - mu_vb)))


def test_banded_solver_matches_dense_reference():
    """The O(M) banded LDL solve + inverse-diagonal recursion must reproduce the
    dense linear-algebra posterior exactly."""
    prior_mean, obs_traj, obs_var, obs_mask, _ = _setup()
    q, sigma0 = 4.0, 5.0
    mu_b, var_b = cs.window_smoother(prior_mean, q, sigma0, obs_traj, obs_var, obs_mask)
    mu_d, var_d = cs.dense_reference(prior_mean, q, sigma0, obs_traj, obs_var, obs_mask)
    assert np.allclose(mu_b, mu_d, atol=1e-8), float(np.max(np.abs(mu_b - mu_d)))
    assert np.allclose(var_b, var_d, atol=1e-8), float(np.max(np.abs(var_b - var_d)))


def test_posterior_covariance_is_full_with_decaying_correlation():
    """The posterior covariance is NOT diagonal: adjacent intervals are
    correlated, and the correlation decays with the lag |t-s|."""
    prior_mean, obs_traj, obs_var, obs_mask, _ = _setup()
    q, sigma0 = 4.0, 5.0
    Cov = cs.cross_covariance(q, sigma0, obs_var, obs_mask)
    sd = np.sqrt(np.diag(Cov))
    corr = Cov / np.outer(sd, sd)
    i = Cov.shape[0] // 2
    c1, c5, c15 = float(corr[i, i + 1]), float(corr[i, i + 5]), float(corr[i, i + 15])
    _narrate(
        "Controller posterior covariance is full, correlation decays with lag",
        [
            f"corr(L(t), L(t+1))  = {c1:.3f}",
            f"corr(L(t), L(t+5))  = {c5:.3f}",
            f"corr(L(t), L(t+15)) = {c15:.3f}",
            "Expectation: clearly non-zero at lag 1, decaying to ~0 with lag.",
            f"Verdict: {'PASS' if c1 > 0.1 and c1 > c5 > c15 >= 0 else 'FAIL'}.",
        ],
    )
    assert c1 > 0.1            # genuinely correlated neighbours (not diagonal)
    assert c1 > c5 > c15 - 1e-9  # decays with lag


def test_posterior_variance_shrinks_with_more_days():
    """More observed days in the window => lower posterior marginal variance."""
    prior_mean, obs_traj, obs_var, obs_mask, _ = _setup(W=8)
    q, sigma0 = 4.0, 5.0
    _, var1 = cs.window_smoother(
        prior_mean, q, sigma0, obs_traj[:1], obs_var[:1], np.array([1.0]))
    _, var8 = cs.window_smoother(prior_mean, q, sigma0, obs_traj, obs_var, obs_mask)
    _narrate(
        "Controller belief sharpens as the window fills",
        [
            f"mean marginal variance: W=1 -> {var1.mean():.3f}   W=8 -> {var8.mean():.3f}",
            "Expectation: more days => lower variance.",
            f"Verdict: {'PASS' if var8.mean() < var1.mean() else 'FAIL'}.",
        ],
    )
    assert var8.mean() < var1.mean()


def test_exact_observation_recovers_trajectory():
    """With negligible observation noise and a weak (large-q) prior, the
    posterior mean equals the observed trajectory."""
    M = 30
    true = np.clip(np.linspace(0, 90, M), 0, None)
    prior_mean = np.zeros(M)
    mu, _ = cs.window_smoother(
        prior_mean, q=1e6, sigma0=5.0,
        obs_traj=true[None, :], obs_var=np.full((1, M), 1e-6), obs_mask=np.array([1.0]),
    )
    assert np.allclose(mu, true, atol=1e-6), float(np.max(np.abs(mu - true)))


def test_no_observations_returns_prior():
    """With every window slot masked off, the posterior mean equals the prior
    mean (here zero)."""
    M, W = 20, 4
    prior_mean = np.zeros(M)
    mu, var = cs.window_smoother(
        prior_mean, q=4.0, sigma0=5.0,
        obs_traj=np.zeros((W, M)), obs_var=np.ones((W, M)), obs_mask=np.zeros(W),
    )
    assert np.allclose(mu, prior_mean, atol=1e-9)
    assert np.all(var > 0)


# --- integration with the controller ---------------------------------------
def _controller():
    spec = AIFControllerSpec(controller_window_size=5)
    net = NetworkParams()
    return AIFController(spec, SignalParams(), net.signalised_links), net


def _fake_day(K, level):
    """A fake realised day: ramped queues + a constant balanced split."""
    L2 = np.clip(np.linspace(0, level, K), 0, None)
    L6 = np.clip(np.linspace(0, level * 0.6, K), 0, None)
    return {
        "queues": {2: L2, 6: L6},
        "phi2": np.full(K, 0.45), "phi6": np.full(K, 0.45),
        "day": 0, "tt_route": {}, "SC": 0.0,
    }


def test_forecast_is_the_smoother_posterior_not_a_rollout():
    """After observing days, the broadcast forecast equals the smoother
    posterior mean (expanded to minutes), NOT an empty-start rollout."""
    from aif_traffic.parameters import Params
    ctrl, net = _controller()
    p = Params.default()
    K = p.sim.K
    ctx = {"net": net, "sim": p.sim, "signal": p.signal,
           "inflow_by_route": {"alpha": np.zeros(K), "beta": np.zeros(K),
                                "gamma": np.zeros(K)}}
    ctrl.prepare_day(ctx)
    for _ in range(3):
        ctrl.observe(_fake_day(K, level=80.0))
    fc = ctrl.forecast(ctx)
    # The forecast mu_L equals the (minute-expanded) posterior mean of L_2.
    expected = cs.expand_to_minutes(ctrl._post_mu[0], ctrl._ci, K)
    assert np.allclose(fc.mu_L, expected, atol=1e-9)
    # It is not the flat empty-queue start (the posterior has learned a ramp).
    assert fc.mu_L.max() > 10.0
    # The broadcast variance is the posterior marginal variance (finite, > 0).
    assert np.all(fc.var_L > 0) and np.all(np.isfinite(fc.var_L))


def test_forecast_cold_start_is_flat_prior():
    """Before any day is observed, the forecast falls back to a flat empty-queue
    prior (mean 0, variance sigma0^2)."""
    from aif_traffic.parameters import Params
    ctrl, net = _controller()
    p = Params.default()
    K = p.sim.K
    ctx = {"net": net, "sim": p.sim, "signal": p.signal,
           "inflow_by_route": {"alpha": np.zeros(K), "beta": np.zeros(K),
                                "gamma": np.zeros(K)}}
    ctrl.prepare_day(ctx)
    fc = ctrl.forecast(ctx)
    assert np.allclose(fc.mu_L, 0.0)
    assert np.allclose(fc.var_L, ctrl.spec.sigma0 ** 2)


def test_controller_belief_variance_shrinks_over_observed_days():
    """Observing more days lowers the controller's mean posterior variance."""
    from aif_traffic.parameters import Params
    ctrl, net = _controller()
    p = Params.default()
    K = p.sim.K
    ctx = {"net": net, "sim": p.sim, "signal": p.signal,
           "inflow_by_route": {"alpha": np.zeros(K), "beta": np.zeros(K),
                                "gamma": np.zeros(K)}}
    ctrl.prepare_day(ctx)
    ctrl.observe(_fake_day(K, 80.0))
    v1 = float(np.mean(ctrl._post_var))
    for _ in range(4):
        ctrl.observe(_fake_day(K, 80.0))
    v5 = float(np.mean(ctrl._post_var))
    _narrate(
        "Controller belief variance shrinks as it observes more days",
        [f"mean posterior variance: 1 day -> {v1:.3f}   5 days -> {v5:.3f}",
         f"Verdict: {'PASS' if v5 < v1 else 'FAIL'}."],
    )
    assert v5 < v1
