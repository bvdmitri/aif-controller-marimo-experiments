"""Traveller population driven by the closed-form Gaussian smoother.

Reused from the IWAI route-choice model, with the two routes relabelled to the
intersection route ``alpha`` (index 0) and the bypass route ``beta`` (index 1),
and macro-coupling additions across **two distinct controller->traveller
channels**:

* a per-agent **compliance** mask (drawn once from ``CohortSpec.compliance_fraction``):
  compliant agents ingest the controller broadcast, the rest ignore it;
* **Cost-offset channel** (Experiment 1, ``theta``): ``begin_day(..., broadcast=...)``
  turns the cost-offset advisory into the EFE
  ``cost_offset = theta * compliance * E_r`` per (agent, route). This affects
  *action selection only* -- the smoother (``filter.py``) is untouched by it.
* **Belief-informing channel** (Experiment 3, CG/SN):
  ``update_beliefs(..., belief_broadcast=...)`` folds the broadcast route queue
  (CG) and/or green split (SN) into the smoother as observations of routes the
  compliant agent did *not* take. This DOES enter the belief update -- a
  deliberate, documented departure from the IWAI "belief update sees only
  first-hand, chosen-route observations" property. The gate
  ``complies * (last_choice != route)`` keeps the chosen route's first-hand
  observation authoritative (no double counting).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..demand import DemandProfile
from ..parameters import CohortSpec, EFEParams, PopulationParams, SignalParams, SimParams
from ..utils import smooth_profile
from .efe import _predictive_moments, efe_route_probabilities
from .filter import (
    C_IDX,
    CohortPriors,
    F_IDX,
    L_IDX,
    PHI_IDX,
    VariationalState,
    init_variational_state,
    window_step,
)


def _cohort_array(cohorts: tuple[CohortSpec, ...], attr: str, n_total: int) -> np.ndarray:
    out = np.empty(n_total, dtype=float)
    idx = 0
    for c in cohorts:
        out[idx : idx + c.n_agents] = float(getattr(c, attr))
        idx += c.n_agents
    return out


class Population:
    """AIF traveller population using a closed-form rolling smoother.

    Route index 0 = ``alpha`` (intersection), 1 = ``beta`` (bypass).
    """

    def __init__(
        self,
        cohorts: tuple[CohortSpec, ...],
        sim: SimParams,
        demand: DemandProfile,
        rng: np.random.Generator,
        route_names: tuple[str, str] = ("alpha", "beta"),
        signal: SignalParams | None = None,
    ):
        self.sim = sim
        self.cohorts = tuple(cohorts)
        self.route_names = route_names
        self.N = sum(c.n_agents for c in cohorts)

        # Route 0 (alpha) is the signalised intersection route; route 1 (beta)
        # is the bypass with no signal. The green-split latent phi couples into
        # travel time only on signalised routes.
        signal = signal if signal is not None else SignalParams()
        self.signalised_route = jnp.asarray([1.0, 0.0])
        self.phi_lo = float(signal.phi_min)
        self.phi_hi = float(signal.phi_sat)

        window_sizes = {int(c.window_size) for c in cohorts}
        if len(window_sizes) != 1:
            raise ValueError(
                "All cohorts must share the same CohortSpec.window_size; "
                f"got {sorted(window_sizes)}."
            )
        self.W: int = next(iter(window_sizes))

        self.cohort_id = np.empty(self.N, dtype=int)
        self.cohort_label = np.empty(self.N, dtype=object)
        self.sigma_pref = _cohort_array(cohorts, "sigma_pref", self.N)
        self.sigma_obs = _cohort_array(cohorts, "sigma_obs", self.N)
        self.sigma_L_obs = _cohort_array(cohorts, "sigma_L_obs", self.N)
        self.gamma = _cohort_array(cohorts, "gamma", self.N)
        self.theta = _cohort_array(cohorts, "theta", self.N)

        F_prior_mu = np.empty((self.N, 2), dtype=float)
        F_prior_sigma = np.empty((self.N, 2), dtype=float)
        C_prior_mu = np.empty((self.N, 2), dtype=float)
        C_prior_sigma = np.empty((self.N, 2), dtype=float)
        L_prior_mu = np.empty((self.N, 2), dtype=float)
        L_prior_sigma = np.empty((self.N, 2), dtype=float)
        phi_prior_mu = np.empty((self.N, 2), dtype=float)
        phi_prior_sigma = np.empty((self.N, 2), dtype=float)
        self.sigma_phi_obs = _cohort_array(cohorts, "sigma_phi_obs", self.N)
        self._sigma_F_drift = np.empty(self.N, dtype=float)
        self._sigma_C_drift = np.empty(self.N, dtype=float)
        self._sigma_L_drift = np.empty(self.N, dtype=float)
        self._sigma_phi_drift = np.empty(self.N, dtype=float)
        self._mean_revert_days = np.empty(self.N, dtype=float)
        self._n_laplace_iters = max(int(c.n_laplace_iters) for c in cohorts)

        # Per-agent compliance: does this agent read the controller broadcast?
        self.complies = np.zeros(self.N, dtype=bool)

        starts = np.cumsum([0] + [c.n_agents for c in cohorts])
        for i, c in enumerate(cohorts):
            s, e = starts[i], starts[i + 1]
            self.cohort_id[s:e] = i
            self.cohort_label[s:e] = c.label

            F_prior_mu[s:e, 0] = c.F_prior_mu_alpha
            F_prior_mu[s:e, 1] = c.F_prior_mu_beta
            F_prior_sigma[s:e, :] = c.F_prior_sigma
            C_prior_mu[s:e, 0] = c.C_prior_mu_alpha
            C_prior_mu[s:e, 1] = c.C_prior_mu_beta
            C_prior_sigma[s:e, 0] = c.C_prior_sigma_alpha
            C_prior_sigma[s:e, 1] = c.C_prior_sigma_beta
            L_prior_mu[s:e, 0] = c.L_prior_mu_alpha
            L_prior_mu[s:e, 1] = c.L_prior_mu_beta
            L_prior_sigma[s:e, :] = c.L_prior_sigma
            phi_prior_mu[s:e, 0] = c.phi_prior_mu_alpha
            phi_prior_mu[s:e, 1] = c.phi_prior_mu_beta
            phi_prior_sigma[s:e, :] = c.phi_prior_sigma

            self._sigma_F_drift[s:e] = float(c.sigma_F_drift)
            self._sigma_C_drift[s:e] = float(c.sigma_C_drift)
            self._sigma_L_drift[s:e] = float(c.sigma_L_drift)
            self._sigma_phi_drift[s:e] = float(c.sigma_phi_drift)
            self._mean_revert_days[s:e] = float(c.mean_revert_days)

            n_comply = int(round(c.compliance_fraction * c.n_agents))
            mask = np.zeros(c.n_agents, dtype=bool)
            mask[:n_comply] = True
            rng.shuffle(mask)
            self.complies[s:e] = mask

        self.cohort_priors = CohortPriors(
            F_mu=jnp.asarray(F_prior_mu),
            F_sigma=jnp.asarray(F_prior_sigma),
            C_mu=jnp.asarray(C_prior_mu),
            C_sigma=jnp.asarray(C_prior_sigma),
            L_mu=jnp.asarray(L_prior_mu),
            L_sigma=jnp.asarray(L_prior_sigma),
            phi_mu=jnp.asarray(phi_prior_mu),
            phi_sigma=jnp.asarray(phi_prior_sigma),
        )

        self.state: VariationalState = init_variational_state(
            cohort_priors=self.cohort_priors,
        )

        self._obs_buffer_route = np.zeros((self.N, self.W), dtype=np.int32)
        self._obs_buffer_tt = np.zeros((self.N, self.W), dtype=float)
        self._obs_buffer_sigma_tt = np.ones((self.N, self.W), dtype=float)
        self._obs_buffer_L = np.zeros((self.N, self.W), dtype=float)
        self._obs_buffer_sigma_L = np.ones((self.N, self.W), dtype=float)
        self._obs_buffer_phi = np.zeros((self.N, self.W), dtype=float)
        self._obs_buffer_sigma_phi = np.ones((self.N, self.W), dtype=float)
        self._obs_mask = np.zeros((self.N, self.W), dtype=float)

        # Belief-informing broadcast buffers (paper Exp 3: CG/SN). Unlike the
        # chosen-route buffers above (one value per agent-day), these carry one
        # value per (agent, route, day): the broadcast informs routes the agent
        # did NOT take. The masks are choice-independent (set in update_beliefs).
        self._belief_L = np.zeros((self.N, 2, self.W), dtype=float)
        self._belief_sigma_L = np.ones((self.N, 2, self.W), dtype=float)
        self._belief_phi = np.zeros((self.N, 2, self.W), dtype=float)
        self._belief_sigma_phi = np.ones((self.N, 2, self.W), dtype=float)
        self._belief_mask_L = np.zeros((self.N, 2, self.W), dtype=float)
        self._belief_mask_phi = np.zeros((self.N, 2, self.W), dtype=float)

        self.day_count: int = 0

        self._last_observed_day = np.full((self.N, 2), -1, dtype=int)

        K = sim.K
        d_ab = np.asarray(demand.d_AB, dtype=float)
        if d_ab.sum() <= 0:
            p = np.full(K, 1.0 / K)
        else:
            p = d_ab / d_ab.sum()
        self.departure_time = rng.choice(K, size=self.N, p=p).astype(int)

        self.last_choice = np.full(self.N, -1, dtype=int)  # 0 = alpha, 1 = beta
        self.last_P_alpha = np.full(self.N, 0.5, dtype=float)

    # ----------------------------------------------------- helper accessors
    @property
    def predictive_moments(self) -> tuple[np.ndarray, np.ndarray]:
        mu_y, var_y = _predictive_moments(
            self.state, self.signalised_route, self.phi_lo, self.phi_hi,
        )
        return np.asarray(mu_y), np.asarray(var_y)

    def latent_summary(self) -> dict:
        mu = self.state.mu
        scale_tril = self.state.scale_tril
        marginal_sd = jnp.sqrt(jnp.sum(scale_tril ** 2, axis=-1))
        return {
            "F_mean": np.asarray(mu[..., F_IDX]),
            "F_sd": np.asarray(marginal_sd[..., F_IDX]),
            "C_mean": np.asarray(mu[..., C_IDX]),
            "C_sd": np.asarray(marginal_sd[..., C_IDX]),
            "L_mean": np.asarray(mu[..., L_IDX]),
            "L_sd": np.asarray(marginal_sd[..., L_IDX]),
            "phi_mean": np.asarray(mu[..., PHI_IDX]),
            "phi_sd": np.asarray(marginal_sd[..., PHI_IDX]),
        }

    def _broadcast_cost_offset(self, broadcast) -> np.ndarray | None:
        """Per-(agent, route) EFE offset ``theta * compliance * E_r`` from a
        broadcast, sampled at each agent's departure minute. ``None`` when
        there is no broadcast (recovers the no-information case)."""
        if broadcast is None:
            return None
        t_i = self.departure_time
        scale = self.theta * self.complies.astype(float)  # (N,)
        offset = np.zeros((self.N, 2), dtype=float)
        for j, route in enumerate(self.route_names):
            vals = np.asarray(broadcast.value[route], dtype=float)
            offset[:, j] = scale * vals[t_i]
        return offset

    def _append_belief_broadcast(self, belief_broadcast, t_i, rng=None) -> None:
        """Shift the belief-broadcast buffers and write today's slot.

        The new slot carries, per (agent, route), the broadcast queue (CG) and
        green split (SN) sampled at the agent's departure minute. The gate
        ``complies * (last_choice != route)`` is the key correctness invariant:
        only compliant agents ingest the broadcast, and never for the route
        they actually took (the first-hand observation wins -- no double
        counting). When no broadcast is shared the new slot's masks stay zero
        (a no-op in the smoother).

        The broadcast is treated as a *deterministic message*: the value is the
        controller's reading and the traveller's trust in it is encoded by the
        fold variance (``sigma_L_obs`` / ``sigma_phi_obs``), not by injecting
        random noise into the value. This keeps the pipeline draw-free here, so
        the baseline (no compliant ingester) is bit-identical to no broadcast.
        ``rng`` is accepted for signature symmetry but unused.
        """
        del rng  # broadcast is a deterministic message (trust encoded by sigma)
        # Shift every belief buffer/mask left by one day.
        for buf in (
            self._belief_L, self._belief_sigma_L,
            self._belief_phi, self._belief_sigma_phi,
            self._belief_mask_L, self._belief_mask_phi,
        ):
            buf[:, :, :-1] = buf[:, :, 1:]
        # Default new slot = no broadcast observation (mask 0 ⇒ no-op).
        self._belief_L[:, :, -1] = 0.0
        self._belief_sigma_L[:, :, -1] = 1.0
        self._belief_phi[:, :, -1] = 0.0
        self._belief_sigma_phi[:, :, -1] = 1.0
        self._belief_mask_L[:, :, -1] = 0.0
        self._belief_mask_phi[:, :, -1] = 0.0

        if belief_broadcast is None:
            return

        comply_f = self.complies.astype(float)  # (N,)

        # CG: route-queue advisory for routes the agent did not take.
        if getattr(belief_broadcast, "L", None) is not None:
            for j, route in enumerate(self.route_names):
                vals = np.asarray(belief_broadcast.L[route], dtype=float)[t_i]
                self._belief_L[:, j, -1] = vals
                self._belief_sigma_L[:, j, -1] = np.maximum(self.sigma_L_obs, 1e-3)
                self._belief_mask_L[:, j, -1] = comply_f * (self.last_choice != j)

        # SN: green-split advisory for the signalised route (index 0) only.
        if getattr(belief_broadcast, "phi", None) is not None:
            route0 = self.route_names[0]
            phi_vals = np.asarray(belief_broadcast.phi[route0], dtype=float)[t_i]
            self._belief_phi[:, 0, -1] = phi_vals
            self._belief_sigma_phi[:, 0, -1] = np.maximum(self.sigma_phi_obs, 1e-3)
            self._belief_mask_phi[:, 0, -1] = comply_f * (self.last_choice != 0)

    # ------------------------------------------------------------------ day
    def begin_day(
        self,
        efe: EFEParams,
        rng: np.random.Generator,
        broadcast=None,
    ) -> None:
        """Each agent samples a route from the closed-form EFE softmax.

        When ``broadcast`` is given, compliant agents fold the advisory into
        their perceived cost via the EFE ``cost_offset``.
        """
        cost_offset = self._broadcast_cost_offset(broadcast)
        P = efe_route_probabilities(
            state=self.state,
            sigma_obs=jnp.asarray(self.sigma_obs),
            sigma_pref=jnp.asarray(self.sigma_pref),
            gamma=jnp.asarray(self.gamma),
            risk_weight=efe.risk_weight,
            info_gain_weight=efe.info_gain_weight,
            signalised=self.signalised_route,
            cost_offset=None if cost_offset is None else jnp.asarray(cost_offset),
            phi_lo=self.phi_lo,
            phi_hi=self.phi_hi,
        )

        P = np.asarray(P)
        self.last_P = P
        self.last_P_alpha = P[:, 0]
        cdf = np.cumsum(P, axis=-1)
        u = rng.uniform(size=(self.N, 1))
        self.last_choice = (u >= cdf[..., :-1]).sum(axis=-1).astype(int)

    def aggregate_route_share(self, smooth_window: int = 13) -> np.ndarray:
        """Empirical P_alpha(t) over departure intervals, smoothed for low-N bins."""
        K = self.sim.K
        chose_alpha = (self.last_choice == 0).astype(float)
        n_total = np.bincount(self.departure_time, minlength=K).astype(float)
        n_alpha = np.bincount(self.departure_time, weights=chose_alpha, minlength=K)

        with np.errstate(divide="ignore", invalid="ignore"):
            P_alpha = np.where(n_total > 0, n_alpha / np.maximum(n_total, 1.0), 0.5)

        if smooth_window > 1:
            P_alpha = smooth_profile(P_alpha, smooth_window)

        return np.clip(P_alpha, 0.02, 0.98)

    def update_beliefs(
        self,
        TT_alpha: np.ndarray,
        TT_beta: np.ndarray,
        L_obs_alpha: np.ndarray,
        L_obs_beta: np.ndarray,
        green_obs_alpha: np.ndarray | None = None,
        belief_broadcast=None,
        rng: np.random.Generator | None = None,
        obs_noise_sd: float = 0.0,
    ) -> None:
        """Append today's TT + queue (+ green-split) observations and re-fit.

        ``green_obs_alpha`` is the realised intersection green split, aligned to
        the traveller's arrival, per departure minute (length ``K``). Only
        agents who chose the intersection observe it; the smoother gates it to
        the signalised route, so others' values are ignored.

        ``belief_broadcast`` is the controller's belief-informing broadcast
        (paper Exp 3: CG/SN; :class:`communication.BeliefBroadcast`). When
        present, **compliant** agents additionally fold the broadcast route
        queue (CG) and/or green split (SN) into their belief about routes they
        did *not* take that day. ``None`` (or ``BeliefBroadcast(None, None)``,
        the baseline BL case) draws no randomness and leaves the belief update
        bit-identical to the chosen-route-only smoother.
        """
        t_i = self.departure_time
        realised_tt_alpha = TT_alpha[t_i]
        realised_tt_beta = TT_beta[t_i]

        if rng is not None and obs_noise_sd > 0.0:
            realised_tt_alpha = realised_tt_alpha + rng.normal(0.0, obs_noise_sd, size=self.N)
            realised_tt_beta = realised_tt_beta + rng.normal(0.0, obs_noise_sd, size=self.N)

        y_tt_today = np.where(self.last_choice == 0, realised_tt_alpha, realised_tt_beta)
        y_L_today = np.where(self.last_choice == 0, L_obs_alpha[t_i], L_obs_beta[t_i])

        if rng is not None:
            y_L_today = y_L_today + rng.normal(0.0, self.sigma_L_obs, size=self.N)

        # Green-split observation (intersection route only; gated downstream).
        if green_obs_alpha is None:
            y_phi_today = np.full(self.N, 0.5, dtype=float)
        else:
            y_phi_today = np.asarray(green_obs_alpha, dtype=float)[t_i]
            if rng is not None and self.sigma_phi_obs.max() > 0.0:
                y_phi_today = y_phi_today + rng.normal(0.0, self.sigma_phi_obs, size=self.N)

        self._obs_buffer_route[:, :-1] = self._obs_buffer_route[:, 1:]
        self._obs_buffer_route[:, -1] = self.last_choice.astype(np.int32)
        self._obs_buffer_tt[:, :-1] = self._obs_buffer_tt[:, 1:]
        self._obs_buffer_tt[:, -1] = y_tt_today
        self._obs_buffer_sigma_tt[:, :-1] = self._obs_buffer_sigma_tt[:, 1:]
        self._obs_buffer_sigma_tt[:, -1] = self.sigma_obs
        self._obs_buffer_L[:, :-1] = self._obs_buffer_L[:, 1:]
        self._obs_buffer_L[:, -1] = y_L_today
        self._obs_buffer_sigma_L[:, :-1] = self._obs_buffer_sigma_L[:, 1:]
        self._obs_buffer_sigma_L[:, -1] = self.sigma_L_obs
        self._obs_buffer_phi[:, :-1] = self._obs_buffer_phi[:, 1:]
        self._obs_buffer_phi[:, -1] = y_phi_today
        self._obs_buffer_sigma_phi[:, :-1] = self._obs_buffer_sigma_phi[:, 1:]
        self._obs_buffer_sigma_phi[:, -1] = self.sigma_phi_obs
        self._obs_mask[:, :-1] = self._obs_mask[:, 1:]
        self._obs_mask[:, -1] = 1.0

        self._append_belief_broadcast(belief_broadcast, t_i, rng)

        self.day_count += 1

        self._last_observed_day[np.arange(self.N), self.last_choice] = self.day_count

        if self.day_count < self.W:
            return

        n_stale = np.where(
            self._last_observed_day < 0,
            self.day_count,
            self.day_count - self._last_observed_day,
        )

        self.state = window_step(
            state=self.state,
            cohort_priors=self.cohort_priors,
            route_chosen_window=jnp.asarray(self._obs_buffer_route),
            y_tt_window=jnp.asarray(self._obs_buffer_tt),
            sigma_tt_window=jnp.asarray(self._obs_buffer_sigma_tt),
            y_L_window=jnp.asarray(self._obs_buffer_L),
            sigma_L_window=jnp.asarray(self._obs_buffer_sigma_L),
            y_phi_window=jnp.asarray(self._obs_buffer_phi),
            sigma_phi_window=jnp.asarray(self._obs_buffer_sigma_phi),
            obs_mask=jnp.asarray(self._obs_mask),
            signalised=self.signalised_route,
            W=self.W,
            y_belief_L_window=jnp.asarray(self._belief_L),
            sigma_belief_L_window=jnp.asarray(self._belief_sigma_L),
            belief_mask_L_window=jnp.asarray(self._belief_mask_L),
            y_belief_phi_window=jnp.asarray(self._belief_phi),
            sigma_belief_phi_window=jnp.asarray(self._belief_sigma_phi),
            belief_mask_phi_window=jnp.asarray(self._belief_mask_phi),
            n_stale_days=jnp.asarray(n_stale),
            sigma_F_drift=jnp.asarray(self._sigma_F_drift)[:, None],
            sigma_C_drift=jnp.asarray(self._sigma_C_drift)[:, None],
            sigma_L_drift=jnp.asarray(self._sigma_L_drift)[:, None],
            sigma_phi_drift=jnp.asarray(self._sigma_phi_drift)[:, None],
            n_laplace_iters=self._n_laplace_iters,
            mean_revert_days=jnp.asarray(self._mean_revert_days)[:, None],
            phi_lo=self.phi_lo,
            phi_hi=self.phi_hi,
        )

    # ----------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        mu_y, var_y = self.predictive_moments
        sigma_y = np.sqrt(var_y)
        latents = self.latent_summary()
        return {
            "cohort_id": self.cohort_id.copy(),
            "cohort_label": self.cohort_label.copy(),
            "departure_time": self.departure_time.copy(),
            "complies": self.complies.copy(),
            "mu_alpha": mu_y[:, 0],
            "mu_beta": mu_y[:, 1],
            "sigma_alpha": sigma_y[:, 0],
            "sigma_beta": sigma_y[:, 1],
            "F_mean_alpha": latents["F_mean"][:, 0],
            "F_mean_beta": latents["F_mean"][:, 1],
            "C_mean_alpha": latents["C_mean"][:, 0],
            "C_mean_beta": latents["C_mean"][:, 1],
            "L_mean_alpha": latents["L_mean"][:, 0],
            "L_mean_beta": latents["L_mean"][:, 1],
            "phi_mean_alpha": latents["phi_mean"][:, 0],
            "phi_sd_alpha": latents["phi_sd"][:, 0],
            "last_P_alpha": self.last_P_alpha.copy(),
            "last_choice": self.last_choice.copy(),
        }


def build_population(
    population_params: PopulationParams,
    sim: SimParams,
    demand: DemandProfile,
    rng: np.random.Generator,
    route_names: tuple[str, str] = ("alpha", "beta"),
    signal: SignalParams | None = None,
) -> Population:
    return Population(
        cohorts=population_params.cohorts,
        sim=sim,
        demand=demand,
        rng=rng,
        route_names=route_names,
        signal=signal,
    )
