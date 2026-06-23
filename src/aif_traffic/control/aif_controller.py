"""Active-Inference signal controller.

The controller is an AIF agent that allocates green time between the two
signalised movements (link 2 for A--B, link 6 for C--D). It keeps a Gaussian
belief over the junction queue state ``(L_2, L_6)`` -- a genuine recursive
Gaussian filter, carried across control epochs -- predicts the queues one
control interval ahead under each candidate green split, and scores the splits
with the Expected-Free-Energy functional ``G = risk - epistemic``.

Its preference is a preferred-observation distribution ``N(0, Sigma_pref)`` over
the queues ("prefer empty queues"), mirroring the traveller's preferred-
observation Gaussian. The low-and-balanced goal lives inside ``Sigma_pref``
(extra precision along the capacity-normalised imbalance direction), so there is
no hand-crafted cost: the controller minimises the same EFE as the travellers
and differs only in its preferred observation.

The epistemic (information-gain) term is **live**: the controller's detectors
sample a movement more accurately the more green it receives, so the predicted
observation precision -- and hence the expected information gain about the queue
state -- depends on the split. The epistemic term therefore pulls green toward
the movement the controller is currently least certain about, traded off against
the pragmatic (risk) term that pulls toward low, balanced queues. This makes the
"propagate a full belief over queues" property substantive: both the predicted
mean *and* the predicted (action-dependent) covariance enter split selection.
See paper Section 4.2 / Appendix.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..network import link_inflows
from ..parameters import AIFControllerSpec, SignalParams
from .interface import BaseController, QueueForecast, project_to_constraint


def _mvn_kl(
    mu0: np.ndarray, S0: np.ndarray, mu1: np.ndarray, S1: np.ndarray,
) -> float:
    """``KL[ N(mu0, S0) || N(mu1, S1) ]`` for small dense covariances."""
    k = mu0.shape[0]
    S1_inv = np.linalg.inv(S1)
    diff = mu1 - mu0
    tr = float(np.trace(S1_inv @ S0))
    quad = float(diff @ S1_inv @ diff)
    _, logdet0 = np.linalg.slogdet(S0)
    _, logdet1 = np.linalg.slogdet(S1)
    return 0.5 * (tr + quad - k + float(logdet1 - logdet0))


class AIFController(BaseController):
    name = "aif"

    def __init__(
        self,
        spec: AIFControllerSpec,
        signal: SignalParams,
        signalised_links: tuple[int, int],
    ):
        self.spec = spec
        self.signal = signal
        self.sig_ab, self.sig_cd = signalised_links
        self.phi2_prev = signal.phi_sat / 2.0
        self._phi_ref = signal.phi_sat / 2.0
        # Recursive queue belief q(L_2, L_6) = N(mu, diag(var)); reset per day.
        self._mu: np.ndarray | None = None
        self._var: np.ndarray | None = None
        # Per-day prediction context, filled by prepare_day.
        self._inflow2: np.ndarray | None = None
        self._inflow6: np.ndarray | None = None
        self._cbar2 = 0.0
        self._cbar6 = 0.0
        self._dt_h = 0.0
        self._K = 0
        self._N2 = 0
        self._N6 = 0
        self._Sigma_pref: np.ndarray | None = None
        self._last_risk = float("nan")
        self._last_info = float("nan")
        self._last_efe = float("nan")

    # -- preference -------------------------------------------------------
    def _build_sigma_pref(self) -> None:
        """``Sigma_pref^{-1} = sigma_pref^{-2} I + omega n n^T`` with ``n`` the
        **unit** capacity-normalised imbalance direction ``prop. (1/Cbar2, -1/Cbar6)``.

        Convention note: ``n`` is unit-normalised so that ``omega`` lives in the
        same ``veh^-2`` units as the isotropic precision ``sigma_pref^-2`` and is
        directly comparable to it. With the defaults this makes the balance
        (imbalance-direction) precision the dominant term, i.e. *balance is
        designed to matter*. (The paper's Table 2 currently lists an
        un-normalised ``n`` with ``omega = 1``, under which the balance term is
        numerically negligible -- that table should be updated to match this
        convention.)
        """
        s = self.spec
        n = np.array([1.0 / self._cbar2, -1.0 / self._cbar6])
        n = n / np.linalg.norm(n)
        prec = (1.0 / s.sigma_pref ** 2) * np.eye(2) + s.omega * np.outer(n, n)
        self._Sigma_pref = np.linalg.inv(prec)

    # -- per-day setup ----------------------------------------------------
    def _setup_context(self, context: Mapping) -> None:
        """Fill the per-day prediction context (inflows, capacities, delays,
        preference) from ``context``. Shared by :meth:`prepare_day` (today's
        realised inflows) and :meth:`forecast` (expected inflows for a future
        day). Pure context only -- does NOT touch the running belief
        ``_mu``/``_var``/``phi2_prev`` or the ``_last_*`` snapshot fields."""
        net = context["net"]
        sim = context["sim"]
        Q_link = link_inflows(context["inflow_by_route"], net)
        self._inflow2 = np.asarray(Q_link[self.sig_ab], dtype=float)
        self._inflow6 = np.asarray(Q_link[self.sig_cd], dtype=float)
        self._cbar2 = net.cbar(self.sig_ab)
        self._cbar6 = net.cbar(self.sig_cd)
        self._dt_h = sim.dt_h
        self._K = sim.K
        nd = net.n_delay(sim.dt_min)
        self._N2 = nd[self.sig_ab]
        self._N6 = nd[self.sig_cd]
        self._build_sigma_pref()

    def prepare_day(self, context: Mapping) -> None:
        self._setup_context(context)
        # Fresh belief each day (the day starts from empty queues).
        self._mu = None
        self._var = None
        self.phi2_prev = self.signal.phi_sat / 2.0

    # -- generative-model pieces -----------------------------------------
    def _obs_var(self, phi2: float, phi6: float) -> np.ndarray:
        """Action-dependent observation variance per movement.

        A movement receiving more green is discharged and counted more
        accurately, so its queue observation is more precise:
        ``R_m(phi_m) = sigma_obs^2 * phi_ref / phi_m`` (decreasing in ``phi_m``).
        This is what makes the EFE epistemic term discriminate between splits.
        """
        s = self.spec
        ref = self._phi_ref
        return (s.sigma_obs ** 2) * np.array([ref / phi2, ref / phi6])

    def _rollout_mean(
        self, L0: np.ndarray, phi2: float, phi6: float, k: int, n_steps: int,
    ) -> np.ndarray:
        """Roll the store-and-forward queue mean forward ``n_steps`` under a
        constant split, returning the predicted-queue mean."""
        cap2 = phi2 * self._cbar2
        cap6 = phi6 * self._cbar6
        L = np.array(L0, dtype=float)
        for j in range(n_steps):
            t = k + j
            if t >= self._K - 1:
                break
            a2 = self._inflow2[t - self._N2] if t - self._N2 >= 0 else 0.0
            a6 = self._inflow6[t - self._N6] if t - self._N6 >= 0 else 0.0
            L[0] = max(0.0, L[0] + self._dt_h * (a2 - cap2))
            L[1] = max(0.0, L[1] + self._dt_h * (a6 - cap6))
        return L

    def _score_best_split(
        self, mu_b: np.ndarray, var_state: np.ndarray, k: int, phi2_prev: float,
    ) -> tuple[float, tuple[float, float, float]]:
        """Grid-search the green split minimising the EFE-plus-smoothness score.

        For each candidate split: roll the queue mean forward (``_rollout_mean``),
        score the pragmatic risk (Gaussian KL to the preference) against the
        action-dependent observation precision, subtract the expected information
        gain, and add the smoothness policy prior ``kappa*(phi2-phi2_prev)^2``.
        Returns ``(best_phi2, (risk, info, efe))``. Used by both :meth:`decide`
        (with the carried ``phi2_prev``) and :meth:`forecast` (with the planned
        previous split)."""
        s = self.spec
        sat = self.signal.phi_sat
        H = max(1, int(s.horizon_min))
        lo = self.signal.phi_min
        hi = sat - self.signal.phi_min
        grid = np.linspace(lo, hi, int(s.phi_grid_size))
        mu_pref = np.zeros(2)

        best_score = -np.inf
        best_phi2 = phi2_prev
        best = (float("nan"), float("nan"), float("nan"))
        for phi2 in grid:
            phi6 = sat - phi2
            mu_o = self._rollout_mean(mu_b, float(phi2), phi6, k, H)
            R_fut = self._obs_var(float(phi2), phi6)
            Sigma_o = np.diag(var_state + R_fut)
            risk = _mvn_kl(mu_o, Sigma_o, mu_pref, self._Sigma_pref)
            # Expected information gain about the queue state from the next
            # (split-dependent) observation: 0.5 sum_m log(1 + var_state/R_m).
            info = 0.5 * float(np.sum(np.log1p(var_state / R_fut)))
            efe = risk - s.info_gain_weight * info
            smooth = s.kappa * (phi2 - phi2_prev) ** 2
            score = -smooth - s.gamma * efe  # log q(phi) up to a constant
            if score > best_score:
                best_score = score
                best_phi2 = float(phi2)
                best = (risk, info, efe)
        return best_phi2, best

    # -- action selection (minimise the EFE: risk - epistemic) ------------
    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        if self._Sigma_pref is None:  # prepare_day not called (bare unit test)
            return project_to_constraint(
                self.phi2_prev, self.signal.phi_sat, self.signal.phi_min,
            )
        s = self.spec
        sat = self.signal.phi_sat
        obs = np.array([
            float(queue_obs.get(self.sig_ab, 0.0)),
            float(queue_obs.get(self.sig_cd, 0.0)),
        ])

        # -- Perception: Bayesian correction of the carried prior by the new
        #    observation. The observation precision reflects the split that was
        #    actually applied over the elapsed interval (more green -> sharper).
        R_app = self._obs_var(self.phi2_prev, sat - self.phi2_prev)
        if self._mu is None:  # first epoch of the day: posterior = observation
            mu_b = obs.copy()
            var_b = R_app.copy()
        else:
            var_b = 1.0 / (1.0 / self._var + 1.0 / R_app)
            mu_b = var_b * (self._mu / self._var + obs / R_app)

        # Prior over the future queue state (state covariance grows with the
        # rollout; this part is split-independent, as in any linear-Gaussian
        # model -- the action enters through the *observation* precision below).
        H = max(1, int(s.horizon_min))
        var_state = var_b + H * (s.sigma_proc ** 2)

        best_phi2, best = self._score_best_split(mu_b, var_state, k, self.phi2_prev)

        # Carry the belief forward one control interval under the chosen split.
        n_int = max(1, int(s.control_interval_min))
        self._mu = self._rollout_mean(mu_b, best_phi2, sat - best_phi2, k, n_int)
        self._var = var_b + n_int * (s.sigma_proc ** 2)
        self.phi2_prev = best_phi2
        self._last_risk, self._last_info, self._last_efe = best
        return project_to_constraint(best_phi2, sat, self.signal.phi_min)

    # -- forecast (belief broadcast for decision-time fusion) -------------
    def forecast(self, context: Mapping) -> QueueForecast:
        """Forward-predict the day's queue belief and planned split, to be
        broadcast to travellers for decision-time fusion (Experiment 3 / 4).

        Given the **expected** inflows for the upcoming day (``context`` carries
        ``inflow_by_route`` -- the simulator passes the most recent realised
        inflows as a persistence forecast), closed-loop dry-run the controller's
        own policy across the day **without** observation correction: start from
        empty queues, and at each control epoch pick the split via the same EFE
        scoring as :meth:`decide`, holding it until the next epoch. The queue
        belief is propagated by the generative model only, so its variance grows
        over the horizon -- the controller's honest predictive uncertainty.

        Returns a :class:`QueueForecast` with per-minute ``mu_L``/``var_L`` over
        the A--B queue ``L_2`` and the planned split ``phi2`` (length ``K``); the
        communication layer applies the traveller's arrival alignment. Uses only
        local state, so it does not disturb the running belief or snapshot."""
        self._setup_context(context)
        s = self.spec
        sat = self.signal.phi_sat
        K = self._K
        ci = max(1, int(s.control_interval_min))
        H = max(1, int(s.horizon_min))

        mu = np.zeros(2)
        var = np.full(2, s.sigma_obs ** 2)
        phi2_plan_prev = sat / 2.0
        cur_phi2 = sat / 2.0

        mu_L = np.zeros(K)
        var_L = np.zeros(K)
        phi_plan = np.zeros(K)
        for k in range(K):
            if k % ci == 0:
                var_state = var + H * (s.sigma_proc ** 2)
                cur_phi2, _ = self._score_best_split(mu, var_state, k, phi2_plan_prev)
                phi2_plan_prev = cur_phi2
            # Record the belief about the queue / the planned split AT minute k.
            mu_L[k] = mu[0]
            var_L[k] = var[0]
            phi_plan[k] = cur_phi2
            # Predict one minute ahead under the planned split (no observation,
            # so the variance only grows).
            mu = self._rollout_mean(mu, cur_phi2, sat - cur_phi2, k, 1)
            var = var + (s.sigma_proc ** 2)

        return QueueForecast(
            mu_L=mu_L, var_L=var_L, phi2=phi_plan,
            var_phi=float(s.sigma_phi_plan ** 2),
        )

    def snapshot(self) -> dict:
        return {"name": self.name, "phi2_last": self.phi2_prev,
                "risk_last": self._last_risk, "info_last": self._last_info,
                "efe_last": self._last_efe}
