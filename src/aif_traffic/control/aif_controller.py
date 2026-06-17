"""Active-Inference signal controller.

The controller is an AIF agent that allocates green time between the two
signalised movements (link 2 for A--B, link 6 for C--D). It keeps a Gaussian
belief over the junction queue state ``(L_2, L_6)``, predicts the queues one
control interval ahead under each candidate green split, and scores the splits
with the fixed Expected-Free-Energy functional.

Its preference is a preferred-observation distribution ``N(0, Sigma_pref)`` over
the queues ("prefer empty queues"), mirroring the traveller's preferred-
observation Gaussian. The low-and-balanced goal lives inside ``Sigma_pref``
(extra precision along the capacity-normalised imbalance direction), so there is
no hand-crafted cost: the controller minimises the same EFE as the travellers
and differs only in its preferred observation.

The epistemic (information-gain) term of the EFE is inert here. The queues are
observed every control interval at fixed precision, so the predicted-observation
covariance does not depend on the split and the epistemic term cannot
discriminate between candidates; selection is governed by the pragmatic (risk)
term. The asymmetry with the travellers (who explore, because a route is
observed only when chosen) is therefore derived, not designed. See paper
Section 4.2 / Appendix.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..network import link_inflows
from ..parameters import AIFControllerSpec, SignalParams
from .interface import BaseController, project_to_constraint


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

    # -- preference -------------------------------------------------------
    def _build_sigma_pref(self) -> None:
        """``Sigma_pref^{-1} = sigma_pref^{-2} I + omega n n^T`` with ``n`` the
        unit capacity-normalised imbalance direction ``(1/Cbar2, -1/Cbar6)``."""
        s = self.spec
        n = np.array([1.0 / self._cbar2, -1.0 / self._cbar6])
        n = n / np.linalg.norm(n)
        prec = (1.0 / s.sigma_pref ** 2) * np.eye(2) + s.omega * np.outer(n, n)
        self._Sigma_pref = np.linalg.inv(prec)

    # -- per-day setup ----------------------------------------------------
    def prepare_day(self, context: Mapping) -> None:
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

    # -- transition (predict the queues forward under a candidate split) --
    def _predict(
        self, L0: np.ndarray, phi2: float, phi6: float, k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the store-and-forward queue map forward over the horizon under a
        constant split, returning predicted-observation mean and covariance."""
        s = self.spec
        H = max(1, int(s.horizon_min))
        cap2 = phi2 * self._cbar2
        cap6 = phi6 * self._cbar6
        L = np.array(L0, dtype=float)
        var = (s.sigma_obs ** 2) * np.ones(2)  # posterior at observation time
        for j in range(H):
            t = k + j
            if t >= self._K - 1:
                break
            a2 = self._inflow2[t - self._N2] if t - self._N2 >= 0 else 0.0
            a6 = self._inflow6[t - self._N6] if t - self._N6 >= 0 else 0.0
            L[0] = max(0.0, L[0] + self._dt_h * (a2 - cap2))
            L[1] = max(0.0, L[1] + self._dt_h * (a6 - cap6))
            var = var + s.sigma_proc ** 2  # random-walk process noise
        Sigma_obs = np.diag(var + s.sigma_obs ** 2)  # plus future obs noise
        return L, Sigma_obs

    # -- action selection (minimise the fixed EFE pragmatic term) ---------
    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        if self._Sigma_pref is None:  # prepare_day not called (bare unit test)
            return project_to_constraint(
                self.phi2_prev, self.signal.phi_sat, self.signal.phi_min,
            )
        # Perception: the observation is the current queue (trivial update).
        L0 = np.array([
            float(queue_obs.get(self.sig_ab, 0.0)),
            float(queue_obs.get(self.sig_cd, 0.0)),
        ])
        lo = self.signal.phi_min
        hi = self.signal.phi_sat - self.signal.phi_min
        grid = np.linspace(lo, hi, int(self.spec.phi_grid_size))
        mu_pref = np.zeros(2)

        best_score = -np.inf
        best_phi2 = self.phi2_prev
        best_risk = float("nan")
        for phi2 in grid:
            phi6 = self.signal.phi_sat - phi2
            mu_o, Sigma_o = self._predict(L0, float(phi2), phi6, k)
            risk = _mvn_kl(mu_o, Sigma_o, mu_pref, self._Sigma_pref)
            smooth = self.spec.kappa * (phi2 - self.phi2_prev) ** 2
            score = -smooth - self.spec.gamma * risk  # log q(phi) up to const
            if score > best_score:
                best_score = score
                best_phi2 = float(phi2)
                best_risk = risk
        self.phi2_prev = best_phi2
        self._last_risk = best_risk
        return project_to_constraint(
            best_phi2, self.signal.phi_sat, self.signal.phi_min,
        )

    def snapshot(self) -> dict:
        return {"name": self.name, "phi2_last": self.phi2_prev,
                "risk_last": self._last_risk}
