"""Anticipatory controller: rolling-horizon predictive optimisation.

Model-based deterministic benchmark. At **each control epoch** it observes the
current junction (and network) queues and grid-searches the constant split that
minimises the predicted total system cost over a short rollout horizon
``H = horizon_min``, then applies that split until the next epoch (receding
horizon). This is the deterministic, point-estimate counterpart of the AIF
controller: same store-and-forward rollout and the same per-interval cadence,
but it optimises a single predicted trajectory (no belief / no uncertainty) and
its objective is the realised system cost rather than an EFE preference.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..network import link_inflows
from ..parameters import AnticipatoryControllerSpec, NetworkParams, SignalParams, SimParams
from .interface import BaseController, project_to_constraint


class AnticipatoryController(BaseController):
    name = "anticipatory"

    def __init__(
        self,
        spec: AnticipatoryControllerSpec,
        signal: SignalParams,
        net: NetworkParams,
        sim: SimParams,
    ):
        self.spec = spec
        self.signal = signal
        self.net = net
        self.sim = sim
        self._best_phi2 = signal.phi_sat / 2.0
        self._inflow_by_route: Mapping[str, np.ndarray] | None = None
        self._Q_link: dict[int, np.ndarray] | None = None
        self._n_delay: Mapping[int, int] | None = None

    def prepare_day(self, context) -> None:
        self._inflow_by_route = context["inflow_by_route"]
        self._Q_link = link_inflows(self._inflow_by_route, self.net)
        self._n_delay = self.net.n_delay(self.sim.dt_min)

    # -- predicted system cost of a constant split over the horizon -------
    def _horizon_cost(
        self, queue_obs: Mapping[int, float], phi2: float, phi6: float, k: int,
    ) -> float:
        net, sim = self.net, self.sim
        dt_h = sim.dt_h
        sig_ab, sig_cd = net.signalised_links
        H = max(1, int(self.spec.horizon_min))
        end = min(k + H, sim.K - 1)

        cap = {}
        for lid in net.link_ids:
            if lid == sig_ab:
                cap[lid] = phi2 * net.cbar(lid)
            elif lid == sig_cd:
                cap[lid] = phi6 * net.cbar(lid)
            else:
                cap[lid] = net.cbar(lid)

        L = {lid: float(queue_obs.get(lid, 0.0)) for lid in net.link_ids}
        cost = 0.0
        for t in range(k, end):
            # Instantaneous route travel times and system-cost increment at t.
            for route in net.routes:
                tt = 0.0
                for lid in net.route_links[route]:
                    tt += net.link(lid).F_min + 60.0 * L[lid] / max(cap[lid], 1e-6)
                q_rt = float(self._inflow_by_route[route][t])
                cost += q_rt * tt * dt_h
            # Advance the store-and-forward queues one step.
            for lid in net.link_ids:
                N_l = self._n_delay[lid]
                arr = self._Q_link[lid][t - N_l] if t - N_l >= 0 else 0.0
                L[lid] = max(0.0, L[lid] + dt_h * (arr - cap[lid]))
        return cost

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        if self._Q_link is None:  # prepare_day not called
            return project_to_constraint(
                self._best_phi2, self.signal.phi_sat, self.signal.phi_min,
            )
        lo = self.signal.phi_min
        hi = self.signal.phi_sat - self.signal.phi_min
        grid = np.linspace(lo, hi, self.spec.phi_grid_size)

        best_cost = np.inf
        best_phi2 = self._best_phi2
        for phi2 in grid:
            phi6 = self.signal.phi_sat - phi2
            cost = self._horizon_cost(queue_obs, float(phi2), phi6, k)
            if cost < best_cost:
                best_cost = cost
                best_phi2 = float(phi2)
        self._best_phi2 = best_phi2
        return project_to_constraint(
            best_phi2, self.signal.phi_sat, self.signal.phi_min,
        )

    def snapshot(self) -> dict:
        return {"name": self.name, "phi2": self._best_phi2}
