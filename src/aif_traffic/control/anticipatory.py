"""Anticipatory controller: predictive optimisation of the green split.

Model-based: at the start of each day it rolls the queue model forward under
each candidate split and picks the one minimising predicted total system cost.
Benchmark paradigm "predictive optimisation".

For tractability (and to keep it clearly distinct from the AIF controller),
this scaffold optimises a *single constant split* held over the whole day via
a grid search over candidates. A finer multi-epoch rollout is a natural later
extension; ``horizon_min`` is reserved for that.
"""

from __future__ import annotations

import numpy as np

from ..network import simulate_link_queues_const_phi
from ..parameters import AnticipatoryControllerSpec, NetworkParams, SignalParams, SimParams
from ..utils import daily_system_cost
from .interface import BaseController


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

    def prepare_day(self, context) -> None:
        inflow_by_route = context["inflow_by_route"]
        lo = self.signal.phi_min
        hi = self.signal.phi_sat - self.signal.phi_min
        grid = np.linspace(lo, hi, self.spec.phi_grid_size)

        best_cost = np.inf
        best_phi2 = self._best_phi2
        for phi2 in grid:
            _, tt_route = simulate_link_queues_const_phi(
                inflow_by_route, float(phi2), self.net, self.sim, self.signal,
            )
            cost = daily_system_cost(inflow_by_route, tt_route, self.sim.dt_h)
            if cost < best_cost:
                best_cost = cost
                best_phi2 = float(phi2)
        self._best_phi2 = best_phi2

    def decide(self, queue_obs, k: int) -> tuple[float, float]:
        return self._best_phi2, self.signal.phi_sat - self._best_phi2

    def snapshot(self) -> dict:
        return {"name": self.name, "phi2": self._best_phi2}
