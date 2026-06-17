"""Fixed-time controller: a constant, non-adaptive green-time split.

Conventional pre-timed signal control: the split is set once and never
responds to traffic. Benchmark paradigm "non-adaptive control".
"""

from __future__ import annotations

from typing import Mapping

from ..parameters import FixedTimeControllerSpec, SignalParams
from .interface import BaseController, project_to_constraint


class FixedTimeController(BaseController):
    name = "fixed_time"

    def __init__(self, spec: FixedTimeControllerSpec, signal: SignalParams):
        self.spec = spec
        self.signal = signal
        self._phi2, self._phi6 = project_to_constraint(
            spec.phi2_frac * signal.phi_sat, signal.phi_sat, signal.phi_min,
        )

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        return self._phi2, self._phi6

    def snapshot(self) -> dict:
        return {"name": self.name, "phi2": self._phi2, "phi6": self._phi6}
