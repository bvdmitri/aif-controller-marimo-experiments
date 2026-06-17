"""Reactive feedback controller (SCOOT-like).

Traffic-responsive: shifts green toward whichever competing movement has the
longer queue. ``phi2 = clip(phi_sat/2 + k_L * (L_2 - L_6))``. Benchmark
paradigm "traffic-responsive control".
"""

from __future__ import annotations

from typing import Mapping

from ..parameters import ReactiveControllerSpec, SignalParams
from .interface import BaseController, project_to_constraint


class ReactiveController(BaseController):
    name = "reactive"

    def __init__(
        self,
        spec: ReactiveControllerSpec,
        signal: SignalParams,
        signalised_links: tuple[int, int],
    ):
        self.spec = spec
        self.signal = signal
        self.sig_ab, self.sig_cd = signalised_links
        self._last = (signal.phi_sat / 2.0, signal.phi_sat / 2.0)

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        L2 = float(queue_obs.get(self.sig_ab, 0.0))
        L6 = float(queue_obs.get(self.sig_cd, 0.0))
        phi2_raw = self.signal.phi_sat / 2.0 + self.spec.k_L * (L2 - L6)
        self._last = project_to_constraint(
            phi2_raw, self.signal.phi_sat, self.signal.phi_min,
        )
        return self._last

    def snapshot(self) -> dict:
        return {"name": self.name, "k_L": self.spec.k_L,
                "phi2_last": self._last[0]}
