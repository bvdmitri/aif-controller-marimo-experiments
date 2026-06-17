"""Active-Inference signal controller -- PLACEHOLDER.

This is the open seam of the project. The AIF controller's generative model
over the signalised-junction queue state, its preferred states (low and
balanced queues), its policy prior (smooth green-time changes), and its
Expected-Free-Energy action selection over the green split are an open design
question to be developed against this structure, taking inspiration from the
IWAI traveller model.

For now this conforms to the controller interface and delegates to the
reactive feedback baseline, so the full traveller<->controller pipeline runs
end to end and we can compare controllers before committing to a formulation.

    TODO(AIF controller): replace the delegated decision below with the
    Active-Inference formulation once the methodology (paper Section 4.2) is
    settled. The generative model / preferences / EFE belong here.
"""

from __future__ import annotations

from typing import Mapping

from ..parameters import AIFControllerSpec, ReactiveControllerSpec, SignalParams
from .interface import BaseController
from .reactive import ReactiveController


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
        # Placeholder behaviour: delegate to the reactive baseline. The real
        # AIF belief + EFE machinery will replace this.
        self._delegate = ReactiveController(
            ReactiveControllerSpec(), signal, signalised_links,
        )

    def prepare_day(self, context: Mapping) -> None:
        self._delegate.prepare_day(context)

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        return self._delegate.decide(queue_obs, k)

    def observe(self, day_state: Mapping) -> None:
        self._delegate.observe(day_state)

    def snapshot(self) -> dict:
        return {"name": self.name, "placeholder": True,
                "delegate": self._delegate.snapshot()}
