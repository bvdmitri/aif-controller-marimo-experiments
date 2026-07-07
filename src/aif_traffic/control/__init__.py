"""Macro-layer signal controllers.

A pluggable family of controllers behind one interface (:mod:`.interface`):

* :class:`FixedTimeController`   : non-adaptive constant split,
* :class:`ReactiveController`    : queue-feedback (SCOOT-like),
* :class:`AnticipatoryController`: predictive grid search,
* :class:`AIFController`         : Active-Inference (placeholder seam).

``build_controller(spec, ...)`` dispatches by spec type so notebooks and the
simulator can swap controllers via config alone.
"""

from __future__ import annotations

from ..parameters import (
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    FixedTimeControllerSpec,
    NetworkParams,
    ReactiveControllerSpec,
    SignalParams,
    SimParams,
)
from .aif_controller import AIFController
from .anticipatory import AnticipatoryController
from .fixed_time import FixedTimeController
from .interface import BaseController, Controller, project_to_constraint
from .reactive import ReactiveController


def build_controller(
    spec: object,
    signal: SignalParams,
    net: NetworkParams,
    sim: SimParams,
) -> Controller:
    """Construct the controller matching ``spec``'s type."""
    if isinstance(spec, FixedTimeControllerSpec):
        return FixedTimeController(spec, signal)
    if isinstance(spec, ReactiveControllerSpec):
        return ReactiveController(spec, signal, net.signalised_links)
    if isinstance(spec, AnticipatoryControllerSpec):
        return AnticipatoryController(spec, signal, net, sim)
    if isinstance(spec, AIFControllerSpec):
        return AIFController(spec, signal, net.signalised_links)
    raise TypeError(
        f"Unsupported controller spec type {type(spec).__name__}. Expected one "
        "of FixedTimeControllerSpec, ReactiveControllerSpec, "
        "AnticipatoryControllerSpec, AIFControllerSpec."
    )


__all__ = [
    "build_controller",
    "Controller",
    "BaseController",
    "project_to_constraint",
    "FixedTimeController",
    "ReactiveController",
    "AnticipatoryController",
    "AIFController",
]
