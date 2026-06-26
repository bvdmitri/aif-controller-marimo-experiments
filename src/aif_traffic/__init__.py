"""AIF-controller traffic experiments.

A signalised-intersection network with a two-layer Active Inference model:

* **Micro layer** -- decentralised AIF travellers choosing between an
  intersection route and a bypass route (reused from the IWAI route-choice
  demonstration).
* **Macro layer** -- a *pluggable* signal controller allocating the green-time
  split between the competing movements. Fixed-time, reactive, anticipatory,
  and an Active-Inference controller (placeholder) share one interface.

The controller broadcasts an information signal travellers may fold into their
perceived route cost; a per-cohort compliance fraction controls who listens.

Submodules:

    parameters    - frozen dataclasses for every numeric knob
    network       - link incidence, queue dynamics, signal capacities
    demand        - A--B and C--D demand profiles
    inference     - closed-form Gaussian smoother and EFE (traveller AIF)
    control       - pluggable signal-controller family
    communication - controller -> traveller broadcast
    simulator     - coupled two-layer day loop and experiment driver
    aggregation   - per-day / summary roll-ups
    plotting      - pure Figure-returning visualisations
    utils         - smoothing, robust limits, daily cost
"""

from .explainers import NOTEBOOK_IDS, explainer_pointer, notebook_explainer
from .parameters import (
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    BeliefSignal,
    CohortSpec,
    CommunicationSpec,
    DemandParams,
    EFEParams,
    FixedTimeControllerSpec,
    NetworkParams,
    NoiseParams,
    ObservationSignal,
    Params,
    PopulationParams,
    ReactiveControllerSpec,
    SignalParams,
    SignalType,
    SimParams,
)

__all__ = [
    "Params",
    "SimParams",
    "NetworkParams",
    "DemandParams",
    "SignalParams",
    "CohortSpec",
    "PopulationParams",
    "EFEParams",
    "NoiseParams",
    "FixedTimeControllerSpec",
    "ReactiveControllerSpec",
    "AnticipatoryControllerSpec",
    "AIFControllerSpec",
    "CommunicationSpec",
    "SignalType",
    "BeliefSignal",
    "ObservationSignal",
    "NOTEBOOK_IDS",
    "explainer_pointer",
    "notebook_explainer",
]
