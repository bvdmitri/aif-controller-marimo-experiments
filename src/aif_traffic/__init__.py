"""AIF-controller traffic experiments.

A signalised-intersection network with a two-layer Active Inference model:

* **Micro layer**: decentralised AIF travellers choosing between an
  intersection route and a bypass route (reused from the IWAI route-choice
  demonstration).
* **Macro layer**: a *pluggable* signal controller allocating the green-time
  split between the competing movements. Fixed-time, reactive, anticipatory,
  and an Active-Inference controller (placeholder) share one interface.

The controller can relay extra observations of the non-chosen routes into the
travellers' belief update and share its own belief for decision-time fusion; a
per-cohort compliance fraction controls who fuses the shared belief.

Submodules:

    parameters    - frozen dataclasses for every numeric knob
    network       - link incidence, queue dynamics, signal capacities
    demand        - A--B and C--D demand profiles
    inference     - closed-form Gaussian smoother and EFE (traveller AIF)
    control       - pluggable signal-controller family
    communication - controller -> traveller communication channels
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
    "BeliefSignal",
    "ObservationSignal",
    "NOTEBOOK_IDS",
    "explainer_pointer",
    "notebook_explainer",
]
