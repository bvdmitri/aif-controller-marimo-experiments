"""The controller abstraction.

Every signal controller implements the same minimal contract so the simulator
can drive any of them polymorphically. A controller decides the **green-time
split between the two competing signalised movements** (link 2 for A--B,
link 6 for C--D) for the upcoming control interval. How it parameterises that
decision, and whether it treats the split as a direct action or as something
it infers, is each controller's private business; the simulator only sees
``decide`` returning ``(phi2, phi6)``.

The protocol:

* ``prepare_day(context)``: optional per-day hook (e.g. a predictive
  controller precomputes a plan from the day's planned inflows). Default no-op.
* ``decide(queue_obs, k)``: return ``(phi2, phi6)`` for control epoch ``k``,
  given the current per-link queue observation. ``phi2 + phi6 == phi_sat``.
* ``observe(day_state)``: optional end-of-day learning hook. Default no-op.
* ``snapshot()``: a small dict for plotting / inspection.

This is intentionally model-agnostic: the AIF controller's internal
formulation is an open question and lives behind this same interface.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Protocol, runtime_checkable

import numpy as np


class QueueForecast(NamedTuple):
    """A controller's belief over an upcoming day, broadcast to travellers for
    decision-time fusion (paper Experiment 3 / 4).

    For the AIF controller this is its **rolling-window smoother posterior** over
    the within-day queue trajectory (built by injecting the realised per-interval
    observations and running inference, :mod:`control.controller_smoother`), not
    a prior-predictive rollout. All arrays are length ``K`` (per within-day
    minute), indexed by the minute the queue/split refers to (the communication
    layer applies the traveller's arrival alignment ``k + N_l``):

    * ``mu_L`` / ``var_L``: the controller's posterior belief over the
      signalised A--B queue ``L_2`` (mean and per-interval marginal variance; the
      variance shrinks as the window fills with observations).
    * ``phi2``: the controller's *planned* green split for the A--B movement.
    * ``var_phi``: the (scalar) variance the controller attaches to its
      planned split when shared.
    """

    mu_L: np.ndarray
    var_L: np.ndarray
    phi2: np.ndarray
    var_phi: float


def project_to_constraint(
    phi2_raw: float,
    phi_sat: float,
    phi_min: float,
) -> tuple[float, float]:
    """Clip ``phi2`` into ``[phi_min, phi_sat - phi_min]`` and set
    ``phi6 = phi_sat - phi2`` so the cycle constraint always holds."""
    lo, hi = phi_min, phi_sat - phi_min
    phi2 = min(max(phi2_raw, lo), hi)
    return phi2, phi_sat - phi2


@runtime_checkable
class Controller(Protocol):
    def prepare_day(self, context: Mapping) -> None: ...

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]: ...

    def observe(self, day_state: Mapping) -> None: ...

    def snapshot(self) -> dict: ...

    def forecast(self, context: Mapping) -> QueueForecast | None: ...

    def belief_trajectory(self) -> tuple[np.ndarray, np.ndarray] | None: ...


class BaseController:
    """Convenience base providing no-op ``prepare_day`` / ``observe`` / ``forecast``
    and a default ``snapshot``. Concrete controllers override ``decide``."""

    name: str = "base"

    def prepare_day(self, context: Mapping) -> None:  # noqa: D401 - no-op hook
        return None

    def observe(self, day_state: Mapping) -> None:  # noqa: D401 - no-op hook
        return None

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        raise NotImplementedError

    def snapshot(self) -> dict:
        return {"name": self.name}

    def forecast(self, context: Mapping) -> QueueForecast | None:  # noqa: D401
        """Controllers that maintain a shareable belief override this; baselines
        have no belief to broadcast, so they forecast nothing."""
        return None

    def belief_trajectory(self) -> tuple[np.ndarray, np.ndarray] | None:  # noqa: D401
        """Controllers that learn a within-day queue belief override this to
        expose it for plotting: ``(mu, sd)`` each shape ``(2, K)`` (movement 0 =
        signalised A--B queue ``L_2``, 1 = C--D queue ``L_6``) per within-day
        minute. Baselines have no belief, so they return ``None``."""
        return None
