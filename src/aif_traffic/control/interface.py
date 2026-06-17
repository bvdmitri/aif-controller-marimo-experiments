"""The controller abstraction.

Every signal controller implements the same minimal contract so the simulator
can drive any of them polymorphically. A controller decides the **green-time
split between the two competing signalised movements** (link 2 for A--B,
link 6 for C--D) for the upcoming control interval. How it parameterises that
decision, and whether it treats the split as a direct action or as something
it infers, is each controller's private business -- the simulator only sees
``decide`` returning ``(phi2, phi6)``.

The protocol:

* ``prepare_day(context)`` -- optional per-day hook (e.g. a predictive
  controller precomputes a plan from the day's planned inflows). Default no-op.
* ``decide(queue_obs, k)`` -- return ``(phi2, phi6)`` for control epoch ``k``,
  given the current per-link queue observation. ``phi2 + phi6 == phi_sat``.
* ``observe(day_state)`` -- optional end-of-day learning hook. Default no-op.
* ``snapshot()`` -- a small dict for plotting / inspection.

This is intentionally model-agnostic: the AIF controller's internal
formulation is an open question and lives behind this same interface.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable


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


class BaseController:
    """Convenience base providing no-op ``prepare_day`` / ``observe`` and a
    default ``snapshot``. Concrete controllers override ``decide``."""

    name: str = "base"

    def prepare_day(self, context: Mapping) -> None:  # noqa: D401 - no-op hook
        return None

    def observe(self, day_state: Mapping) -> None:  # noqa: D401 - no-op hook
        return None

    def decide(self, queue_obs: Mapping[int, float], k: int) -> tuple[float, float]:
        raise NotImplementedError

    def snapshot(self) -> dict:
        return {"name": self.name}
