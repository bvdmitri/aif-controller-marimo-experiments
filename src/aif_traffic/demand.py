"""Within-day demand profiles for the A--B and C--D movements.

Both streams follow the paper's shifted-sine form
$D(d,t) = D_{\\min} + (D_{\\max}-D_{\\min}) \\sin^2(\\pi t / T)$, kept in the
equivalent cosine form so endpoint values match the closed form exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import DemandParams, SimParams


def shifted_sine_demand(
    t: np.ndarray,
    d_min: float,
    d_max: float,
    h: int,
) -> np.ndarray:
    return d_min + 0.5 * (d_max - d_min) * (1.0 - np.cos(2.0 * np.pi * t / h))


@dataclass(frozen=True)
class DemandProfile:
    """Per-time-step demand on the A--B and C--D movements (veh/h)."""

    d_AB: np.ndarray
    d_CD: np.ndarray

    @classmethod
    def from_params(cls, sim: SimParams, demand: DemandParams) -> "DemandProfile":
        d_ab = shifted_sine_demand(sim.time, demand.d_AB_min, demand.d_AB_max, sim.h_min)
        d_cd = shifted_sine_demand(sim.time, demand.d_CD_min, demand.d_CD_max, sim.h_min)
        return cls(d_AB=d_ab, d_CD=d_cd)
