"""Numerical helpers shared across modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smooth_profile(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred rolling mean with edge handling via ``min_periods=1``."""
    return (
        pd.Series(np.asarray(x, dtype=float))
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def robust_limits(values, low: int = 2, high: int = 98, pad: float = 0.05):
    """Robust min/max for colourmap limits, ignoring NaN/inf."""
    vals = pd.Series(np.ravel(values)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(vals, [low, high])

    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 1.0)
        return vmin - delta, vmax + delta

    delta = (vmax - vmin) * pad
    return vmin - delta, vmax + delta


def daily_system_cost(
    inflows_by_route: dict,
    tt_by_route: dict,
    dt_h: float,
) -> float:
    """Daily total system cost: sum_r sum_t q_r(t) * TT_r(t) * dt_h.

    ``inflows_by_route[r]`` is the per-minute route inflow q_r(t) in veh/h
    (NOT the queue length), so the integrand has units veh-min.
    """
    total = 0.0
    for r in inflows_by_route:
        total += float(np.sum(inflows_by_route[r] * tt_by_route[r] * dt_h))
    return total
