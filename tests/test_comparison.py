"""Cross-controller comparison helpers: the signal-variation metric and the
summary table."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from aif_traffic.parameters import (
    AIFControllerSpec,
    FixedTimeControllerSpec,
    Params,
    SimParams,
)
from aif_traffic.plotting.comparison import (
    _daily_signal_variation,
    controller_summary,
)
from aif_traffic.simulator import run_experiment


def _step(phi2_by_tau: list[float]) -> pd.DataFrame:
    n = len(phi2_by_tau)
    return pd.DataFrame({
        "day": [0] * n,
        "tau": list(range(n)),
        "phi2": phi2_by_tau,
        "phi6": [0.9 - p for p in phi2_by_tau],
        "L2": [0.0] * n,
        "L6": [0.0] * n,
        "SC": [1.0] * n,
    })


def test_signal_variation_zero_for_constant_split():
    s = _daily_signal_variation(_step([0.4, 0.4, 0.4, 0.4]))
    assert float(s.loc[0]) == 0.0


def test_signal_variation_sums_absolute_steps():
    # steps: 0.4->0.6 (0.2), 0.6->0.5 (0.1) => total variation 0.3
    s = _daily_signal_variation(_step([0.4, 0.6, 0.5]))
    assert float(s.loc[0]) == np.float64(0.3) or abs(float(s.loc[0]) - 0.3) < 1e-9


def test_controller_summary_one_row_per_controller():
    base = Params()
    results = {}
    for name, spec in [("fixed_time", FixedTimeControllerSpec()),
                       ("aif", AIFControllerSpec())]:
        p = replace(base, sim=replace(SimParams(), days=4), controller=spec)
        results[name] = run_experiment(p, seeds=[0])
    summ = controller_summary(results)
    assert list(summ.columns) == [
        "controller", "mean_SC", "std_SC",
        "mean_signal_variation", "mean_peak_queue",
    ]
    assert len(summ) == 2
    assert np.isfinite(summ["mean_SC"]).all()
    assert np.isfinite(summ["mean_peak_queue"]).all()
