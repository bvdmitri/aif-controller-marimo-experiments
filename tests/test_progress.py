"""The ``on_step`` per-day tick callback of ``run_experiment``.

This is the plumbing behind the fused sweep progress bar
(``notebook_io.sweep_progress_bar``): a sweep creates one bar and passes its
``.update`` as ``on_step`` so the bar advances per simulated day across every
experiment. The marimo bar rendering itself is a UI concern; here we pin the
contract headlessly: the callback fires exactly once per simulated day (every
seed, burn-in included), and omitting it is a no-op.
"""

from __future__ import annotations

import numpy as np


def test_on_step_called_once_per_simulated_day(small_params):
    """``on_step`` fires exactly ``(burn_in + days) * n_seeds`` times."""
    from aif_traffic.simulator import run_experiment

    sim = small_params.sim
    seeds = [1, 2, 3]
    calls = {"n": 0}

    run_experiment(small_params, seeds=seeds, on_step=lambda: calls.__setitem__("n", calls["n"] + 1))

    expected = (sim.burn_in + sim.days) * len(seeds)
    assert calls["n"] == expected, f"expected {expected} ticks, got {calls['n']}"


def test_on_step_none_is_a_noop(small_params):
    """Omitting ``on_step`` does not perturb the run (determinism preserved)."""
    from aif_traffic.simulator import run_experiment

    a = run_experiment(small_params, seeds=[7])
    b = run_experiment(small_params, seeds=[7], on_step=lambda: None)
    assert np.allclose(a.step["P_alpha"].values, b.step["P_alpha"].values)
