"""The network-state chart: per-link data plumbing and the plot helper."""

from __future__ import annotations

from dataclasses import replace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from aif_traffic.parameters import NetworkParams, Params, SimParams  # noqa: E402
from aif_traffic.plotting import plot_network_state  # noqa: E402
from aif_traffic.plotting.network import _link_flows  # noqa: E402
from aif_traffic.simulator import run_experiment  # noqa: E402

NET = NetworkParams()


def _small_run():
    p = replace(Params(), sim=replace(SimParams(days=4, burn_in=1, h_min=60, dt_min=1)))
    return p, run_experiment(p, seeds=[0])


def test_simulator_persists_all_link_queues():
    _, res = _small_run()
    for lid in NET.link_ids:
        assert f"L{lid}" in res.step.columns, f"missing queue column L{lid}"


def test_link_flows_match_incidence():
    """Per-link flow equals the sum of the route flows that traverse the link."""
    _, res = _small_run()
    row = res.step.iloc[len(res.step) // 2]
    flows = _link_flows(row, NET)
    qa, qb, qg = row["Q_alpha"], row["Q_beta"], row["Q_gamma"]
    assert flows[1] == pytest.approx(qa + qb)   # link 1: alpha + beta
    assert flows[2] == pytest.approx(qa)        # link 2: alpha (intersection)
    assert flows[4] == pytest.approx(qa + qb)   # link 4: alpha + beta (merge)
    assert flows[5] == pytest.approx(qb)        # link 5: beta (bypass)
    assert flows[6] == pytest.approx(qg)        # link 6: gamma (C-D)
    assert flows[7] == pytest.approx(qg)        # link 7: gamma


@pytest.mark.parametrize("color_by", ["travellers", "queue"])
def test_plot_network_state_returns_figure(color_by):
    p, res = _small_run()
    fig = plot_network_state(res.step, p.network, color_by=color_by)
    assert isinstance(fig, Figure)


def test_plot_network_state_rejects_unknown_metric():
    p, res = _small_run()
    with pytest.raises(ValueError):
        plot_network_state(res.step, p.network, color_by="speed")
