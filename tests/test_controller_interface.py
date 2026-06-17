"""Controller abstraction: every controller honours the cycle constraint and
the documented directional behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from aif_traffic.control import build_controller
from aif_traffic.parameters import (
    AIFControllerSpec,
    AnticipatoryControllerSpec,
    FixedTimeControllerSpec,
    NetworkParams,
    ReactiveControllerSpec,
    SignalParams,
    SimParams,
)

NET = NetworkParams()
SIGNAL = SignalParams()
SIM = SimParams(h_min=60, dt_min=1)

ALL_SPECS = [
    FixedTimeControllerSpec(),
    ReactiveControllerSpec(),
    AnticipatoryControllerSpec(phi_grid_size=5),
    AIFControllerSpec(),
]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: type(s).__name__)
def test_decision_respects_cycle_constraint(spec):
    ctrl = build_controller(spec, SIGNAL, NET, SIM)
    ctrl.prepare_day({
        "inflow_by_route": {
            "alpha": np.full(SIM.K, 1000.0),
            "beta": np.full(SIM.K, 200.0),
            "gamma": np.full(SIM.K, 300.0),
        },
        "net": NET, "sim": SIM, "signal": SIGNAL, "day": 0,
    })
    for k in (0, 10, 30):
        obs = {2: 50.0 * k, 6: 10.0 * k}
        phi2, phi6 = ctrl.decide(obs, k)
        assert phi2 + phi6 == pytest.approx(SIGNAL.phi_sat, abs=1e-9)
        assert SIGNAL.phi_min - 1e-9 <= phi2 <= SIGNAL.phi_sat - SIGNAL.phi_min + 1e-9


def test_fixed_time_is_constant():
    ctrl = build_controller(FixedTimeControllerSpec(phi2_frac=0.5), SIGNAL, NET, SIM)
    a = ctrl.decide({2: 0.0, 6: 0.0}, 0)
    b = ctrl.decide({2: 999.0, 6: 0.0}, 10)
    assert a == b


def test_reactive_shifts_green_toward_longer_queue():
    ctrl = build_controller(ReactiveControllerSpec(k_L=1e-3), SIGNAL, NET, SIM)
    phi2_ab_heavy, _ = ctrl.decide({2: 200.0, 6: 0.0}, 0)
    phi2_cd_heavy, _ = ctrl.decide({2: 0.0, 6: 200.0}, 0)
    assert phi2_ab_heavy > SIGNAL.phi_sat / 2.0
    assert phi2_cd_heavy < SIGNAL.phi_sat / 2.0


def test_anticipatory_favours_overloaded_movement():
    """With A--B demand far above C--D, the predictive controller allocates
    more than an even share of green to the A--B movement (link 2)."""
    ctrl = build_controller(
        AnticipatoryControllerSpec(phi_grid_size=9), SIGNAL, NET, SIM,
    )
    ctrl.prepare_day({
        "inflow_by_route": {
            "alpha": np.full(SIM.K, 1600.0),
            "beta": np.full(SIM.K, 0.0),
            "gamma": np.full(SIM.K, 150.0),
        },
        "net": NET, "sim": SIM, "signal": SIGNAL, "day": 0,
    })
    phi2, phi6 = ctrl.decide({2: 0.0, 6: 0.0}, 0)
    assert phi2 > SIGNAL.phi_sat / 2.0
