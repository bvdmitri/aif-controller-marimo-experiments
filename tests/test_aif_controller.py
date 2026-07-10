"""The Active Inference signal controller: preference KL, directional behaviour,
and an end-to-end run that stays within the cycle constraint."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aif_traffic.control import build_controller
from aif_traffic.control.aif_controller import _mvn_kl
from aif_traffic.parameters import (
    AIFControllerSpec,
    NetworkParams,
    Params,
    SignalParams,
    SimParams,
)
from aif_traffic.simulator import run_experiment

NET = NetworkParams()
SIGNAL = SignalParams()
SIM = SimParams(h_min=60, dt_min=1)


def test_mvn_kl_zero_when_equal():
    mu = np.array([3.0, -1.0])
    S = np.array([[2.0, 0.3], [0.3, 1.5]])
    assert _mvn_kl(mu, S, mu, S) == pytest.approx(0.0, abs=1e-9)


def test_mvn_kl_grows_with_mean_gap():
    S = np.eye(2)
    near = _mvn_kl(np.array([1.0, 0.0]), S, np.zeros(2), S)
    far = _mvn_kl(np.array([5.0, 0.0]), S, np.zeros(2), S)
    assert far > near > 0.0


def _prepare(ctrl, alpha, gamma):
    ctrl.prepare_day({
        "inflow_by_route": {
            "alpha": np.full(SIM.K, float(alpha)),
            "beta": np.zeros(SIM.K),
            "gamma": np.full(SIM.K, float(gamma)),
        },
        "net": NET, "sim": SIM, "signal": SIGNAL, "day": 0,
    })


def test_aif_favours_overloaded_movement():
    """With A--B demand far above C--D, the controller gives the A--B movement
    (link 2) more than an even share of green."""
    ctrl = build_controller(AIFControllerSpec(), SIGNAL, NET, SIM)
    _prepare(ctrl, alpha=1600.0, gamma=150.0)
    phi2, phi6 = ctrl.decide({2: 0.0, 6: 0.0}, 0)
    assert phi2 > SIGNAL.phi_sat / 2.0
    # The mirror case gives the C--D movement the larger share.
    ctrl2 = build_controller(AIFControllerSpec(), SIGNAL, NET, SIM)
    _prepare(ctrl2, alpha=150.0, gamma=1600.0)
    phi2_cd, _ = ctrl2.decide({2: 0.0, 6: 0.0}, 0)
    assert phi2_cd < SIGNAL.phi_sat / 2.0


def test_aif_observation_precision_depends_on_split():
    """A movement receiving more green is observed more precisely; this is
    what makes the EFE epistemic term action-dependent (no longer inert)."""
    ctrl = build_controller(AIFControllerSpec(), SIGNAL, NET, SIM)
    _prepare(ctrl, alpha=800.0, gamma=800.0)
    R = ctrl._obs_var(0.8, SIGNAL.phi_sat - 0.8)  # lots of green to movement 2
    assert R[0] < R[1]


def test_aif_epistemic_term_is_live_and_positive():
    """The information-gain term is actually computed and contributes to EFE."""
    ctrl = build_controller(AIFControllerSpec(), SIGNAL, NET, SIM)
    _prepare(ctrl, alpha=800.0, gamma=800.0)
    ctrl.decide({2: 20.0, 6: 20.0}, 0)
    snap = ctrl.snapshot()
    assert np.isfinite(snap["info_last"]) and snap["info_last"] > 0.0
    assert np.isfinite(snap["efe_last"])


def test_aif_end_to_end_respects_constraint():
    params = replace(Params(), sim=replace(SimParams(), days=3),
                     controller=AIFControllerSpec())
    res = run_experiment(params, seeds=[0])
    step = res.step
    lo = SIGNAL.phi_min - 1e-9
    hi = SIGNAL.phi_sat - SIGNAL.phi_min + 1e-9
    assert np.isfinite(step["phi2"]).all()
    assert (step["phi2"] >= lo).all() and (step["phi2"] <= hi).all()
    assert np.allclose(step["phi2"] + step["phi6"], SIGNAL.phi_sat, atol=1e-9)
