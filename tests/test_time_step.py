"""The within-day time step (``dt_min``) and the minutes-to-steps conversion.

The controller's cadence and prediction horizon are specified in *minutes*, but
the within-day arrays are indexed in ``dt_min``-minute *steps*. Regression: for a
while these minute values were used directly as step counts, so at ``dt_min != 1``
a "10 minute" control cadence silently became ``10 * dt_min`` minutes while the
free-flow propagation delays (correctly divided by ``dt_min``) stayed in minutes,
so the two disagreed. ``SimParams.n_steps`` now does the conversion once and every
consumer uses it.

These are fast, narrated checks: they print what they expect and observe so the
behaviour can be read back, not just pass/fail-checked.
"""

from __future__ import annotations

import numpy as np

from aif_traffic.network import run_within_day
from aif_traffic.parameters import Params, SimParams


def test_n_steps_converts_minutes_to_steps():
    """``n_steps(minutes)`` divides by ``dt_min`` (floored at one step)."""
    assert SimParams(dt_min=1).n_steps(10) == 10   # identity at 1 min
    assert SimParams(dt_min=2).n_steps(10) == 5    # 10 min / 2 = 5 steps
    assert SimParams(dt_min=5).n_steps(10) == 2    # 10 min / 5 = 2 steps
    assert SimParams(dt_min=3).n_steps(10) == 3    # round(10/3)
    assert SimParams(dt_min=2).n_steps(1) == 1     # never below a single step
    print("n_steps(minutes) = round(minutes / dt_min), min 1: verified for "
          "dt_min in {1,2,3,5}.")


def _epoch_minutes(dt_min: int, control_interval_min: int = 10):
    """Return the within-day *minutes* at which the controller is asked to act,
    when the day is integrated at the given ``dt_min``.

    Spies on ``run_within_day``: it fires ``green_split_fn(queue_obs, k)`` every
    ``control_interval`` steps, where the caller passes the minute cadence through
    ``sim.n_steps``. Step ``k`` sits at wall-clock minute ``k * dt_min``.
    """
    base = Params.default()
    sim = SimParams(days=1, h_min=60, dt_min=dt_min)
    net, signal = base.network, base.signal
    K = sim.K
    inflow_by_route = {r: np.full(K, 500.0) for r in net.routes}

    epochs: list[int] = []

    def spy(queue_obs, k):  # noqa: ANN001 - test stub
        epochs.append(k)
        return signal.phi_sat / 2.0, signal.phi_sat / 2.0

    stride = sim.n_steps(control_interval_min)
    run_within_day(inflow_by_route, spy, stride, net, sim, signal)
    return [k * dt_min for k in epochs], stride


def test_control_cadence_is_in_wall_clock_minutes_regardless_of_dt_min():
    """A 10-minute control cadence fires at the same *minutes* (0, 10, 20, ...)
    whether the day is stepped at 1, 2, or 5 minutes."""
    ci_min = 10
    got = {}
    for dt_min in (1, 2, 5):
        epoch_minutes, stride = _epoch_minutes(dt_min, ci_min)
        got[dt_min] = epoch_minutes
        # Every epoch lands on a multiple of the requested cadence in minutes.
        assert all(m % ci_min == 0 for m in epoch_minutes), (
            f"dt_min={dt_min}: epochs at minutes {epoch_minutes} are not all "
            f"multiples of the {ci_min}-min cadence (stride={stride} steps)"
        )
        # First epoch is the start of the day.
        assert epoch_minutes[0] == 0

    # The epoch minute-grids agree across dt_min (same wall-clock cadence).
    ref = got[1]
    for dt_min, minutes in got.items():
        assert minutes == ref, (
            f"dt_min={dt_min} epochs {minutes} differ from dt_min=1 {ref}"
        )
    print(f"10-min cadence -> epoch minutes {ref} at dt_min in {{1,2,5}} "
          "(identical wall-clock grid). Before the fix, dt_min=2/5 would have "
          "fired every 20/50 min.")


def test_free_flow_delays_and_cadence_share_the_same_minute_clock():
    """Both the propagation delays and the control cadence are measured against
    the same ``dt_min`` minute clock: a link's delay in *minutes* is stable
    across ``dt_min`` (up to the floor), matching the cadence conversion."""
    base = Params.default()
    # Pick a signalised link with a non-trivial free-flow time.
    lid = base.network.signalised_links[0]
    F = base.network.link(lid).F_min
    delays_min = {}
    for dt_min in (1, 2):
        n = base.network.n_delay(dt_min)[lid]      # delay in STEPS
        delays_min[dt_min] = n * dt_min            # back to minutes
    # floor(F/dt)*dt approximates F from below; both are within one dt of F.
    for dt_min, dm in delays_min.items():
        assert F - dt_min < dm <= F, (
            f"dt_min={dt_min}: delay {dm} min not within one step of F={F}"
        )
    print(f"link {lid} F={F} min -> delay {delays_min} min across dt_min "
          "(same minute clock the cadence now uses).")
