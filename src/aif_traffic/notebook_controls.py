"""Single source of truth for the experiment notebooks' parameter controls.

Every experiment notebook (``notebooks/0{1..4}_*.py``) builds its parameter panel
from this module instead of hand-defining ``mo.ui`` widgets inline, so the
controls stay **consistent across experiments**: the same control always has the
same range, default, label and one-line description, and the panel always renders
in the same order and grouping.

Usage in a notebook controls cell::

    from aif_traffic import notebook_controls as nc

    days = nc.days(); seed = nc.seed(); control_interval = nc.control_interval()
    ...                                   # build the controls this experiment needs
    run_btn = mo.ui.run_button(label="Run experiment")
    controls = nc.standard_panel({
        "days": days, "seed": seed, "control_interval": control_interval, ...
    }, run_btn)
    controls

Each control is created as a named global in the cell (so downstream cells read
``days.value`` etc. and marimo's reactivity works); :func:`standard_panel` lays out
whichever subset was passed.

Convention (see CLAUDE.md): do **not** hand-define ``mo.ui`` sliders/checkboxes for
the parameter panel in a notebook -- add or change a control here. ``theta`` is
only meaningful where the externality cost-offset is broadcast (Experiments 1, 2);
``compliance`` only where a controller->traveller channel is gated (Experiments
1-3; swept in Experiment 4). The per-notebook ``day_sel`` / ``tod_sel`` day-
inspection sliders are *not* parameter controls and are exempt.
"""

from __future__ import annotations

import marimo as mo


# ---------------------------------------------------------------------------
# Canonical one-line descriptions (channel-agnostic, reused verbatim everywhere).
# ---------------------------------------------------------------------------
DESCRIPTIONS: dict[str, str] = {
    "days": "Recorded days to simulate (after any warm-up days).",
    "warmup": (
        "Warm-up days run before recording starts, to let the two layers settle "
        "out of their cold-start transient; these days are discarded from the "
        "output. 0 by default (the whole run, including the cold start, is "
        "recorded)."
    ),
    "seed": "Master seed; redraws all stochastic elements.",
    "time_step": (
        r"Within-day discretisation interval $\Delta t$ (minutes): the simulation "
        r"time step. $1$ min is the finest; larger steps are coarser and faster "
        r"but blur the within-day dynamics, and also set the free-flow propagation "
        r"delays $\lfloor F_\ell/\Delta t\rfloor$."
    ),
    "control_interval": (
        "Minutes between green-split decisions, and the controller's prediction "
        "horizon for scoring each split."
    ),
    "demand_scale": (
        r"Scales peak A--B and C--D demand. $>1$ pushes the junction toward "
        r"saturation and makes the control problem harder."
    ),
    "bypass_capacity_scale": (
        r"Scales the nominal capacity of the bypass link (link 5). $<1$ throttles "
        r"the bypass so it can congest and become a bottleneck (making the "
        r"intersection route relatively more attractive); $1.0$ = the default "
        r"network."
    ),
    "traveller_window": (
        "Days each traveller's rolling-window smoother remembers when forming "
        r"route beliefs. Shorter $\to$ more reactive; longer $\to$ steadier but "
        "slower to adapt."
    ),
    "controller_window": (
        "Days of past queue observations the macro AIF controller smooths over "
        r"before acting (and broadcasting its belief). Shorter $\to$ faster but "
        "noisier; longer $\to$ steadier."
    ),
    "learn_noise": (
        "Learn the observation-noise SD instead of fixing it: a conjugate-Gamma "
        "posterior over each precision, fit by variational Bayes (controller per "
        "movement, travellers per agent). On by default; the belief band is then "
        "data-driven rather than set by `sigma_obs`."
    ),
    "stationary": (
        "Assume the environment is **stationary**: both layers do continuous "
        "filtering (keep the whole run, never forget) instead of rolling-window "
        "smoothing, so beliefs accumulate all evidence and converge (no periodic "
        "spikes from the window dropping old days). On by default; the traveller / "
        "controller **window sliders are ignored while this is on**. Turn off to "
        "recover the rolling-window smoother (the non-stationary / disruption case)."
    ),
    "noise_regime": (
        "How much measurement noise the environment injects into what travellers "
        "observe (travel time / queue / green split). **off** = fully "
        "deterministic (exact observations + deterministic route choice, for "
        "clean convergence); **low / medium / high** scale the noise, with "
        "**medium** the realistic default and **low**/**high** half/double it."
    ),
    "comm_mechanism": (
        "Which controller->traveller information channel Experiment 3 sweeps. "
        "**Extra observations** relays the *true realised* queue (CG) / green "
        "split (SN) of the routes a traveller did not take into its belief update "
        "(works with any controller, reaches everyone). **Belief sharing** has the "
        "AIF controller share its own forecast belief (QB/SP) for transient "
        "decision-time fusion (compliant travellers only). **Both** runs each "
        "channel alone and combined; **Disable** is the no-information baseline."
    ),
    "theta": (
        r"Social internalisation $\theta$: how much travellers fold the congestion "
        r"externality $E_r$ into their perceived cost $\zeta_r = TT_r + \theta E_r$. "
        r"$0$ = selfish (user equilibrium), $1$ = fully cooperative (system "
        r"optimum). Only bites when the externality is broadcast (Experiments 1, 2)."
    ),
    "compliance": (
        "Fraction of travellers that act on the controller's broadcast (the "
        "externality advisory and/or the shared belief). At $0$ the broadcast is "
        "ignored and the setting collapses onto the baseline."
    ),
    "gamma": (
        r"AIF action precision $\gamma^c$. Higher $\to$ a sharper preference for "
        r"the lowest-EFE split (more decisive control)."
    ),
    "omega": (
        r"AIF balance weight in the preference $\Sigma^c_{\mathrm{pref}}$: "
        r"penalises capacity-normalised queue imbalance between the two movements. "
        r"$0$ = only total queue matters."
    ),
    "sigma_pref": (
        r"AIF preferred-queue tolerance (veh): the SD of the *empty queues* "
        r"preference. Smaller $\to$ less tolerant of any residual queue."
    ),
    "phi_grid": "Number of candidate green splits the AIF controller scores each decision.",
    "k_L": "Reactive (SCOOT-like) feedback gain on the queue imbalance $L_2-L_6$.",
}


# ---------------------------------------------------------------------------
# Control builders -- one per control, returning a fresh widget with the
# canonical range / default / label. (marimo needs a new element per cell run.)
# ---------------------------------------------------------------------------
def days():
    return mo.ui.slider(10, 180, value=90, label="days")


def warmup():
    return mo.ui.slider(0, 90, value=0, label="warm-up days")


def seed():
    return mo.ui.slider(0, 100, value=42, label="seed")


def time_step():
    return mo.ui.slider(1, 10, value=1, label="time step [min]")


def control_interval():
    return mo.ui.slider(1, 30, value=10, label="control interval [min]")


def demand_scale():
    return mo.ui.slider(0.5, 2.5, step=0.1, value=1.0, label="demand scale")


def bypass_capacity_scale():
    return mo.ui.slider(0.1, 1.5, step=0.05, value=1.0,
                        label="bypass capacity scale")


def traveller_window(disabled: bool = False):
    return mo.ui.slider(0, 60, value=30, label="traveller window [days]",
                        disabled=disabled)


def controller_window(disabled: bool = False):
    return mo.ui.slider(0, 60, value=30, label="controller window [days]",
                        disabled=disabled)


def learn_noise():
    return mo.ui.checkbox(value=True, label="learn observation noise (VB)")


def stationary():
    return mo.ui.checkbox(value=True, label="assume stationary environment")


def noise_regime():
    return mo.ui.dropdown(
        options=["off", "low", "medium", "high"],
        value="medium",
        label="noise regime",
    )


def comm_mechanism():
    return mo.ui.dropdown(
        options=["Disable", "Extra observations", "Belief sharing", "Both"],
        value="Extra observations",
        label="communication mechanism",
    )


def theta():
    return mo.ui.slider(0.0, 1.0, step=0.05, value=0.0, label="theta")


def compliance():
    return mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="compliance")


def gamma():
    return mo.ui.slider(0.5, 20.0, step=0.5, value=4.0, label="AIF gamma")


def omega():
    return mo.ui.slider(0.0, 0.2, step=0.005, value=0.02, label="AIF omega")


def sigma_pref():
    return mo.ui.slider(5.0, 60.0, step=1.0, value=20.0, label="AIF sigma_pref [veh]")


def phi_grid():
    return mo.ui.slider(3, 21, value=9, label="candidate splits K")


def k_L():
    return mo.ui.slider(1e-4, 5e-3, step=1e-4, value=1e-3, label="reactive k_L")


# ---------------------------------------------------------------------------
# Canonical layout: fixed order + grouping. ``standard_panel`` renders only the
# controls that were passed, so notebooks select a subset but the structure,
# order and descriptions are identical across experiments.
# ---------------------------------------------------------------------------
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Simulation", (
        "days", "warmup", "seed", "time_step", "control_interval", "demand_scale",
        "bypass_capacity_scale",
        "learn_noise", "noise_regime",
        # The window sliders sit under (and are disabled by) the stationary
        # toggle -- they only bite in the rolling-window (non-stationary) mode.
        "stationary", "traveller_window", "controller_window",
    )),
    ("Communication / social", ("comm_mechanism", "theta", "compliance")),
    ("AIF controller", ("gamma", "omega", "sigma_pref", "phi_grid", "k_L")),
)


def _row(widget, desc: str):
    """A control row: the widget beside its one-line explanation."""
    return mo.hstack([widget, mo.md(desc)], widths=[2, 3], align="center", gap=1)


def standard_panel(widgets: dict, run_btn, *, title: str = "### Parameters you can play with"):
    """Render the parameter panel for the controls in ``widgets`` (name -> widget).

    Lays out the controls in the canonical :data:`_GROUPS` order with group
    headers, each paired with its :data:`DESCRIPTIONS` text via :func:`_row`, and
    appends ``run_btn`` last. Only groups with at least one present control are
    shown, so every notebook gets an identically-structured panel for whatever
    subset it uses. Raises ``KeyError`` if an unknown control name is passed."""
    unknown = set(widgets) - set(DESCRIPTIONS)
    if unknown:
        raise KeyError(f"unknown control(s) not defined in notebook_controls: {sorted(unknown)}")
    rows = [mo.md(title)]
    for header, names in _GROUPS:
        present = [(n, widgets[n]) for n in names if n in widgets]
        if not present:
            continue
        rows.append(mo.md(f"**{header}**"))
        rows.extend(_row(w, DESCRIPTIONS[n]) for n, w in present)
    rows.append(run_btn)
    return mo.vstack(rows, gap=0.5)
