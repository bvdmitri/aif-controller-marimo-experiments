"""Shared, semantic colour palette.

One place defines what colour/label/abbreviation each *thing* gets (the four
controllers, the three routes, and the four information-communication settings)
so a controller (or route, or setting) keeps the **same** colour across every
figure in the paper. Xue's review asks for this consistency repeatedly.

Colours are served through small accessor functions rather than bare dicts so a
later research-paper style can override them centrally (via
:func:`aif_traffic.plotting.style.active_style`) without touching call sites.
For this session there is a single style, so the accessors just return the
marimo colours below.
"""

from __future__ import annotations

from .style import active_style

# --- controllers ------------------------------------------------------------
# Keys match each controller's ``snapshot()["name"]`` (and the notebook keys).
CTRL_ORDER: tuple[str, ...] = ("fixed_time", "reactive", "anticipatory", "aif")
CTRL_LABELS: dict[str, str] = {
    "fixed_time": "Fixed-time (FT)",
    "reactive": "Reactive feedback (RF)",
    "anticipatory": "Anticipatory control (AC)",
    "aif": "Active Inference (AIF)",
}
# Short abbreviations for heatmap titles / compact legends (Xue, paper Table 3).
CTRL_ABBR: dict[str, str] = {
    "fixed_time": "FT",
    "reactive": "RF",
    "anticipatory": "AC",
    "aif": "AIF",
}
# Distinct but unbiased hues; every controller gets the same line weight so no
# controller is visually favoured; colour only tells them apart.
_CTRL_COLOURS_MARIMO: dict[str, str] = {
    "fixed_time": "#9e9e9e",    # grey
    "reactive": "#4393c3",      # blue
    "anticipatory": "#ff9800",  # amber
    "aif": "#1b5e20",           # green
}

# --- routes -----------------------------------------------------------------
# Match the colours already used by ``plot_route_flows`` / the demand plot.
_ROUTE_COLOURS_MARIMO: dict[str, str] = {
    "alpha": "tab:blue",    # A--B intersection route (traverses signalised L2)
    "beta": "tab:green",    # A--B bypass route (unsignalised)
    "gamma": "tab:orange",  # exogenous C--D stream (traverses signalised L6)
}

# --- controller signal (green split phi_2) ----------------------------------
# The controller's *action* line (the realised / believed green split phi_2)
# drawn in the coupled within-day figures. Kept out of the route palette (it is
# not a traveller route) but served through the same override seam so the paper
# style can give it a distinct vivid hue instead of the on-screen grey. Keyed
# ``"phi2"`` under the ``"signal"`` override kind.
_SIGNAL_COLOUR_MARIMO: str = "0.25"   # dark grey on screen

# --- information-communication settings (Experiment 3) ----------------------
COMM_ORDER: tuple[str, ...] = ("BL", "CG", "SN", "CG+SN")
COMM_LABELS: dict[str, str] = {
    "BL": "Baseline (BL)",
    "CG": "Route congestion (CG)",
    "SN": "Signal control (SN)",
    "CG+SN": "Route congestion + signal control (CG+SN)",
}
_COMM_COLOURS_MARIMO: dict[str, str] = {
    "BL": "#9e9e9e",     # grey, the reference
    "CG": "#4393c3",     # blue
    "SN": "#d6604d",     # red
    "CG+SN": "#762a83",  # purple
}


# --- line styles (accessibility) --------------------------------------------
# Colour tells series apart on screen; a distinct dash pattern tells them apart
# again in greyscale / black-and-white print, so every figure stays readable
# when the paper is printed without colour. Applied for both styles (harmless on
# screen, essential in print). Keyed the same way as the colours above; the most
# important series in each family gets the solid line.
_CTRL_LINESTYLES: dict[str, str] = {
    "fixed_time": ":",       # dotted (non-adaptive baseline)
    "reactive": "-.",        # dash-dot
    "anticipatory": "--",    # dashed
    "aif": "-",              # solid (the proposed controller)
}
_COMM_LINESTYLES: dict[str, str] = {
    "BL": ":",       # dotted (reference)
    "CG": "--",      # dashed
    "SN": "-",       # solid (the setting with the lowest cost)
    "CG+SN": "-.",   # dash-dot
}
_ROUTE_LINESTYLES: dict[str, str] = {
    "alpha": "-",    # solid (A--B intersection)
    "beta": "--",    # dashed (A--B bypass)
    "gamma": "-.",   # dash-dot (C--D)
}
# Generic dash cycle for keyless sweeps (demand scales, compliance): index in.
_SWEEP_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", (0, (1, 1)))


def _override(kind: str) -> dict[str, str]:
    """Per-style colour overrides, keyed by palette kind. Empty for marimo;
    the future paper style registers print-safe variants here."""
    return getattr(active_style(), "palette_overrides", {}).get(kind, {})


def controller_colour(name: str) -> str:
    return _override("controller").get(name, _CTRL_COLOURS_MARIMO.get(name, "k"))


def controller_linestyle(name: str) -> str:
    return _CTRL_LINESTYLES.get(name, "-")


def comm_linestyle(name: str) -> str:
    return _COMM_LINESTYLES.get(str(name), "-")


def route_linestyle(name: str) -> str:
    return _ROUTE_LINESTYLES.get(name, "-")


def sweep_linestyle(i: int):
    """Dash pattern for the ``i``-th line of a keyless sweep (demand scale /
    compliance), so the overlaid lines stay distinct in greyscale."""
    return _SWEEP_LINESTYLES[int(i) % len(_SWEEP_LINESTYLES)]


def controller_label(name: str, *, abbr: bool = False) -> str:
    table = CTRL_ABBR if abbr else CTRL_LABELS
    return table.get(name, name)


def route_colour(name: str) -> str:
    return _override("route").get(name, _ROUTE_COLOURS_MARIMO.get(name, "k"))


def signal_colour() -> str:
    """Colour of the controller's green-split ``phi_2`` line (realised / plan).

    Not a traveller route, so it lives outside the route palette; served through
    the style override seam (``"signal"`` kind, key ``"phi2"``) so the paper
    style can render it as a distinct vivid hue rather than the on-screen grey.
    """
    return _override("signal").get("phi2", _SIGNAL_COLOUR_MARIMO)


def comm_colour(name: str) -> str:
    return _override("comm").get(name, _COMM_COLOURS_MARIMO.get(name, "k"))


def comm_label(name: str) -> str:
    return COMM_LABELS.get(name, name)


def ordered_controllers(present) -> list[str]:
    """Controller keys in canonical order, restricted to those present."""
    return [k for k in CTRL_ORDER if k in present]
