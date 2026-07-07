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


def _override(kind: str) -> dict[str, str]:
    """Per-style colour overrides, keyed by palette kind. Empty for marimo;
    the future paper style registers print-safe variants here."""
    return getattr(active_style(), "palette_overrides", {}).get(kind, {})


def controller_colour(name: str) -> str:
    return _override("controller").get(name, _CTRL_COLOURS_MARIMO.get(name, "k"))


def controller_label(name: str, *, abbr: bool = False) -> str:
    table = CTRL_ABBR if abbr else CTRL_LABELS
    return table.get(name, name)


def route_colour(name: str) -> str:
    return _override("route").get(name, _ROUTE_COLOURS_MARIMO.get(name, "k"))


def comm_colour(name: str) -> str:
    return _override("comm").get(name, _COMM_COLOURS_MARIMO.get(name, "k"))


def comm_label(name: str) -> str:
    return COMM_LABELS.get(name, name)


def ordered_controllers(present) -> list[str]:
    """Controller keys in canonical order, restricted to those present."""
    return [k for k in CTRL_ORDER if k in present]
