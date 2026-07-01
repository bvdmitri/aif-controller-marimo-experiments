"""Central styling seam.

The plotting layer has **one** place that decides visual constants: the
*active style*. It is process-global — set once by the notebook / CI harness —
so switching the whole app between looks is a single call, not an edit across
the ~25 chart functions.

This session ships a single style (``"marimo"``, the current look). The
research-paper style and the CI PDF export land in a later session; the seam is
already here so they drop in without churning call sites:

* chart functions read semantic sizes/weights from :func:`active_style` (e.g.
  ``active_style().line_main``) instead of hard-coding them, and
* colours come from :mod:`.palette`, which itself routes through the active
  style, so a print-safe paper palette can be swapped in centrally.

Adding the paper style later is: register another :class:`StyleContext` in
``_STYLES`` and teach :mod:`.palette` its colour overrides. Nothing else moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib as mpl


@dataclass(frozen=True)
class StyleContext:
    """All visual constants for one look, resolved in one object.

    ``rc`` is pushed into ``matplotlib.rcParams`` by :func:`apply_style`; the
    remaining fields are *semantic* handles charts read directly so a later
    style can retune weights/alphas without every figure repeating a literal.
    """

    name: str
    # LLNCS figure-width budget (inches); mirrors ``primitives.TEXT_W(_HALF)``.
    text_w: float = 4.8
    text_w_half: float = 3.3
    # On-screen display width (inches) that notebook figures are widened to so
    # they fill the marimo content column (height is left unchanged). Applied
    # centrally in ``notebook_io.figure_block``; a paper style would set this to
    # ``text_w`` so figures keep their authored print width.
    fig_display_w: float = 9.0
    # Semantic line weights / band opacities charts read instead of literals.
    line_main: float = 1.5
    line_ref: float = 0.9
    band_alpha: float = 0.20
    band_alpha_light: float = 0.12
    rc: dict = field(default_factory=dict)


# The current marimo look, factored out of the old ``setup_style``.
_MARIMO = StyleContext(
    name="marimo",
    rc={
        "font.family": "Arial",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
        "legend.framealpha": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
    },
)

# Registry of known styles. A ``"paper"`` entry is added in a later session.
_STYLES: dict[str, StyleContext] = {"marimo": _MARIMO}

_ACTIVE: StyleContext = _MARIMO


def apply_style(name: str = "marimo") -> StyleContext:
    """Make ``name`` the active style and push its rcParams into matplotlib.

    Called once by the notebook (via :func:`aif_traffic.plotting.setup_style`)
    or, in the future, by the CI exporter with ``name="paper"`` before saving.
    Returns the resolved :class:`StyleContext`.
    """
    global _ACTIVE
    try:
        style = _STYLES[name]
    except KeyError as exc:  # pragma: no cover - guards a typo'd style name
        raise ValueError(
            f"unknown style {name!r}; known styles: {sorted(_STYLES)}"
        ) from exc
    _ACTIVE = style
    mpl.rcParams.update(style.rc)
    return style


def active_style() -> StyleContext:
    """The style currently in effect (defaults to the marimo look)."""
    return _ACTIVE
