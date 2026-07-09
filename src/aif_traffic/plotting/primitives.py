"""Low-level helpers shared by every figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import text_w, text_w_half

# --- figure-width budget (inches) -------------------------------------------
# Figures are authored at ``text_w()`` (full block) / ``text_w_half()``
# (side-by-side pair), which come from the **active style** so the same chart
# code renders at the marimo width or the paper (elsarticle 3p, ~6.72 in) width
# without edits. Chart functions call ``text_w()`` / ``text_w_half()`` in their
# ``figsize``. ``TEXT_W`` / ``TEXT_W_HALF`` remain as the marimo constants for
# back-compat (and any non-figsize use). These live here (not in ``__init__``)
# so every plot module imports them without a circular import.
TEXT_W = 4.8
TEXT_W_HALF = 3.3

__all__ = [
    "TEXT_W", "TEXT_W_HALF", "text_w", "text_w_half", "within_day_profile_size",
    "figure_placeholder", "place_legend_above", "panel_label", "light_borders",
]


def within_day_profile_size() -> tuple[float, float]:
    """Authored size (inches) of a single-day within-day profile panel.

    The paper's Figure 5 stitches three of these (route flows / travel times /
    queues) side by side. Authoring all three at this identical size (about a
    third of the text width, so they sit near 1:1 at ``0.32\\linewidth`` and keep
    their fonts legible) keeps their aspect ratios matched, so they render at a
    consistent height in the row.
    """
    return (text_w() / 3.0, text_w() * 0.42)


def figure_placeholder(
    title: str,
    subtitle: str = "Click *Run experiment* to generate this figure.",
    figsize: tuple[float, float] = (7.0, 3.5),
):
    """A neutral placeholder shown in plot cells before any run has happened."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    ax.set_facecolor("#fafafa")
    ax.text(0.5, 0.58, title, ha="center", va="center",
            transform=ax.transAxes, fontsize=13, color="#666666")
    ax.text(0.5, 0.40, subtitle, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="#999999", style="italic")
    fig.tight_layout()
    return fig


def place_legend_above(
    ax,
    *,
    handles=None,
    labels=None,
    ncol: int | None = None,
    fontsize: float = 8,
    title_pad: float = 18,
    columnspacing: float = 1.4,
    handlelength: float | None = None,
):
    """Place ``ax``'s legend outside the data area, centred above the plot
    and below the title, as a single horizontal row, instead of floating
    inside the data.

    The existing axes title is re-applied with extra padding so it clears the
    legend strip. Mirrors the placement already used by the per-agent and
    route-split figures. No-op when the axes has no labelled artists.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = len(handles)
    kw = {} if handlelength is None else {"handlelength": handlelength}
    leg = ax.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, fontsize=fontsize, frameon=False, borderaxespad=0.0,
        columnspacing=columnspacing, **kw,
    )
    title = ax.get_title()
    if title:
        ax.set_title(title, pad=title_pad)
    return leg


def panel_label(ax, s: str, *, x: float = 0.025, y: float = 0.97,
                fontsize: float = 9) -> None:
    """Stamp a ``(a)`` style panel label inside the top-left corner of ``ax``.

    Placed in axes-fraction coordinates *inside* the data area (with a faint
    white backing box for legibility over lines or a heatmap), so it never
    extends the tight bounding box: the saved figure keeps exactly the same
    dimensions. Used to mark the panels a multi-PDF paper figure stitches
    together so in-text ``Figure~Xa`` references map to the right panel.
    """
    ax.text(
        x, y, f"({s})", transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", ha="left", va="top", zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="none", alpha=0.7),
    )


def light_borders(axes, *, color: str = "#bbbbbb", linewidth: float = 0.6) -> None:
    """Give each panel a thin, light box border on all four sides.

    Used in place of inter-panel separator lines: every subplot keeps its own
    full border, drawn light and thin so the grid reads as a set of framed
    panels without heavy black boxes.
    """
    for ax in np.atleast_2d(axes).ravel():
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(linewidth)
