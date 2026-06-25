"""Small helpers for the marimo notebooks (kept out of the plotting layer).

The notebooks expose a "save PDFs to outputs/" control that writes figures to
the *local* repository's ``outputs/`` directory. On the public / deployed
marimo build that control is meaningless (and would let visitors write to the
server filesystem), so it is hidden when a deployment environment variable is
set -- the env-var gating pattern suggested in marimo-team/marimo#6049.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set any of these to a truthy value on the deployed server to hide the
# PDF-export controls. ``MARIMO_DEPLOY`` doubles as a generic flag if the
# hosting setup already exports one.
_DEPLOY_ENV_VARS = ("IWAI_DEPLOY", "MARIMO_DEPLOY")
_FALSEY = ("", "0", "false", "no", "off")


def is_deployed() -> bool:
    """True when running on the public/deployed marimo server.

    Controlled by the env vars in :data:`_DEPLOY_ENV_VARS`; locally these are
    unset, so the PDF-export tick is shown as usual.
    """
    return any(
        os.environ.get(v, "").strip().lower() not in _FALSEY
        for v in _DEPLOY_ENV_VARS
    )


def outputs_dir() -> Path:
    """Absolute path to the repository's ``outputs/`` directory."""
    return Path(__file__).resolve().parents[2] / "outputs"


def figure_block(chart_id: str, content, *, extra: str | None = None):
    """Render a figure with its "how to read" caption, centred (CLAUDE.md hard rule).

    ``chart_id`` is the plotting-function name registered in
    :data:`aif_traffic.explainers.CHART_GUIDE`; ``content`` is the figure to show
    (a matplotlib ``Figure``, a ``figure_placeholder``, or any marimo-renderable
    such as an ``mo.hstack`` of figures). Returns a centred ``mo.vstack`` of the
    caption (markdown, with a slider badge when applicable) above the figure, so
    every notebook figure is captioned and centred uniformly. ``extra`` appends a
    notebook-specific note (e.g. another control that filters the chart)."""
    import marimo as mo

    from .explainers import chart_caption

    caption = mo.md(chart_caption(chart_id, extra=extra))
    return mo.center(mo.vstack([caption, content], align="center"))
