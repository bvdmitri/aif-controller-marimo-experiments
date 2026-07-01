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

    from .explainers import chart_caption_parts

    _widen_for_display(content)
    headline, details = chart_caption_parts(chart_id, extra=extra)
    # Short one-liner always visible; the full "what / how to read" folds away
    # under a collapsed accordion so it does not eat vertical space.
    caption = mo.accordion({headline: mo.md(details)})
    return mo.center(mo.vstack([caption, content], align="center"))


def table_block(table_id: str, df, *, extra: str | None = None, round_to: int = 2):
    """Render a summary table with its caption, centred (mirrors ``figure_block``).

    ``table_id`` is registered in :data:`aif_traffic.explainers.TABLE_GUIDE`;
    ``df`` is a pandas ``DataFrame`` (rounded to ``round_to`` decimals and shown
    via ``mo.ui.table``), or ``None`` before a run (a placeholder is shown).
    ``extra`` appends a notebook-specific note. Returns a centred
    ``mo.vstack`` of the caption above the table."""
    import marimo as mo

    from .explainers import table_caption_parts

    headline, details = table_caption_parts(table_id, extra=extra)
    caption = mo.accordion({headline: mo.md(details)})
    if df is None:
        body = mo.md("*Run the experiment above to populate this table.*")
    else:
        body = mo.ui.table(df.round(round_to), selection=None)
    return mo.center(mo.vstack([caption, body], align="center"))


def _widen_for_display(content) -> None:
    """Widen a matplotlib ``Figure`` to the notebook display width (keeping its
    height) so charts fill the marimo content column instead of rendering at the
    narrow paper text width. No-op for non-figures (gifs, ``mo`` layouts) and for
    figures already at least that wide. The target lives in the active style, so
    a future paper style leaves figures at their authored print width."""
    try:
        from matplotlib.figure import Figure
    except Exception:  # pragma: no cover - matplotlib always present in practice
        return
    if not isinstance(content, Figure):
        return
    from .plotting.style import active_style

    target = getattr(active_style(), "fig_display_w", None)
    if not target:
        return
    w, h = content.get_size_inches()
    if target > w:
        content.set_size_inches(target, h)


def sweep_progress_bar(n_experiments: int, sim, *, title: str, n_seeds: int = 1):
    """A single fused progress bar over every simulated day of a sweep.

    A sweep runs many experiments back-to-back (e.g. one per communication
    setting). Wrapping only the outer loop makes the bar advance once per
    *finished* experiment -- it sits at ``0/N`` for a whole 90-day run and gives
    no useful ETA. Instead, create this bar once and pass its ``.update`` to
    ``run_experiment(..., on_step=...)`` so it advances **per simulated day**
    across all experiments::

        with sweep_progress_bar(len(settings), base.sim, title="...") as bar:
            for name, p in settings.items():
                results[name] = run_experiment(p, seeds=[seed], on_step=bar.update)

    The total is ``n_experiments * n_seeds * (sim.burn_in + sim.days)`` -- one
    tick per simulated day (``run_experiment`` ticks every day, burn-in included).
    marimo shows the rate + ETA automatically.
    """
    import marimo as mo

    days_each = int(sim.burn_in) + int(sim.days)
    total = int(n_experiments) * int(n_seeds) * days_each
    return mo.status.progress_bar(total=total, title=title)
