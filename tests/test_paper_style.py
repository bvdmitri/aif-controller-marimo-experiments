"""The publication (`paper`) figure style and the paper-figure export path.

Fast checks (not the full-scale render): that `apply_style("paper")` flips the
style seam (IWAI-style Arial/sans-serif fonts, paper width, print-safe palette
overrides), that charts
render under it, and that the export script's render path runs headless on a
reduced config. A fixture always restores the marimo style so the process-global
style can't leak into other tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root ``scripts/`` package importable (it is not installed).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from aif_traffic import plotting as pl  # noqa: E402
from aif_traffic.plotting import palette, style  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_marimo_style():
    """Never let the process-global paper style leak past a test."""
    yield
    pl.apply_style("marimo")


def test_paper_style_flips_the_seam():
    pl.apply_style("paper")
    st = style.active_style()
    assert st.name == "paper"
    # Elsevier 3p single-column authoring width, no marimo widening.
    assert st.text_w == pytest.approx(6.72)
    assert st.fig_display_w == st.text_w
    # IWAI-figure fonts (Arial/sans-serif) + embedded (editable) PDF text.
    assert "sans-serif" in mpl.rcParams["font.family"]
    assert "Arial" in mpl.rcParams["font.sans-serif"]
    assert mpl.rcParams["pdf.fonttype"] == 42
    # Print-safe palette overrides are populated and actually consulted.
    assert st.palette_overrides
    assert palette.controller_colour("aif") == st.palette_overrides["controller"]["aif"]
    assert palette.route_colour("alpha") == st.palette_overrides["route"]["alpha"]


def test_style_switch_does_not_leak():
    pl.apply_style("paper")
    assert "sans-serif" in mpl.rcParams["font.family"]
    pl.apply_style("marimo")
    # Back to the marimo look: sans family, marimo colours, marimo width.
    assert mpl.rcParams["font.family"] == ["Arial"]
    assert style.active_style().text_w == pytest.approx(4.8)
    assert palette.controller_colour("aif") == "#1b5e20"  # marimo green


def test_charts_render_at_paper_width():
    pl.apply_style("paper")
    from dataclasses import replace

    from aif_traffic.parameters import (
        CohortSpec, Params, PopulationParams, SimParams,
    )
    from aif_traffic.simulator import run_experiment

    p = replace(
        Params.default(),
        sim=SimParams(days=6, h_min=40, dt_min=1, burn_in=0, seed=3),
        population=PopulationParams(cohorts=(CohortSpec(n_agents=60),)),
    ).with_noise_regime("off").with_stationary(True)
    res = run_experiment(p, seeds=[3])
    fig = pl.plot_controller_metrics({"aif": res})
    # Authored at the paper full-column width (~6.72 in).
    assert fig.get_size_inches()[0] == pytest.approx(6.72, abs=0.01)
    plt.close(fig)


def test_export_paper_figures_runs_headless(tmp_path: Path):
    """The export path renders + writes PDFs on a reduced config (no full run)."""
    import scripts.export_paper_figures as exp

    written = exp.export_all(tmp_path, quick=True)
    assert written, "no figures written"
    assert (tmp_path / "INDEX.md").exists()
    # A representative few exist as PDFs.
    for stem in ("exp1_coupled_within_day", "exp2_controller_metrics",
                 "exp3_within_day_by_setting"):
        assert (tmp_path / f"{stem}.pdf").exists(), stem
