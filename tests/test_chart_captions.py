"""The per-chart "how to read" guide and its hard consistency rule.

`explainers.CHART_GUIDE` is the single source of truth for every notebook
figure's caption (CLAUDE.md hard rule). These tests pin:

  1. every CHART_GUIDE entry is well formed and every NOTEBOOK_CHARTS id is known;
  2. `chart_caption` rejects unknown charts and emits a slider badge exactly for
     the slider charts;
  3. each experiment notebook renders every chart it calls through
     `figure_block("<id>", ...)`, and the set it renders matches NOTEBOOK_CHARTS;
  4. a chart that follows a day / time-of-day slider only appears in a notebook
     that actually defines that slider.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aif_traffic import explainers as ex

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parent.parent / "notebooks").glob("0[1-4]_*.py")
)


def _nb_id(path: Path) -> str:
    """Map a notebook filename (``01_social_internalisation.py``) to its
    explainer id (``social_internalisation``)."""
    return re.sub(r"^\d+_", "", path.stem)


def _charts_called(src: str) -> set[str]:
    """The CHART_GUIDE plotting functions actually *called* in a notebook (a call
    is ``name(`` -- import lines and return tuples use ``name,``)."""
    return {
        cid for cid in ex.CHART_GUIDE
        if re.search(rf"\b{re.escape(cid)}\(", src)
    }


def _figure_block_ids(src: str) -> list[str]:
    """The chart ids passed to ``figure_block("<id>", ...)`` in a notebook."""
    return re.findall(r"figure_block\(\s*[\"']([^\"']+)[\"']", src)


# -- 1. registry is well formed ------------------------------------------------

def test_chart_guide_entries_well_formed():
    for cid, g in ex.CHART_GUIDE.items():
        assert set(g) >= {"title", "what", "read", "slider"}, cid
        for field in ("title", "what", "read"):
            assert isinstance(g[field], str) and g[field].strip(), f"{cid}.{field}"
        assert g["slider"] in ex._VALID_SLIDERS, f"{cid} slider={g['slider']!r}"


def test_notebook_charts_reference_known_ids():
    for nb_id, ids in ex.NOTEBOOK_CHARTS.items():
        assert nb_id in ex.NOTEBOOK_IDS, nb_id
        for cid in ids:
            assert cid in ex.CHART_GUIDE, f"{nb_id} -> unknown chart {cid!r}"
        assert len(ids) == len(set(ids)), f"{nb_id} lists a chart twice"
    # every notebook id has a chart list
    assert set(ex.NOTEBOOK_CHARTS) == set(ex.NOTEBOOK_IDS)


# -- 2. chart_caption behaviour ------------------------------------------------

def test_chart_caption_rejects_unknown():
    with pytest.raises(KeyError):
        ex.chart_caption("not_a_chart")


def test_chart_caption_slider_badge_matches_slider_field():
    for cid, g in ex.CHART_GUIDE.items():
        cap = ex.chart_caption(cid)
        assert g["title"] in cap
        if g["slider"] is None:
            assert "🎚️" not in cap, f"{cid} has a badge but no slider"
        else:
            assert "🎚️" in cap, f"{cid} missing slider badge"
            if "day" in g["slider"]:
                assert "inspect day" in cap
            if "tod" in g["slider"]:
                assert "time of day" in cap


def test_chart_caption_extra_is_appended():
    cap = ex.chart_caption("plot_queue_belief_day", extra="EXTRA_NOTE_XYZ")
    assert "EXTRA_NOTE_XYZ" in cap


def test_charts_section_lists_each_chart():
    for nb_id, ids in ex.NOTEBOOK_CHARTS.items():
        section = ex.charts_section(nb_id)
        assert "How to read the charts" in section
        for cid in ids:
            assert ex.CHART_GUIDE[cid]["title"] in section, f"{nb_id}:{cid}"


# -- 3. notebooks render every chart via figure_block --------------------------

@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_every_called_chart_is_captioned_and_matches_registry(nb: Path):
    src = nb.read_text()
    nb_id = _nb_id(nb)
    called = _charts_called(src)
    block_ids = _figure_block_ids(src)

    # every figure_block id is a real chart, listed for this notebook
    for cid in block_ids:
        assert cid in ex.CHART_GUIDE, f"{nb.name}: figure_block unknown id {cid!r}"
        assert cid in ex.NOTEBOOK_CHARTS[nb_id], (
            f"{nb.name}: figure_block({cid!r}) not in NOTEBOOK_CHARTS[{nb_id!r}]"
        )

    # every chart actually called is rendered through figure_block (captioned)
    missing = called - set(block_ids)
    assert not missing, (
        f"{nb.name} calls {sorted(missing)} without figure_block(...) -- every "
        "figure must be displayed via notebook_io.figure_block so it is captioned."
    )

    # the registry's notebook map matches what the notebook really renders
    assert called == set(ex.NOTEBOOK_CHARTS[nb_id]), (
        f"{nb.name}: charts called {sorted(called)} != "
        f"NOTEBOOK_CHARTS[{nb_id!r}]={sorted(ex.NOTEBOOK_CHARTS[nb_id])}"
    )


# -- 4. slider charts only where the slider exists -----------------------------

@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_slider_charts_only_in_notebooks_with_the_slider(nb: Path):
    src = nb.read_text()
    nb_id = _nb_id(nb)
    has_day = bool(re.search(r"day_sel\s*=\s*mo\.ui\.", src))
    has_tod = bool(re.search(r"tod_sel\s*=\s*mo\.ui\.", src))
    for cid in ex.NOTEBOOK_CHARTS[nb_id]:
        slider = ex.CHART_GUIDE[cid]["slider"]
        if slider and "day" in slider:
            assert has_day, f"{nb.name} shows {cid} (needs day slider) but defines none"
        if slider and "tod" in slider:
            assert has_tod, f"{nb.name} shows {cid} (needs tod slider) but defines none"
