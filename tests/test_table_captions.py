"""The summary-table registry and its consistency rule (mirrors
`tests/test_chart_captions.py` for the quantitative tables).

`explainers.TABLE_GUIDE` is the single source of truth for every notebook
table's caption. These tests pin:

  1. every TABLE_GUIDE entry is well formed and every NOTEBOOK_TABLES id is known;
  2. `table_caption` rejects unknown table ids;
  3. each notebook renders exactly its NOTEBOOK_TABLES set via
     `table_block("<id>", ...)`;
  4. each table function returns a non-empty DataFrame with the expected columns.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from aif_traffic import explainers as ex

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parent.parent / "notebooks").glob("0[1-5]_*.py")
)


def _nb_id(path: Path) -> str:
    return re.sub(r"^\d+_", "", path.stem)


def _table_block_ids(src: str) -> list[str]:
    return re.findall(r"table_block\(\s*[\"']([^\"']+)[\"']", src)


# -- 1. registry is well formed ------------------------------------------------

def test_table_guide_entries_well_formed():
    for tid, g in ex.TABLE_GUIDE.items():
        assert set(g) >= {"title", "what", "read"}, tid
        for field in ("title", "what", "read"):
            assert isinstance(g[field], str) and g[field].strip(), f"{tid}.{field}"


def test_notebook_tables_reference_known_ids():
    for nb_id, ids in ex.NOTEBOOK_TABLES.items():
        assert nb_id in ex.NOTEBOOK_IDS, nb_id
        for tid in ids:
            assert tid in ex.TABLE_GUIDE, f"{nb_id} -> unknown table {tid!r}"
        assert len(ids) == len(set(ids)), f"{nb_id} lists a table twice"
    assert set(ex.NOTEBOOK_TABLES) == set(ex.NOTEBOOK_IDS)


# -- 2. table_caption behaviour ------------------------------------------------

def test_table_caption_rejects_unknown():
    with pytest.raises(KeyError):
        ex.table_caption("not_a_table")


def test_table_caption_includes_title_and_extra():
    cap = ex.table_caption("controller_summary", extra="EXTRA_NOTE_XYZ")
    assert ex.TABLE_GUIDE["controller_summary"]["title"] in cap
    assert "EXTRA_NOTE_XYZ" in cap


def test_tables_section_lists_each_table():
    for nb_id, ids in ex.NOTEBOOK_TABLES.items():
        section = ex.tables_section(nb_id)
        if not ids:
            assert section == ""
            continue
        for tid in ids:
            assert ex.TABLE_GUIDE[tid]["title"] in section, f"{nb_id}:{tid}"


# -- 3. notebooks render exactly their NOTEBOOK_TABLES set ---------------------

@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_renders_its_tables(nb: Path):
    src = nb.read_text()
    nb_id = _nb_id(nb)
    block_ids = _table_block_ids(src)
    for tid in block_ids:
        assert tid in ex.TABLE_GUIDE, f"{nb.name}: table_block unknown id {tid!r}"
        assert tid in ex.NOTEBOOK_TABLES[nb_id], (
            f"{nb.name}: table_block({tid!r}) not in NOTEBOOK_TABLES[{nb_id!r}]"
        )
    assert set(block_ids) == set(ex.NOTEBOOK_TABLES[nb_id]), (
        f"{nb.name}: tables rendered {sorted(set(block_ids))} != "
        f"NOTEBOOK_TABLES[{nb_id!r}]={sorted(ex.NOTEBOOK_TABLES[nb_id])}"
    )


# -- 4. the table functions actually produce their columns ---------------------

def _small(**kw):
    from aif_traffic.parameters import (
        CohortSpec, Params, PopulationParams, SimParams,
    )
    return replace(
        Params.default(),
        sim=SimParams(days=6, h_min=40, dt_min=1, burn_in=0, seed=3),
        population=PopulationParams(cohorts=(CohortSpec(n_agents=40),)),
        **kw,
    ).with_noise_regime("off").with_stationary(True)


def test_table_functions_return_expected_columns():
    from aif_traffic.parameters import FixedTimeControllerSpec, ObservationSignal
    from aif_traffic.plotting import (
        communication_summary_table,
        controller_summary,
        run_summary_table,
        theta_summary_table,
    )
    from aif_traffic.simulator import run_experiment

    res = run_experiment(_small(), seeds=[3])

    run_t = run_summary_table(res, n_last=3)
    assert not run_t.empty and set(run_t.columns) == {"metric", "mean", "std"}

    cs = controller_summary({"aif": res})
    assert {"mean_SC", "std_signal_variation", "mean_peak_L6"} <= set(cs.columns)

    nested = {
        "aif": {0.0: res},
        "fixed_time": {0.0: run_experiment(
            _small(controller=FixedTimeControllerSpec()), seeds=[3])},
    }
    tt = theta_summary_table(nested, n_last=3)
    assert not tt.empty
    assert {"controller", "theta", "mean_SC", "mean_peak_queue",
            "mean_P_alpha"} <= set(tt.columns)

    _CG = ObservationSignal.ROUTE_CONGESTION
    base = _small()
    by_set = {
        "BL": run_experiment(base.with_extra_observations(), seeds=[3]),
        "CG": run_experiment(base.with_extra_observations(_CG), seeds=[3]),
    }
    ct = communication_summary_table(by_set, n_last=3)
    assert list(ct["setting"]) == ["BL", "CG"]
    assert {"mean_SC", "dSC_vs_BL_pct", "belief_SD_TT_alpha",
            "belief_SD_TT_beta", "mean_P_alpha"} <= set(ct.columns)
