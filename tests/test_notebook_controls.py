"""The shared experiment-controls module and the hard consistency rule.

`notebook_controls.py` is the single source of truth for the experiment
notebooks' parameter sliders/checkboxes (CLAUDE.md hard rule). These tests pin:

  1. every control has a builder, a description, and a slot in the layout;
  2. `standard_panel` builds for any subset and rejects unknown names;
  3. the four experiment notebooks actually use the shared module and do NOT
     hand-define a parameter control inline (the `day_sel`/`tod_sel` day-
     inspection sliders and the `make_gif` render toggle are exempt).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aif_traffic import notebook_controls as nc

NOTEBOOKS = sorted(
    (Path(__file__).resolve().parent.parent / "notebooks").glob("0[1-5]_*.py")
)


def _layout_names() -> set[str]:
    return {name for _header, names in nc._GROUPS for name in names}


def test_descriptions_builders_layout_are_in_sync():
    """Every control appears in DESCRIPTIONS, has a callable builder, and sits in
    exactly one layout group -- no orphans in any of the three."""
    desc = set(nc.DESCRIPTIONS)
    layout = _layout_names()
    assert desc == layout, f"DESCRIPTIONS vs layout mismatch: {desc ^ layout}"
    for name in desc:
        builder = getattr(nc, name, None)
        assert callable(builder), f"no builder for control {name!r}"
    # no duplicate placement across groups
    all_names = [n for _h, names in nc._GROUPS for n in names]
    assert len(all_names) == len(set(all_names)), "a control is in two groups"


def test_builders_produce_widgets_and_learn_noise_defaults_on():
    """Each builder returns a fresh marimo UI element; learn_noise defaults True."""
    for name in nc.DESCRIPTIONS:
        w = getattr(nc, name)()
        assert hasattr(w, "value"), f"{name} builder did not return a UI element"
    assert nc.learn_noise().value is True  # VB on by default


def test_standard_panel_builds_subset_and_rejects_unknown():
    import marimo as mo

    widgets = {"days": nc.days(), "theta": nc.theta(), "compliance": nc.compliance()}
    panel = nc.standard_panel(widgets, mo.ui.run_button(label="Run"))
    assert panel is not None
    with pytest.raises(KeyError):
        nc.standard_panel({"not_a_control": nc.days()}, mo.ui.run_button(label="Run"))


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_shared_controls_and_no_inline_params(nb: Path):
    """Each experiment notebook imports `notebook_controls` and never hand-defines
    a parameter control inline (which would let the panels drift apart again)."""
    src = nb.read_text()
    assert "notebook_controls" in src, f"{nb.name} does not import notebook_controls"

    # No canonical parameter control may be assigned straight from `mo.ui.*`.
    offenders = []
    for name in nc.DESCRIPTIONS:
        if re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*mo\.ui\.", src):
            offenders.append(name)
    assert not offenders, (
        f"{nb.name} hand-defines parameter control(s) {offenders} inline; "
        "build them via aif_traffic.notebook_controls instead."
    )
