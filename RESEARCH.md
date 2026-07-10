# Open research concerns

Findings worth reconciling with the paper. Surfaced by the report-style tests
in `tests/test_narrative_reports.py` (run `pytest --runslow -s`).

## 1. Communication ranking (resolved)

Full-scale run (90 days, n=2000): **SN alone is the cheapest** setting; CG and
CG+SN are *not* cheaper than baseline; **CG gives the sharpest route-TT beliefs**;
and **CG+SN does not improve on SN**. The Experiment-3 narrative and the
report-style test (`test_report_communication_value`, which holds when SN is
cheapest) now agree with this, so this is no longer an open concern (the earlier
"paper expects CG+SN best" framing is superseded).

(The controller benchmark is consistent with the paper: AIF reaches a cost
comparable to the anticipatory controller, ~5% cheaper at full scale, with a
smoother green split.)
