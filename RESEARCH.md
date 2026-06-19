# Open research concerns

Findings worth reconciling with the paper. Surfaced by the report-style tests
in `tests/test_narrative_reports.py` (run `pytest --runslow -s`).

## 1. theta is coupled to the externality *broadcast*

theta enters perceived cost as `zeta_r = TT_r + theta * E_r`, and the code folds
`E_r` in only through the broadcast channel (`SignalType.EXTERNALITY`/`MSC`).
So with "no broadcast" the offset is `theta * 0` and **every theta gives an
identical result** (this is why the notebook 01 theta-sweep first showed zero
difference). But the paper's Experiment 1 varies theta *with* "no broadcast" —
contradictory as written. Workaround: Experiment 1 now broadcasts EXTERNALITY at
full compliance (theta then lowers cost ~7% at full scale). Cleaner fix to
consider: compute/internalise `E_r` independently of the broadcast abstraction.

## 2. CG+SN is not the lowest-cost communication setting

Paper (Experiment 3) expects CG+SN best on cost. Full-scale run (90 days, n=2000)
instead shows **SN alone cheapest (~-16% vs baseline)**, while CG and CG+SN are
*slightly more expensive than baseline*; CG+SN only wins on belief uncertainty,
not cost. Either the model or the Experiment-3 narrative needs revisiting before
the paper commits to "CG+SN is best".

(Controller benchmark and the theta direction, once the externality is on, are
both consistent with the paper.)
