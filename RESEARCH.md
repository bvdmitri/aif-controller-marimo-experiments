# Open research concerns

Findings worth reconciling with the paper. Surfaced by the report-style tests
in `tests/test_narrative_reports.py` (run `pytest --runslow -s`).

## 1. theta acts only through the externality *broadcast*

theta enters perceived cost as `zeta_r = TT_r + theta * E_r`, and the code folds
`E_r` in only through the broadcast channel (`SignalType.EXTERNALITY`/`MSC` and
their sequential variants). With no broadcast the offset is `theta * 0` and every
theta gives an identical result. The paper's Experiment 1 text now states the
externality *is* broadcast (at full compliance), so the earlier "contradictory as
written" concern is resolved.

Remaining open point: theta's cost effect is advisory- and network-dependent. At
the default (uncongested-bypass) network the **raw** advisory barely helps and can
slightly *raise* cost at theta=1 (`test_report_theta_effect_on_system_cost`
observes +5.3% under the raw advisory). theta reliably lowers cost only once the
bypass is a bottleneck **and** the advisory is temporally stable (Experiment 4),
or under the **sequential-from-belief** advisory. Section 5.3.1 still needs
reported theta-sweep results consistent with whichever advisory the figures use
(the exported figures now use sequential-from-belief).

## 2. Communication ranking (resolved)

Full-scale run (90 days, n=2000): **SN alone is the cheapest** setting; CG and
CG+SN are *not* cheaper than baseline; **CG gives the sharpest route-TT beliefs**;
and **CG+SN does not improve on SN**. The Experiment-3 narrative and the
report-style test (`test_report_communication_value`, which holds when SN is
cheapest) now agree with this, so this is no longer an open concern (the earlier
"paper expects CG+SN best" framing is superseded).

(The controller benchmark, once the externality is on, is consistent with the
paper: AIF reaches a cost comparable to the anticipatory controller, ~5% cheaper
at full scale, with a smoother green split.)
