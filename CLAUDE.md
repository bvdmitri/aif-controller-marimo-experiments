# Project guidance for Claude Code

This repository is the simulator + marimo companion for the paper extending the
IWAI route-choice work with a **macro-layer Active Inference signal
controller** on a signalised-intersection network.

The repo holds the network, the reused IWAI traveller model, a pluggable
controller abstraction with baselines, the **implemented Active Inference
controller**, the communication + compliance mechanism, and the first
experiment notebook.

- **The AIF controller is implemented** in `control/aif_controller.py` (numpy)
  as **one big Active-Inference agent**, the macro analogue of the (thousands of)
  tiny traveller agents. Its latent is the **entire within-day queue trajectory**
  of both signalised movements `(L_2(t), L_6(t))_t` (a large state), estimated
  from the per-interval queue observations by a **rolling-window Gaussian
  smoother over the last `controller_window_size` days** (mirroring the
  travellers' smoother), with a **full covariance** capturing temporal
  correlations. The smoother is in `control/controller_smoother.py`: a random-walk
  trajectory prior (tridiagonal precision) + linear identity observations →
  banded `O(M)` solve + `O(M)` marginal variances (a dense reference validates
  it). The controller still **acts** each control interval by minimising the
  **fixed** Expected Free Energy (pragmatic MVN-KL from the predicted queue to the
  preferred `N(0, Sigma_pref)`, minus a split-dependent epistemic info gain, plus
  a smoothness prior on the split), now using its smoother posterior as the prior
  (posterior-as-prior). The only designed object is the preference; the
  low-and-balanced goal lives inside `Sigma_pref` (extra precision `omega` along
  the unit capacity-normalised imbalance direction), not in a hand-built cost.
  Keep it EFE-based; do not reintroduce a hand-crafted scalar cost.
- **Three communication channels.** `communication.py` carries three distinct
  controller→traveller channels (paper Sections 4.3 / Experiment 3):
  - **Cost-offset advisory** (`SignalType`, `build_broadcast`, `Broadcast`):
    a per-route signal folded into the *perceived cost* `zeta_r = TT_r + theta*E_r`.
    Affects route choice only (`begin_day`), never the belief update. Carries the
    `theta` social-internalisation (Experiment 1). MSC/externality use the
    finite-difference re-roll; travel-time/congestion are direct proxies.
  - **Extra observations** (`ObservationSignal` = `ROUTE_CONGESTION` / `SIGNAL_CONTROL`,
    `build_observation_broadcast`, `ObservationBroadcast`; settings BL/CG/SN/CG+SN —
    the **Experiment 3 default story**): travellers natively observe only the route
    they chose; this relays the **true realised** route queue `L_r` (CG) and/or
    green split `phi_r` (SN) of the *non-chosen* routes into the **end-of-day
    belief update** (`population._append_observation_broadcast` → `update_beliefs`
    → a choice-independent fold in `filter.window_step`/`_laplace_iter_step`,
    gated `last_choice != route` so first-hand obs wins — no double counting). It
    **persists** into the smoother (a documented departure from IWAI first-hand-only),
    reaches **all** travellers (NOT gated by compliance), and works with **any**
    controller. Built from the **same-day** realised values. Empty `obs_signals` (BL)
    is bit-identical (masks all-zero → exact no-op).
  - **Controller-belief broadcast** (`BeliefSignal` = `QUEUE_BELIEF` / `SPLIT_PLAN`,
    `build_belief_broadcast`, `BeliefBroadcast`; settings BL/QB/SP/QB+SP — **parked
    for a future "heterogeneity" paper, kept off by default and NOT part of the
    current paper**): the controller shares its **smoother posterior** over the
    upcoming day — its queue belief `N(L_hat, var)` (QB) and/or its planned green
    split (SP) — *before* travellers choose. A *compliant* traveller **fuses** that
    Gaussian into a **copy** of its own posterior at decision time
    (`population._fuse_controller_belief`, reusing `filter._kalman_one_obs`), gated
    by compliance. The fusion is **transient** — it informs the choice but is never
    written back, so the traveller smoother stays **first-hand-only**
    (IWAI-verbatim). Empty `belief_signals` (BL), or zero compliance, is
    bit-identical to no information. (Compliance, being human-level trust toward an
    assistant agent, is likewise deferred to that future paper — not a paper
    experiment here.)
  - The Experiment 3 notebook (`03_information_communication.py`) picks the channel
    via the `nc.comm_mechanism()` dropdown (Disable / Extra observations [default] /
    Belief sharing / Both); the **extra-observations** channel is the paper's
    Experiment-3 story, belief sharing is the off-by-default exploratory option
    (run at compliance=1).

## Conventions

- **Pure simulation, pure plotting.** Simulator returns DataFrames; every plot
  function returns a `matplotlib.figure.Figure` and never calls `plt.show` /
  `plt.savefig`. Notebooks handle display and saving.
- **The controller is an abstraction.** Anything that drives signals implements
  the `Controller` protocol in `control/interface.py` and is built via
  `control/build_controller`. The simulator never special-cases a controller.
- **The traveller smoother (`inference/filter.py`) follows IWAI** and is
  route-agnostic. `theta` / broadcast affect only action selection
  (`inference/efe.py`, `population.begin_day`), never the belief update. The one
  belief-update extension is the optional per-agent VB observation-noise learning
  (`learn_obs_noise`, on by default); with it `False` the smoother is the
  IWAI-verbatim fixed-noise filter (and deterministic, bit-identical).
- **Determinism.** Inference is closed-form; with noise knobs at 0 the pipeline
  is reproducible. RNG streams are spawned from one `SeedSequence`.
- **Notebooks** gate heavy work behind `mo.ui.run_button`; the smoke harness
  exercises the same code path without the marimo runtime. Wrap a *single* long
  run via `run_experiment(..., progress=mo.status.progress_bar)` (its own per-day
  bar). For a **sweep** of experiments, create one fused bar with
  `notebook_io.sweep_progress_bar(n_experiments, sim, title=...)` and pass its
  `.update` as `run_experiment(..., on_step=bar.update)`, so the single bar
  advances per simulated day across all experiments (`k/(N*days)`, real ETA)
  instead of once per finished experiment.
- **HARD RULE — experiment controls come from `notebook_controls.py`.** Every
  experiment notebook builds its parameter panel from `aif_traffic.notebook_controls`
  (`nc.days()`, `nc.theta()`, … as named globals) and renders it with
  `nc.standard_panel({...}, run_btn)`, which fixes the canonical order, grouping,
  labels and one-line descriptions. **Never hand-define an `mo.ui` slider/checkbox
  for the parameter panel inline** — add or change a control (its range/default/
  label/description) once in `notebook_controls.py`, so all four experiments stay
  consistent. A notebook shows only the subset it needs, but each shown control is
  identical across experiments. Placement: `theta` only where the externality
  cost-offset is broadcast (Exp 1, 2); `compliance` where a controller→traveller
  channel is gated (Exp 1–3; the swept variable in Exp 4). The per-notebook
  `day_sel` / `tod_sel` day-inspection sliders are *not* parameter controls and are
  exempt. (Enforced by `tests/test_notebook_controls.py`.)
- **Observation-noise learning (VB) is ON by default** (`CohortSpec.learn_obs_noise`
  / `AIFControllerSpec.learn_obs_noise` default `True`): both smoothers learn their
  observation-noise SD via a conjugate-Gamma variational update. Set `False` (e.g.
  `params.with_learn_obs_noise(False)`) to recover the fixed-noise IWAI-verbatim
  smoother.
- **`explainers.py` is part of the spec.** Each simulation notebook ends with a
  "How the simulation actually works" cell rendered from `notebook_explainer`.
  When you change the per-day loop, the controller's generative model /
  preference / EFE, or the belief update, update `explainers.py` in the same
  commit. The same module also owns the per-chart reading guide (`CHART_GUIDE`,
  `chart_caption`, `NOTEBOOK_CHARTS`, `charts_section`) — see the next rule.
- **HARD RULE — every chart carries a "how to read" caption from
  `explainers.CHART_GUIDE`.** Every figure rendered in an experiment notebook is
  displayed through `notebook_io.figure_block("<plot_fn>", fig)`, which renders a
  **centred** caption-above-figure block; the caption text is generated from the
  chart's `CHART_GUIDE` entry (`{title, what, read, slider}`), and the *same*
  registry generates the end-of-notebook "How to read the charts" section
  (`charts_section`), so the two never drift. **Never write a figure's reading
  guidance inline** — add or change it once in `CHART_GUIDE`, and list the chart
  in `NOTEBOOK_CHARTS[nb_id]`. Captions must be **descriptive**: say *what is on
  screen and how to read it* ("if you see X, that is Y"); they must **not** assert
  the experiment's conclusion or inject analysis. When a chart is governed by the
  `day` / time-of-day inspection slider, set its `slider` field so the automatic
  affordance badge makes clear the view can be re-pointed. Charts whose Y-axis (or
  colour scale) should be comparable across days fix it across the whole run
  (`shared_ylim` / `shared_scale`, default on). Update `CHART_GUIDE` in the same
  commit as any new/changed chart. (Enforced by `tests/test_chart_captions.py`.)

## Run

- Tests: `uv run --extra dev pytest tests/ -q`
- Full-scale characterization reports (slow, opt-in):
  `uv run --extra dev pytest tests/test_narrative_reports.py --runslow -s`
- Headless smoke: `uv run python scripts/smoke_notebooks.py`
- The fast tests and the smoke must pass before committing behaviour-changing
  diffs; read the `--runslow` report narration when a paper claim is in question.

## Behavioural tests (verbal, self-documenting)

Beyond unit tests, this repo keeps **behavioural characterization tests** in
`tests/test_behaviour.py`: they run a small default experiment and assert
*emergent* facts about the coupled two-layer dynamics (e.g. the queue dip at the
demand peak is route diversion; travellers learn the green split only by taking
the intersection). Each test **prints its reasoning** — what it expected, the
observed numbers, and a verdict — so the behaviour can be audited, not just
pass/fail-checked. Read the narration with:

    uv run --extra dev pytest tests/test_behaviour.py -s

**When you discover or change a non-obvious emergent behaviour, add (or update) a
verbal behavioural test for it.** The goal is that a future agent can run pytest,
read the narration, and directly confirm or disconfirm the documented
understanding of how the model behaves. Guidelines:

- Assert robust *qualitative* facts (with generous margins), not brittle exact
  numbers — emergent equilibria shift slightly with parameters/seeds.
- Narrate via `print(...)` (pytest shows it on `-s` and on failure); state the
  expectation, the evidence, and a one-line verdict.
- If a behavioural test starts failing, that is a signal to re-investigate and
  consciously revise the documented understanding — not to silently loosen it.
- **Run the real experiment, not a reduced stand-in.** Behavioural tests use the
  full default configuration — at least 90 days and the default 2000-traveller
  population (no shrunk `n_agents`/horizon) — so the verdicts reflect the model
  as actually used. Share one `run_experiment` via a module-scoped fixture and
  prefer the deterministic (noise-free) path to keep them tractable.
- The **full-scale behavioural characterization modules** themselves
  (`tests/test_behaviour.py`, `tests/test_belief_informing.py`) are now marked
  `@pytest.mark.slow` (module-level `pytestmark`), together with the externality
  re-roll, multi-run sweeps, and the narrative reports. All `slow` tests are
  **skipped unless** you pass `--runslow` (see `tests/conftest.py`), so the fast
  `pytest tests/` and the per-push CI stay quick. Run the heavy tier locally on
  demand with `uv run --extra dev pytest --runslow` — it is deliberately not
  run in CI.
- When the throwaway `/tmp` analysis you wrote to understand a behaviour proves a
  point worth keeping, promote it into `tests/test_behaviour.py`.

### Two flavours of characterization test

Both print their reasoning; they differ in what they assert (this is the classic
*characterization / golden-master* idea — describe what the code actually does so
a human or another agent can read it back):

- **Direction-asserting** (`tests/test_behaviour.py`,
  `tests/test_belief_informing.py`): pin a qualitative fact and assert its
  direction with a generous margin (e.g. "sharing the controller's queue belief
  QB shifts route choice; non-compliant travellers ignore it").
- **Report-style** (`tests/test_narrative_reports.py`): assert only *sanity*
  (finite, in range, not NaN) and **print** whether each paper claim holds —
  `PAPER CLAIMS … / OBSERVED … / consistent | MISMATCH`. The direction is
  deliberately NOT asserted, so a discrepancy between the code's behaviour and
  the paper text gets *surfaced* (for a human to reconcile) rather than asserted
  away or silently encoded. **If a report reads MISMATCH, that is a finding to
  raise against the paper, not a test to "fix".** Use this flavour when you are
  fine with either outcome and want the result reported, not enforced. (As of
  writing, the communication report runs the **extra-observations** settings
  BL/CG/SN/CG+SN and confirms the paper's Experiment-3 narrative: SN gives the
  lowest system cost, CG the sharpest beliefs, and CG+SN does *not* improve on SN.)

## Notebooks

The three experiment notebooks mirror the paper's Experiment section one-to-one
(explainer IDs in `explainers.py` match the filenames):

- `00_introduction.py` — markdown landing page (two-layer model, two
  communication channels, the three experiments).
- `01_social_internalisation.py` — **Experiment 1**: fix the AIF controller,
  sweep `theta in {0,0.25,0.5,0.75,1}`. A single-run section (within-day and
  day-to-day charts: `plot_signal_day`, `plot_green_split_heatmap`,
  `plot_daily_system_cost`, `plot_route_share_over_days`, optional `animate_days`)
  plus a `theta`-sweep overlay (`plot_sweep_metrics`).
- `02_controller_benchmark.py` — **Experiment 2**: runs all four controllers and
  compares them (`plotting/comparison.py`): scalar day-series overlaid
  (`plot_controller_metrics`: system cost, peak queue, green-split variation),
  per-controller green-split heatmaps (`plot_green_split_heatmaps_by_controller`),
  a `controller_summary` table, and an optional faceted gif
  (`animate_controller_comparison`).
- `03_information_communication.py` — **Experiment 3**: fix the AIF controller and
  sweep the **extra-observations** settings BL/CG/SN/CG+SN via
  `with_extra_observations(...)` (the relayed *realised* queue / green split of
  non-chosen routes), overlaying outcomes and belief uncertainty
  (`plot_sweep_metrics`) plus per-day route-choice heatmaps
  (`plot_route_choice_heatmaps`). The dropdown can also run the parked
  belief-sharing channel (QB/SP) for exploration. A fourth notebook
  `04_compliance_robustness.py` sweeps the compliance fraction that gates the
  belief-sharing fusion — **exploratory, parked for a future heterogeneity paper,
  not part of the current paper's experiment set.**

Heavy per-experiment analysis charts (e.g. Experiment 2's theta×controller grid,
Experiment 3's route-choice heatmaps) are scaffolded but not all built yet.

## Scope reminders

- Plotting and `explainers.py` grow only as notebooks actually need them — no
  speculative code.
- Don't over-formalise: design code structure first; keep model math out until
  the paper section settles it.
