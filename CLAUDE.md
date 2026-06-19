# Project guidance for Claude Code

This repository is the simulator + marimo companion for the paper extending the
IWAI route-choice work with a **macro-layer Active Inference signal
controller** on a signalised-intersection network.

The repo holds the network, the reused IWAI traveller model, a pluggable
controller abstraction with baselines, the **implemented Active Inference
controller**, the communication + compliance mechanism, and the first
experiment notebook.

- **The AIF controller is implemented** in `control/aif_controller.py` (numpy),
  following the corrected paper Section 4.2. It keeps a Gaussian belief over the
  two signalised queues `(L_2, L_6)` and selects the green split by minimising
  the **fixed** Expected Free Energy: a pragmatic term that is the multivariate
  Gaussian KL from the predicted-queue belief to a preferred observation
  `N(0, Sigma_pref)` ("prefer empty queues"), minus an epistemic term that is
  *inert* here (queues are observed every interval at fixed precision, so it
  cannot distinguish splits). The only designed object is the preference; the
  low-and-balanced goal lives inside `Sigma_pref` (extra precision `omega` along
  the capacity-normalised imbalance direction), not in a hand-built cost. Keep it
  EFE-based: do not reintroduce a hand-crafted scalar cost function.
- **Two communication channels.** `communication.py` carries two distinct
  controller→traveller channels (paper Sections 4.3 / Experiment 3):
  - **Cost-offset advisory** (`SignalType`, `build_broadcast`, `Broadcast`):
    a per-route signal folded into the *perceived cost* `zeta_r = TT_r + theta*E_r`.
    Affects route choice only (`begin_day`), never the belief update. Carries the
    `theta` social-internalisation (Experiment 1). MSC/externality use the
    finite-difference re-roll; travel-time/congestion are direct proxies.
  - **Belief-informing broadcast** (`BeliefSignal`, `build_belief_broadcast`,
    `BeliefBroadcast`; settings BL/CG/SN/CG+SN): the controller broadcasts the
    route queue `L_hat_r` (CG) and/or green split `phi_hat_r` (SN), which
    *compliant* travellers fold into the smoother as observations of routes they
    did NOT take. This **intentionally enters the belief update** — a documented
    departure from "the smoother is reused verbatim / broadcast affects only
    action selection" (see `filter.py`, `population.py`). The gate
    `complies * (last_choice != route)` keeps first-hand observations
    authoritative (no double counting); empty `belief_signals` (BL) is bit-identical
    to no information.

## Conventions

- **Pure simulation, pure plotting.** Simulator returns DataFrames; every plot
  function returns a `matplotlib.figure.Figure` and never calls `plt.show` /
  `plt.savefig`. Notebooks handle display and saving.
- **The controller is an abstraction.** Anything that drives signals implements
  the `Controller` protocol in `control/interface.py` and is built via
  `control/build_controller`. The simulator never special-cases a controller.
- **The traveller smoother (`inference/filter.py`) is reused verbatim** from
  IWAI and is route-agnostic. `theta` / broadcast affect only action selection
  (`inference/efe.py`, `population.begin_day`), never the belief update.
- **Determinism.** Inference is closed-form; with noise knobs at 0 the pipeline
  is reproducible. RNG streams are spawned from one `SeedSequence`.
- **Notebooks** gate heavy work behind `mo.ui.run_button`; the smoke harness
  exercises the same code path without the marimo runtime. Wrap long runs via
  `run_experiment(..., progress=mo.status.progress_bar)`, and pair each slider
  with a one-line explanation (the `_row(widget, desc)` pattern).
- **`explainers.py` is part of the spec.** Each simulation notebook ends with a
  "How the simulation actually works" cell rendered from `notebook_explainer`.
  When you change the per-day loop, the controller's generative model /
  preference / EFE, or the belief update, update `explainers.py` in the same
  commit.

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
- Tests that are genuinely expensive at full scale (the externality re-roll, or
  multi-run sweeps) are marked `@pytest.mark.slow` and **skipped unless** you pass
  `--runslow` (see `tests/conftest.py`). The fast `pytest tests/` still runs the
  in-suite behavioural tests; the full-scale reports are run on demand.
- When the throwaway `/tmp` analysis you wrote to understand a behaviour proves a
  point worth keeping, promote it into `tests/test_behaviour.py`.

### Two flavours of characterization test

Both print their reasoning; they differ in what they assert (this is the classic
*characterization / golden-master* idea — describe what the code actually does so
a human or another agent can read it back):

- **Direction-asserting** (`tests/test_behaviour.py`,
  `tests/test_belief_informing.py`): pin a qualitative fact and assert its
  direction with a generous margin (e.g. "CG lowers unchosen-route uncertainty").
- **Report-style** (`tests/test_narrative_reports.py`): assert only *sanity*
  (finite, in range, not NaN) and **print** whether each paper claim holds —
  `PAPER CLAIMS … / OBSERVED … / consistent | MISMATCH`. The direction is
  deliberately NOT asserted, so a discrepancy between the code's behaviour and
  the paper text gets *surfaced* (for a human to reconcile) rather than asserted
  away or silently encoded. **If a report reads MISMATCH, that is a finding to
  raise against the paper, not a test to "fix".** Use this flavour when you are
  fine with either outcome and want the result reported, not enforced. (As of
  writing, the communication report flags that CG+SN is *not* the lowest-cost
  setting, contrary to the Experiment-3 narrative — worth a look.)

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
  sweep the belief-informing settings BL/CG/SN/CG+SN via `with_belief_signals(...)`,
  overlaying outcomes and belief uncertainty (`plot_sweep_metrics`). Per-day
  route-choice heatmaps / animations are deferred.

Heavy per-experiment analysis charts (e.g. Experiment 2's theta×controller grid,
Experiment 3's route-choice heatmaps) are scaffolded but not all built yet.

## Scope reminders

- Plotting and `explainers.py` grow only as notebooks actually need them — no
  speculative code.
- Don't over-formalise: design code structure first; keep model math out until
  the paper section settles it.
