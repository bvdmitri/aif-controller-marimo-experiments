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
- **The communication signal definitions stay open.** `communication.py` uses
  cheap per-route proxies. The paper-faithful marginal social cost
  (finite-difference re-roll) and externality are a deferred extension; the
  *mechanism* is what is concrete for now.

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
- Headless smoke: `uv run python scripts/smoke_notebooks.py`
- Both must pass before committing behaviour-changing diffs.

## Notebooks

- `00_introduction.py` — markdown landing page.
- `01_aif_controller.py` — the AIF-controller experiment: sliders (with
  explanations), a progress-bar run, within-day and day-to-day charts
  (`plot_signal_day`, `plot_green_split_heatmap`, `plot_daily_system_cost`,
  `plot_route_share_over_days`), and an optional per-day gif (`animate_days`,
  needs `pillow`).
- `02_controller_comparison.py` — runs all four controllers and compares them
  (`plotting/comparison.py`): scalar day-series overlaid on one chart
  (`plot_controller_metrics`: system cost, peak queue, green-split variation),
  per-controller green-split heatmaps (`plot_green_split_heatmaps_by_controller`),
  a `controller_summary` table, and an optional faceted gif
  (`animate_controller_comparison`). Disruptions / communication sweeps are not
  in it yet.

## Scope reminders

- Plotting and `explainers.py` grow only as notebooks actually need them — no
  speculative code.
- Don't over-formalise: design code structure first; keep model math out until
  the paper section settles it.
