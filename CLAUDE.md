# Project guidance for Claude Code

This repository is the simulator + marimo companion for the paper extending the
IWAI route-choice work with a **macro-layer Active Inference signal
controller** on a signalised-intersection network.

The repo is deliberately **structure-first**: the network, the reused IWAI
traveller model, a pluggable controller abstraction with baselines, and the
communication + compliance mechanism. Two things are intentionally left open
and must NOT be prematurely locked in:

1. **The AIF controller's internal model.** `control/aif_controller.py` is a
   placeholder that delegates to the reactive baseline. Its generative model /
   preferred states / EFE action selection over the green split are an open
   design question developed against the paper's Section 4.2. Do not invent a
   formulation without agreement.
2. **The exact communication signal definitions.** `communication.py` uses cheap
   per-route proxies. The paper-faithful marginal social cost (finite-difference
   re-roll) and externality are a deferred extension; the *mechanism* is what is
   concrete for now.

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
  exercises the same code path without the marimo runtime.

## Run

- Tests: `uv run --extra dev pytest tests/ -q`
- Headless smoke: `uv run python scripts/smoke_notebooks.py`
- Both must pass before committing behaviour-changing diffs.

## Scope reminders

- **Only notebook 00 exists for now.** Do not pre-build experiment notebooks;
  the notebook set is decided later, once the methodology is settled.
- Plotting and `explainers.py` grow only as notebooks actually need them — no
  speculative code.
- Don't over-formalise: design code structure first; keep model math out until
  the paper section settles it.
