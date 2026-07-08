# AIF-controller traffic experiments

Interactive companion to the paper extending our IWAI route-choice work with a
**macro-layer Active Inference signal controller**. It studies how the
controller should communicate with the Active Inference travellers, which
information is most useful to share, and what happens when travellers ignore
it.

This repository provides the network, the traveller model (reused from IWAI), a
pluggable controller abstraction with baseline controllers, the **implemented
Active Inference controller**, the communication + compliance mechanism, the
experiment notebooks (`notebooks/00`-`05`), and tests.

## The model

* **Network** — a signalised intersection with a bypass (links 1–7). A–B
  travellers choose between the intersection route `alpha` (1-2-3-4) and the
  bypass `beta` (1-5-4); a competing C–D stream `gamma` (6-7) shares the
  junction. Links 2 (A–B) and 6 (C–D) are signalised; their effective capacity
  is the controller's green-time split.
* **Micro layer (travellers)** — decentralised AIF agents with a closed-form
  rolling-window Gaussian belief over `(F, C, L, phi)` per route (the green-split
  belief `phi` is active on the signalised route) and Expected Free Energy route
  choice. `theta` sets social internalisation; `compliance_fraction` sets how
  many agents read the controller broadcast.
* **Macro layer (controller)** — a pluggable family behind one interface
  (`src/aif_traffic/control/`): `fixed_time`, `reactive`, `anticipatory`, and
  `aif` (the implemented Active Inference controller; see paper Section 4.2).
* **Communication** — `src/aif_traffic/communication.py` builds the broadcast;
  travellers fold it into `zeta_r = TT_r + theta * E_r`.

## Layout

```
src/aif_traffic/
  parameters.py     frozen dataclasses (network table, signal, controller specs, comm)
  network.py        incidence, queue dynamics, signal capacities, within-day loop
  demand.py         A--B and C--D shifted-sine demand
  inference/        traveller AIF: closed-form smoother (filter.py) + EFE (efe.py)
  control/          pluggable controllers + build_controller()
  communication.py  controller -> traveller broadcast
  simulator.py      coupled two-layer day loop + run_experiment()
  plotting/         pure Figure-returning helpers
notebooks/          00_introduction + experiment notebooks 01-05
scripts/            smoke_notebooks.py (headless pipeline smoke)
tests/              pytest contract tests
```

## Getting started

```bash
uv sync --extra dev
uv pip install -e .

uv run --extra dev pytest tests/ -q          # contract tests
uv run python scripts/smoke_notebooks.py     # headless pipeline smoke
uv run marimo edit notebooks/00_introduction.py
```

Inference is closed-form and deterministic, so a given configuration and seed
reproduces exactly.

## Figure styles & paper export

Plotting has two styles, switched centrally via
`aif_traffic.plotting.apply_style(name)` (the single style seam in
`plotting/style.py`):

- **`"marimo"`** (default) — the on-screen notebook look (sans fonts, figures
  widened to the notebook content column).
- **`"paper"`** — publication style for the manuscript (Elsevier `elsarticle`
  3p, Times/serif, ~6.72 in text width, vector PDF, colourblind/greyscale-safe
  palette). The same chart functions render in either style with no per-chart
  edits (widths come from the active style; colours from a style-aware palette).

Render the publication figures locally:

```bash
uv run python scripts/export_paper_figures.py            # full paper scale -> paper_figures/
uv run python scripts/export_paper_figures.py --quick    # fast reduced-scale check
```

This writes vector PDFs (+ PNG previews) and an `INDEX.md` (file → figure) into
`paper_figures/`; drop the PDFs straight into the LaTeX `\includegraphics`.

**CI.** Per-push CI (`.github/workflows/ci.yml`) runs the fast tests + notebook
smoke only. One heavier workflow runs off the critical path:

- **`paper-figures.yml`** (manual `workflow_dispatch` + version tags) renders the
  paper figures and uploads `paper_figures/` as a **downloadable artifact**.

The full-scale behavioural characterization + narrative tests
(`pytest --runslow`, `@pytest.mark.slow`) are not run in CI — run them locally
on demand with `uv run --extra dev pytest --runslow`.
