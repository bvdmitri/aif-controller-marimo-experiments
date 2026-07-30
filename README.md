# Active Inference for Adaptive Traffic Management

Companion code for the paper

> **Active Inference for Adaptive Traffic Management: A Unified Framework for Route Choice and
> Signal Control**
> Xue Yao, Dmitry Bagaev, Yanan Xin, J.W.C. van Lint, Bert de Vries, Serge P. Hoogendoorn
> *Submitted to Transportation Research Part C.*

The paper models travellers and a traffic signal controller as instances of the *same* Active
Inference agent template, differing only in what their latent states represent, what they prefer, and
the timescale on which they act. Coordination is not imposed by a shared objective or a central
planner; it emerges from each agent repeatedly minimising its own free energy.

**Interactive version:** <https://aif-controller.experiments.bvdmitri.me/> — the notebooks served as a
read-only web app, with hyperparameters exposed as sliders and the simulation re-running live. No
installation needed.

## Quickstart

```bash
uv sync --extra dev
uv pip install -e .

uv run --extra dev pytest tests/ -q          # contract tests
uv run python scripts/smoke_notebooks.py     # headless pipeline smoke (10 smokes)
uv run marimo edit notebooks/00_introduction.py
```

Inference is closed-form and deterministic, so a given configuration and seed reproduces exactly.

## A first experiment

Why the Active Inference controller and not a simpler one? Run all four controllers on the same
network and compare what they cost. This is a deliberately reduced scale so it finishes in seconds:

```python
from dataclasses import replace
from aif_traffic.parameters import (
    Params, SimParams, PopulationParams, CohortSpec,
    FixedTimeControllerSpec, ReactiveControllerSpec,
    AnticipatoryControllerSpec, AIFControllerSpec,
)
from aif_traffic.simulator import run_experiment

controllers = {
    "fixed-time":     FixedTimeControllerSpec(),
    "reactive":       ReactiveControllerSpec(),
    "anticipatory":   AnticipatoryControllerSpec(),
    "active-inference": AIFControllerSpec(),
}

for name, spec in controllers.items():
    params = replace(
        Params.default(),
        sim=SimParams(days=20, burn_in=0, seed=42),
        population=PopulationParams(cohorts=(CohortSpec(n_agents=400),)),
        controller=spec,
    ).with_stationary(True)

    result = run_experiment(params, seeds=[42])
    daily_cost = result.step.groupby("day")["SC"].first()
    print(f"{name:>17}: {daily_cost.tail(5).mean():>9,.0f} veh-min")
```

```
       fixed-time:   226,839 veh-min
         reactive:    93,494 veh-min
     anticipatory:    60,812 veh-min
 active-inference:    59,078 veh-min
```

Fixed-time control cannot serve the peak at all. Reactive feedback reacts to the queue it already has.
The anticipatory controller predicts ahead but optimises a single point estimate of the split, so it
chatters between near-equivalent settings; the Active Inference controller propagates a full belief
over the within-day queue trajectory and settles. At full paper scale (90 days, 2000 travellers) the
same ordering holds and the last two separate further — see `notebooks/02_controller_benchmark.py`.

### A caution on scale

The controller ordering above is robust and reproduces at almost any scale. **The
information-communication result is not.** The paper's finding — that relaying realised queue lengths
(CG) *raises* system cost while relaying the green split (SN) lowers it — depends on having enough
travellers and enough days for the self-fulfilling low-queue equilibrium to form. At 800 travellers
over 45 days the CG effect inverts. Reproduce that experiment at the paper's full scale (90 days,
2000 travellers, seed 42) or not at all; `notebooks/03_information_communication.py` and
`scripts/export_paper_figures.py` both use the full configuration.

## The model

* **Network** — a signalised intersection with a bypass (links 1–7). A–B travellers choose between the
  intersection route `alpha` (1-2-3-4) and the bypass `beta` (1-5-4); a competing C–D stream `gamma`
  (6-7) shares the junction. Links 2 (A–B) and 6 (C–D) are signalised, and their effective capacity is
  the controller's green-time split. Topology and link characteristics follow Taale (2008).
* **Traveller layer** (paper Section 4.2) — decentralised AIF agents, each with a closed-form Gaussian
  belief over `(F, C, L, phi)` per route and Expected Free Energy route choice over predicted travel
  time. A traveller observes only the route it takes, so beliefs about the alternative go stale.
* **Controller layer** (paper Section 4.3) — a pluggable family behind one interface
  (`src/aif_traffic/control/`): `fixed_time`, `reactive`, `anticipatory`, and `aif`. The AIF controller
  infers the whole within-day queue *trajectory* and scores candidate splits by expected free energy.
* **Information relay** (paper Section 4.4) — `src/aif_traffic/communication.py` relays the realised
  conditions of the route a traveller did *not* take, as queue lengths (CG), the green split (SN), or
  both, folded into the end-of-day belief update.

> **Scope note.** The codebase also contains a second, richer channel in which the controller shares
> its *beliefs* rather than observations, gated by a `compliance_fraction` per cohort
> (`build_belief_broadcast`, `BeliefSignal`). That channel and its experiment were **removed from the
> paper**; they are kept here because they work and may be of independent interest. Nothing reported
> in the manuscript uses them.

## Reproducing the paper

```bash
uv run python scripts/export_paper_figures.py            # full paper scale -> paper_figures/
uv run python scripts/export_paper_figures.py --quick    # fast reduced-scale check
```

This writes vector PDFs and an `INDEX.md` into `paper_figures/`; the PDFs drop straight into the LaTeX
`\includegraphics`. The full-scale run takes on the order of tens of minutes.

| Paper item | Produced by |
|---|---|
| Fig. 3 day-to-day learning profiles | `notebooks/01_coordination_mechanism.py` |
| Fig. 4 within-day coupled profiles | `scripts/export_paper_figures.py` (export-only) |
| Fig. 5 belief uncertainty by setting | `notebooks/03_information_communication.py` |
| Fig. 6 queue `L2` heatmaps by setting | `scripts/export_paper_figures.py` (export-only) |
| Fig. 7 system cost by setting | `notebooks/03_information_communication.py` |
| Figs. 8–9 robustness to demand | `notebooks/05_robustness.py` |
| Figs. 10–12 controller comparison | `notebooks/02_controller_benchmark.py` |
| Table 3 belief-level mechanism | `result.cohort` columns, full-scale Exp. 3 run |
| Table 5 controller metrics | `plotting.comparison.controller_summary` |
| Table A.1 hyperparameters | `src/aif_traffic/parameters.py` (the single source of truth) |

Notebooks are `00_introduction`, `01_coordination_mechanism`, `02_controller_benchmark`,
`03_information_communication`, and `05_robustness`. There is deliberately no `04` — that notebook
covered the advisory channel that was cut from the paper.

## Layout

```
src/aif_traffic/
  parameters.py     frozen dataclasses (network table, signal, controller specs, comm)
  network.py        incidence, queue dynamics, signal capacities, within-day loop
  demand.py         A--B and C--D shifted-sine demand
  inference/        traveller AIF: closed-form smoother (filter.py) + EFE (efe.py)
  control/          pluggable controllers + build_controller()
  communication.py  controller -> traveller relay
  simulator.py      coupled two-layer day loop + run_experiment()
  plotting/         pure Figure-returning helpers
notebooks/          00_introduction + experiment notebooks
scripts/            export_paper_figures.py, smoke_notebooks.py
tests/              pytest contract + behavioural characterization tests
```

## Figure styles

Plotting has two styles, switched centrally via `aif_traffic.plotting.apply_style(name)` (the single
style seam, in `plotting/style.py`):

- **`"marimo"`** (default) — the on-screen notebook look, figures widened to the content column.
- **`"paper"`** — publication style (Elsevier `elsarticle` 3p, ~6.72 in text width, vector PDF,
  colourblind- and greyscale-safe palette). The same chart functions render in either style with no
  per-chart edits.

## Tests

```bash
uv run --extra dev pytest tests/ -q            # fast contract tests
uv run --extra dev pytest --runslow -s         # full-scale behavioural characterization
```

The `--runslow` suite is the interesting one for readers of the paper: it asserts the reported
mechanisms rather than just the plumbing. `test_relaying_queue_lengths_backfires` pins the whole
causal chain behind the Experiment-3 result, and `test_peak_demand_queue_dip_is_route_diversion`
establishes that the intersection queue is low at the peak *because* travellers avoid it. Both print a
narrated summary with `-s`.

Per-push CI (`.github/workflows/ci.yml`) runs the fast tests and the notebook smoke.
`paper-figures.yml` (manual dispatch and version tags) renders the paper figures and uploads
`paper_figures/` as a downloadable artifact. The `--runslow` suite is not run in CI.

## Citation

```bibtex
@article{yao2026active,
  title  = {Active Inference for Adaptive Traffic Management: A Unified Framework
            for Route Choice and Signal Control},
  author = {Yao, Xue and Bagaev, Dmitry and Xin, Yanan and van Lint, J.W.C.
            and de Vries, Bert and Hoogendoorn, Serge P.},
  note   = {Submitted to Transportation Research Part C},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
