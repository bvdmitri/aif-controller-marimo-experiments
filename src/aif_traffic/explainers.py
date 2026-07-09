"""Simulation-mechanics explainers rendered at the end of simulation notebooks.

Centralised prose so the per-notebook "How the simulation actually works" cells
share one source of truth. Right now the repository ships only the markdown
landing page (notebook 00), so there are no per-notebook addenda yet; this
module grows as the experiment notebooks are added, in lockstep with the code
paths they describe.

Public API mirrors the IWAI companion repo:

* ``notebook_explainer(nb_id)``: full markdown for an end-of-notebook cell.
* ``explainer_pointer()``       : short top-of-notebook pointer.
* ``NOTEBOOK_IDS``              : the canonical set of simulation-notebook IDs.

Per-chart "how to read" guidance lives in one place too (CLAUDE.md hard rule):

* ``CHART_GUIDE``     : registry keyed by plotting-function name; each entry is
  ``{title, what, read, slider}`` (``slider`` in ``{None,"day","tod","day+tod"}``).
* ``NOTEBOOK_CHARTS`` : which charts each notebook shows, in display order.
* ``chart_caption(id)``-- short markdown rendered *next to* a figure (with a slider
  badge when the chart follows a day / time-of-day slider).
* ``charts_section(nb_id)``: the "How to read the charts" list appended to the
  end-of-notebook explainer, generated from the same registry so the two never drift.
"""

from __future__ import annotations

NOTEBOOK_IDS: tuple[str, ...] = (
    "social_internalisation",
    "controller_benchmark",
    "information_communication",
    "capacity_sensitivity",
    "robustness",
)


def explainer_pointer() -> str:
    return (
        "> **Implementation reference.** Scroll to *How the simulation actually "
        "works* at the end of this notebook for a concise spec of the coupled "
        "traveller and controller loop, against which the code can be verified."
    )




_SHARED = r"""
### How the simulation actually works

The network is a single signalised intersection with a bypass. Each **day** runs
in two stages.

1. **Route choice (once per day).** Every A--B traveller is an Active Inference
   agent that picks the intersection route $\alpha$ or the bypass $\beta$ by
   minimising expected free energy. The outcome it has preferences about is the
   **perceived generalized cost** $\zeta_r = TT_r + \theta\,E_r$ (its predicted
   travel time plus the internalised externality; $\theta=0$ makes it the
   private travel time). The pragmatic term is the divergence of that predicted
   cost from a preference centred on the free-flow ideal; the epistemic term
   rewards resolving uncertainty in the belief (the IWAI rolling-window Gaussian
   smoother) and is independent of $\theta$. Each route's latent is
   $(F, C, L, \phi)$: free-flow time, capacity, queue, and, extending the IWAI
   model, the **expected green split** $\phi$. On the signalised intersection
   route $C$ is the saturation flow and the effective capacity is $\phi\,C$, so
   $TT_\alpha = F + 60\,L/(\phi\,C)$; the bypass has no signal and keeps
   $TT_\beta = F + 60\,L/C$ with $\phi$ inert. The green split is observed
   *directly* only when the intersection is actually chosen (which is what
   separates $\phi$ from the saturation flow), and between observations it
   decays/widens like the other latents. The chosen shares set the route
   inflows; a competing C--D stream $\gamma$ is exogenous.

2. **Within-day control and queues.** Demand enters minute by minute. Every
   *control interval* the signal controller observes the junction queues and
   sets the green split $(\phi_2, \phi_6)$, held until the next decision. Link
   queues follow store-and-forward dynamics
   $L_\ell(t{+}1)=\max\!\big(0,\,L_\ell(t)+\tfrac{\Delta t}{60}\,[\,Q_\ell(t{-}N_\ell)-C_\ell(t)\,]\big)$,
   with the signalised capacities $C_2=\phi_2\bar C_2$, $C_6=\phi_6\bar C_6$.

At the end of the day travellers update their route beliefs from the realised
travel times. Across days the two layers co-adapt through the shared network.

**Stationary environment vs rolling window (default: stationary).** Both layers'
smoothers can run in two modes. With **"assume stationary environment"** on (the
default), they do *continuous filtering*: every observed day is kept (the window
spans the whole run), the prior is not re-widened, and beliefs simply accumulate
evidence, so posteriors **tighten monotonically toward convergence**. This suits a
fixed environment and removes the periodic cost spikes that appear when a rolling
window first drops its oldest day. Turning it **off** recovers the IWAI
**rolling-window** smoother, which keeps only the last *window* days and re-inflates
uncertainty on routes not seen recently, the right choice when the environment is
*non-stationary* (disruptions), at the cost of some forgetting-driven churn. While
stationary is on, the traveller/controller **window sliders have no effect**.

By default travellers also observe a *noisy* trip, a small measurement noise on
the realised travel time / queue / green split, and route choice is a finite
sample, so realised shares jitter day to day. The **"noise-free environment"**
toggle removes all of this (exact observations, deterministic route choice,
identical demand each day) for clean, reproducible convergence; it is off by
default (a realistic run is noisy).

**Controller$\to$traveller communication channels.** The controller has a
network-wide view and can share it in three distinct ways.

- *Cost-offset advisory* ($\theta$): a per-route signal (e.g. the congestion
  externality $E_r$) enters the traveller's **preference** as the internalised
  term of its outcome, the perceived generalized cost
  $\zeta_r = TT_r + \theta\,E_r$, where $\theta\in[0,1]$ is the traveller's
  social internalisation ($0$ = purely selfish / user equilibrium, $1$ = fully
  cooperative / system optimum). It is a *goal* term (it shapes what the agent
  prefers, via the pragmatic EFE term), **not** a distortion of its belief about
  its own trip: it shifts route *choice* only and never enters the belief update
  or the epistemic term.
- *Extra observations* (CG / SN): travellers natively observe only the route
  they took. The controller relays the **true realised** route congestion
  $L_r$ (**CG**) and/or signal green split $\phi_r$ (**SN**) of the routes they
  did *not* take, folded into the end-of-day **belief update** for those routes
  (gated so first-hand experience stays authoritative). This reaches **every**
  traveller and works with **any** controller. The baseline (**BL**) relays
  nothing.
- *Controller-belief broadcast* (QB / SP): the controller has its own
  *belief* (a probability distribution) about the upcoming day. Before
  travellers choose, it shares its forward-predicted queue belief
  $\mathcal N(\hat L,\widehat{\mathrm{var}})$ (**QB**) and/or its planned green
  split $\hat\phi$ (**SP**). A **compliant** traveller *fuses* that Gaussian into
  its own posterior (precision-weighted) to make the route decision; a
  non-compliant traveller uses only its own posterior. The fusion is
  **transient**; it shapes the choice but is never written back into the
  smoother. Requires an AIF controller (only it holds beliefs). At zero
  compliance every QB/SP setting collapses onto BL exactly.
"""

_CONTROLLER = r"""
### The Active Inference signal controller

The controller is the same kind of agent as the travellers, differing only in
its preferred observation, and, like them, it **learns with a rolling-window
Gaussian smoother**. Its latent is the **entire within-day queue trajectory** of
both signalised movements $(L_2(t), L_6(t))_t$ (one big state), which it estimates
from the per-interval queue observations over a window of the last few days, with
a full covariance capturing the temporal correlations. So the macro layer is one
big AIF agent and the micro layer is thousands of tiny ones, under the same
inference scheme. Within the day it acts at each control interval, using its
learned belief as the prior, and selects the split by minimising the expected
free energy

$$G(\phi)=\underbrace{\mathrm{KL}\!\big[q(o^c\mid\phi)\,\|\,\tilde p(o^c)\big]}_{\text{pragmatic (risk)}}
-\underbrace{\mathbb E\big[\mathrm{KL}(q(s^c\mid o^c,\phi)\,\|\,q(s^c\mid\phi))\big]}_{\text{epistemic (info gain)}},\qquad
\tilde p(o^c)=\mathcal N(\mathbf 0,\Sigma^c_{\mathrm{pref}}).$$

The only designed object is the preference $\tilde p(o^c)$ ("prefer empty
queues"); the *low and balanced* goal lives inside $\Sigma^c_{\mathrm{pref}}$
(extra precision $\omega$ along the **unit** capacity-normalised imbalance
direction, so $\omega$ is comparable to the isotropic precision and balance
genuinely matters), **not** in a hand-built cost. For each candidate split it
rolls the queue belief forward one control interval, scores the pragmatic risk,
subtracts the expected information gain, adds a smoothness prior on $\phi$, and
takes the most probable (MAP) split. The epistemic term is **live**: the
controller's detectors sample a movement more accurately the more green it
receives, so the predicted observation precision depends on the split and the
information gain pulls green toward the movement the controller is least certain
about. The travellers' exploration, by contrast, arises because a route is seen
only when chosen; both layers act under one expected-free-energy objective.

**Learning the observation noise (optional).** By default the queue
observation-noise SD $\sigma_{obs}$ is a fixed knob. With *learn observation
noise* on, it is instead **inferred**: a conjugate $\mathrm{Gamma}$ prior on the
precision $\tau=1/\sigma_{obs}^2$, fit by mean-field coordinate-ascent
variational Bayes inside the smoother (the controller learns one scale per
movement; each traveller learns one per observation channel). The split-dependent
structure is kept; only the magnitude is learned. The belief band below then
becomes *data-driven*, and the learned-noise chart shows how it settles over
days. (Off by default, so the fixed-noise model is unchanged.)

A per-chart "how to read" guide is appended automatically below (one entry per
figure shown in this notebook); it also flags which charts follow the day /
time-of-day inspection sliders.
"""

_COMPARISON = r"""
### Comparing the four controllers

The same network and demand are run under four signal controllers, swapping only
`params.controller`:

- **Fixed-time** holds a constant green split; non-adaptive.
- **Reactive (SCOOT-like)** shifts green toward the longer queue each interval.
- **Anticipatory (predictive)** re-optimises each control interval: it
  grid-searches the constant split minimising predicted system cost over a
  rollout horizon from the current queues (receding horizon, point estimate).
- **AIF (proposed)** keeps a belief over the junction queues and minimises the
  Expected Free Energy (risk minus information gain) each control interval,
  the belief-propagated counterpart of the anticipatory controller (Section 4.2).

The *summary table* collects, per controller, mean cost, day-to-day cost
stability ($\mathrm{std}$ of daily cost), mean signal variation, and mean peak
queue. A per-chart "how to read" guide is appended automatically below.
"""

_SOCIAL = r"""
### Sweeping social internalisation $\theta$

This experiment fixes the AIF controller and varies how cooperative the
travellers are. At $\theta=0$ (the default, the user equilibrium) the within-day
and day-to-day charts above show the coupled traveller--controller adaptation.
Sweeping
$\theta\in\{0,0.25,0.5,0.75,1\}$ then traces the spectrum from the **user
equilibrium** ($\theta=0$, travellers minimise only their own travel time) to
the **system optimum** ($\theta=1$, travellers fully internalise the congestion
externality $E_r$ they impose). The summary panels compare route shares, travel
times, queue imbalance, and total system cost across $\theta$: higher $\theta$
is expected to spread demand more evenly between the intersection and bypass,
especially at the demand peak, and lower the total system cost.

Note that $\theta$ enters the perceived cost as $\zeta_r = TT_r + \theta E_r$,
so it only changes behaviour when the externality $E_r$ is actually communicated
(otherwise the offset is $\theta \times 0$ and every $\theta$ coincides). This
experiment therefore broadcasts the externality at full compliance; the
belief-informing CG/SN broadcasts of Experiment 3 are not used here.
"""

_COMMUNICATION = r"""
### What information should travellers receive?

This experiment fixes the controller and a single traveller population, and
varies only **what travellers learn about the network**. The *communication
mechanism* dropdown selects which of two channels is swept.

**Extra observations (the default channel).** A traveller natively has only a
**partial view**: it observes the realised travel time / queue (and, on the
intersection, the green split) of the route it actually took that day, and learns
nothing first-hand about the other route. This channel relays the **true realised
values** of the routes the traveller did *not* take, folded into its end-of-day
belief update (its rolling-window smoother), comparing four settings:

- **BL (baseline)**: nothing relayed; each traveller keeps its partial view.
- **CG (route congestion)**: the realised route queue $L_r(d,t)$ of the
  non-chosen route is relayed.
- **SN (signal control)**: the realised green split $\phi_r(d,t)$ is relayed to
  travellers who did not take the intersection.
- **CG+SN**: both.

The relayed value is the day's true reading; the smoother folds it in with the
same observation variance the traveller uses for its own first-hand readings,
gated so it only informs the route the traveller did *not* take (its own
experience stays authoritative, no double counting). This channel reaches
**every** traveller and works with **any** controller (it does not require the
controller to hold beliefs); it simply lifts each traveller from a partial view
toward a fuller one.

**Belief sharing (the optional channel).** Instead of realised values, the AIF
controller can share its **own forward-predicted belief** before travellers
choose: its queue belief $\mathcal N(\hat L,\widehat{\mathrm{var}})$ (**QB**)
and/or its *planned* green split $\hat\phi$ (**SP**), settings BL/QB/SP/QB+SP. The
controller is itself **one big Active-Inference agent** whose latent is the whole
within-day queue trajectory, estimated by a rolling-window smoother; QB is that
**smoother posterior** (a genuine inference object, not a forward guess). A
*compliant* traveller **fuses** the shared Gaussian into its own posterior at
decision time (precision-weighted: a confident belief pulls harder); the fusion is
transient and never enters the traveller's smoother. Unlike extra observations,
this channel requires an AIF controller and reaches only compliant travellers; it
is studied at full compliance. The **Both** setting runs each channel alone and
combined.

A per-chart "how to read" guide is appended automatically below (one entry per
figure shown in this notebook), including which chart follows the inspect-day
slider.

A caveat the model makes explicit: both channels sharpen each traveller's
*private* travel-time anticipation, which drives behaviour toward the **user
equilibrium**; neither carries an externality/social term, so (unlike the
cost-offset $\theta\,E_r$ channel of Experiment 1) there is no guarantee
that fuller information alone reaches the lowest *system* cost. Better
anticipation can still help by reducing over-/under-reaction; read the
value-of-information question as empirical, not assumed.
"""

_CAPACITY = r"""
### Does internalisation help when the bypass is a bottleneck?

Sweeping social internalisation $\theta$ barely moves system cost at the default
network, because the bypass route $\beta$ is high-capacity spare: diverting off
the intersection is nearly free, so the congestion externality that $\theta$ acts
on lives almost entirely on the intersection and there is little to redistribute.
This experiment fixes the AIF controller (externality advisory on, full
compliance) and sweeps $\theta \in \{0,0.25,0.5,0.75,1.0\}$ across several
**bypass-capacity scales** (link 5 saturation flow $\times\,1.0, 0.5, 0.25$),
throttling the bypass into a real bottleneck so that diverting onto it carries a
genuine social cost.

The twist is a route-choice **cobweb**. The cost-offset advisory is built from a
day's realised state and acted on the *next* day, a one-day-stale feedback. The
**raw** externality broadcasts a *single* per-route value (the marginal social
cost of one extra vehicle) to the whole population, so once the bypass is
congestible every compliant traveller reads the same "the other route is cheaper"
nudge and swings there en masse; the next day that route is overloaded and the
signal flips. Travellers oscillate between the two routes day to day and the
alternating overloads send system cost far *above* the $\theta=0$ baseline, so
$\theta$ appears to backfire. The root cause is that the finite difference
measures the cost of adding *one* vehicle, while broadcasting it identically in
effect moves *thousands* onto one route.

Two levers break the cobweb, both selectable in the notebook:

* **Advisory smoothing $W$.** Travellers act on the mean advisory over the last
  $W$ days rather than only yesterday's. Larger $W$ damps the swing; past a
  threshold (around $W\approx25$ days here) the oscillation collapses and $\theta$
  helps again even with the bypass throttled. $W=1$ is the raw act-on-yesterday
  advisory.
* **Sequential (per-traveller) advisory.** Instead of one shared value, the
  controller builds a per-departure-minute *schedule*: it redistributes the whole
  day's A--B demand *from empty* in $M$ increments, each minute's chunk going to
  its currently-cheaper route, re-computing the marginal social cost as each route
  fills. Travellers read the bin at their stable within-minute *rank*, so early
  ranks are pushed toward the emptier route and later ranks toward the other, and
  the population *splits* toward the system optimum rather than herding. Filling
  from empty is what matters: the total A--B demand is independent of how it was
  split yesterday, so the schedule depends only on that demand, the exogenous
  $C$--$D$ flow and the green splits, making it a stable day-to-day fixed point
  rather than a cobweb. It is also heterogeneous across travellers, so the
  coordinated herd that caused the overshoot is gone; the oscillation collapses
  even at $W=1$. The **sequential seed** selects the starting point: *from empty*
  (rebuild from zero, stable by construction) or *from belief* (start from the
  controller's believed split and reassign the marginal travellers toward the
  balance, posterior-as-prior); both reach the same balanced split, with the
  belief seed making the minimal, least-disruptive move.

A per-chart "how to read" guide is appended automatically below.
"""

_ROBUSTNESS = r"""
### How do the coupled agents behave under different traffic demand?

This experiment fixes the AIF controller (externality advisory on, full
compliance) and re-runs the coupled system at several **traffic-demand scales**,
multiplying the peak A--B and C--D demand by $\{0.8, 1.0, 1.2, 1.4\}$. It asks
whether the two-layer Active Inference framework keeps coordinating rather than
relying on a single fixed operating point as the network fills up.

Two views, one coloured line per demand scale:

* **Within-day.** At a representative day after learning, the intersection-route
  flow $Q_\alpha$, the bypass-route flow $Q_\beta$ and the controller's green
  split $\phi_2$ across the day. As the load grows, more travellers divert to the
  bypass at the peak while the controller allocates more green time to the
  congested approach; the two adapt together rather than independently.
* **Across-day.** The daily route share $P_\alpha$, the daily mean green split
  $\phi_2$, and the daily total system cost (with the controller's cost-belief SD
  on a right axis). Each load should re-settle to its own stable route split and
  signal policy, with cost and controller uncertainty falling together as before.

A per-chart "how to read" guide is appended automatically below.
"""

_ADDENDA: dict[str, str] = {
    "social_internalisation": _CONTROLLER + "\n" + _SOCIAL,
    "controller_benchmark": _COMPARISON,
    "information_communication": _CONTROLLER + "\n" + _COMMUNICATION,
    "capacity_sensitivity": _CONTROLLER + "\n" + _CAPACITY,
    "robustness": _CONTROLLER + "\n" + _ROBUSTNESS,
}


def notebook_explainer(nb_id: str) -> str:
    if nb_id not in NOTEBOOK_IDS:
        raise KeyError(
            f"Unknown notebook id {nb_id!r}. Known ids: {NOTEBOOK_IDS}. "
            "Add an entry here when you add the corresponding simulation notebook."
        )
    return _SHARED + "\n" + _ADDENDA[nb_id] + "\n" + charts_section(nb_id)


# ---------------------------------------------------------------------------
# Per-chart "how to read" guide: single source of truth (CLAUDE.md hard rule)
# ---------------------------------------------------------------------------
#
# Every figure rendered in an experiment notebook is captioned from this registry
# via ``chart_caption`` (rendered next to the figure by
# ``notebook_io.figure_block``) and listed in the end-of-notebook guide via
# ``charts_section``. Keep captions **descriptive**: say what is on screen and how
# to read it ("if you see X, that is Y"); do NOT assert the experiment's
# conclusion. The ``slider`` field drives an automatic affordance badge so it is
# obvious which charts can be re-pointed with a day / time-of-day slider.

_SLIDER_BADGE: dict[str, str] = {
    "day": (
        "🎚️ **Shows a single day, whichever the _inspect day_ slider is set "
        "to.** Move the slider to inspect any other day."
    ),
    "tod": (
        "🎚️ **Shows a single time of day, whichever the _time of day_ slider "
        "is set to.** Move it to scan across the day."
    ),
    "day+tod": (
        "🎚️ **Shows a single day _and_ time of day, set by the _inspect day_ "
        "and _time of day_ sliders.** Move either to look elsewhere."
    ),
}

_VALID_SLIDERS = (None, "day", "tod", "day+tod")

CHART_GUIDE: dict[str, dict] = {
    "plot_demand_profile": {
        "title": "Demand profile",
        "slider": None,
        "what": "The two demand streams over the time of day: A--B (blue) and "
        "the exogenous C--D (orange), in veh/h.",
        "read": "This is the model's input, identical every day; both rise to a "
        "midday peak and A--B is the larger stream. Use the peak's position here as "
        "the reference for when queues build in the within-day charts.",
    },
    "plot_signal_day": {
        "title": "Within-day queues & green split",
        "slider": "day",
        "what": "Top: within-day queues on the two signalised links, L_2 (A--B, "
        "blue) and L_6 (C--D, orange), in veh. Bottom: the green split phi_2, phi_6 "
        "they receive (the two sum to a fixed total below 1; the rest is lost time).",
        "read": "Read the panels together: a queue grows whenever its movement's "
        "green time (capacity) sits below its arrival rate, and drains when it sits "
        "above. The Y-axis is fixed across all days, so a lower curve is a genuinely "
        "lower-queue day rather than a rescaling.",
    },
    "plot_queue_belief_day": {
        "title": "Controller belief vs realised queue",
        "slider": "day",
        "what": "Per link (L_2 top, L_6 bottom): the realised queue (solid) and the "
        "controller's belief: its smoother posterior mean (dashed) with a +/-1 sigma "
        "band.",
        "read": "The belief is the controller's estimate of a typical day after "
        "folding this day into its window. A wide band means it is still uncertain; "
        "the band narrows on later days as the window fills. Where the dashed line "
        "sits away from the solid line, the typical-day estimate differs from this "
        "particular day. The Y-axis is fixed across days.",
    },
    "plot_route_flows": {
        "title": "Per-route traveller flow",
        "slider": "day",
        "what": "Within-day flow (veh/h) on each route: the A--B intersection route "
        "alpha (blue) and bypass beta (green), the exogenous C--D stream gamma "
        "(orange), and total A--B demand (dashed) for reference.",
        "read": "alpha and beta together make up the A--B demand. If alpha dips "
        "while beta rises at some time, travellers shifted from the intersection to "
        "the bypass then. The Y-axis is fixed across days.",
    },
    "plot_network_state": {
        "title": "Network state",
        "slider": "day+tod",
        "what": "The seven-link network as a graph at one day and one time of day; "
        "each link is coloured and labelled by traveller flow or queue (a toggle), "
        "with the green split shown on the two signalised links.",
        "read": "More intense colour means more of the selected metric. The colour "
        "scale is fixed to the global maximum across all days, so snapshots are "
        "comparable both across the time of day and across days.",
    },
    "plot_learned_obs_noise": {
        "title": "Learned observation noise",
        "slider": None,
        "what": "The controller's learned queue observation-noise SD per movement "
        "(L_2 blue, L_6 orange) over days, with the fixed default as a dashed "
        "reference.",
        "read": "These lines only move when observation-noise learning is on; "
        "otherwise they sit on the dashed default. A gap between the two lines means "
        "the controller observes one movement less precisely than the other.",
    },
    "plot_green_split_heatmap": {
        "title": "Green-split / queue heatmap",
        "slider": None,
        "what": "A within-day quantity over (day x time-of-day): the green split "
        "phi_2 by default, or a queue L_2 / L_6, depending on the panel. X = day, "
        "Y = minute, colour = value.",
        "read": "Each column is one day's within-day profile: scan left-to-right to "
        "see how the daily pattern changes over the run, and top-to-bottom for the "
        "within-day shape.",
    },
    "plot_daily_system_cost": {
        "title": "Daily system cost",
        "slider": None,
        "what": "Total daily system cost (summed travel time, veh-min) against day.",
        "read": "One point per day. A downward trend means the coupled system is "
        "settling into a cheaper state; a flat line means it has converged.",
    },
    "plot_route_share_over_days": {
        "title": "Route share over days",
        "slider": None,
        "what": "The demand-weighted share of travellers taking the intersection "
        "route alpha, against day (in [0, 1]).",
        "read": "One point per day. Drift over days shows route choice re-adapting; "
        "a flat line means the split between intersection and bypass has settled.",
    },
    "plot_controller_metrics": {
        "title": "Controller metrics",
        "slider": None,
        "what": "Three stacked day-series, one line per controller: daily system "
        "cost, daily peak total queue L_2+L_5+L_6, and daily green-split variation "
        "sum_t|phi_2(t)-phi_2(t-1)|.",
        "read": "Compare the lines within each panel. Lower cost and lower peak "
        "queue are better, and lower variation means a steadier signal; read the "
        "cost/queue panels together with the variation panel.",
    },
    "plot_green_split_heatmaps_by_controller": {
        "title": "Green split by controller",
        "slider": None,
        "what": "One green-split heatmap per controller (phi_2 over day x "
        "time-of-day) on a shared colour scale.",
        "read": "Columns are controllers, so the policies can be read side by side; "
        "within each, X = day and Y = minute. The shared scale makes intensities "
        "comparable between controllers.",
    },
    "plot_controller_theta_grid": {
        "title": "theta x controller cost grid",
        "slider": None,
        "what": "Heatmap of steady-state mean system cost over (theta x "
        "controller), each cell annotated with its value.",
        "read": "Rows are the social-internalisation theta, columns the controllers, "
        "on a shared colour scale (lighter = cheaper). Read down a column for one "
        "controller across theta, along a row to compare controllers at one theta.",
    },
    "plot_cost_vs_theta_by_capacity": {
        "title": "Cost vs theta, by bypass capacity",
        "slider": None,
        "what": "Steady-state total system cost against social internalisation "
        "theta, one line per bypass-capacity scale (link 5 saturation flow "
        "x1.0/0.5/0.25). Left panel: absolute cost; right panel: cost relative to "
        "that scale's theta=0, so every line starts at 1.0.",
        "read": "On the right panel a line bending BELOW 1.0 means theta lowers "
        "system cost at that capacity; bending ABOVE 1.0 means it backfires (the "
        "advisory cobweb). At full capacity the line is nearly flat (theta "
        "near-inert); throttling with a raw advisory bends it up, a smoothed "
        "advisory (larger W) bends it back down.",
    },
    "plot_sweep_metrics": {
        "title": "Sweep metrics",
        "slider": None,
        "what": "Stacked day-series panels, one line per swept variant: by "
        "default daily system cost, mean intersection route share, peak total "
        "queue, and traveller belief uncertainty over the intersection travel "
        "time (some views swap the last panel for the daily mean green split "
        "phi_2).",
        "read": "Each variant is one line (the colour ramp follows the sweep order). "
        "Compare lines within a panel across days; the belief-uncertainty panel "
        "shows how sure travellers are about the intersection travel time.",
    },
    "plot_belief_sd_sweep": {
        "title": "Belief uncertainty by setting",
        "slider": None,
        "what": "Two stacked day-series, one line per swept variant: (top) the "
        "travellers' daily mean posterior SD on the intersection travel time "
        "TT_alpha, and (bottom) the controller's posterior SD on the total system "
        "travel time TT^tot (its system-cost belief SD, in veh-min).",
        "read": "Falling curves mean that agent layer is getting surer of that "
        "quantity. Compare a variant against the baseline: route-congestion "
        "information should shrink the traveller curve (top), while signal "
        "information should sharpen the controller's total-travel-time belief "
        "(bottom); the direct value-of-information readout.",
    },
    "plot_within_day_communication": {
        "title": "Within-day realised vs belief by setting",
        "slider": "day",
        "what": "A 4x2 grid for one inspected day, one line per communication "
        "setting (full names in the legend). Columns are realised (left) and "
        "belief (right); rows are route-alpha travel time (via L_2), route-beta "
        "travel time (via L_5), and the signalised queues L_2 and L_6. The "
        "travel-time belief is the travellers' mean predictive-TT per departure "
        "minute; the queue belief is the controller's mean. Each row shares its "
        "y-scale between the realised and belief columns.",
        "read": "Read a row left-to-right: where a setting's belief (right) matches "
        "its realised (left), that agent layer holds an accurate picture under that "
        "setting. Comparing colours within a panel shows how the settings shift "
        "both what happens and what is believed. Travellers hold beliefs per route "
        "(alpha/beta), so the two travel-time rows cover the A--B routes, not the "
        "exogenous L_6.",
    },
    "plot_controller_queue_comparison": {
        "title": "Controller day-series: cost & total queue",
        "slider": None,
        "what": "Two day-series panels, one line per controller (full names in "
        "the legend): (a) daily system cost, and (b) the daily total network "
        "queue L_2+L_5+L_6: the within-day mean total queue as a solid line with "
        "the within-day min--max range shaded.",
        "read": "Compare controllers within each panel: lower cost and lower "
        "total queue are better, and a narrower band on (b) means the total queue "
        "varies less within the day.",
    },
    "plot_within_day_queue_by_controller": {
        "title": "Within-day queue by controller",
        "slider": None,
        "what": "One square panel per controller (FT/RF/AC/AIF), sharing a "
        "y-axis: the within-day realised queue on the three critical links L_2 "
        "(A--B intersection), L_5 (A--B bypass) and L_6 (C--D) at a "
        "representative day.",
        "read": "Compare the panels on the common scale: a controller that keeps "
        "all three link queues low and balanced across the day, without a single "
        "movement spiking, is coordinating the competing streams well.",
    },
    "plot_route_choice_heatmaps": {
        "title": "Route-choice heatmaps",
        "slider": None,
        "what": "One heatmap per swept variant: the intersection-route share "
        "P_alpha over (day x time-of-day), on a shared colour scale pinned to [0, 1].",
        "read": "Columns are variants; X = day and Y = minute. The colour shows when "
        "within the day travellers pick the intersection and how that pattern settles "
        "across the run.",
    },
    "plot_day_overview_grid": {
        "title": "Multi-day overview",
        "slider": None,
        "what": "A 4x3 grid: columns are three representative days (first, middle, "
        "last); rows are the queues (L_2,L_6), the green split (phi_2,phi_6), and "
        "belief-vs-realised for L_2 then L_6. The Y-axis is shared across each row.",
        "read": "Scan a row left-to-right to compare the same quantity on the first, "
        "middle and last day on one scale, e.g. whether queues shrink over the run. "
        "The inspect-day slider below drills into any single day.",
    },
    "animate_days": {
        "title": "Within-day animation",
        "slider": None,
        "what": "An animation, one frame per day, of the within-day queues (L_2, "
        "L_6) and the green split; axis limits are fixed across frames.",
        "read": "Play it to watch the within-day profile change day by day; because "
        "the axes are fixed, shrinking curves mean genuinely smaller queues over the "
        "run.",
    },
    "animate_route_flows": {
        "title": "Within-day traveller-flow animation",
        "slider": None,
        "what": "An animation, one frame per day, of the within-day per-route "
        "traveller flow: the A--B total (Q_AB) and its split into the intersection "
        "route (Q_alpha) and the bypass (Q_beta), with the exogenous C--D stream "
        "(Q_CD); axis limits are fixed across frames.",
        "read": "Play it to watch the population redistribute between routes day by "
        "day; because the axes are fixed, a Q_alpha dip with a matching Q_beta rise "
        "means travellers have shifted from the intersection to the bypass.",
    },
    "animate_network_state": {
        "title": "Network-flow animation",
        "slider": "day",
        "what": "An animation, one frame per time of day within the inspected day, "
        "of the seven-link network graph; each link is coloured by traveller flow "
        "(or queue) and labelled with flow, queue and the green split.",
        "read": "Play it to watch the flow (or congestion) wave build up and clear "
        "across the junction over the day; the colour scale is fixed across frames "
        "and days, so brighter links mean genuinely more of the metric. The "
        "inspect-day slider re-points the animation at another day.",
    },
    "animate_controller_comparison": {
        "title": "Controller comparison animation",
        "slider": None,
        "what": "An animation, one frame per day, with one column per controller: "
        "within-day queues (top) and green split (bottom), on shared axes across "
        "frames and controllers.",
        "read": "Play it to compare how the controllers' within-day behaviour "
        "evolves over the run on a common scale.",
    },
    "plot_within_day_tt_vs_belief": {
        "title": "Within-day travel time: realised vs belief",
        "slider": None,
        "what": "A grid with one column per representative day and two rows (route "
        "alpha top, beta bottom). In each panel: the realised within-day travel "
        "time (line) and the travellers' mean predictive-TT belief (dots), against "
        "the within-day departure minute.",
        "read": "The line is what actually happened; the dots are what travellers "
        "expected. Read a row left-to-right (earliest to latest day): the dots "
        "should settle onto the line as the travellers learn the within-day "
        "travel-time profile of the route they take.",
    },
    "plot_belief_reality_queues": {
        "title": "Belief vs realised queue",
        "slider": "day",
        "what": "Rows are the route-carrying links (L_2, L_5, L_6); columns are the "
        "inspected day(s): the paper figure shows the first, middle, and last "
        "recorded day so the beliefs sharpening over days is visible, while the "
        "notebook's day slider re-points a single-day view. Each panel: the "
        "realised within-day queue (solid) and, on the signalised L_2/L_6, the "
        "controller's queue belief (dashed mean + band; it holds no belief over "
        "the unsignalised bypass L_5). On L_2 and L_5, the queue belief of the "
        "traveller route that traverses the link (alpha and beta) is a dotted "
        "line + band with markers: each A--B traveller that took the route "
        "placed at the minute it meets the queue (its arrival minute), agents in "
        "a minute averaged, the band their across-agent spread. L_6 (exogenous "
        "C--D) shows the controller only.",
        "read": "Where the belief lines/bands sit on the realised curve, that agent "
        "type has a consistent picture of the queue. Across columns (days) the "
        "belief closes onto the realised queue as it is learned. The traveller line "
        "forms a within-day profile because travellers departing at different "
        "minutes meet and learn different queues; the band width is how much they "
        "disagree, and the markers show which minutes had takers of that route "
        "(sparse at the peak on L_2). L_6 shows the controller alone. Rows share a "
        "y-axis so the days are directly comparable.",
    },
    "plot_coupled_within_day": {
        "title": "Coupled within-day: flow & green split",
        "slider": None,
        "what": "A grid with one column per representative day and two rows. Top: "
        "within-day traveller flow on the intersection route alpha (blue) and "
        "bypass beta (green). Bottom: the controller's green split phi_2: "
        "realised (solid) vs planned/believed (dots).",
        "read": "Read with the travel-time belief chart: it shows the *controller* "
        "half of the coupled decision, day by day across the columns. Where "
        "realised and believed phi_2 diverge, the controller reacted within the "
        "day to queues that differed from its typical-day belief.",
    },
    "plot_co_adaptation": {
        "title": "Day-to-day co-adaptation",
        "slider": None,
        "what": "A compact grid grouped (a)-(b). (a) side-by-side heatmaps of the "
        "intersection share P_alpha(d,t) and the green split phi_2(d,t) over "
        "(day x time-of-day), with their colourbars on top; (b) total system cost "
        "(full width), with the controller's cost-belief SD as a red dashed line "
        "on a right axis when recorded.",
        "read": "Scan the heatmaps left-to-right to see the daily route-choice and "
        "signal patterns settle, and read (b) to see whether system cost falls "
        "while the controller's uncertainty (red dashed) shrinks. Together they "
        "show the two layers co-adapting over days.",
    },
    "plot_learning_uncertainty": {
        "title": "Learning uncertainty over days",
        "slider": None,
        "what": "Top: traveller posterior SD on the route travel times TT_alpha "
        "(blue) and TT_beta (green). Bottom: the controller's learned queue "
        "observation-noise SD per movement (L_2, L_6). (Its cost-belief SD is "
        "shown in the co-adaptation figure's system-cost panel.)",
        "read": "Falling curves mean the agents are getting surer. The top panel is "
        "traveller uncertainty, the bottom controller uncertainty; compare how fast "
        "each layer's confidence grows over the run.",
    },
    "plot_theta_summary": {
        "title": "theta-sweep performance summary",
        "slider": None,
        "what": "Four panels: mean and SD of daily system cost, and mean and SD of "
        "the daily peak queue L_2+L_5+L_6, against social internalisation theta, one "
        "line per controller (canonical colour), over the last days of each run.",
        "read": "Read each controller's line across theta: a flat line means theta "
        "barely changes that metric for that controller. Comparing controllers "
        "shows whether an adaptive controller absorbs the effect of theta that a "
        "fixed one would expose.",
    },
    "plot_within_day_by_setting": {
        "title": "Within-day belief vs reality by setting",
        "slider": None,
        "what": "One panel per communication setting (BL/CG/SN/CG+SN): the "
        "realised within-day travel time on the intersection route (line) vs the "
        "travellers' mean predictive-TT belief (dots), a few learning days "
        "overlaid as a shade gradient.",
        "read": "Compare panels: the setting whose belief dots sit closest to the "
        "realised line is the one that best resolves travellers' uncertainty about "
        "what they will actually face. Darker (later) days should track the line "
        "more tightly as learning proceeds.",
    },
    "plot_msc_tt_by_route": {
        "title": "Route cost decomposition: TT, MSC, externality",
        "slider": None,
        "what": "Three stacked day-series, one line per traveller route (alpha "
        "intersection, beta bypass): daily mean travel time TT_r, daily mean "
        "marginal social cost MSC_r (finite-difference cost of one extra "
        "vehicle, recorded while the externality advisory is broadcast), and "
        "the raw externality E_r = MSC_r - TT_r (unclipped; the broadcast "
        "clips it at zero).",
        "read": "Compare the two routes within each panel: where the curves "
        "coincide, user equilibrium and system optimum coincide and theta has "
        "no lever; a gap between the routes' MSC or externality is exactly "
        "what the theta advisory can act on. The zero line in the bottom "
        "panel separates routes that impose congestion on others (above) from "
        "ones that do not (below).",
    },
    "plot_msc_vs_theta": {
        "title": "MSC & travel time vs theta",
        "slider": None,
        "what": "A 2x2 grid, one line per controller: columns are the traveller "
        "routes alpha and beta, the top row the steady-state mean daily "
        "marginal social cost MSC_r, the bottom row the mean daily travel time "
        "TT_r, each against social internalisation theta.",
        "read": "Read each controller's line across theta: a flat line means "
        "theta does not move that route's cost for that controller. Comparing "
        "the alpha and beta columns shows whether the two routes' costs "
        "differ at all; if they are alike, UE and SO coincide and the theta "
        "channel has nothing to redistribute.",
    },
    "plot_theta_route_choice": {
        "title": "theta behavioural mechanism",
        "slider": None,
        "what": "Grouped box plots of the daily intersection share P_alpha at each "
        "theta, one box per controller within each theta group (canonical colour), "
        "over the last days of each run.",
        "read": "If the boxes shift as theta grows, route choice responds to social "
        "internalisation; if they barely move, the behavioural response is small. "
        "Compare controllers to see whether the signal policy masks that response.",
    },
    "plot_within_day_by_demand": {
        "title": "Within-day adaptation by demand",
        "slider": None,
        "what": "Three panels at a representative day, one coloured line per "
        "traffic-demand scale: (a) the intersection-route flow Q_alpha, (b) the "
        "bypass-route flow Q_beta, and (c) the controller's green split phi_2, "
        "across the within-day time axis.",
        "read": "Compare the lines within each panel: as the demand scale grows, a "
        "lower Q_alpha with a higher Q_beta at the peak means more travellers "
        "divert to the bypass, and a shifting phi_2 shows the controller "
        "re-allocating green time in step with them.",
    },
    "plot_across_day_by_demand": {
        "title": "Across-day learning by demand",
        "slider": None,
        "what": "Three day-series panels, one coloured line per traffic-demand "
        "scale: (a) the daily route share P_alpha, (b) the daily mean green split "
        "phi_2, and (c) the daily total system cost (left axis) with the "
        "controller's cost-belief SD (dashed, matching colour) on a shared right "
        "axis.",
        "read": "Read each scale's line across days: a route share and green split "
        "that flatten out mean the coupled system re-settled at that load, and on "
        "(c) a falling cost tracked by a falling dashed SD means cost and "
        "controller uncertainty shrink together as before.",
    },
}

# Which charts each notebook shows, in display order. Drives both the generated
# end-of-notebook guide and the enforcing test (tests/test_chart_captions.py).
NOTEBOOK_CHARTS: dict[str, tuple[str, ...]] = {
    "social_internalisation": (
        "plot_demand_profile",
        "plot_day_overview_grid",
        "plot_within_day_tt_vs_belief",
        "plot_coupled_within_day",
        "plot_signal_day",
        "plot_queue_belief_day",
        "plot_belief_reality_queues",
        "plot_learned_obs_noise",
        "plot_learning_uncertainty",
        "plot_route_flows",
        "plot_network_state",
        "plot_green_split_heatmap",
        "plot_daily_system_cost",
        "plot_route_share_over_days",
        "plot_co_adaptation",
        "plot_msc_tt_by_route",
        "plot_msc_vs_theta",
        "animate_days",
        "animate_route_flows",
        "animate_network_state",
        "plot_sweep_metrics",
        "plot_theta_summary",
        "plot_theta_route_choice",
    ),
    "controller_benchmark": (
        "plot_controller_metrics",
        "plot_controller_queue_comparison",
        "plot_within_day_queue_by_controller",
        "plot_learned_obs_noise",
        "plot_green_split_heatmaps_by_controller",
        "animate_controller_comparison",
        "plot_controller_theta_grid",
    ),
    "information_communication": (
        "plot_sweep_metrics",
        "plot_belief_sd_sweep",
        "plot_within_day_by_setting",
        "plot_within_day_communication",
        "plot_day_overview_grid",
        "plot_queue_belief_day",
        "plot_route_choice_heatmaps",
    ),
    "capacity_sensitivity": (
        "plot_cost_vs_theta_by_capacity",
        "plot_sweep_metrics",
    ),
    "robustness": (
        "plot_within_day_by_demand",
        "plot_across_day_by_demand",
    ),
}

# --- summary-table registry (the quantitative companions to the charts) -----
# Parallel to CHART_GUIDE but for the DataFrame summary tables; no slider field
# (tables are static steady-state summaries). Rendered via
# ``notebook_io.table_block`` with a caption from :func:`table_caption`.
TABLE_GUIDE: dict[str, dict] = {
    "run_summary_table": {
        "title": "Run summary",
        "what": "Steady-state values (mean over the last recorded days, with "
        "their day-to-day std) for this single run: system cost, peak queue "
        "L_2+L_5+L_6, intersection share P_alpha, green-split variation, and the "
        "traveller / controller belief SDs.",
        "read": "The numbers behind the day-series charts above: `mean` is the "
        "level the run settles at, `std` how much it still wobbles day to day "
        "(smaller = more converged).",
    },
    "controller_summary": {
        "title": "Controller summary",
        "what": "One row per controller: mean and day-to-day std of the daily "
        "system cost, the green-split variation (mean and std, signal "
        "stability), the mean daily peak queue L_2+L_5+L_6, and the mean daily peak "
        "on the C--D movement L_6.",
        "read": "Compare controllers row by row: lower `mean_SC` is cheaper, "
        "lower `*_signal_variation` is a steadier signal, lower peak-queue "
        "columns mean less congestion. Pairs with the controller-metrics chart.",
    },
    "theta_summary_table": {
        "title": "theta x controller summary",
        "what": "One row per (controller, theta): mean/std of daily system cost, "
        "mean/std of the peak queue L_2+L_5+L_6, and mean/std of the intersection "
        "share P_alpha, over the last recorded days.",
        "read": "Read a controller's rows down theta: if the metrics barely move "
        "with theta, that controller is 'absorbing' the social-internalisation "
        "effect; compare controllers to see which expose it. Pairs with the "
        "theta-sweep charts.",
    },
    "communication_summary_table": {
        "title": "Communication summary",
        "what": "One row per information setting (BL/CG/SN/CG+SN): mean system "
        "cost and its change vs the baseline (%), the traveller belief SD on "
        "TT_alpha and TT_beta (uncertainty), and the mean intersection share.",
        "read": "Lower `mean_SC` / more-negative `dSC_vs_BL_pct` is better; lower "
        "`belief_SD_*` means sharper beliefs. Read against the sweep chart to see "
        "which channel helps cost vs which sharpens beliefs.",
    },
    "communication_cost_table": {
        "title": "Communication system-cost summary",
        "what": "One row per information setting (BL/CG/SN/CG+SN): the average, "
        "best (lowest), worst (highest) and standard deviation of the daily "
        "system cost over the steady-state window.",
        "read": "The numbers behind the day-by-day cost curves: lower `mean_SC` is "
        "cheaper on average, a smaller gap between `best_SC` and `worst_SC` (and a "
        "smaller `std_SC`) is a steadier day-to-day cost.",
    },
    "capacity_theta_summary": {
        "title": "Capacity x theta summary",
        "what": "One row per bypass-capacity scale: system cost at theta=0 and "
        "theta=1 and the change (%), the best theta in the sweep and its cost, and "
        "the day-to-day route-share oscillation (P_alpha std) at theta=1.",
        "read": "A large positive `dSC_pct` alongside a large `Palpha_std_theta1` "
        "is the advisory cobweb (theta backfiring); smoothing the advisory shrinks "
        "both and can turn `dSC_pct` negative (theta helps). `best_theta` is where "
        "cost is lowest for that capacity.",
    },
}

# Which tables each notebook renders, in display order (drives the table test
# and the end-of-notebook guide).
NOTEBOOK_TABLES: dict[str, tuple[str, ...]] = {
    "social_internalisation": ("run_summary_table", "theta_summary_table"),
    "controller_benchmark": ("controller_summary",),
    "information_communication": ("communication_cost_table",
                                  "communication_summary_table"),
    "capacity_sensitivity": ("capacity_theta_summary",),
    "robustness": (),
}

_SLIDER_NOTE: dict[str, str] = {
    "day": " *(follows the inspect-day slider)*",
    "tod": " *(follows the time-of-day slider)*",
    "day+tod": " *(follows the inspect-day and time-of-day sliders)*",
}


def chart_caption(chart_id: str, *, extra: str | None = None) -> str:
    """Short markdown for the caption rendered next to a figure.

    Leads with a slider-affordance badge when the chart follows a day /
    time-of-day slider, then ``**title.** what read``, then any ``extra`` note.
    Raises ``KeyError`` for an unregistered ``chart_id`` (forces registration)."""
    g = CHART_GUIDE[chart_id]
    parts: list[str] = []
    badge = _SLIDER_BADGE.get(g["slider"])
    if badge:
        parts.append("> " + badge)
    parts.append(f"**{g['title']}.** {g['what']} {g['read']}")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def charts_section(nb_id: str) -> str:
    """The "How to read the charts" markdown list for an end-of-notebook cell,
    generated from CHART_GUIDE so it stays in sync with the per-figure captions."""
    ids = NOTEBOOK_CHARTS[nb_id]
    lines = ["### How to read the charts", ""]
    for cid in ids:
        g = CHART_GUIDE[cid]
        note = _SLIDER_NOTE.get(g["slider"], "")
        lines.append(f"- **{g['title']}**{note}: {g['what']} {g['read']}")
    return "\n".join(lines)


def table_caption(table_id: str, *, extra: str | None = None) -> str:
    """Short markdown for the caption rendered above a summary table.

    ``**title.** what read`` then any ``extra`` note. Raises ``KeyError`` for an
    unregistered ``table_id`` (forces registration in ``TABLE_GUIDE``)."""
    g = TABLE_GUIDE[table_id]
    parts = [f"**{g['title']}.** {g['what']} {g['read']}"]
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def tables_section(nb_id: str) -> str:
    """The "Summary tables" markdown list for an end-of-notebook cell, generated
    from TABLE_GUIDE. Empty string when the notebook has no tables."""
    ids = NOTEBOOK_TABLES.get(nb_id, ())
    if not ids:
        return ""
    lines = ["### Summary tables", ""]
    for tid in ids:
        g = TABLE_GUIDE[tid]
        lines.append(f"- **{g['title']}**: {g['what']} {g['read']}")
    return "\n".join(lines)
