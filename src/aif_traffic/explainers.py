"""Simulation-mechanics explainers rendered at the end of simulation notebooks.

Centralised prose so the per-notebook "How the simulation actually works" cells
share one source of truth. Right now the repository ships only the markdown
landing page (notebook 00), so there are no per-notebook addenda yet -- this
module grows as the experiment notebooks are added, in lockstep with the code
paths they describe.

Public API mirrors the IWAI companion repo:

* ``notebook_explainer(nb_id)`` -- full markdown for an end-of-notebook cell.
* ``explainer_pointer()``        -- short top-of-notebook pointer.
* ``NOTEBOOK_IDS``               -- the canonical set of simulation-notebook IDs.

Per-chart "how to read" guidance lives in one place too (CLAUDE.md hard rule):

* ``CHART_GUIDE``      -- registry keyed by plotting-function name; each entry is
  ``{title, what, read, slider}`` (``slider`` in ``{None,"day","tod","day+tod"}``).
* ``NOTEBOOK_CHARTS``  -- which charts each notebook shows, in display order.
* ``chart_caption(id)``-- short markdown rendered *next to* a figure (with a slider
  badge when the chart follows a day / time-of-day slider).
* ``charts_section(nb_id)`` -- the "How to read the charts" list appended to the
  end-of-notebook explainer, generated from the same registry so the two never drift.
"""

from __future__ import annotations

NOTEBOOK_IDS: tuple[str, ...] = (
    "social_internalisation",
    "controller_benchmark",
    "information_communication",
    "compliance_robustness",
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
   minimising expected free energy over its belief about route travel times
   (the IWAI rolling-window Gaussian smoother). Each route's latent is
   $(F, C, L, \phi)$: free-flow time, capacity, queue, and -- extending the IWAI
   model -- the **expected green split** $\phi$. On the signalised intersection
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

**Controller$\to$traveller communication channels.** The controller has a
network-wide view and can share it in three distinct ways.

- *Cost-offset advisory* ($\theta$): a per-route signal (e.g. the congestion
  externality $E_r$) is folded into the traveller's **perceived cost**
  $\zeta_r = TT_r + \theta\,E_r$, where $\theta\in[0,1]$ is the traveller's
  social internalisation ($0$ = purely selfish / user equilibrium, $1$ = fully
  cooperative / system optimum). This shifts route *choice* only; it never
  touches the belief update.
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
  **transient** -- it shapes the choice but is never written back into the
  smoother. Requires an AIF controller (only it holds beliefs). At zero
  compliance every QB/SP setting collapses onto BL exactly.
"""

_CONTROLLER = r"""
### The Active Inference signal controller

The controller is the same kind of agent as the travellers, differing only in
its preferred observation -- and, like them, it **learns with a rolling-window
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
becomes *data-driven* — and the learned-noise chart shows how it settles over
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
  Expected Free Energy (risk minus information gain) each control interval --
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
is expected to spread demand more evenly between the intersection and bypass --
especially at the demand peak -- and lower the total system cost.

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

- **BL (baseline)** -- nothing relayed; each traveller keeps its partial view.
- **CG (route congestion)** -- the realised route queue $L_r(d,t)$ of the
  non-chosen route is relayed.
- **SN (signal control)** -- the realised green split $\phi_r(d,t)$ is relayed to
  travellers who did not take the intersection.
- **CG+SN** -- both.

The relayed value is the day's true reading; the smoother folds it in with the
same observation variance the traveller uses for its own first-hand readings,
gated so it only informs the route the traveller did *not* take (its own
experience stays authoritative -- no double counting). This channel reaches
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
equilibrium** -- neither carries an externality/social term, so (unlike the
cost-offset $\theta\,E_r$ channel of Experiment 1) there is no guarantee
that fuller information alone reaches the lowest *system* cost. Better
anticipation can still help by reducing over-/under-reaction; read the
value-of-information question as empirical, not assumed.
"""

_COMPLIANCE = r"""
### Sweeping traveller compliance

This experiment fixes the AIF controller and shares the controller's full belief
(**QB+SP** -- queue belief and planned split) before travellers choose, then
sweeps the **compliance fraction**: the share of travellers that actually fuse
the controller's belief into their decision. The rest ignore it and decide on
their own posterior alone.

At compliance $0$ nobody fuses, so the broadcast is an exact no-op and the
setting is **bit-identical** to the baseline; at compliance $1$ every traveller
anticipates the day using the controller's shared belief. This isolates *who
listens* (the complement of Experiment 3, which fixes full compliance and varies
*what* is shared).

A per-chart "how to read" guide is appended automatically below. The question
this experiment asks is whether the coordination effect of shared anticipation
degrades **gracefully** with compliance -- fading smoothly rather than collapsing
at a cliff.
"""

_ADDENDA: dict[str, str] = {
    "social_internalisation": _CONTROLLER + "\n" + _SOCIAL,
    "controller_benchmark": _COMPARISON,
    "information_communication": _CONTROLLER + "\n" + _COMMUNICATION,
    "compliance_robustness": _CONTROLLER + "\n" + _COMPLIANCE,
}


def notebook_explainer(nb_id: str) -> str:
    if nb_id not in NOTEBOOK_IDS:
        raise KeyError(
            f"Unknown notebook id {nb_id!r}. Known ids: {NOTEBOOK_IDS}. "
            "Add an entry here when you add the corresponding simulation notebook."
        )
    return _SHARED + "\n" + _ADDENDA[nb_id] + "\n" + charts_section(nb_id)


# ---------------------------------------------------------------------------
# Per-chart "how to read" guide -- single source of truth (CLAUDE.md hard rule)
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
        "🎚️ **Shows a single day -- whichever the _inspect day_ slider is set "
        "to.** Move the slider to inspect any other day."
    ),
    "tod": (
        "🎚️ **Shows a single time of day -- whichever the _time of day_ slider "
        "is set to.** Move it to scan across the day."
    ),
    "day+tod": (
        "🎚️ **Shows a single day _and_ time of day -- set by the _inspect day_ "
        "and _time of day_ sliders.** Move either to look elsewhere."
    ),
}

_VALID_SLIDERS = (None, "day", "tod", "day+tod")

CHART_GUIDE: dict[str, dict] = {
    "plot_demand_profile": {
        "title": "Demand profile",
        "slider": None,
        "what": "The two demand streams over the time of day -- A--B (blue) and "
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
        "they receive (the two sum to a fixed total below 1 -- the rest is lost time).",
        "read": "Read the panels together: a queue grows whenever its movement's "
        "green time (capacity) sits below its arrival rate, and drains when it sits "
        "above. The Y-axis is fixed across all days, so a lower curve is a genuinely "
        "lower-queue day rather than a rescaling.",
    },
    "plot_queue_belief_day": {
        "title": "Controller belief vs realised queue",
        "slider": "day",
        "what": "Per link (L_2 top, L_6 bottom): the realised queue (solid) and the "
        "controller's belief -- its smoother posterior mean (dashed) with a +/-1 sigma "
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
        "cost, daily peak total queue L_2+L_6, and daily green-split variation "
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
    "plot_sweep_metrics": {
        "title": "Sweep metrics",
        "slider": None,
        "what": "Four stacked day-series, one line per swept variant: daily system "
        "cost, mean intersection route share, peak total queue, and traveller belief "
        "uncertainty over the intersection travel time.",
        "read": "Each variant is one line (the colour ramp follows the sweep order). "
        "Compare lines within a panel across days; the belief-uncertainty panel "
        "shows how sure travellers are about the intersection travel time.",
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
        "middle and last day on one scale -- e.g. whether queues shrink over the run. "
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
        "what": "Per route (alpha top, beta bottom): the realised within-day "
        "travel time (line) and the travellers' mean predictive-TT belief (dots), "
        "both against the within-day departure minute. A few learning days are "
        "overlaid as a shade gradient (earliest dim, last saturated).",
        "read": "The line is what actually happened; the dots are what travellers "
        "expected. On later (darker) days the dots should sit closer to the line -- "
        "that tightening is the travellers learning the within-day travel-time "
        "profile of the route they take.",
    },
    "plot_belief_reality_queues": {
        "title": "Belief vs realised queue",
        "slider": "day",
        "what": "Per signalised link (L_2 top, L_6 bottom): the realised within-day "
        "queue (solid) and the controller's queue belief (dashed mean + band). On "
        "L_2, the traveller route-alpha queue belief is a dotted line + band with "
        "markers -- each A--B traveller that took the intersection placed at the "
        "minute it meets the queue (its arrival minute), agents in a minute "
        "averaged, the band their across-agent spread. L_6 (exogenous C--D) shows "
        "the controller only.",
        "read": "Where the belief lines/bands sit on the realised curve, that agent "
        "type has a consistent picture of the queue. The traveller line forms a "
        "within-day profile because travellers departing at different minutes meet "
        "and learn different queues; the band width is how much they disagree, and "
        "the markers show which minutes actually had intersection-takers (sparse at "
        "the peak). L_6 shows the controller alone. The Y-axis is fixed across days.",
    },
    "plot_coupled_within_day": {
        "title": "Coupled within-day: flow & green split",
        "slider": None,
        "what": "Top: within-day traveller flow on the intersection route alpha "
        "(blue) and bypass beta (green). Bottom: the controller's green split "
        "phi_2 -- realised (solid) vs planned/believed (dots). A few learning days "
        "are overlaid as a shade gradient.",
        "read": "Read with the travel-time belief chart: it shows the *controller* "
        "half of the coupled decision. Where realised and believed phi_2 diverge, "
        "the controller reacted within the day to queues that differed from its "
        "typical-day belief.",
    },
    "plot_co_adaptation": {
        "title": "Day-to-day co-adaptation",
        "slider": None,
        "what": "Top and middle: heatmaps of the intersection share P_alpha(d,t) "
        "and the green split phi_2(d,t) over (day x time-of-day). Bottom: daily "
        "demand-weighted P_alpha and mean phi_2 (left axis) with total system cost "
        "(right axis).",
        "read": "Scan the heatmaps left-to-right to see the daily patterns settle; "
        "the bottom panel ties route choice and signal control to whether system "
        "cost falls. Together they show the two layers co-adapting over days.",
    },
    "plot_learning_uncertainty": {
        "title": "Learning uncertainty over days",
        "slider": None,
        "what": "Top: traveller posterior SD on the route travel times TT_alpha "
        "(blue) and TT_beta (green). Bottom: the controller's learned queue "
        "observation-noise SD per movement (L_2, L_6) and, on a right axis, its "
        "belief SD on the daily queue-delay (a proxy for system cost).",
        "read": "Falling curves mean the agents are getting surer. The top panel is "
        "traveller uncertainty, the bottom controller uncertainty; compare how fast "
        "each layer's confidence grows over the run.",
    },
    "plot_theta_summary": {
        "title": "theta-sweep performance summary",
        "slider": None,
        "what": "Four panels -- mean and SD of daily system cost, and mean and SD of "
        "the daily peak queue L_2+L_6 -- against social internalisation theta, one "
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
        "animate_days",
        "animate_route_flows",
        "animate_network_state",
        "plot_sweep_metrics",
        "plot_theta_summary",
        "plot_theta_route_choice",
    ),
    "controller_benchmark": (
        "plot_controller_metrics",
        "plot_learned_obs_noise",
        "plot_green_split_heatmaps_by_controller",
        "animate_controller_comparison",
        "plot_controller_theta_grid",
    ),
    "information_communication": (
        "plot_sweep_metrics",
        "plot_within_day_by_setting",
        "plot_day_overview_grid",
        "plot_queue_belief_day",
        "plot_route_choice_heatmaps",
    ),
    "compliance_robustness": (
        "plot_sweep_metrics",
    ),
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
        lines.append(f"- **{g['title']}**{note} -- {g['what']} {g['read']}")
    return "\n".join(lines)
