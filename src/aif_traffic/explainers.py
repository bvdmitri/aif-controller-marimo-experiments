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

**Two controller$\to$traveller communication channels.** The controller has a
network-wide view and can share it in two distinct ways.

- *Cost-offset advisory* ($\theta$): a per-route signal (e.g. the congestion
  externality $E_r$) is folded into the traveller's **perceived cost**
  $\zeta_r = TT_r + \theta\,E_r$, where $\theta\in[0,1]$ is the traveller's
  social internalisation ($0$ = purely selfish / user equilibrium, $1$ = fully
  cooperative / system optimum). This shifts route *choice* only; it never
  touches the belief update.
- *Controller-belief broadcast* (QB / SP): the controller has its own
  *belief* (a probability distribution) about the upcoming day. Before
  travellers choose, it shares its forward-predicted queue belief
  $\mathcal N(\hat L,\widehat{\mathrm{var}})$ (**QB**) and/or its planned green
  split $\hat\phi$ (**SP**). A **compliant** traveller *fuses* that Gaussian into
  its own posterior (precision-weighted) to make the route decision; a
  non-compliant traveller uses only its own posterior. The fusion is
  **transient** -- it shapes the choice but is never written back into the
  smoother, so the traveller's first-hand belief is untouched. The baseline
  (**BL**) shares nothing, and at zero compliance every setting collapses onto
  BL exactly.
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

**What the charts show.** *Within-day queues and green split* trace one day's
$L_2,L_6$ and $\phi_2,\phi_6$. *Controller belief vs realised queue* overlays the
controller's learned belief -- its rolling-window smoother posterior over the
within-day queue (mean $\pm 1\sigma$) -- on the day's realised $L_2,L_6$; the
posterior shown for day $N$ is the one *after* folding day $N$ into the window (the
controller's estimate of a typical day given days up to $N$, compared against day
$N$'s single realised sample), and its $\pm 1\sigma$ band narrows on later days as
the window fills. *Per-route traveller flow* shows, for the same
day, how the A--B demand splits between the intersection route $\alpha$ (link 2)
and the bypass $\beta$ (link 5), alongside the exogenous C--D stream $\gamma$
(link 6): when travellers divert away from the congested intersection near the
demand peak, $Q_\alpha$ dips and $Q_\beta$ rises, which relieves $L_2$ there.
The *network-state diagram* draws the seven links as a graph for a chosen day
**and time of day**, colouring and labelling each link by either the traveller
flow or the queue length (a switch); every link label carries both, and the two
signalised links are annotated with the current green split $\phi_2,\phi_6$.
The *green-split heatmap* shows $\phi_2$ over (day $\times$ time-of-day); the
*queue heatmaps* do the same for $L_2,L_6$. The *system cost* and *route share*
curves track day-to-day evolution, and the gif animates the within-day profiles
one day per frame.
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

**What the charts show.** Scalar day-series are overlaid on one chart, one line
per controller: *daily system cost* (total travel time, lower is better), *daily
peak total queue* $L_2+L_6$, and the *green-split variation* $\sum_t|\phi_2(t)-
\phi_2(t{-}1)|$ within a day (how much the signal moves; lower is steadier
operation). The *green-split heatmaps* show $\phi_2$ over (day $\times$
time-of-day), one column per controller, so the policies can be read side by
side. The *summary table* collects mean cost, day-to-day cost stability
($\mathrm{std}$ of daily cost), mean signal variation, and mean peak queue. The
optional gif animates each controller's within-day queues and split, one frame
per day.

A good controller reaches **low cost and low queues** without paying for it with
**erratic signal switching**; read the cost/queue panels together with the
variation panel. Finally, a *$\theta\times$controller grid* re-runs every
controller across the social-internalisation sweep
$\theta\in\{0,0.25,0.5,0.75,1\}$ and shows the steady-state system cost as a
heatmap, so the controller's advantage can be read off across the whole
user-equilibrium-to-system-optimum spectrum, not just at one $\theta$.
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
### What belief should the controller share?

This experiment fixes the AIF controller and a single traveller population, and
varies only **what the controller shares from its own belief** before travellers
choose, comparing four settings:

- **BL (baseline)** -- the controller shares nothing; travellers decide on their
  own posterior alone.
- **QB (queue belief)** -- the controller shares its forward-predicted belief
  over the intersection queue, $\mathcal N(\hat L,\widehat{\mathrm{var}})$.
- **SP (split plan)** -- the controller shares its *planned* green split
  $\hat\phi$ for the upcoming day.
- **QB+SP** -- both.

The controller is itself **one big Active-Inference agent**: its latent is the
whole within-day queue trajectory of both signalised movements, and it estimates
it from the per-interval queue observations with a **rolling-window Gaussian
smoother over the last few days** (the macro analogue of the travellers' window
smoother), with a full covariance capturing the temporal correlations between
intervals. QB is therefore the controller's **smoother posterior** over the queue
(mean + variance, the variance shrinking as the window fills) -- a genuine
inference object, not a forward guess. A compliant traveller **fuses** the shared
Gaussian into its own posterior over the intersection-route latent at decision
time (precision-weighted by the controller's variance: a confident belief pulls
harder). The fusion is transient -- it never enters the *traveller's* smoother.
QB tells travellers what queue to expect; SP lets them anticipate the
intersection's effective capacity, which they cannot observe first-hand.

**What the charts show.** Day-to-day *route share* and *total system cost*
overlay the four settings; *queue evolution on the critical links* checks whether
the shared belief reduces imbalance; *green-split evolution* shows the
controller's policy; a *belief-uncertainty* panel compares the traveller
posterior SD across settings; and a row of *route-choice heatmaps* (one per
setting) shows the intersection share $P_\alpha$ over (day $\times$ time-of-day).
A *controller belief vs realised queue* chart (pick a setting and day) overlays
exactly the posterior the controller broadcasts as QB (mean $\pm 1\sigma$) on the
realised $L_2,L_6$, so what is shared can be read against what actually happened.

A caveat the model makes explicit: sharing the controller's belief sharpens each
traveller's *private* travel-time anticipation, which drives behaviour toward the
**user equilibrium** -- it carries no externality/social term, so (unlike the
cost-offset $\theta\,E_r$ channel of Experiments 1 and 4) there is no guarantee
that QB/SP alone reach the lowest *system* cost. Better anticipation can still
help by reducing over-/under-reaction; read the value-of-information question as
empirical, not assumed.
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

**What the charts show.** The same four overlay panels as the sweep experiments,
one line per compliance level: day-to-day *system cost* (does it change smoothly
as more travellers listen, or jump at a threshold?), *intersection route share*,
*peak queue* $L_2+L_6$, and traveller *belief uncertainty*. The question is
whether the coordination effect of shared anticipation degrades **gracefully**
with compliance -- fading smoothly rather than collapsing at a cliff.
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
    return _SHARED + "\n" + _ADDENDA[nb_id]
