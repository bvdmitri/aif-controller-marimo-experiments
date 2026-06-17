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

NOTEBOOK_IDS: tuple[str, ...] = ("aif_controller",)


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
   (the reused IWAI rolling-window Gaussian smoother). The chosen shares set the
   route inflows; a competing C--D stream $\gamma$ is exogenous.

2. **Within-day control and queues.** Demand enters minute by minute. Every
   *control interval* the signal controller observes the junction queues and
   sets the green split $(\phi_2, \phi_6)$, held until the next decision. Link
   queues follow store-and-forward dynamics
   $L_\ell(t{+}1)=\max\!\big(0,\,L_\ell(t)+\tfrac{\Delta t}{60}\,[\,Q_\ell(t{-}N_\ell)-C_\ell(t)\,]\big)$,
   with the signalised capacities $C_2=\phi_2\bar C_2$, $C_6=\phi_6\bar C_6$.

At the end of the day travellers update their route beliefs from the realised
travel times. Across days the two layers co-adapt through the shared network.
"""

_CONTROLLER = r"""
### The Active Inference signal controller

The controller is the same kind of agent as the travellers, differing only in
its preferred observation. It keeps a Gaussian belief over the two signalised
queues $(L_2, L_6)$ and selects the split by minimising the **fixed** expected
free energy

$$G(\phi)=\underbrace{\mathrm{KL}\!\big[q(o^c\mid\phi)\,\|\,\tilde p(o^c)\big]}_{\text{pragmatic}}
-\underbrace{\text{(epistemic value)}}_{\approx\,0\ \text{here}},\qquad
\tilde p(o^c)=\mathcal N(\mathbf 0,\Sigma^c_{\mathrm{pref}}).$$

The only designed object is the preference $\tilde p(o^c)$ ("prefer empty
queues"); the *low and balanced* goal lives inside $\Sigma^c_{\mathrm{pref}}$
(extra precision $\omega$ along the capacity-normalised imbalance direction),
**not** in a hand-built cost. For each candidate split it rolls the queue belief
forward one control interval, scores the pragmatic risk, adds a smoothness prior
on $\phi$, and takes the most probable (MAP) split. The epistemic term is inert
because the queues are observed every interval at fixed precision, so it cannot
distinguish splits; the travellers, by contrast, explore because a route is seen
only when chosen. That asymmetry is derived, not designed.

**What the charts show.** *Within-day queues and green split* trace one day's
$L_2,L_6$ and $\phi_2,\phi_6$. The *green-split heatmap* shows $\phi_2$ over
(day $\times$ time-of-day); the *queue heatmaps* do the same for $L_2,L_6$. The
*system cost* and *route share* curves track day-to-day evolution, and the gif
animates the within-day profiles one day per frame.
"""

_ADDENDA: dict[str, str] = {
    "aif_controller": _CONTROLLER,
}


def notebook_explainer(nb_id: str) -> str:
    if nb_id not in NOTEBOOK_IDS:
        raise KeyError(
            f"Unknown notebook id {nb_id!r}. Known ids: {NOTEBOOK_IDS}. "
            "Add an entry here when you add the corresponding simulation notebook."
        )
    return _SHARED + "\n" + _ADDENDA[nb_id]
