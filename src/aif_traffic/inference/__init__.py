"""Closed-form Gaussian inference backend for AIF agents.

Each agent maintains a posterior over the paper's latent generative-model
state :math:`x^{(i)}_{d,k} = (F_{r_k}, C_{r_k}, L_{r_k})` per route,
updated by a **closed-form rolling-window Gaussian smoother**: each day,
the prior is rebuilt with mean carry-forward + Σ-reset (plus stale-route
σ inflation), and the W chosen-route likelihoods are folded in via
iterated linearised Kalman / Laplace updates. F is shared across the
window; L is a Gaussian Markov chain inside the window.

Public entry points:

* :class:`Population` -- agent population with begin_day /
  update_beliefs / snapshot.
* :func:`build_population` -- factory from :class:`PopulationParams`.
* :func:`window_step` -- the underlying closed-form smoother.
* :func:`efe_route_probabilities` -- closed-form EFE action selection.
"""

from .efe import efe_route_probabilities
from .filter import (
    ObsNoisePosterior,
    VariationalState,
    init_variational_state,
    window_step,
)
from .population import Population, build_population

__all__ = [
    "VariationalState",
    "ObsNoisePosterior",
    "init_variational_state",
    "window_step",
    "efe_route_probabilities",
    "Population",
    "build_population",
]
