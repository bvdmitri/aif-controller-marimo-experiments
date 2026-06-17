"""Frozen parameter dataclasses for the AIF-controller traffic experiments.

This repository studies a signalised-intersection network with a *macro-layer
signal controller* on top of the IWAI decentralised route-choice travellers.

Two timescales, two layers:

* **Micro layer (travellers).** A--B travellers choose between an intersection
  route ``alpha`` (links 1-2-3-4) and a bypass route ``beta`` (links 1-5-4).
  They are Active-Inference agents reusing the IWAI closed-form rolling-window
  Gaussian smoother over a per-route latent ``(F, C, L)`` and Expected Free
  Energy route choice. A competing C--D stream ``gamma`` (links 6-7) shares the
  signalised junction.
* **Macro layer (controller).** A pluggable signal controller allocates the
  green-time split between the two competing signalised movements (link 2 for
  A--B, link 6 for C--D). The controller is an *abstraction* (see
  ``control/``): fixed-time, reactive, anticipatory, or an Active-Inference
  controller can be dropped in and compared. The AIF controller's internal
  model is deliberately left open for now.

Communication: the controller broadcasts an information signal that travellers
may fold into their perceived route cost ``zeta_r = TT_r + theta * E_r``. A
per-cohort ``compliance_fraction`` controls how many travellers actually read
the broadcast (the rest ignore it).

Nothing here commits to a particular AIF-controller formulation; that is
developed later against this structure.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np


# ============================================================================
#  Time
# ============================================================================
@dataclass(frozen=True)
class SimParams:
    seed: int = 42
    days: int = 90
    burn_in: int = 30
    h_min: int = 300
    dt_min: int = 1
    selected_days: tuple[int, ...] = (0, 1, 3, 7, 14, 21, 29)

    @property
    def dt_h(self) -> float:
        return self.dt_min / 60.0

    @property
    def time(self) -> np.ndarray:
        return np.arange(0, self.h_min + self.dt_min, self.dt_min)

    @property
    def K(self) -> int:  # noqa: N802 - keep paper notation
        return len(self.time)


# ============================================================================
#  Network: signalised intersection with a bypass (Problem Formulation, Table 1)
# ============================================================================
@dataclass(frozen=True)
class LinkSpec:
    """A single directed link. Free-flow time ``F_min`` is in minutes,
    capacity ``cbar`` in veh/h (the *nominal* saturation flow; signalised
    links realise only a green-time fraction of this)."""

    link_id: int
    length_m: float
    lanes: int
    cbar: float
    speed_kmh: float
    F_min: float


# Link table from Problem_formulation_v3 (following Taale 2008).
_DEFAULT_LINKS: tuple[LinkSpec, ...] = (
    LinkSpec(1, 999.0, 2, 4000.0, 50.0, 1.2),
    LinkSpec(2, 1570.0, 1, 2000.0, 50.0, 1.9),  # signalised A--B movement
    LinkSpec(3, 571.0, 1, 2000.0, 50.0, 0.7),
    LinkSpec(4, 499.0, 2, 4000.0, 50.0, 0.6),
    LinkSpec(5, 4492.0, 2, 4000.0, 80.0, 3.4),  # bypass
    LinkSpec(6, 941.0, 1, 2000.0, 50.0, 1.1),   # signalised C--D movement
    LinkSpec(7, 553.0, 1, 2000.0, 50.0, 0.7),
)


@dataclass(frozen=True)
class NetworkParams:
    """Link-level intersection network.

    Routes:
      * ``alpha`` -- intersection route A->B via links 1-2-3-4,
      * ``beta``  -- bypass route A->B via links 1-5-4,
      * ``gamma`` -- competing C->D stream via links 6-7 (exogenous demand).

    Links 2 (A--B) and 6 (C--D) are the two competing signalised movements;
    their effective discharge capacity is set by the controller's green-time
    split each control interval.
    """

    links: tuple[LinkSpec, ...] = _DEFAULT_LINKS
    routes: tuple[str, ...] = ("alpha", "beta", "gamma")
    route_links: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: {
            "alpha": (1, 2, 3, 4),
            "beta": (1, 5, 4),
            "gamma": (6, 7),
        }
    )
    # Travellers (A--B OD) choose between these two; gamma is exogenous.
    traveller_routes: tuple[str, str] = ("alpha", "beta")
    signalised_links: tuple[int, int] = (2, 6)  # (A--B movement, C--D movement)

    # ---- derived helpers ------------------------------------------------
    @property
    def link_ids(self) -> tuple[int, ...]:
        return tuple(ls.link_id for ls in self.links)

    def link(self, link_id: int) -> LinkSpec:
        for ls in self.links:
            if ls.link_id == link_id:
                return ls
        raise KeyError(f"No link with id {link_id}.")

    def cbar(self, link_id: int) -> float:
        return self.link(link_id).cbar

    def route_free_flow(self, route: str) -> float:
        """Sum of free-flow times over the route's links (minutes)."""
        return sum(self.link(l).F_min for l in self.route_links[route])

    def route_bottleneck_cap(self, route: str) -> float:
        """Nominal bottleneck capacity (min cbar) over the route's links."""
        return min(self.link(l).cbar for l in self.route_links[route])

    def n_delay(self, dt_min: int) -> Mapping[int, int]:
        """Per-link free-flow propagation delay N_l = floor(F_l / dt)."""
        return {ls.link_id: int(math.floor(ls.F_min / dt_min)) for ls in self.links}


@dataclass(frozen=True)
class SignalParams:
    """Signal timing for the two competing movements (links 2 and 6)."""

    cycle_length_s: float = 60.0
    lost_time_s: float = 6.0
    phi_min: float = 0.1
    """Feasibility floor on each movement's green fraction."""

    @property
    def phi_sat(self) -> float:
        """Total usable green fraction phi_2 + phi_6 = (C_cyc - L_cyc)/C_cyc."""
        return (self.cycle_length_s - self.lost_time_s) / self.cycle_length_s


# ============================================================================
#  Demand: two shifted-sine streams (A--B and C--D)
# ============================================================================
@dataclass(frozen=True)
class DemandParams:
    """Shifted-sine demand for the A--B and C--D movements (veh/h)."""

    d_AB_min: float = 800.0
    d_AB_max: float = 2400.0
    d_CD_min: float = 500.0
    d_CD_max: float = 1500.0


# ============================================================================
#  Travellers (micro layer) -- reuse the IWAI AIF agent, two routes alpha/beta
# ============================================================================
@dataclass(frozen=True)
class CohortSpec:
    r"""One traveller cohort.

    Reuses the IWAI closed-form rolling-window Gaussian smoother over the
    per-route latent ``(F, C, L)`` and EFE route choice. Route index 0 is the
    intersection route ``alpha`` and index 1 is the bypass route ``beta`` (the
    smoother is route-agnostic; only the prior labels change).

    New macro-coupling knobs:

    * ``theta``               -- social internalisation in ``[0, 1]``: the
      fraction of the broadcast congestion externality folded into the
      perceived route cost ``zeta_r = TT_r + theta * E_r``.
    * ``compliance_fraction`` -- fraction of the cohort that actually reads the
      controller broadcast. The rest ignore it (fall back to private TT).
    """

    n_agents: int = 2000
    label: str = "default"

    # Social / communication coupling.
    theta: float = 0.5
    compliance_fraction: float = 1.0

    # EFE preference: p_tilde_r(y) = N(mu_F_r, sigma_pref^2).
    sigma_pref: float = 4.0
    sigma_obs: float = 5.0
    sigma_L_obs: float = 30.0
    gamma: float = 1.0

    # Priors over the per-route latent (F, C, L).
    # alpha = intersection route (lower free-flow, signal-limited capacity);
    # beta  = bypass route (higher free-flow, high capacity).
    F_prior_mu_alpha: float = 4.4
    F_prior_mu_beta: float = 5.2
    F_prior_sigma: float = 1.0

    C_prior_mu_alpha: float = 1000.0
    C_prior_mu_beta: float = 4000.0
    C_prior_sigma_alpha: float = 600.0
    C_prior_sigma_beta: float = 800.0

    L_prior_mu_alpha: float = 50.0
    L_prior_mu_beta: float = 20.0
    L_prior_sigma: float = 100.0

    # Between-window drift SDs (stale-route prior inflation).
    sigma_F_drift: float = 0.2
    sigma_C_drift: float = 150.0
    sigma_L_drift: float = 10.0
    mean_revert_days: float = 60.0

    window_size: int = 10
    n_laplace_iters: int = 3


@dataclass(frozen=True)
class PopulationParams:
    """Configuration of the traveller population (AIF cohorts only)."""

    cohorts: tuple[CohortSpec, ...] = (CohortSpec(),)
    route_share_smooth_window: int = 13

    @property
    def total_agents(self) -> int:
        return sum(c.n_agents for c in self.cohorts)


@dataclass(frozen=True)
class EFEParams:
    """Global EFE-objective weights: G(a) = risk_weight*risk - info_gain_weight*info_gain."""

    risk_weight: float = 1.0
    info_gain_weight: float = 1.0


@dataclass(frozen=True)
class NoiseParams:
    """Stochasticity knobs. Defaults 0 so the simulator is deterministic."""

    obs_noise_sd: float = 0.0
    demand_noise_cv: float = 0.0


# ============================================================================
#  Controller (macro layer) -- a pluggable spec family
# ============================================================================
@dataclass(frozen=True)
class FixedTimeControllerSpec:
    """Non-adaptive: a constant green-time split."""

    phi2_frac: float = 0.5
    """Fraction of phi_sat allocated to the A--B movement (link 2)."""
    control_interval_min: int = 10


@dataclass(frozen=True)
class ReactiveControllerSpec:
    """Traffic-responsive feedback on the queue imbalance L_2 - L_6 (SCOOT-like)."""

    k_L: float = 1.0e-3
    control_interval_min: int = 10


@dataclass(frozen=True)
class AnticipatoryControllerSpec:
    """Predictive: at each control epoch, grid-search the constant split that
    minimises predicted system cost over a rollout horizon (rolling horizon)."""

    horizon_min: int = 20
    phi_grid_size: int = 9
    control_interval_min: int = 10


@dataclass(frozen=True)
class AIFControllerSpec:
    """Active-Inference signal controller.

    The controller keeps a Gaussian belief over the two signalised queues
    ``(L_2, L_6)``, predicts them one control interval ahead under each
    candidate green split, and selects the split by minimising the fixed
    Expected-Free-Energy functional. Its preference is a preferred-observation
    distribution ``N(0, Sigma_pref)`` over the queues ("prefer empty queues"),
    with the low-and-balanced goal encoded in ``Sigma_pref``. See
    ``control/aif_controller.py`` and paper Section 4.2.
    """

    control_interval_min: int = 10
    horizon_min: int = 10
    """Prediction horizon for scoring a candidate split (defaults to one interval)."""
    phi_grid_size: int = 9
    """Number of candidate green splits evaluated each control epoch."""

    # Preference N(0, Sigma_pref) over the queues (the only designed object).
    sigma_pref: float = 20.0
    """Preferred-queue level tolerance (veh): isotropic SD of the preference."""
    omega: float = 0.02
    """Balance precision along the *unit* capacity-normalised imbalance
    direction (veh^-2), directly comparable to the isotropic precision
    ``sigma_pref^-2``. With the defaults the balance term dominates, so the
    'low and balanced' preference genuinely shapes the split. See
    ``aif_controller._build_sigma_pref`` for the convention (and the note that
    paper Table 2 lists a different, un-normalised convention to be reconciled)."""

    # Generative-model noise for the queue belief.
    sigma_obs: float = 5.0
    """Queue observation-noise SD (veh) at the reference (balanced) split. The
    per-movement observation precision scales with the green allocated to that
    movement, which makes the EFE epistemic term action-dependent."""
    sigma_proc: float = 2.0
    """Per-step random-walk process-noise SD on the queue belief (veh)."""

    kappa: float = 1.0       # green-split smoothness (policy-prior) weight
    gamma: float = 4.0       # action precision
    info_gain_weight: float = 1.0
    """Weight on the EFE epistemic (information-gain) term, EFE = risk - lambda*info."""


ControllerSpecLike = (
    "FixedTimeControllerSpec | ReactiveControllerSpec | "
    "AnticipatoryControllerSpec | AIFControllerSpec"
)


# ============================================================================
#  Communication
# ============================================================================
class SignalType(enum.Enum):
    """What the controller broadcasts to travellers.

    All variants are folded into the perceived route cost via a per-route
    externality-like offset; ``build_broadcast`` converts each into that
    common form. Which signal is most effective is an experimental question.
    """

    NONE = "none"
    TRAVEL_TIME = "travel_time"   # \hat{TT}_r
    CONGESTION = "congestion"     # \hat{L}_r (queue)
    EXTERNALITY = "externality"   # \hat{E}_r
    MSC = "msc"                   # \widehat{MSC}_r (marginal social cost)


@dataclass(frozen=True)
class CommunicationSpec:
    """How the controller communicates with travellers."""

    signal_type: SignalType = SignalType.NONE


# ============================================================================
#  Top-level container
# ============================================================================
@dataclass(frozen=True)
class Params:
    """Top-level container bundling every parameter group."""

    sim: SimParams = field(default_factory=SimParams)
    network: NetworkParams = field(default_factory=NetworkParams)
    demand: DemandParams = field(default_factory=DemandParams)
    signal: SignalParams = field(default_factory=SignalParams)
    population: PopulationParams = field(default_factory=PopulationParams)
    efe: EFEParams = field(default_factory=EFEParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    controller: object = field(default_factory=AIFControllerSpec)
    comm: CommunicationSpec = field(default_factory=CommunicationSpec)

    @classmethod
    def default(cls) -> "Params":
        return cls()

    def with_days(self, days: int) -> "Params":
        return replace(self, sim=replace(self.sim, days=days))

    def with_seed(self, seed: int) -> "Params":
        return replace(self, sim=replace(self.sim, seed=seed))

    def with_noise(
        self,
        obs_noise_sd: float | None = None,
        demand_noise_cv: float | None = None,
    ) -> "Params":
        n = self.noise
        if obs_noise_sd is not None:
            n = replace(n, obs_noise_sd=obs_noise_sd)
        if demand_noise_cv is not None:
            n = replace(n, demand_noise_cv=demand_noise_cv)
        return replace(self, noise=n)

    def with_cohorts(self, cohorts: tuple[CohortSpec, ...]) -> "Params":
        return replace(self, population=replace(self.population, cohorts=tuple(cohorts)))

    def with_theta(self, theta: float) -> "Params":
        """Set the same social-internalisation theta on every cohort."""
        cohorts = tuple(replace(c, theta=theta) for c in self.population.cohorts)
        return self.with_cohorts(cohorts)

    def with_compliance(self, fraction: float) -> "Params":
        """Set the same compliance fraction on every cohort."""
        cohorts = tuple(replace(c, compliance_fraction=fraction)
                        for c in self.population.cohorts)
        return self.with_cohorts(cohorts)

    def with_controller(self, spec: object) -> "Params":
        return replace(self, controller=spec)

    def with_comm(self, signal_type: SignalType) -> "Params":
        return replace(self, comm=replace(self.comm, signal_type=signal_type))
