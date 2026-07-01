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
    theta: float = 0.0
    compliance_fraction: float = 1.0

    # EFE preference: p_tilde_r(y) = N(mu_F_r, sigma_pref^2).
    sigma_pref: float = 4.0
    sigma_obs: float = 5.0
    """Assumed travel-time observation SD (min) -- the smoother likelihood /
    VB-prior centre. Added TT measurement noise is separate (``obs_noise_sd``,
    default 0)."""
    sigma_L_obs: float = 3.0
    """Queue observation SD (veh) -- both the *added* measurement noise (applied
    every day in ``population.update_beliefs``) and the assumed likelihood SD.
    Default ~3 veh ~= 10% of a typical route queue (tens of veh); keep it well
    below the queue magnitude or it swamps the signal."""
    sigma_phi_obs: float = 0.03
    """Green-split observation SD (fraction) -- added measurement noise on the
    directly-sensed split and the assumed likelihood SD. Default 0.03 ~= a few %
    of the split, below the candidate-split grid step (~0.1), so the sensor is
    informative without pretending to be exact."""
    gamma: float = 1.0

    # Priors over the per-route latent (F, C, L, phi).
    # alpha = intersection route (lower free-flow, signal-limited capacity);
    # beta  = bypass route (higher free-flow, high capacity).
    F_prior_mu_alpha: float = 4.4
    F_prior_mu_beta: float = 5.2
    F_prior_sigma: float = 1.0

    # On the signalised route, C is the *saturation flow* (green-independent);
    # effective capacity is phi * C. On the bypass C is the effective capacity
    # (no signal, phi inert). C_prior_mu_alpha * phi_prior_mu_alpha ~ 900 matches
    # the previous effective-capacity prior (1000) for day-0 consistency.
    C_prior_mu_alpha: float = 2000.0
    C_prior_mu_beta: float = 4000.0
    C_prior_sigma_alpha: float = 600.0
    C_prior_sigma_beta: float = 800.0

    L_prior_mu_alpha: float = 50.0
    L_prior_mu_beta: float = 20.0
    L_prior_sigma: float = 100.0

    # Green-split belief phi (fraction in [phi_min, phi_sat]). On the bypass it
    # is inert (never observed, does not enter travel time).
    phi_prior_mu_alpha: float = 0.45
    phi_prior_mu_beta: float = 0.45
    phi_prior_sigma: float = 0.2

    # Between-window drift SDs (stale-route prior inflation).
    sigma_F_drift: float = 0.2
    sigma_C_drift: float = 150.0
    sigma_L_drift: float = 10.0
    sigma_phi_drift: float = 0.02
    mean_revert_days: float = 60.0

    window_size: int = 30
    n_laplace_iters: int = 3

    # -- Assume a stationary environment (continuous filtering) ----------------
    stationary: bool = True
    """When ``True`` (the default) the traveller smoother assumes the environment
    is **stationary** and does *continuous filtering* instead of rolling-window
    forgetting: it keeps the **entire run** in its window (never drops a day),
    fits from day 1, and disables the per-route staleness re-inflation, so the
    posterior accumulates all evidence and tightens toward convergence.
    ``window_size`` (and the between-window drift SDs / ``mean_revert_days``) are
    then ignored. Set ``False`` to recover the rolling ``window_size``-day
    smoother with forgetting (the IWAI non-stationary / disruption setting)."""

    # -- Noise-free (fully deterministic) environment --------------------------
    noise_free: bool = False
    """When ``True`` the environment injects **no** stochastic noise: travellers
    fold in the *exact* realised travel time / queue / green split (no measurement
    noise is added -- note that the queue-observation noise ``sigma_L_obs`` is
    otherwise applied every day regardless of ``NoiseParams.obs_noise_sd``), and
    each agent's route choice is a **deterministic** function of its beliefs
    (a frozen per-agent decision threshold instead of a fresh random draw), so
    finite-population sampling no longer jitters the realised route shares.
    Combined with zero demand noise this makes the whole run smooth and
    reproducible. Off by default. Set via ``Params.with_noise_free``. (The fixed
    ``sigma_obs`` / ``sigma_L_obs`` still parameterise the smoother's *assumed*
    likelihood; only the *added* measurement noise is removed.)"""

    # -- Learn the observation noise (per-agent variational Gamma) -------------
    learn_obs_noise: bool = True
    """When ``True`` (the default), each traveller *learns* its observation-noise
    SD per channel (TT, L, phi) instead of fixing them at ``sigma_obs`` /
    ``sigma_L_obs`` / ``sigma_phi_obs``: a conjugate ``Gamma`` precision posterior
    per agent per channel, fit by mean-field VB interleaved with the smoother's
    Laplace iterations. Those fixed sigmas then only set the (weakly-informative)
    prior centre. Per-agent windows are short (<= ``window_size``), so the shared
    prior provides shrinkage. Set ``False`` to recover the fixed-noise
    IWAI-verbatim smoother (bit-identical and deterministic)."""
    obs_noise_prior_shape: float = 1.0
    """Shape ``a0`` of each channel's ``Gamma(a0, b0)`` precision prior
    (weakly-informative at ``1``); the rate is ``b0 = a0 * sigma_channel^2`` so
    the prior mean precision is centred on the fixed default."""
    obs_noise_vb_iters: int = 8
    """Coordinate-ascent iterations when learning the observation noise (the
    smoother runs ``max(n_laplace_iters, obs_noise_vb_iters)`` iterations)."""


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
    """Environment stochasticity knobs.

    Travellers observe a *noisy* realisation of their trip; a fully deterministic
    run is obtained with ``Params.with_noise_free(True)`` (the notebook toggle),
    which zeros these and removes the queue/green-split measurement noise too.
    """

    obs_noise_sd: float = 0.5
    """Travel-time observation-noise SD (min) added to each traveller's realised
    trip time. Default ~0.5 min ~= 10% of the route free-flow travel time (the
    default network's routes are ~4-6 min free-flow) -- a realistic probe /
    perceived-time measurement error, well below the smoother's assumed
    ``sigma_obs`` so the filter stays stable. Set to 0 (or use
    ``with_noise_free``) for an exact, noise-free trip time."""
    demand_noise_cv: float = 0.0
    """Coefficient of variation of the per-day demand multiplier (lognormal).
    0 = identical demand every day."""


# Named measurement-noise regimes for ``Params.with_noise_regime`` (the notebook
# dropdown): (obs_noise_sd [min], sigma_L_obs [veh], sigma_phi_obs [fraction]).
# "medium" is the default; "low" is half, "high" is double. "off" is handled
# separately (fully deterministic via ``with_noise_free``).
_NOISE_REGIMES: dict[str, tuple[float, float, float]] = {
    "low": (0.25, 1.5, 0.015),
    "medium": (0.5, 3.0, 0.03),
    "high": (1.0, 6.0, 0.06),
}


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
    sigma_phi_plan: float = 0.02
    """SD (green-split fraction) the controller attaches to its *planned* split
    when it broadcasts it to travellers (decision-time belief fusion, Experiment
    3 SPLIT_PLAN). Small: the controller knows its own intended action, but its
    realised split may still drift as it adapts to queues. Used as the fusion
    observation variance ``sigma_phi_plan**2``."""

    # -- Trajectory-state rolling-window smoother (the controller's belief) ----
    # The controller is one big AIF agent whose latent is the within-day queue
    # trajectory of both signalised movements; it estimates that trajectory from
    # the per-interval observations via a rolling-window Gaussian smoother over
    # the last ``controller_window_size`` days (the macro analogue of the
    # travellers' window smoother). The smoothed posterior is what it broadcasts.
    controller_window_size: int = 30
    """Number of past days the controller smooths over (its window ``W``).
    Ignored when ``stationary`` is ``True`` (the window then spans the whole run)."""
    stationary: bool = True
    """When ``True`` (the default) the controller assumes a **stationary**
    environment and does *continuous filtering*: its window spans the **entire
    run** (never drops a day) and any across-day drift (``sigma_proc_day``) is
    forced to zero, so its within-day queue-trajectory posterior accumulates all
    days and tightens toward convergence. ``controller_window_size`` is then
    ignored. Set ``False`` for the rolling ``controller_window_size``-day window
    with forgetting (the non-stationary setting)."""
    controller_state_resolution: str = "minute"
    """Grid for the trajectory latent: ``"minute"`` (one node per within-day
    minute -- a genuinely big state) or ``"epoch"`` (one node per control
    interval, coarser/cheaper; the broadcast is then zero-order-hold expanded)."""
    sigma0: float = 5.0
    """SD (veh) of the start-of-day anchor ``L(0) ~ N(0, sigma0^2)`` that makes
    the random-walk trajectory prior proper."""
    sigma_proc_day: float = 0.0
    """Optional extra per-day random-walk drift SD (veh) inflating the trajectory
    prior across the window, so the smoother can track non-stationarity. ``0``
    disables it (the within-day process noise still couples adjacent nodes)."""

    # -- Learn the observation noise (variational Gamma on precision) ----------
    learn_obs_noise: bool = True
    """When ``True`` (the default), the controller *learns* its queue
    observation-noise scale instead of fixing it at ``sigma_obs``: a conjugate
    ``Gamma`` prior on the precision ``tau = 1/sigma_obs^2`` per movement, fit by
    mean-field coordinate-ascent VB inside the smoother (the split-dependent
    weighting is kept as a known structure; only the scale is learned). The
    learned ``E[sigma_obs^2]`` then feeds both the belief band and the EFE
    epistemic term. Set ``False`` to recover the fixed-noise smoother."""
    obs_noise_prior_shape: float = 1.0
    """Shape ``a0`` of the ``Gamma(a0, b0)`` precision prior (weakly-informative
    at ``1``). The rate ``b0 = a0 * sigma_obs^2`` is derived so the prior mean
    precision is ``1/sigma_obs^2`` (centred on the fixed default)."""
    obs_noise_vb_iters: int = 8
    """Coordinate-ascent iterations for the observation-noise VB (it converges in
    a handful; see ``controller_smoother.window_smoother_vb``)."""


ControllerSpecLike = (
    "FixedTimeControllerSpec | ReactiveControllerSpec | "
    "AnticipatoryControllerSpec | AIFControllerSpec"
)


# ============================================================================
#  Communication
# ============================================================================
class SignalType(enum.Enum):
    """What the controller broadcasts to travellers as a *cost-offset advisory*.

    All variants are folded into the perceived route cost via a per-route
    externality-like offset ``zeta_r = TT_r + theta * value_r``;
    ``build_broadcast`` converts each into that common form. This channel
    affects *action selection only* (the EFE risk term), never the belief
    update. It carries the theta social-internalisation of Experiment 1.

    This is distinct from the belief-informing channel (:class:`BeliefSignal`),
    which feeds observations into the smoother.
    """

    NONE = "none"
    TRAVEL_TIME = "travel_time"   # \hat{TT}_r
    CONGESTION = "congestion"     # \hat{L}_r (queue)
    EXTERNALITY = "externality"   # \hat{E}_r
    MSC = "msc"                   # \widehat{MSC}_r (marginal social cost)


class BeliefSignal(enum.Enum):
    """What the controller shares from its *own belief* with travellers, for
    decision-time fusion (paper Experiment 3 settings BL/QB/SP/QB+SP).

    Unlike :class:`SignalType` (a cost-offset that shifts the perceived cost),
    these are the controller's forward-predicted belief about the upcoming day:
    a **distribution**, not a realised reading. Before travellers choose, a
    *compliant* traveller fuses the controller's Gaussian into its own posterior
    (precision-weighted) for the route-choice decision only; the fusion is
    **transient** -- it never enters the traveller's smoother (see
    ``inference/population.begin_day`` and ``control/aif_controller.forecast``).

    Experiment-3 settings map to subsets of this enum::

        BL    -> frozenset()                  (baseline, no info shared)
        QB    -> {QUEUE_BELIEF}               (share queue belief N(mu_L, var_L))
        SP    -> {SPLIT_PLAN}                 (share the planned green split)
        QB+SP -> {QUEUE_BELIEF, SPLIT_PLAN}   (both)
    """

    QUEUE_BELIEF = "qb"   # controller's predicted queue belief N(mu_L, var_L)
    SPLIT_PLAN = "sp"     # controller's planned green split (with its variance)


class ObservationSignal(enum.Enum):
    """What the controller relays to travellers as an *extra observation* of the
    routes they did *not* take (paper Experiment 3 settings BL/CG/SN/CG+SN).

    Travellers natively have only **partial observation**: each observes the
    realised travel time / queue (and, on the intersection, the green split) of
    the route it actually took that day, and learns nothing first-hand about the
    other route. This channel relays the **true realised values** of the
    non-chosen routes -- route congestion ``L_r(d,t)`` (CG) and/or the signal
    green split ``phi_r(d,t)`` (SN) -- which travellers fold into their
    **end-of-day belief update** (the smoother, see ``inference/population`` and
    ``inference/filter``). Unlike the belief-sharing channel
    (:class:`BeliefSignal`), this carries *realised readings* (not a controller
    forecast), it **persists** into the smoother, it reaches **all** travellers
    (not gated by compliance), and it works with **any** controller (it does not
    require the controller to hold beliefs).

    Experiment-3 settings map to subsets of this enum::

        BL    -> frozenset()                       (baseline, partial observation)
        CG    -> {ROUTE_CONGESTION}                (relay true queue L_r)
        SN    -> {SIGNAL_CONTROL}                  (relay true green split phi_r)
        CG+SN -> {ROUTE_CONGESTION, SIGNAL_CONTROL} (both)
    """

    ROUTE_CONGESTION = "cg"   # true realised queue L_r for non-chosen routes
    SIGNAL_CONTROL = "sn"     # true realised green split phi_r (signalised route)


@dataclass(frozen=True)
class CommunicationSpec:
    """How the controller communicates with travellers.

    Three orthogonal channels that can be combined:

    * ``signal_type`` -- the cost-offset advisory (Experiment 1, theta);
    * ``obs_signals`` -- **extra observations** of the non-chosen routes relayed
      into the traveller's belief update (Experiment 3 default, BL/CG/SN/CG+SN);
    * ``belief_signals`` -- the controller's own belief shared for decision-time
      fusion (Experiment 3 optional, BL/QB/SP/QB+SP). Empty set = baseline
      (nothing shared).
    """

    signal_type: SignalType = SignalType.NONE
    obs_signals: frozenset[ObservationSignal] = frozenset()
    belief_signals: frozenset[BeliefSignal] = frozenset()


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

    def with_window_size(self, window_size: int) -> "Params":
        """Set the same rolling-window smoother length on every cohort."""
        cohorts = tuple(replace(c, window_size=int(window_size))
                        for c in self.population.cohorts)
        return self.with_cohorts(cohorts)

    def with_learn_obs_noise(self, flag: bool = True) -> "Params":
        """Toggle variational observation-noise learning on both layers: every
        cohort's traveller smoother and (if the controller is the AIF one) the
        controller smoother. Off by default everywhere, so this is the single
        switch the experiment notebooks flip."""
        cohorts = tuple(replace(c, learn_obs_noise=bool(flag))
                        for c in self.population.cohorts)
        out = self.with_cohorts(cohorts)
        if isinstance(self.controller, AIFControllerSpec):
            out = replace(out, controller=replace(self.controller,
                                                   learn_obs_noise=bool(flag)))
        return out

    def with_stationary(self, flag: bool = True) -> "Params":
        """Toggle the stationary-environment assumption on both layers: every
        cohort's traveller smoother and (if the controller is the AIF one) the
        controller smoother. When ``True`` (the default) both do continuous
        filtering over the whole run instead of rolling-window forgetting, so
        posteriors accumulate all evidence and converge. This is the single
        switch the experiment notebooks flip."""
        cohorts = tuple(replace(c, stationary=bool(flag))
                        for c in self.population.cohorts)
        out = self.with_cohorts(cohorts)
        if isinstance(self.controller, AIFControllerSpec):
            out = replace(out, controller=replace(self.controller,
                                                   stationary=bool(flag)))
        return out

    def with_noise_regime(self, regime: str) -> "Params":
        """Set the environment noise to a named regime (the notebook dropdown):

        * ``"off"``    -- fully deterministic (delegates to ``with_noise_free``);
        * ``"low"``    -- half the medium measurement noise;
        * ``"medium"`` -- the default measurement noise (TT 0.5 min, queue 3 veh,
          split 0.03);
        * ``"high"``   -- twice the medium measurement noise.

        Regimes scale the *added measurement noise* on the travel-time / queue /
        green-split observations (which is also the smoother's assumed likelihood
        SD); demand stays deterministic. ``"low"/"medium"/"high"`` also clear the
        ``noise_free`` flag so a prior "off" selection is undone."""
        r = str(regime).strip().lower()
        if r == "off":
            return self.with_noise_free(True)
        try:
            obs_sd, sig_L, sig_phi = _NOISE_REGIMES[r]
        except KeyError as exc:  # pragma: no cover - guards a bad dropdown value
            raise ValueError(
                f"unknown noise regime {regime!r}; options: "
                f"off, {', '.join(_NOISE_REGIMES)}"
            ) from exc
        out = self.with_noise_free(False).with_noise(obs_noise_sd=obs_sd)
        cohorts = tuple(
            replace(c, sigma_L_obs=sig_L, sigma_phi_obs=sig_phi)
            for c in out.population.cohorts
        )
        return out.with_cohorts(cohorts)

    def with_noise_free(self, flag: bool = True) -> "Params":
        """Toggle a fully deterministic, noise-free environment. When ``True``
        it zeros the demand and observation noise knobs and marks every cohort
        ``noise_free`` (no added measurement noise, deterministic route choice),
        so the run is smooth and reproducible. Off by default; this is the single
        switch the notebooks flip."""
        out = self
        if flag:
            out = out.with_noise(obs_noise_sd=0.0, demand_noise_cv=0.0)
        cohorts = tuple(replace(c, noise_free=bool(flag))
                        for c in out.population.cohorts)
        return out.with_cohorts(cohorts)

    def with_controller(self, spec: object) -> "Params":
        return replace(self, controller=spec)

    def with_comm(self, signal_type: SignalType) -> "Params":
        return replace(self, comm=replace(self.comm, signal_type=signal_type))

    def with_belief_signals(self, *signals: BeliefSignal) -> "Params":
        """Set which parts of the controller's belief are shared for
        decision-time fusion (Experiment 3 settings).

        ``with_belief_signals()`` (no args) is the baseline BL case;
        ``with_belief_signals(BeliefSignal.QUEUE_BELIEF)`` is QB; pass both for
        QB+SP.
        """
        return replace(self, comm=replace(self.comm, belief_signals=frozenset(signals)))

    def with_extra_observations(self, *signals: ObservationSignal) -> "Params":
        """Set which extra observations of the non-chosen routes are relayed into
        the traveller belief update (Experiment 3 default settings).

        ``with_extra_observations()`` (no args) is the baseline BL case;
        ``with_extra_observations(ObservationSignal.ROUTE_CONGESTION)`` is CG;
        pass both for CG+SN.
        """
        return replace(self, comm=replace(self.comm, obs_signals=frozenset(signals)))
