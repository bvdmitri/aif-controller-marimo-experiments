"""Coupled two-layer day loop and experiment driver.

Each day interleaves:

1. **Travellers choose** routes (alpha/beta) via EFE, folding in yesterday's
   controller broadcast (compliant agents only).
2. **Within-day physics + control**: queues evolve while the controller sets
   the green-time split between links 2 (A--B) and 6 (C--D) each control
   interval. See :func:`network.run_within_day`.
3. **System cost** over all routes (including the competing C--D stream).
4. **Broadcast** for the next day is assembled from the realised state.
5. **Belief updates**: travellers update from realised route travel time and a
   route-level queue reading; the controller gets an end-of-day ``observe``
   hook.

With both noise knobs at 0 the whole pipeline is deterministic and the AIF
controller is a placeholder, so the structure can be validated before the real
controller model exists.
"""

from __future__ import annotations

from typing import Callable, Iterable, NamedTuple

import numpy as np
import pandas as pd

from .communication import (
    build_belief_broadcast,
    build_broadcast,
    build_observation_broadcast,
    empty_belief_broadcast,
    empty_broadcast,
    empty_observation_broadcast,
)
from .control import build_controller
from .demand import DemandProfile
from .inference.population import Population, build_population
from .network import route_arrival_queues, run_within_day
from .parameters import Params
from .utils import daily_system_cost


class ExperimentResult(NamedTuple):
    step: pd.DataFrame
    cohort: pd.DataFrame
    controller: pd.DataFrame
    snapshots: dict | None = None


def _draw_demand_factors(days: int, K: int, cv: float, rng: np.random.Generator) -> np.ndarray:
    mu = -0.5 * cv * cv
    return rng.lognormal(mu, cv, size=(days, K))


def _prior_predictive_summary(population: Population) -> dict[str, np.ndarray]:
    mu_y, var_y = population.predictive_moments
    return {
        "mu_alpha": mu_y[:, 0],
        "mu_beta": mu_y[:, 1],
        "sigma_alpha": np.sqrt(var_y[:, 0]),
        "sigma_beta": np.sqrt(var_y[:, 1]),
    }


def simulate_one_day(
    population: Population,
    controller,
    day_index: int,
    params: Params,
    demand: DemandProfile,
    rng_choice: np.random.Generator,
    rng_obs: np.random.Generator | None,
    broadcast_prev,
    belief_broadcast_prev=None,
    demand_factor: np.ndarray | None = None,
) -> dict:
    """Run one coupled day; return per-step arrays and the next-day broadcasts."""
    net, sim, signal = params.network, params.sim, params.signal
    prior_state = _prior_predictive_summary(population)

    population.begin_day(
        params.efe, rng_choice,
        broadcast=broadcast_prev,
        belief_broadcast=belief_broadcast_prev,
    )
    P_alpha = population.aggregate_route_share(
        smooth_window=params.population.route_share_smooth_window,
    )

    d_ab = demand.d_AB if demand_factor is None else demand.d_AB * demand_factor
    Q_alpha = P_alpha * d_ab
    Q_beta = (1.0 - P_alpha) * d_ab
    Q_gamma = np.asarray(demand.d_CD, dtype=float)
    inflow_by_route = {"alpha": Q_alpha, "beta": Q_beta, "gamma": Q_gamma}

    controller.prepare_day({
        "inflow_by_route": inflow_by_route,
        "net": net, "sim": sim, "signal": signal,
        "day": day_index,
    })

    control_interval = int(getattr(params.controller, "control_interval_min", 10))
    queues, tt_route, phi2, phi6 = run_within_day(
        inflow_by_route, controller.decide, control_interval, net, sim, signal,
    )

    SC = daily_system_cost(inflow_by_route, tt_route, sim.dt_h)

    route_q = route_arrival_queues(queues, net, sim)
    # Arrival-aligned intersection green split: the split the traveller meets
    # when discharging through the signalised link, mirroring the forward look
    # used for queue/TT observations (k + N_l).
    sig_ab, _sig_cd = net.signalised_links
    N_ab = net.n_delay(sim.dt_min)[sig_ab]
    k_arr = np.minimum(np.arange(sim.K) + N_ab, sim.K - 1)
    green_obs_alpha = np.asarray(phi2, dtype=float)[k_arr]
    # Extra-observation relay (Experiment 3 default, CG/SN): the controller
    # relays the *true realised* route queues / green split of THIS day so every
    # traveller can fold the non-chosen route into its end-of-day belief update.
    # Empty obs_signals returns an empty broadcast (a no-op in the smoother). The
    # belief-sharing (QB/SP) broadcast is separate and is fused at decision time
    # in begin_day, never here.
    obs_broadcast = (
        build_observation_broadcast(params.comm, queues, phi2, phi6, net, sim)
        if params.comm.obs_signals
        else empty_observation_broadcast()
    )
    population.update_beliefs(
        tt_route["alpha"], tt_route["beta"],
        route_q["alpha"], route_q["beta"],
        green_obs_alpha=green_obs_alpha,
        obs_broadcast=obs_broadcast,
        rng=rng_obs, obs_noise_sd=params.noise.obs_noise_sd,
    )

    # When the EXTERNALITY / MSC advisory is on, the broadcast build already
    # re-rolls the day for the finite-difference marginal social cost; capture
    # the raw per-route MSC so it can be recorded (None otherwise).
    comm_diag: dict = {}
    broadcast_next = build_broadcast(
        params.comm, tt_route, queues, net, sim,
        inflow_by_route=inflow_by_route, phi2=phi2, phi6=phi6,
        out_diagnostics=comm_diag,
    )
    controller.observe({
        "day": day_index, "queues": queues, "tt_route": tt_route,
        "phi2": phi2, "phi6": phi6, "SC": SC,
    })

    # The controller's posterior belief over today's within-day queue trajectory
    # (mean +/- sd per minute, movement 0 = L_2, 1 = L_6), recorded so the
    # belief-vs-realised chart can be drawn per day. ``None`` for controllers
    # with no learned belief (baselines) or before the first observed day.
    belief = controller.belief_trajectory()

    # The controller's *planned / believed* green split for the day, scored from
    # its typical-day belief alone (no within-day reaction), recorded so a
    # realised-vs-believed phi_2 chart can be drawn. ``None`` for controllers
    # with no such plan (baselines) or before the first observed day.
    plan_fn = getattr(controller, "planned_split", None)
    phi2_plan = plan_fn() if callable(plan_fn) else None

    # Controller-belief broadcast for the NEXT day's route choice (Experiment
    # 3/4): the controller forward-predicts tomorrow's queue belief + planned
    # split using today's realised inflows as a persistence forecast, and shares
    # it (one-day lag, mirroring the cost-offset advisory). Skip the (costly)
    # forecast entirely when no belief is being shared.
    if params.comm.belief_signals:
        forecast = controller.forecast(
            {"inflow_by_route": inflow_by_route, "net": net, "sim": sim}
        )
        belief_broadcast_next = build_belief_broadcast(
            params.comm, forecast, net, sim,
        )
    else:
        belief_broadcast_next = empty_belief_broadcast()

    return {
        "day": day_index,
        "P_alpha": P_alpha,
        "Q_alpha": Q_alpha,
        "Q_beta": Q_beta,
        "Q_gamma": Q_gamma,
        "TT_alpha": tt_route["alpha"],
        "TT_beta": tt_route["beta"],
        "TT_gamma": tt_route["gamma"],
        # Per-link queues for every link (L1..L7); the network-state chart needs
        # all of them, the within-day charts use L2/L5/L6.
        **{f"L{lid}": queues[lid] for lid in net.link_ids},
        "phi2": phi2,
        "phi6": phi6,
        "SC": SC,
        "prior": prior_state,
        "broadcast_next": broadcast_next,
        "belief_broadcast_next": belief_broadcast_next,
        # (mu, sd) each (2, K) for movements (L_2, L_6), or None.
        "belief_trajectory": belief,
        # Planned/believed green split phi_2 over the day (length K), or None.
        "phi2_plan": phi2_plan,
        # Raw per-route marginal social cost {route: length-K}, or None when
        # the EXTERNALITY/MSC advisory is off (it is only computed then).
        "msc": comm_diag.get("msc"),
    }


def _cohort_record(seed: int, day_index: int, population: Population,
                   prior_state: dict, controller_signal: str) -> list[dict]:
    records = []
    mu_y, var_y = population.predictive_moments
    sigma_y = np.sqrt(var_y)
    latents = population.latent_summary()

    for cid, c in enumerate(population.cohorts):
        mask = population.cohort_id == cid
        if not mask.any():
            continue
        records.append({
            "seed": seed,
            "day": day_index,
            "cohort_id": cid,
            "cohort_label": c.label,
            "theta": c.theta,
            "compliance_fraction": c.compliance_fraction,
            "n_agents": int(mask.sum()),
            "mu_alpha_prior": float(prior_state["mu_alpha"][mask].mean()),
            "mu_beta_prior": float(prior_state["mu_beta"][mask].mean()),
            "mu_alpha_post": float(mu_y[mask, 0].mean()),
            "mu_beta_post": float(mu_y[mask, 1].mean()),
            "sigma_alpha_post": float(sigma_y[mask, 0].mean()),
            "sigma_beta_post": float(sigma_y[mask, 1].mean()),
            "F_alpha_post": float(latents["F_mean"][mask, 0].mean()),
            "F_beta_post": float(latents["F_mean"][mask, 1].mean()),
            "C_alpha_post": float(latents["C_mean"][mask, 0].mean()),
            "C_beta_post": float(latents["C_mean"][mask, 1].mean()),
            # Traveller *queue* belief L (mean +/- SD) per route -- the
            # IWAI-translated latent, surfaced so belief-vs-realised queue
            # charts can be drawn. Route alpha traverses the signalised link L2.
            "L_alpha_post": float(latents["L_mean"][mask, 0].mean()),
            "L_beta_post": float(latents["L_mean"][mask, 1].mean()),
            "sigma_L_alpha_post": float(latents["L_sd"][mask, 0].mean()),
            "sigma_L_beta_post": float(latents["L_sd"][mask, 1].mean()),
            # Traveller *green-split* belief phi (mean +/- SD) per route.
            "phi_alpha_post": float(latents["phi_mean"][mask, 0].mean()),
            "phi_beta_post": float(latents["phi_mean"][mask, 1].mean()),
            "sigma_phi_alpha_post": float(latents["phi_sd"][mask, 0].mean()),
            "sigma_phi_beta_post": float(latents["phi_sd"][mask, 1].mean()),
            "P_alpha_efe": float(population.last_P_alpha[mask].mean()),
            "frac_chose_alpha": float((population.last_choice[mask] == 0).mean()),
        })
    return records


def run_experiment(
    params: Params,
    seeds: Iterable[int] | None = None,
    progress: bool | Callable = False,
    snapshot_days: Iterable[int] | None = None,
    on_step: Callable[[], None] | None = None,
) -> ExperimentResult:
    """Run the coupled simulator for one or more seeds.

    ``progress`` wraps this run's own per-day iterator in a fresh progress bar
    (single-run notebooks). ``on_step``, by contrast, is a per-day tick callback
    advancing an **externally owned** bar: it is called once per simulated day
    (over every seed), so a sweep can fuse many ``run_experiment`` calls into one
    progress bar spanning ``n_experiments * (burn_in + days)`` ticks (see
    :func:`aif_traffic.notebook_io.sweep_progress_bar`). It is a pure side effect
    and never touches the RNG, so determinism is unaffected.
    """
    if seeds is None:
        seeds = [params.sim.seed]
    seeds = list(seeds)
    snap_set = set(int(d) for d in snapshot_days) if snapshot_days is not None else set()

    step_records: list[dict] = []
    cohort_records: list[dict] = []
    controller_records: list[dict] = []
    snapshots: dict = {}

    for seed in seeds:
        params_seed = params.with_seed(seed)
        net, sim, signal = params_seed.network, params_seed.sim, params_seed.signal
        demand = DemandProfile.from_params(sim, params_seed.demand)

        ss = np.random.SeedSequence(seed)
        construct_ss, choice_ss, obs_ss, demand_ss = ss.spawn(4)
        construct_rng = np.random.default_rng(construct_ss)
        choice_rng = np.random.default_rng(choice_ss)
        obs_rng = np.random.default_rng(obs_ss)
        demand_rng = np.random.default_rng(demand_ss)

        population = build_population(
            params_seed.population, sim, demand, construct_rng,
            route_names=net.traveller_routes, signal=signal,
        )
        controller = build_controller(params_seed.controller, signal, net, sim)

        # Noise-free environment: withhold the observation RNG so travellers fold
        # in the *exact* realised TT / queue / green split (every added
        # measurement noise term in update_beliefs is gated on ``rng is not
        # None``); demand noise is off via cv=0 (see with_noise_free).
        noise_free = bool(getattr(params_seed.population.cohorts[0], "noise_free", False))
        if noise_free:
            obs_rng = None

        cv = params_seed.noise.demand_noise_cv
        burn_in = sim.burn_in
        total_days = burn_in + sim.days
        demand_factors = (
            _draw_demand_factors(total_days, sim.K, cv, demand_rng) if cv > 0 else None
        )

        broadcast_prev = empty_broadcast(net, sim)
        belief_broadcast_prev = empty_belief_broadcast()
        signal_name = params_seed.comm.signal_type.value

        all_iter = range(total_days)
        if callable(progress):
            all_iter = progress(all_iter, total=total_days, title=f"seed {seed}")

        for i in all_iter:
            factor = demand_factors[i] if demand_factors is not None else None
            d = i - burn_in
            out = simulate_one_day(
                population, controller, d, params_seed, demand,
                rng_choice=choice_rng, rng_obs=obs_rng,
                broadcast_prev=broadcast_prev,
                belief_broadcast_prev=belief_broadcast_prev,
                demand_factor=factor,
            )
            broadcast_prev = out["broadcast_next"]
            belief_broadcast_prev = out["belief_broadcast_next"]

            if on_step is not None:
                on_step()

            if i < burn_in:
                continue

            belief = out["belief_trajectory"]
            phi2_plan = out["phi2_plan"]
            msc = out["msc"]
            for k, tau in enumerate(sim.time):
                rec = {
                    "seed": seed, "day": d, "tau": int(tau),
                    "P_alpha": float(out["P_alpha"][k]),
                    "Q_alpha": float(out["Q_alpha"][k]),
                    "Q_beta": float(out["Q_beta"][k]),
                    "Q_gamma": float(out["Q_gamma"][k]),
                    "TT_alpha": float(out["TT_alpha"][k]),
                    "TT_beta": float(out["TT_beta"][k]),
                    "TT_gamma": float(out["TT_gamma"][k]),
                    **{f"L{lid}": float(out[f"L{lid}"][k]) for lid in net.link_ids},
                    "phi2": float(out["phi2"][k]),
                    "phi6": float(out["phi6"][k]),
                    # SC is a daily scalar; repeated on every step row so the
                    # per-day value is recovered with a groupby(...).first().
                    "SC": out["SC"],
                }
                # Controller's posterior belief over the within-day queue
                # (mean +/- sd per minute); only present when the controller
                # exposes one (the AIF controller after its first observed day).
                if belief is not None:
                    mu_b, sd_b = belief
                    rec["L2_belief_mu"] = float(mu_b[0, k])
                    rec["L2_belief_sd"] = float(sd_b[0, k])
                    rec["L6_belief_mu"] = float(mu_b[1, k])
                    rec["L6_belief_sd"] = float(sd_b[1, k])
                # Controller's planned/believed green split (realised is phi2).
                if phi2_plan is not None:
                    rec["phi2_plan"] = float(phi2_plan[k])
                # Raw marginal social cost per traveller route; only present
                # when the EXTERNALITY/MSC advisory computed it for this day.
                if msc is not None:
                    rec["MSC_alpha"] = float(msc["alpha"][k])
                    rec["MSC_beta"] = float(msc["beta"][k])
                step_records.append(rec)
            cohort_records.extend(
                _cohort_record(seed, d, population, out["prior"], signal_name)
            )
            snap = controller.snapshot()
            controller_records.append({"seed": seed, "day": d, **snap})
            if d in snap_set:
                snapshots[(seed, d)] = population.snapshot()

    return ExperimentResult(
        step=pd.DataFrame(step_records),
        cohort=pd.DataFrame(cohort_records),
        controller=pd.DataFrame(controller_records),
        snapshots=snapshots,
    )
