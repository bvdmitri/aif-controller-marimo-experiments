"""Shared fixtures for the AIF-controller test suite."""

from __future__ import annotations

from dataclasses import replace

import pytest

from aif_traffic.parameters import (
    CohortSpec,
    FixedTimeControllerSpec,
    NoiseParams,
    Params,
    PopulationParams,
    SimParams,
)


@pytest.fixture
def small_params() -> Params:
    """A tiny, fast configuration for end-to-end pipeline tests."""
    return replace(
        Params.default(),
        sim=SimParams(days=3, h_min=20, dt_min=1, burn_in=0, seed=7,
                      selected_days=(0, 1, 2)),
        population=PopulationParams(cohorts=(CohortSpec(n_agents=50, window_size=2),)),
        noise=NoiseParams(obs_noise_sd=0.0),
        controller=FixedTimeControllerSpec(),
    )


@pytest.fixture
def default_params() -> Params:
    return Params.default()


# --------------------------------------------------------------------------
# Opt-in for the full-scale (slow) characterization tests.
# --------------------------------------------------------------------------
def pytest_addoption(parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run the slow, full-scale characterization tests (real 90-day "
             "experiment); they are skipped by default.",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="full-scale: pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
