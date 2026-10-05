"""Shared fixtures and helpers for the nested-sim tests.

Run on palladium with: .venv/bin/python -m pytest -n 64 tests
Tests marked `slow` (figure smoke test) are skipped unless --runslow is given.
T1_N (env NESTED_SIM_T1_N, default 100) sets the number of random theta in T1.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nested_sim import SolverSettings  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T1_N = int(os.environ.get("NESTED_SIM_T1_N", "100"))
SETTINGS = SolverSettings()
VOLUMES = ["Vla", "Vlv", "Vas", "Vvs", "Vra", "Vrv", "Vap", "Vvp"]


# theta_for seeds whose low-fi beat converges (seed 1 is unstable).
GOOD_SEEDS = [0, 2, 3, 4, 5]


def theta_for(seed):
    return sample_theta(np.random.default_rng(seed))


def max_abs_diff(a, b, keys=None):
    """Max |a[k] - b[k]| over shared (or given) channels of two dense dicts."""
    keys = keys if keys is not None else sorted(set(a) & set(b))
    return max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in keys)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test, needs --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
