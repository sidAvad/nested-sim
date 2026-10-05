"""T6: measurement layer (off = identity) and catheter filter accuracy."""

import math

import numpy as np
import pytest
from conftest import SETTINGS

from nested_sim import PHI_OFF, measure, simulate
from nested_sim.measurement import CATHETER_CHANNELS, catheter_filter


@pytest.fixture(scope="module")
def sim():
    return simulate({}, settings=SETTINGS)


def test_t6_measurement_off_is_identity(sim):
    out = measure(sim, PHI_OFF, np.random.default_rng(0))
    for k in sim.dense:
        assert np.array_equal(out["dense"][k], sim.dense[k])
        assert np.array_equal(out["waves201"][k], sim.waves201[k])
    for k, v in sim.summaries.items():
        assert out["scalars"][k] == v


def test_only_cath_channels_change(sim):
    out = measure(sim, {"fn_cath": 12.0, "zeta_cath": 0.3, "offset_cath": 1.0, "sigma_cath": 0.2},
                  np.random.default_rng(0))
    for k in sim.dense:
        changed = not np.array_equal(out["dense"][k], sim.dense[k])
        assert changed == (k in CATHETER_CHANNELS), k
    for k in ("sbp", "dbp", "map", "sv"):
        assert out["scalars"][k] == sim.summaries[k]


@pytest.mark.parametrize("fn,zeta", [(10.0, 0.15), (25.0, 0.6)])
def test_catheter_filter_matches_analog_response(fn, zeta):
    T, n = 800.0, 2000
    dt = T / n
    t = np.arange(n + 1) * dt
    wn = 2 * math.pi * fn / 1000
    for k in (1, 8, 20):
        w = 2 * math.pi * k / T
        y = catheter_filter(np.sin(w * t), dt, fn, zeta)
        H = wn ** 2 / (-(w ** 2) + 2j * zeta * wn * w + wn ** 2)
        ref = np.abs(H) * np.sin(w * t + np.angle(H))
        assert np.max(np.abs(y - ref)) < 1e-3 * max(1.0, abs(H))
    assert np.allclose(catheter_filter(np.full(n + 1, 7.0), dt, fn, zeta), 7.0)
