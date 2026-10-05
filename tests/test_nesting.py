"""T1 / T2: the high-fi model reduces to the low-fi model when phi is off."""

import numpy as np
import pytest
from conftest import GOOD_SEEDS, SETTINGS, T1_N, max_abs_diff, theta_for

from nested_sim import PHI_OFF, simulate
from nested_sim.hifi import HiFiModel
from nested_sim.lowfi import LowFiModel

TOL = 1e-10


def assert_same(lo, hi):
    assert lo.success == hi.success, (lo.message, hi.message)
    if not lo.success:
        return
    assert lo.t0 == hi.t0
    assert max_abs_diff(lo.dense, hi.dense) <= TOL
    assert max(abs(lo.summaries[k] - hi.summaries[k]) for k in lo.summaries) <= TOL


@pytest.mark.parametrize("seed", range(T1_N))
def test_t1_nesting_random_theta(seed):
    """T1: simulate(theta, PHI_OFF) == simulate(theta), all channels + scalars."""
    th = theta_for(seed)
    assert_same(simulate(th, settings=SETTINGS), simulate(th, PHI_OFF, SETTINGS))


def test_t1_state_vector_identical_when_off():
    m = HiFiModel({}, PHI_OFF)
    assert m.state_names == LowFiModel({}).state_names
    assert m.period == m.T and m.beat_index == 0


@pytest.mark.parametrize("phi", [
    {"Zc_s": 40.0, "L_s": 0.0},                      # L = 0 short-circuits Zc
    {"Zc_s": 0.0, "L_s": 5000.0},                    # Zc = 0 removes the Windkessel
    {"Zc_p": 20.0, "L_p": 0.0},
    {"Zc_p": 0.0, "L_p": 3000.0},
    {"k_av_l": 0.0, "k_av_r": 0.0},
    {"resp_amp": 0.0, "resp_rate": 12.0, "resp_phase": 0.7},
    {"w_wedge": 0.0, "tau_wedge": 50.0},
    {"Lv_av": 0.0, "Lv_pv": 0.0},
    {"c_spt": 0.0},
], ids=["Zc_s-only", "L_s-only", "Zc_p-only", "L_p-only", "k_av-0", "resp-amp-0", "wedge-w-0", "Lv-0", "c_spt-0"])
def test_t2_partial_off(phi):
    """T2: a partner parameter alone (or its switch at 0) reproduces low-fi."""
    for seed in GOOD_SEEDS[:2]:
        th = theta_for(seed)
        assert_same(simulate(th, settings=SETTINGS), simulate(th, phi, SETTINGS))


def test_freed_resistances_at_lowfi_values_are_identity():
    phi = {k: PHI_OFF[k] for k in ("Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp")}
    assert_same(simulate({}, settings=SETTINGS), simulate({}, phi, SETTINGS))


def test_hifi_with_phi_differs():
    """Guard against a vacuous T1: switching phi on must change the output."""
    lo = simulate({}, settings=SETTINGS)
    hi = simulate({}, {"Zc_s": 40.0, "L_s": 8000.0}, SETTINGS)
    assert np.max(np.abs(lo.dense["Pas"] - hi.dense["Pas"])) > 1.0
