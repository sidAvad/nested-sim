"""Properties of phi sampling (configs/phi_priors.toml, sample_phi)."""

import math

import numpy as np
import pytest
from conftest import theta_for

from nested_sim.params import full_theta
from nested_sim.phi import PHI_GROUPS, PHI_MEAS_OFF, PHI_OFF, PHI_STRUCT_OFF, load_priors, sample_phi

PRIORS = load_priors()


@pytest.mark.parametrize("seed", range(5))
def test_alpha_zero_is_off(seed):
    assert sample_phi(np.random.default_rng(seed), 0.0, theta_for(seed)) == PHI_OFF


def test_measurement_off_by_default():
    for seed in range(5):
        phi = sample_phi(np.random.default_rng(seed), 1.0, theta_for(seed))
        assert all(phi[k] == PHI_MEAS_OFF[k] for k in PHI_MEAS_OFF)
        phi_m = sample_phi(np.random.default_rng(seed), 0.0, theta_for(seed), alpha_meas=1.0)
        assert all(phi_m[k] == PHI_STRUCT_OFF[k] for k in PHI_STRUCT_OFF)
        assert math.isfinite(phi_m["fn_cath"])


def test_same_seed_same_draws_across_alpha():
    th = theta_for(3)
    full = sample_phi(np.random.default_rng(7), 1.0, th)
    half = sample_phi(np.random.default_rng(7), 0.5, th)
    for k in ("Zc_s", "L_s", "k_av_l", "resp_amp"):
        assert half[k] == pytest.approx(0.5 * full[k])
    # tau_L = L / Zc is invariant along alpha
    assert half["L_s"] / half["Zc_s"] == pytest.approx(full["L_s"] / full["Zc_s"])
    # nuisances are not alpha-scaled
    assert half["resp_rate"] == full["resp_rate"] and half["resp_phase"] == full["resp_phase"]


@pytest.mark.parametrize("seed", range(20))
def test_conditional_priors_within_ranges(seed):
    th = full_theta(theta_for(seed))
    phi = sample_phi(np.random.default_rng(seed), 1.0, th)
    zs, zp = phi["Zc_s"] / th["Ras"], phi["Zc_p"] / th["Rap"]
    assert PRIORS["Zc_s"]["low"] <= zs <= PRIORS["Zc_s"]["high"]
    assert phi["Zc_p"] <= PRIORS["Zc_p"]["max"] + 1e-9
    if phi["Zc_p"] < PRIORS["Zc_p"]["max"]:
        assert PRIORS["Zc_p"]["low"] <= zp <= PRIORS["Zc_p"]["high"]
    for side, R, E in (("av", "Rav", "Emax_LV"), ("pv", "Rpv", "Emax_RV")):
        floor = PRIORS[f"Lv_{side}"].get("zero_below", 0.0)
        if phi[f"Lv_{side}"] == 0.0:  # draw below the zero_below floor
            assert floor > 0
            continue
        assert phi[f"Lv_{side}"] >= floor
        zeta = phi[R] / (2 * math.sqrt(th[E] * phi[f"Lv_{side}"]))
        assert PRIORS[f"Lv_{side}"]["low"] - 1e-9 <= zeta <= PRIORS[f"Lv_{side}"]["high"] + 1e-9
    for k in ("Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp"):
        assert 0.5 - 1e-9 <= phi[k] / PHI_OFF[k] <= 2.0 + 1e-9


def test_alpha_range_checked():
    with pytest.raises(ValueError):
        sample_phi(np.random.default_rng(0), 1.5)
    with pytest.raises(ValueError):
        sample_phi(np.random.default_rng(0), 0.5, alpha_meas=-0.1)


def test_groups_cover_all_phi_once():
    keys = [k for g in PHI_GROUPS.values() for k in g]
    assert sorted(keys) == sorted(PHI_OFF)


def test_groups_select_elements():
    th = theta_for(0)
    full = sample_phi(np.random.default_rng(1), 1.0, th)
    sub = sample_phi(np.random.default_rng(1), 1.0, th, groups=["wk_s", "lv_pv"])
    on = set(PHI_GROUPS["wk_s"] + PHI_GROUPS["lv_pv"])
    for k in PHI_OFF:
        assert sub[k] == (full[k] if k in on else PHI_OFF[k]), k
    with pytest.raises(KeyError):
        sample_phi(np.random.default_rng(1), 1.0, th, groups=["nope"])
