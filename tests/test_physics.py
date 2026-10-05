"""T3 / T4 / T5 and properties of the new high-fi mechanisms."""

import numpy as np
import pytest
from conftest import GOOD_SEEDS, SETTINGS, VOLUMES, max_abs_diff, theta_for

from nested_sim import SolverSettings, simulate, simulate_breath
from nested_sim.activation import make_epsilon
from nested_sim.params import FIXED, full_theta

WK = {"Zc_s": 45.0, "L_s": 12000.0, "Zc_p": 15.0, "L_p": 6000.0}
ALL_ON = {**WK, "Lv_av": 40.0, "Lv_pv": 20.0, "k_av_l": 0.3, "k_av_r": 0.3, "c_spt": 0.1,
          "Rav": 30.0, "Rmv": 3.5, "resp_amp": 3.0, "resp_rate": 15.0, "resp_phase": 0.5}


def beat_mean(x):
    """Mean over one period of a dense beat (uniform grid, x[-1] = next beat)."""
    return float(np.mean(np.asarray(x)[:-1]))


@pytest.mark.parametrize("seed", GOOD_SEEDS[:4])
@pytest.mark.parametrize("lv", [0.0, 40.0])
def test_t3_dc_resistance(seed, lv):
    """T3: mean(P_prox - P_down) / mean(Q_v) = R_art on both outlets."""
    th = full_theta(theta_for(seed))
    r = simulate(th, {**WK, "Lv_av": lv, "Lv_pv": lv / 2}, SETTINGS)
    assert r.success, r.message
    d = r.dense
    r_sys = (beat_mean(d["Pas"]) - beat_mean(d["Pvs"])) / beat_mean(d["Qas"])
    r_pul = (beat_mean(d["Pap"]) - beat_mean(d["Pvp"])) / beat_mean(d["Qap"])
    assert r_sys == pytest.approx(th["Ras"], rel=2e-3)
    assert r_pul == pytest.approx(th["Rap"], rel=2e-3)


@pytest.mark.parametrize("seed", GOOD_SEEDS[:3])
def test_t4_volume_conservation(seed):
    """T4: stressed volume equals Vs throughout the beat, all phi on."""
    th = full_theta(theta_for(seed))
    for r in simulate_breath(th, ALL_ON, SETTINGS):
        assert r.success, r.message
        unstressed = sum(FIXED[f"V0_{c}"] for c in ("lv", "rv", "la", "ra"))
        total = sum(np.asarray(r.dense[v]) for v in VOLUMES) - unstressed
        assert np.max(np.abs(total - th["Vs"])) < 1e-6 * th["Vs"]


def test_t5_valve_inertance_limit():
    """T5: as Lv -> 0 the output converges (linearly) to the Lv = 0 branch.

    Lv = 0.1 is ~100x below the prior and very stiff, so the rhs budget is raised.
    """
    base = simulate({}, settings=SETTINGS)
    stiff = SolverSettings(max_rhs_per_beat=5_000_000)
    errs = []
    for lv in (10.0, 1.0, 0.1):
        r = simulate({}, {"Lv_av": lv, "Lv_pv": lv}, stiff)
        assert r.success, r.message
        errs.append(max_abs_diff(base.waves201, r.waves201, ["Pas", "Pap", "Plv", "Prv"]))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.01
    assert errs[1] / errs[2] == pytest.approx(10.0, rel=0.3)


def test_respiration_breath_structure():
    th = full_theta({"HR": 75.0})
    beats = simulate_breath(th, {"resp_amp": 3.0, "resp_rate": 15.0}, SETTINGS)
    assert len(beats) == 5  # round(75 / 15)
    assert all(b.success for b in beats)
    means = [beat_mean(b.dense["Pra"]) for b in beats]
    assert np.ptp(means) > 0.05  # beats differ across the breath
    # Reported pressures are transmural: P_it is removed but present as a channel.
    assert min(b.dense["P_it"].min() for b in beats) == pytest.approx(-3.0, abs=0.05)


def test_respiration_continuity_at_zero_amplitude():
    """resp_amp -> 0 converges to the steady-state low-fi beat."""
    lo = simulate({}, settings=SETTINGS)
    for b in simulate_breath({}, {"resp_amp": 1e-6, "resp_rate": 15.0}, SETTINGS):
        assert max_abs_diff(lo.waves201, b.waves201, ["Prv", "Pra", "Pap", "Pvp", "Pas"]) < 1e-3


def test_resp_phase_selects_beat():
    phi = {"resp_amp": 3.0, "resp_rate": 15.0}
    beats = simulate_breath({}, phi, SETTINGS)
    for j in (0, 2, 4):
        r = simulate({}, {**phi, "resp_phase": (j + 0.5) / 5}, SETTINGS)
        assert max_abs_diff(beats[j].dense, r.dense) < 1e-8


def test_av_coupling_continuity():
    lo = simulate({}, settings=SETTINGS)
    r = simulate({}, {"k_av_l": 1e-7, "k_av_r": 1e-7}, SETTINGS)
    assert max_abs_diff(lo.waves201, r.waves201) < 1e-4


def test_activation_starts_at_zero_and_peaks_near_tmax():
    eps = make_epsilon(200.0, 25.0)
    assert eps(0.0) == 0.0
    t = np.linspace(0, 800, 8001)
    v = np.array([eps(x) for x in t])
    assert abs(t[np.argmax(v)] - 200.0) < 5.0
    assert 0.99 < v.max() < 1.01


def test_interdependence_continuity_and_residual():
    from nested_sim.activation import make_epsilon
    from nested_sim.hifi import HiFiModel, septal_volume
    from nested_sim.lowfi import chamber_pressure
    lo = simulate({}, settings=SETTINGS)
    r = simulate({}, {"c_spt": 1e-8}, SETTINGS)
    assert max_abs_diff(lo.waves201, r.waves201) < 1e-4
    # V_spt solves V = c (P_lv(Vlv - V) - P_rv(Vrv + V)) across the cycle
    m = HiFiModel({}, {"c_spt": 0.2})
    p = m.p
    eps_v = make_epsilon(p["Tmax"], p["tau"])
    for e in np.linspace(0.0, 1.0, 11):
        for vlv, vrv in ((60.0, 150.0), (140.0, 40.0), (100.0, 100.0)):
            v = septal_volume(vlv, vrv, e, m.cc["lv"], m.cc["rv"], 0.2)
            rhs = 0.2 * (chamber_pressure(vlv - v, e, m.cc["lv"]) - chamber_pressure(vrv + v, e, m.cc["rv"]))
            assert abs(v - rhs) < 1e-8
    assert eps_v(0.0) == 0.0


def test_interdependence_direction():
    """LV contraction raises RV systolic pressure; septal shift raises LV filling pressure."""
    lo = simulate({}, settings=SETTINGS)
    r = simulate({}, {"c_spt": 0.1}, SETTINGS)
    assert r.summaries["rv_s"] > lo.summaries["rv_s"]
    assert r.summaries["lvedp"] > lo.summaries["lvedp"]
