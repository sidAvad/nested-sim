"""Low-fi (cv8Eed) parameter definitions.

Internal names use ASCII (`tau`, `tau_a`); the Cv8SimApp binary uses the
Unicode names `τ`, `τ_a`. Resistances are in mmHg·ms/mL, compliances in
mL/mmHg, elastances in mmHg/mL, volumes in mL, times in ms, HR in bpm.

Name map between the 8-compartment model write-up (PDF) and the binary:
    PDF Rav (aortic valve)       <-> binary Rcs
    PDF Rpv (pulmonic valve)     <-> binary Rcp
    PDF Rvs (sys vein -> RA)     <-> binary Rra
    PDF Rvp (pulm vein -> LA)    <-> binary Rla
    PDF Cap                      <-> binary 1 / Eap
    PDF Tes (activation peak)    <-> binary Tmax
"""

import math

# 24 inferred parameters + HR, with the binary's prior bounds (README_sim.md).
THETA_BOUNDS = {
    "Emax_LV": (0.5, 5.0),
    "Emax_RV": (0.4, 4.0),
    "Emax_LA": (0.1, 1.0),
    "Emax_RA": (0.1, 0.8),
    "Blv": (0.02, 0.06),
    "Brv": (0.02, 0.06),
    "Bla": (0.03, 0.12),
    "Bra": (0.03, 0.1),
    "Eedref_lv": (0.08, 2.0),
    "Eedref_rv": (0.03, 1.5),
    "Eedref_la": (0.1, 2.5),
    "Eedref_ra": (0.03, 1.5),
    "tau": (15.0, 80.0),
    "Tmax": (100.0, 400.0),
    "tau_a": (20.0, 80.0),
    "Tmax_a": (50.0, 250.0),
    "HR": (43.0, 133.0),
    "AVD": (60.0, 300.0),
    "Eap": (0.03, 5.0),
    "Cvp": (1.0, 9.0),
    "Cas": (0.35, 2.8),
    "Cvs": (30.0, 120.0),
    "Ras": (650.0, 2200.0),
    "Rap": (10.0, 1200.0),
    "Vs": (300.0, 3000.0),
}
THETA_NAMES = list(THETA_BOUNDS)

THETA_DEFAULT = {
    "Emax_LV": 3.0, "Emax_RV": 0.7, "Emax_LA": 0.48, "Emax_RA": 0.38,
    "Blv": 0.033, "Brv": 0.023, "Bla": 0.058, "Bra": 0.046,
    "Eedref_lv": 0.2, "Eedref_rv": 0.08, "Eedref_la": 0.46, "Eedref_ra": 0.13,
    "tau": 25.0, "Tmax": 200.0, "tau_a": 20.0, "Tmax_a": 125.0,
    "HR": 75.0, "AVD": 120.0, "Eap": 0.26, "Cvp": 8.0, "Cas": 2.5,
    "Cvs": 70.0, "Ras": 900.0, "Rap": 23.0, "Vs": 850.0,
}

# Fixed in the low-fi model (not part of theta).
FIXED = {
    "Rmv": 2.5, "Rtv": 2.5,
    "Rcs": 20.0, "Rcp": 10.0,  # aortic / pulmonic valve resistances
    "Rra": 25.0, "Rla": 15.0,  # sys vein -> RA, pulm vein -> LA
    "V0_lv": 5.0, "V0_rv": 5.0, "V0_la": 5.0, "V0_ra": 5.0,
    "Vref_lv": 120.0, "Vref_rv": 105.0, "Vref_la": 60.0, "Vref_ra": 45.0,
}

_TO_BINARY = {"tau": "τ", "tau_a": "τ_a"}
_FROM_BINARY = {v: k for k, v in _TO_BINARY.items()}


def to_binary_names(d):
    return {_TO_BINARY.get(k, k): v for k, v in d.items()}


def from_binary_names(d):
    return {_FROM_BINARY.get(k, k): v for k, v in d.items()}


def full_theta(theta=None):
    """Theta with defaults filled in; raises on unknown keys."""
    theta = dict(theta or {})
    unknown = set(theta) - set(THETA_NAMES)
    if unknown:
        raise KeyError(f"unknown theta keys: {sorted(unknown)}")
    return {**THETA_DEFAULT, **theta}


def sample_theta(rng, n=None):
    """Uniform draw inside THETA_BOUNDS (one dict, or a list of n dicts)."""
    def one():
        return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in THETA_BOUNDS.items()}
    return one() if n is None else [one() for _ in range(n)]


# Parameters sampled log-uniformly by the Cv8SimApp training simsets
# (simset_gen params_to_transform = (Emax_RV, Eap, Rap)).
LOG_SAMPLED = ("Emax_RV", "Eap", "Rap")


def sample_theta_benchmark(rng):
    """One theta draw as in the binary training sets: uniform within THETA_BOUNDS,
    log-uniform for LOG_SAMPLED (the binary's Sobol/uniform mix is replaced by
    plain uniform draws)."""
    th = {}
    for k, (lo, hi) in THETA_BOUNDS.items():
        u = rng.uniform()
        th[k] = float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))) if k in LOG_SAMPLED \
            else float(lo + u * (hi - lo))
    return th
