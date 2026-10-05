"""High-fi extra parameters phi: off values and sampling.

With every phi at its PHI_OFF value the high-fi model and the measurement layer
reduce exactly to the low-fi simulator.

phi always holds physical values (Zc_s in mmHg*ms/mL, Lv_av in mmHg*ms^2/mL,
...). Some priors are conditional on theta (or on other phi): the prior config
can express a parameter as a draw times a reference quantity (`times`), or a
valve inertance through its damping ratio (`form = "damping"`). See
configs/phi_priors.toml.
"""

import math
import tomllib
from pathlib import Path

from .params import FIXED, full_theta

DEFAULT_PRIORS = Path(__file__).resolve().parents[1] / "configs" / "phi_priors.toml"

# Structural phi (used by HiFiModel).
PHI_STRUCT_OFF = {
    "Zc_s": 0.0, "L_s": 0.0,    # systemic 4-element Windkessel
    "Zc_p": 0.0, "L_p": 0.0,    # pulmonary 4-element Windkessel
    "Lv_av": 0.0, "Lv_pv": 0.0,  # semilunar valve inertance
    "k_av_l": 0.0, "k_av_r": 0.0,  # AV-plane coupling (LV -> LA, RV -> RA)
    "c_spt": 0.0,          # septal compliance, ventricular interdependence (mL/mmHg)
    "w_wedge": 0.0, "tau_wedge": 0.0,  # wedge observation of the Pvp channel
    "resp_amp": 0.0,       # respiratory intrathoracic pressure swing, mmHg
    "resp_rate": 15.0,     # breaths/min; no effect while resp_amp = 0
    "resp_phase": 0.0,     # which beat of the breath is reported, in [0, 1)
    # Freed low-fi resistances; PDF names (binary names in params.py).
    "Rmv": FIXED["Rmv"], "Rav": FIXED["Rcs"], "Rtv": FIXED["Rtv"],
    "Rpv": FIXED["Rcp"], "Rvs": FIXED["Rra"], "Rvp": FIXED["Rla"],
}

# Measurement phi (used by measurement.py).
PHI_MEAS_OFF = {
    "fn_cath": math.inf,   # catheter natural frequency, Hz (inf: no filter)
    "zeta_cath": 0.3,      # damping ratio; irrelevant while fn_cath = inf
    "offset_cath": 0.0,    # zero offset b, mmHg
    "sigma_cath": 0.0,     # catheter white-noise std, mmHg
}

PHI_OFF = {**PHI_STRUCT_OFF, **PHI_MEAS_OFF}

# Independently switchable high-fi elements (each group off = low-fi).
PHI_GROUPS = {
    "wk_s": ["Zc_s", "L_s"],                  # systemic 4-element Windkessel
    "wk_p": ["Zc_p", "L_p"],                  # pulmonary 4-element Windkessel
    "lv_av": ["Lv_av"],                       # aortic valve inertance
    "lv_pv": ["Lv_pv"],                       # pulmonic valve inertance
    "res": ["Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp"],  # freed resistances
    "av_l": ["k_av_l"],                       # AV-plane coupling, left heart
    "av_r": ["k_av_r"],                       # AV-plane coupling, right heart
    "vi": ["c_spt"],                          # ventricular interdependence
    "resp": ["resp_amp", "resp_rate", "resp_phase"],    # respiration
    "wedge": ["w_wedge", "tau_wedge"],        # wedge observation (parked)
    "meas": list(PHI_MEAS_OFF),               # measurement model
}
PHI_NAMES = list(PHI_OFF)

# Sampling order: anything a prior refers to via `times` or `form = "damping"`
# must come earlier.
SAMPLE_ORDER = ["Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp",
                "Zc_s", "L_s", "Zc_p", "L_p", "Lv_av", "Lv_pv",
                "k_av_l", "k_av_r", "c_spt", "w_wedge", "tau_wedge",
                "resp_amp", "resp_rate", "resp_phase",
                "fn_cath", "zeta_cath", "offset_cath", "sigma_cath"]
assert sorted(SAMPLE_ORDER) == sorted(PHI_NAMES)

_TO_LOWFI = {"Rmv": "Rmv", "Rav": "Rcs", "Rtv": "Rtv", "Rpv": "Rcp", "Rvs": "Rra", "Rvp": "Rla"}


def full_phi(phi=None):
    phi = dict(phi or {})
    unknown = set(phi) - set(PHI_NAMES)
    if unknown:
        raise KeyError(f"unknown phi keys: {sorted(unknown)}")
    return {**PHI_OFF, **phi}


def phi_to_lowfi_fixed(phi):
    """Freed resistances in the low-fi (binary) naming."""
    return {_TO_LOWFI[k]: phi[k] for k in _TO_LOWFI}


def load_priors(path=None):
    with open(path or DEFAULT_PRIORS, "rb") as fh:
        priors = tomllib.load(fh)
    unknown = set(priors) - set(PHI_NAMES)
    if unknown:
        raise KeyError(f"prior config has unknown phi keys: {sorted(unknown)}")
    return priors


def _draw(rng, spec):
    lo, hi = spec["low"], spec["high"]
    if spec["dist"] == "uniform":
        return float(rng.uniform(lo, hi))
    if spec["dist"] == "loguniform":
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    raise ValueError(f"unknown dist {spec['dist']!r}")


def _physical(name, draw, spec, theta, full):
    """Map a prior draw to the physical value of phi `name` (at alpha = 1)."""
    ref = spec.get("times")
    if spec.get("form") == "damping":
        # draw is the valve damping ratio zeta_v = R / (2 sqrt(Emax * Lv)).
        R = full[spec["R"]]
        value = R * R / (4.0 * draw * draw * theta[spec["E"]])
    elif ref is None:
        value = draw
    elif ref == "off":
        value = draw * PHI_OFF[name]
    elif ref in full:
        value = draw * full[ref]
    else:
        value = draw * theta[ref]
    if "max" in spec:
        value = min(value, spec["max"])
    if "zero_below" in spec and value < spec["zero_below"]:
        value = PHI_OFF[name]  # negligible effect, very stiff: treat as off
    return value


def sample_phi(rng, alpha, theta=None, priors=None, alpha_meas=0.0, groups=None):
    """phi = off + a * (phi_full - off) for each phi with a prior.

    a = alpha for structural phi and a = alpha_meas for measurement phi
    (PHI_MEAS_OFF keys), so the measurement model is a separate axis that is
    off by default. phi_full is the physical value implied by a prior draw at
    a = 1; priors may depend on theta (missing keys = defaults) and on earlier
    phi_full values. a = 0 returns the off values exactly. A draw is taken for
    every prior entry regardless of alpha, so the same rng seed gives the same
    phi_full across alpha values (Zc and L then scale together, keeping
    tau_L = L / Zc fixed along alpha).

    groups: optional list of PHI_GROUPS keys; phi outside these groups is set
    off (draws are still taken, so a seed gives the same values with or
    without `groups`).
    """
    if groups is not None:
        unknown = set(groups) - set(PHI_GROUPS)
        if unknown:
            raise KeyError(f"unknown phi groups: {sorted(unknown)}")
    for a in (alpha, alpha_meas):
        if not 0.0 <= a <= 1.0:
            raise ValueError("alpha and alpha_meas must be in [0, 1]")
    priors = load_priors() if priors is None else priors
    theta = full_theta(theta)
    full = dict(PHI_OFF)
    for name in SAMPLE_ORDER:
        if name in priors:
            full[name] = _physical(name, _draw(rng, priors[name]), priors[name], theta, full)
    phi = dict(PHI_OFF)
    for name in SAMPLE_ORDER:
        if name not in priors:
            continue
        a = alpha_meas if name in PHI_MEAS_OFF else alpha
        off, target = PHI_OFF[name], full[name]
        interp = priors[name].get("interp")
        if interp == "none":
            phi[name] = target if a > 0.0 else off
        elif interp == "inverse":
            inv_off = 0.0 if math.isinf(off) else 1.0 / off
            inv = inv_off + a * (1.0 / target - inv_off)
            phi[name] = math.inf if inv == 0.0 else 1.0 / inv
        else:
            phi[name] = off + a * (target - off)
    if groups is not None:
        keep = {k for g in groups for k in PHI_GROUPS[g]}
        phi = {k: (v if k in keep else PHI_OFF[k]) for k, v in phi.items()}
    return phi
