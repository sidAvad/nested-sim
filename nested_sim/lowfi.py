"""Low-fi model: Python port of the cv8Eed model (reference/julia/cv8Eed.jl).

8 volume states, 2-element Windkessel arterial outlets, hard-diode valves.
Time origin as in the Julia model: t = 0 is atrial contraction onset and the
ventricles contract AVD later. All 8 volumes are integrated (the Julia model
eliminates Vla through the Vs constraint; the sum is conserved either way).

This is the reference path; the high-fi model (hifi.py) must reduce to it
when phi is off.
"""

import math

import numpy as np

from .activation import make_epsilon
from .params import FIXED, full_theta

STATES = ["Vla", "Vlv", "Vas", "Vvs", "Vra", "Vrv", "Vap", "Vvp"]


def chamber_constants(p):
    """(Emax, A, B, V0) per chamber, A from the Eedref reparameterisation."""
    out = {}
    for ch in ("lv", "rv", "la", "ra"):
        B = p["B" + ch]
        V0 = p["V0_" + ch]
        A = p["Eedref_" + ch] / (B * math.exp(B * (p["Vref_" + ch] - V0)))
        out[ch] = (p["Emax_" + ch.upper()], A, B, V0)
    return out


def chamber_pressure(V, e, c):
    Emax, A, B, V0 = c
    return e * Emax * (V - V0) + (1.0 - e) * A * math.expm1(B * (V - V0))


MIN_VVS = 50.0  # u0_cv8Eed keeps at least this much stressed volume in Vvs


def initial_state(p):
    """Port of u0_cv8Eed: mean-circulatory-filling volumes, Vvs absorbs the Vs residual."""
    cc = chamber_constants(p)
    Cch = {ch: 1.0 / (A * B) for ch, (_, A, B, _) in cc.items()}
    Cves = {"as": p["Cas"], "vs": p["Cvs"], "ap": 1.0 / p["Eap"], "vp": p["Cvp"]}
    Pmc = p["Vs"] / (sum(Cch.values()) + sum(Cves.values()))
    V = {k: C * Pmc for k, C in Cves.items()}
    for ch, (_, A, B, V0) in cc.items():
        V[ch] = V0 + math.log1p(Pmc / A) / B
    stressed_ch = sum(V[ch] - cc[ch][3] for ch in cc)
    stressed_no_vvs = stressed_ch + V["as"] + V["ap"] + V["vp"]
    V["vs"] = p["Vs"] - stressed_no_vvs
    if V["vs"] < MIN_VVS:
        target = max(p["Vs"] - MIN_VVS, 0.0)
        scale = target / stressed_no_vvs if stressed_no_vvs > 0 else 0.0
        for ch, c in cc.items():
            V[ch] = c[3] + (V[ch] - c[3]) * scale
        for k in ("as", "ap", "vp"):
            V[k] *= scale
        V["vs"] = MIN_VVS
    return np.array([V["la"], V["lv"], V["as"], V["vs"], V["ra"], V["rv"], V["ap"], V["vp"]])


class LowFiModel:
    n_states = 8
    state_names = STATES
    state_kind = ["V"] * 8

    def __init__(self, theta=None):
        p = {**FIXED, **full_theta(theta)}
        self.p = p
        self.T = 60000.0 / p["HR"]
        self.cc = chamber_constants(p)
        self._build()

    def y0(self):
        return initial_state(self.p)

    def _build(self):
        p, T = self.p, self.T
        c_lv, c_rv, c_la, c_ra = (self.cc[k] for k in ("lv", "rv", "la", "ra"))
        eps_v, eps_a, AVD = make_epsilon(p["Tmax"], p["tau"]), make_epsilon(p["Tmax_a"], p["tau_a"]), p["AVD"]
        Cas, Cvs, Eap, Cvp = p["Cas"], p["Cvs"], p["Eap"], p["Cvp"]
        Ras, Rap, Rmv, Rtv = p["Ras"], p["Rap"], p["Rmv"], p["Rtv"]
        Rcs, Rcp, Rra, Rla = p["Rcs"], p["Rcp"], p["Rra"], p["Rla"]

        def pressures(t, y):
            Vla, Vlv, Vas, Vvs, Vra, Vrv, Vap, Vvp = y
            ea = eps_a(t % T)
            ev = eps_v((t - AVD) % T)
            return (chamber_pressure(Vla, ea, c_la), chamber_pressure(Vlv, ev, c_lv),
                    Vas / Cas, Vvs / Cvs,
                    chamber_pressure(Vra, ea, c_ra), chamber_pressure(Vrv, ev, c_rv),
                    Vap * Eap, Vvp / Cvp)

        def flows(P):
            Pla, Plv, Pas, Pvs, Pra, Prv, Pap, Pvp = P
            Qmv = (Pla - Plv) / Rmv if Pla > Plv else 0.0
            Qav = (Plv - Pas) / Rcs if Plv > Pas else 0.0
            Qtv = (Pra - Prv) / Rtv if Pra > Prv else 0.0
            Qpv = (Prv - Pap) / Rcp if Prv > Pap else 0.0
            return (Qmv, Qav, (Pas - Pvs) / Ras, (Pvs - Pra) / Rra,
                    Qtv, Qpv, (Pap - Pvp) / Rap, (Pvp - Pla) / Rla)

        def rhs(t, y):
            Qmv, Qav, Qas, Qvs, Qtv, Qpv, Qap, Qvp = flows(pressures(t, y))
            return [Qvp - Qmv, Qmv - Qav, Qav - Qas, Qas - Qvs,
                    Qvs - Qtv, Qtv - Qpv, Qpv - Qap, Qap - Qvp]

        def pv_gap(t, y):
            P = pressures(t, y)
            return P[5] - P[6]

        def observe(t, y):
            P = pressures(t, y)
            Pla, Plv, Pas, Pvs, Pra, Prv, Pap, Pvp = P
            Qmv, Qav, Qas, Qvs, Qtv, Qpv, Qap, Qvp = flows(P)
            Vla, Vlv, Vas, Vvs, Vra, Vrv, Vap, Vvp = y
            return {
                "Vla": Vla, "Vlv": Vlv, "Vas": Vas, "Vvs": Vvs,
                "Vra": Vra, "Vrv": Vrv, "Vap": Vap, "Vvp": Vvp,
                "Pla": Pla, "Plv": Plv, "Pas": Pas, "Pvs": Pvs,
                "Pra": Pra, "Prv": Prv, "Pap": Pap, "Pvp": Pvp,
                # Flows are labelled by the compartment they enter (binary convention).
                "Qlv": Qmv, "Qas": Qav, "Qvs": Qas, "Qra": Qvs,
                "Qrv": Qtv, "Qap": Qpv, "Qvp": Qap, "Qla": Qvp,
                "mv": float(Pla > Plv), "av": float(Plv > Pas),
                "tv": float(Pra > Prv), "pv": float(Prv > Pap),
            }

        self.rhs, self.pv_gap, self.observe = rhs, pv_gap, observe
