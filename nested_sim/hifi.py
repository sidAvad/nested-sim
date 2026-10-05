"""High-fi model: the low-fi cv8Eed model plus, on both arterial outlets,

  * a 4-element Windkessel: characteristic impedance Zc in parallel with an
    inertance L, between the valve and the arterial compliance;
  * semilunar valve inertance Lv (aortic, pulmonic);
  * the low-fi fixed resistances (Rmv, Rav, Rtv, Rpv, Rvs, Rvp) freed as phi;
  * atrioventricular-plane coupling (k_av_l, k_av_r): as a ventricle empties,
    the AV plane descends and enlarges its atrium, so the atrial pressure uses
    V_atrium - k (Vref_ventricle - V_ventricle) in place of V_atrium. Blood
    volumes are unchanged (Vs conserved); k = 0 is the low-fi atrium;
  * ventricular interdependence (c_spt, mL/mmHg): a septum with compliance
    c_spt moves V_spt between the ventricles' free walls according to their
    pressure difference, V_spt = c_spt (P_lv - P_rv), with
    P_lv = P(V_lv - V_spt), P_rv = P(V_rv + V_spt), solved by Newton at every
    evaluation (g(V) = V - c_spt (P_lv - P_rv) has g' >= 1: unique root).
    LV contraction then raises RV pressure, and RV overload squeezes LV
    filling. Blood volumes are unchanged (Vs conserved); c_spt = 0 is low-fi;
  * respiration (resp_amp, resp_rate, resp_phase): intrathoracic pressure
    P_it(t) = -resp_amp (1 - cos(2 pi t / P_resp)) / 2 adds to the absolute
    pressure of the chambers, pulmonary artery and pulmonary veins (systemic
    arteries and veins are extrathoracic). P_resp is a whole number of beats,
    m = max(2, round(HR / resp_rate)), so the forced system is periodic over m
    beats; the reported beat is beat floor(resp_phase * m) of that super-cycle.
    Intrathoracic channels are reported transmural (P_it subtracted): the
    first-order respiratory signal is removed, its effect on filling and
    ejection (second order) is kept. resp_amp = 0 is the low-fi model;
  * a wedge observation of the "Pvp" channel (apply_wedge): the channel the
    real data calls Pvp is a pulmonary capillary wedge pressure, i.e. LA
    pressure transmitted through the occluded vessel, modelled as
    (1 - w) Pvp + w lag_tau(Pla) with a first-order lag; w = 0 is the low-fi
    channel. The vessel pressure itself stays available as "Pvp_vessel".

Per outlet (systemic: LV -> aortic valve -> Cas -> Ras -> Pvs; pulmonary:
RV -> pulmonic valve -> Cap=1/Eap -> Rap -> Pvp), with Qv the valve flow and
P_C = V_art / C_art:

    P_prox   = P_C + Zc (Qv - Q_L)          observed arterial pressure
    dQ_L/dt  = (Zc / L) (Qv - Q_L)
    dV_art/dt = Qv - (P_C - P_down) / R_art

Lv = 0: Qv = max(0, (P_up - P_C + Zc Q_L) / (R_v + Zc))
Lv > 0: Qv is a state; the delivered flow is max(Qv, 0) and, with
        dP = P_up - P_prox,
            dQv/dt = (dP - R_v Qv) / Lv                 if dP >= 0
            dQv/dt = (dP g(Qv) - R_v Qv) / Lv           if dP < 0,
        g(Qv) = clip(Qv / Q_GATE, 0, 1). Flow decelerates under a reverse
        gradient and the valve closes when it reaches zero; g only acts within
        Q_GATE of closure. Without it the right-hand side jumps at Qv = 0 under
        a reverse gradient (full deceleration just above 0, none just below),
        and solver noise around 0 makes the step size collapse. While closed,
        the state relaxes to 0 from below at rate R_v / Lv instead of being
        frozen, which keeps its Jacobian column nonzero.

The Windkessel is inactive unless both Zc > 0 and L > 0 (L = 0 short-circuits
Zc; Zc = 0 removes it). With every phi at its off value the equations reduce to
lowfi.py term by term, and theta keeps its meaning: R_art is the mean-flow
resistance (mean(P_prox) = mean(P_C) at periodic steady state), C_art the
arterial compliance, and the new states hold no volume, so Vs is unchanged.

State vector: 8 volumes (lowfi.STATES order), then only the active extra
states, in the order QL_s, QL_p, Qv_s, Qv_p (Q_L when that outlet's Windkessel
is on, Qv when that valve's Lv > 0). Inactive states are left out rather than
held at 0: a column with identically zero derivative makes the solvers'
finite-difference Jacobian blow up its perturbation to inf. With phi off the
state vector is exactly the low-fi one.
"""

import math

import numpy as np

from .activation import make_epsilon
from .lowfi import STATES as LOWFI_STATES
from .lowfi import chamber_constants, chamber_pressure, initial_state
from .params import FIXED, full_theta
from .phi import full_phi, phi_to_lowfi_fixed

EXTRA_STATES = ["QL_s", "QL_p", "Qv_s", "Qv_p"]
SPT_TOL, SPT_MAXIT = 1e-11, 50


def _chamber_dpdv(V, e, c):
    Emax, A, B, V0 = c
    return e * Emax + (1.0 - e) * A * B * math.exp(B * (V - V0))


def septal_volume(Vlv, Vrv, ev, c_lv, c_rv, c_spt):
    """Solve V = c_spt (P_lv(Vlv - V) - P_rv(Vrv + V)) for the septal volume V."""
    x = c_spt * (chamber_pressure(Vlv, ev, c_lv) - chamber_pressure(Vrv, ev, c_rv))
    for _ in range(SPT_MAXIT):
        g = x - c_spt * (chamber_pressure(Vlv - x, ev, c_lv) - chamber_pressure(Vrv + x, ev, c_rv))
        dg = 1.0 + c_spt * (_chamber_dpdv(Vlv - x, ev, c_lv) + _chamber_dpdv(Vrv + x, ev, c_rv))
        step = g / dg
        x -= step
        if abs(step) <= SPT_TOL * (1.0 + abs(x)):
            return x
    raise ValueError("septal volume Newton iteration did not converge")

Q_GATE = 1e-2  # mL/ms (~1% of peak flow), band over which a reverse gradient is gated in;
#                1e-3 made closures extremely stiff (failures, see README); effect of 1e-2 vs 1e-3 <= 0.13 mmHg
Q_OPEN = 1e-9  # mL/ms, valve indicator threshold (ignores solver noise at 0)


def _outlet(P_up, P_C, QL, Qv_state, Zc, L, Lv, Rv):
    """Return (Qv, P_prox, dQL, dQv, open) for one arterial outlet."""
    wk = Zc > 0.0 and L > 0.0
    Zc_eff = Zc if wk else 0.0
    if Lv > 0.0:
        Qv = Qv_state if Qv_state > 0.0 else 0.0
        P_prox = P_C + Zc_eff * (Qv - QL)
        dP = P_up - P_prox
        if dP >= 0.0:
            dQv = (dP - Rv * Qv_state) / Lv
        else:
            g = Qv / Q_GATE if Qv < Q_GATE else 1.0
            dQv = (dP * g - Rv * Qv_state) / Lv
        is_open = Qv > Q_OPEN or dP > 0.0
    else:
        num = P_up - P_C + Zc_eff * QL
        Qv = num / (Rv + Zc_eff) if num > 0.0 else 0.0
        P_prox = P_C + Zc_eff * (Qv - QL)
        is_open = num > 0.0
        dQv = 0.0
    dQL = (Zc_eff / L) * (Qv - QL) if wk else 0.0
    return Qv, P_prox, dQL, dQv, is_open


class HiFiModel:
    def __init__(self, theta=None, phi=None):
        self.phi = ph = full_phi(phi)
        p = {**FIXED, **phi_to_lowfi_fixed(ph), **full_theta(theta)}
        self.p = p
        self.T = 60000.0 / p["HR"]
        self.cc = chamber_constants(p)
        if ph["resp_amp"] > 0.0:
            m = max(2, round(p["HR"] / ph["resp_rate"]))
            self.period = m * self.T
            self.beat_index = min(m - 1, int(ph["resp_phase"] * m))
        else:
            self.period, self.beat_index = self.T, 0
        active = {
            "QL_s": ph["Zc_s"] > 0.0 and ph["L_s"] > 0.0,
            "QL_p": ph["Zc_p"] > 0.0 and ph["L_p"] > 0.0,
            "Qv_s": ph["Lv_av"] > 0.0,
            "Qv_p": ph["Lv_pv"] > 0.0,
        }
        extras = [k for k in EXTRA_STATES if active[k]]
        self.state_names = LOWFI_STATES + extras
        self.state_kind = ["V"] * 8 + ["Q"] * len(extras)
        self.n_states = len(self.state_names)
        self._idx = {k: 8 + i for i, k in enumerate(extras)}
        self._build()

    def y0(self):
        return np.concatenate([initial_state(self.p), np.zeros(self.n_states - 8)])

    def _build(self):
        p, T, ph = self.p, self.T, self.phi
        c_lv, c_rv, c_la, c_ra = (self.cc[k] for k in ("lv", "rv", "la", "ra"))
        eps_v, eps_a, AVD = make_epsilon(p["Tmax"], p["tau"]), make_epsilon(p["Tmax_a"], p["tau_a"]), p["AVD"]
        Cas, Cvs, Eap, Cvp = p["Cas"], p["Cvs"], p["Eap"], p["Cvp"]
        Ras, Rap, Rmv, Rtv = p["Ras"], p["Rap"], p["Rmv"], p["Rtv"]
        Rav, Rpv, Rvs, Rvp = p["Rcs"], p["Rcp"], p["Rra"], p["Rla"]
        Zs, Ls, Lvs = ph["Zc_s"], ph["L_s"], ph["Lv_av"]
        Zp, Lp, Lvp = ph["Zc_p"], ph["L_p"], ph["Lv_pv"]
        k_l, k_r = ph["k_av_l"], ph["k_av_r"]
        c_spt = ph["c_spt"]
        Vref_lv, Vref_rv = p["Vref_lv"], p["Vref_rv"]
        A_it, w_it = ph["resp_amp"], 2.0 * math.pi / self.period

        i_QLs, i_QLp, i_Qvs, i_Qvp = (self._idx.get(k, -1) for k in EXTRA_STATES)
        extras_order = [k for k in EXTRA_STATES if k in self._idx]
        n = self.n_states

        def evaluate(t, y):
            Vla, Vlv, Vas, Vvs, Vra, Vrv, Vap, Vvp = y[:8]
            QLs = y[i_QLs] if i_QLs >= 0 else 0.0
            QLp = y[i_QLp] if i_QLp >= 0 else 0.0
            Qvs_ = y[i_Qvs] if i_Qvs >= 0 else 0.0
            Qvp_ = y[i_Qvp] if i_Qvp >= 0 else 0.0
            ea = eps_a(t % T)
            ev = eps_v((t - AVD) % T)
            Pla = chamber_pressure(Vla - k_l * (Vref_lv - Vlv) if k_l else Vla, ea, c_la)
            if c_spt:
                v_spt = septal_volume(Vlv, Vrv, ev, c_lv, c_rv, c_spt)
                Plv = chamber_pressure(Vlv - v_spt, ev, c_lv)
            else:
                v_spt = 0.0
                Plv = chamber_pressure(Vlv, ev, c_lv)
            Pra = chamber_pressure(Vra - k_r * (Vref_rv - Vrv) if k_r else Vra, ea, c_ra)
            Prv = chamber_pressure(Vrv + v_spt, ev, c_rv)
            PCs, Pvs, PCp, Pvp = Vas / Cas, Vvs / Cvs, Vap * Eap, Vvp / Cvp
            if A_it:  # absolute pressures inside the thorax
                pit = -0.5 * A_it * (1.0 - math.cos(w_it * t))
                Pla, Plv, Pra, Prv, PCp, Pvp = Pla + pit, Plv + pit, Pra + pit, Prv + pit, PCp + pit, Pvp + pit
            else:
                pit = 0.0
            Qav, Pas, dQLs, dQvs, av = _outlet(Plv, PCs, QLs, Qvs_, Zs, Ls, Lvs, Rav)
            Qpv, Pap, dQLp, dQvp, pv = _outlet(Prv, PCp, QLp, Qvp_, Zp, Lp, Lvp, Rpv)
            Qmv = (Pla - Plv) / Rmv if Pla > Plv else 0.0
            Qtv = (Pra - Prv) / Rtv if Pra > Prv else 0.0
            P = (Pla, Plv, Pas, Pvs, Pra, Prv, Pap, Pvp, PCs, PCp)
            Q = (Qmv, Qav, (PCs - Pvs) / Ras, (Pvs - Pra) / Rvs,
                 Qtv, Qpv, (PCp - Pvp) / Rap, (Pvp - Pla) / Rvp)
            return P, Q, (dQLs, dQLp, dQvs, dQvp), (av, pv, pit, v_spt)

        def rhs(t, y):
            _, (Qmv, Qav, Qas, Qvs, Qtv, Qpv, Qap, Qvp), dQ, _ = evaluate(t, y)
            out = [Qvp - Qmv, Qmv - Qav, Qav - Qas, Qas - Qvs,
                   Qvs - Qtv, Qtv - Qpv, Qpv - Qap, Qap - Qvp]
            if n > 8:
                d = dict(zip(EXTRA_STATES, dQ))
                out.extend(d[k] for k in extras_order)
            return out

        def pv_gap(t, y):
            P = evaluate(t, y)[0]
            return P[5] - P[6]

        def observe(t, y):
            P, Q, _, (av, pv, pit, v_spt) = evaluate(t, y)
            Pla, Plv, Pas, Pvs, Pra, Prv, Pap, Pvp, PCs, PCp = P
            # Valve states and flows use absolute pressures; intrathoracic
            # channels are reported transmural (first-order respiration removed).
            if pit:
                Pla, Plv, Pra, Prv, Pap, Pvp, PCp = (v - pit for v in (Pla, Plv, Pra, Prv, Pap, Pvp, PCp))
            Qmv, Qav, Qas, Qvs, Qtv, Qpv, Qap, Qvp = Q
            out = dict(zip(LOWFI_STATES, (float(v) for v in y[:8])))
            out.update({
                "Pla": Pla, "Plv": Plv, "Pas": Pas, "Pvs": Pvs,
                "Pra": Pra, "Prv": Prv, "Pap": Pap, "Pvp": Pvp,
                # Flows are labelled by the compartment they enter (binary convention).
                "Qlv": Qmv, "Qas": Qav, "Qvs": Qas, "Qra": Qvs,
                "Qrv": Qtv, "Qap": Qpv, "Qvp": Qap, "Qla": Qvp,
                "mv": float(Pla > Plv), "av": float(av),
                "tv": float(Pra > Prv), "pv": float(pv),
                # High-fi extras: compartment (distal) pressures, inertance and
                # valve flows (Q_L = 0 when that Windkessel is off).
                "Pas_C": PCs, "Pap_C": PCp, "P_it": pit, "V_spt": v_spt,
                "QL_s": float(y[i_QLs]) if i_QLs >= 0 else 0.0,
                "QL_p": float(y[i_QLp]) if i_QLp >= 0 else 0.0,
            })
            return out

        self.rhs, self.pv_gap, self.observe = rhs, pv_gap, observe


def apply_wedge(res, phi):
    """Replace the "Pvp" channel of a SimResult by the wedge observation.

    Pvp_obs = (1 - w) Pvp + w lag(Pla), lag = periodic steady-state first-order
    low-pass with time constant tau_wedge (ms; 0 = no lag). The vessel pressure
    is kept as "Pvp_vessel"; summaries are recomputed. No-op when w_wedge = 0.
    """
    w, tau = phi["w_wedge"], phi["tau_wedge"]
    if w == 0.0 or not res.success:
        return res
    from scipy import signal

    from .solver import summaries

    dt = res.t_dense[1] - res.t_dense[0]
    pla = np.asarray(res.dense["Pla"], float)
    if tau > 0.0:
        period = pla[:-1]
        a = math.exp(-dt / tau)
        tiles = max(2, math.ceil(12.0 * tau / (dt * period.size)) + 1)
        y, _ = signal.lfilter([1.0 - a], [1.0, -a], np.tile(period, tiles), zi=[a * period[0]])
        lagged = np.append(y[-period.size:], y[-period.size])
    else:
        lagged = pla
    res.dense["Pvp_vessel"] = res.dense["Pvp"]
    res.dense["Pvp"] = (1.0 - w) * res.dense["Pvp"] + w * lagged
    factor = (len(res.t_dense) - 1) // (len(res.t201) - 1)
    res.waves201 = {k: v[::factor] for k, v in res.dense.items()}
    res.summaries = summaries(res.waves201)
    return res
