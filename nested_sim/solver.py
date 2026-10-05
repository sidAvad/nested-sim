"""Periodic steady-state driver shared by the low-fi and high-fi models.

A model object provides: n_states, state_kind, T, y0(), rhs(t, y),
optionally `period` (forcing period, a whole number of beats; default T) and
`beat_index` (which beat of that period is reported; default 0),
pv_gap(t, y) (pulmonic valve pressure gradient; its upward zero crossing is the
PV_OPEN landmark) and observe(t, y) -> dict of scalar outputs.

Output beat convention (matches Cv8SimApp cycle_type=PV_OPEN): the beat starts
at the first point of the sampling grid j*T/200 (j integer, t = 0 at atrial
onset) at or after pulmonic valve opening, so the landmark is quantised to T/200
as in the binary. t runs 0..T and the 201-point grid includes both endpoints.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

N_OUT = 201  # points of the standard output beat (nt = 200 intervals)

# Scale floors for the periodicity check, per state kind.
_SS_FLOOR = {"V": 1.0, "Q": 0.1}


@dataclass
class SolverSettings:
    method: str = "LSODA"  # validated against the binary; Radau/BDF agree but are 5-10x slower
    rtol: float = 1e-8
    atol: float = 1e-8
    ss_tol: float = 1e-7  # max scaled change of the state between cycle starts
    min_cycles: int = 5
    max_cycles: int = 400
    dense_factor: int = 10  # dense points per 201-grid interval
    max_step: float = np.inf
    # Cap on right-hand-side evaluations per simulated beat (typical: ~1e3
    # low-fi, ~7e3 high-fi with valve inertance). A runaway solve is reported
    # as a failure instead of grinding on (and storing every tiny step).
    max_rhs_per_beat: int = 100_000


@dataclass
class SimResult:
    success: bool
    message: str
    T: float
    n_cycles: int
    ss_error: float
    t0: float = np.nan  # absolute model time of the PV_OPEN landmark
    t_dense: np.ndarray | None = None  # 0..T, (dense_factor*200 + 1,)
    dense: dict = field(default_factory=dict)  # channel -> (n_dense,)
    t201: np.ndarray | None = None
    waves201: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)


class SolverBudgetExceeded(RuntimeError):
    pass


SOLVER_ERRORS = (OverflowError, ValueError, ZeroDivisionError, SolverBudgetExceeded)


def _integrate(model, t_span, y0, s, **kw):
    n_beats = max(1, int(np.ceil((t_span[1] - t_span[0]) / model.T - 1e-9)))
    budget = s.max_rhs_per_beat * n_beats
    calls = [0]

    def rhs(t, y):
        calls[0] += 1
        if calls[0] > budget:
            raise SolverBudgetExceeded(f"more than {budget} rhs evaluations over {n_beats} beat(s)")
        return model.rhs(t, y)

    return solve_ivp(rhs, t_span, y0, method=s.method, rtol=s.rtol,
                     atol=s.atol, max_step=s.max_step, **kw)


def _observe_many(model, t, Y):
    rows = [model.observe(ti, Y[:, i]) for i, ti in enumerate(t)]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def periodic_steady_state(model, s=None, y_init=None):
    """Iterate whole periods from t=0 until the period-start state repeats.

    The period is model.period if set (a whole number of beats), else T.
    Returns (y at period start, period index k so that the state is at t=k*P,
    last scaled change, message or None on success).
    """
    s = s or SolverSettings()
    T = getattr(model, "period", model.T)
    y = np.asarray(model.y0() if y_init is None else y_init, float)
    scale_floor = np.array([_SS_FLOOR[k] for k in model.state_kind])
    err = np.inf
    for k in range(s.max_cycles):
        try:
            sol = _integrate(model, (k * T, (k + 1) * T), y, s)
        except SOLVER_ERRORS as exc:
            return y, k, err, f"solver failed in cycle {k}: {exc!r}"
        if not sol.success:
            return y, k, err, f"solver failed in cycle {k}: {sol.message}"
        y_new = sol.y[:, -1]
        if not np.all(np.isfinite(y_new)):
            return y, k, err, f"non-finite state in cycle {k}"
        err = float(np.max(np.abs(y_new - y) / np.maximum(np.abs(y), scale_floor)))
        y = y_new
        if k + 1 >= s.min_cycles and err < s.ss_tol:
            return y, k + 1, err, None
    return y, s.max_cycles, err, f"no periodic steady state after {s.max_cycles} cycles (err={err:.2e})"


def simulate_model(model, s=None):
    """Periodic steady state, then the reported beat (model.beat_index)."""
    s = s or SolverSettings()
    y, k, err, msg = periodic_steady_state(model, s)
    res = SimResult(success=msg is None, message=msg or "ok", T=model.T, n_cycles=k, ss_error=err)
    if msg is not None:
        return res
    return _extract_beat(model, y, k, getattr(model, "beat_index", 0), s, res)


def simulate_beats(model, s=None):
    """Periodic steady state, then every beat of the forcing period.

    Returns a list of SimResult, one per beat (a single beat when the model has
    no forcing period).
    """
    s = s or SolverSettings()
    y, k, err, msg = periodic_steady_state(model, s)
    n_beats = round(getattr(model, "period", model.T) / model.T)
    out = []
    for j in range(n_beats):
        res = SimResult(success=msg is None, message=msg or "ok", T=model.T, n_cycles=k, ss_error=err)
        out.append(res if msg is not None else _extract_beat(model, y, k, j, s, res))
    return out


def _output_beat(model, y, k, j, s, res):
    """Integrate to beat j's PV_OPEN grid point and over that beat."""
    T = model.T
    tk = k * getattr(model, "period", T)
    if j:
        sol = _integrate(model, (tk, tk + j * T), y, s)
        y = sol.y[:, -1]
        tk += j * T
    ev = lambda t, yy: model.pv_gap(t, yy)  # noqa: E731
    ev.direction = 1.0
    sol = _integrate(model, (tk, tk + T), y, s, events=ev)
    dt = T / (N_OUT - 1)
    if sol.t_events[0].size:
        t_open = sol.t_events[0][0]
        t0 = np.ceil(t_open / dt - 1e-9) * dt
    else:  # pulmonic valve never opens/closes: fall back to atrial onset
        t0 = tk
        res.message = "no PV_OPEN event; beat starts at atrial onset"
    if t0 > tk:
        sol = _integrate(model, (tk, t0), y, s)
        y0 = sol.y[:, -1]
    else:
        y0 = y

    n_dense = s.dense_factor * (N_OUT - 1) + 1
    t_rel = np.linspace(0.0, T, n_dense)
    sol = _integrate(model, (t0, t0 + T), y0, s, t_eval=t0 + t_rel)
    if not sol.success:
        res.success, res.message = False, f"solver failed on output beat: {sol.message}"
    return res, t0, t_rel, sol


def _extract_beat(model, y, k, j, s, res):
    """Fill res with beat j of the steady-state period starting at t = k * period."""
    try:
        res, t0, t_rel, sol = _output_beat(model, y, k, j, s, res)
    except SOLVER_ERRORS as exc:
        res.success, res.message = False, f"solver failed on output beat: {exc!r}"
        return res
    if not res.success:
        return res
    n_dense = s.dense_factor * (N_OUT - 1) + 1
    res.t0 = t0
    res.t_dense = t_rel
    res.dense = _observe_many(model, sol.t, sol.y)
    idx = np.arange(0, n_dense, s.dense_factor)
    res.t201 = t_rel[idx]
    res.waves201 = {k_: v[idx] for k_, v in res.dense.items()}
    res.summaries = summaries(res.waves201)
    return res


def _edp(P, valve):
    """P at the last inflow-valve closure on the grid, else P[0].

    Approximates the binary's lvedp/rvedp (exact for most beats, not all);
    not used by any downstream pipeline.
    """
    closes = np.where((valve[:-1] == 1) & (valve[1:] == 0))[0]
    return float(P[closes[-1]]) if closes.size else float(P[0])


def summaries(w):
    """Scalar summaries on the 201-point beat, as defined by Cv8SimApp."""
    return {
        "sv": float(w["Vlv"].max() - w["Vlv"].min()),
        "sbp": float(w["Pas"].max()), "dbp": float(w["Pas"].min()), "map": float(w["Pas"].mean()),
        "spa": float(w["Pap"].max()), "dpa": float(w["Pap"].min()), "mpap": float(w["Pap"].mean()),
        "pcw": float(w["Pvp"].mean()), "cvp": float(w["Pra"].mean()),
        "rv_s": float(w["Prv"].max()), "rv_d": float(w["Prv"].min()), "rv_m": float(w["Prv"].mean()),
        "lv_s": float(w["Plv"].max()), "lv_d": float(w["Plv"].min()), "lv_m": float(w["Plv"].mean()),
        "rvedp": _edp(w["Prv"], w["tv"]), "lvedp": _edp(w["Plv"], w["mv"]),
    }
