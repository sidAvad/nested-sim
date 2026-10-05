"""Sweep structural phi families over prior theta to calibrate phi ranges.

Each config switches on one family (others off):
  wk_s : Zc_s = f * Ras, L_s = tau_L * Zc_s
  wk_p : Zc_p = f * Rap, L_p = tau_L * Zc_p
  lv   : Lv_av = Lv_pv = Lv
For every (config, theta) it records waveform features of high-fi relative to
low-fi on the same theta, then prints per-config medians and 90th percentiles.

Usage: python scripts/prior_sweep.py OUT.csv [--grid v2] [--n 300] [--workers 250]
"""

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import full_theta, sample_theta  # noqa: E402


def hangout(r, side):
    """ms from ventricular pressure falling below arterial pressure to the end of forward flow."""
    pv, pa, q = (("Plv", "Pas", "Qas") if side == "s" else ("Prv", "Pap", "Qap"))
    t, Pv, Pa, Q = r.t_dense, r.dense[pv], r.dense[pa], r.dense[q]
    i_peak = int(np.argmax(Q))
    after = np.where(Q[i_peak:] <= 1e-6)[0]
    if after.size == 0:
        return np.nan
    i_end = i_peak + after[0]
    seg = slice(i_peak, i_end + 1)
    cross = np.where((Pv[seg][:-1] >= Pa[seg][:-1]) & (Pv[seg][1:] < Pa[seg][1:]))[0]
    return float(t[i_end] - t[i_peak + cross[0] + 1]) if cross.size else 0.0


def rebound(r, side):
    """Largest rise of arterial pressure after it starts falling from the systolic
    peak, before the next valve opening (dicrotic-wave / undershoot amplitude)."""
    P, Q = (r.dense["Pas"], r.dense["Qas"]) if side == "s" else (r.dense["Pap"], r.dense["Qap"])
    i_peak = int(np.argmax(P))
    end = len(P)
    while end > i_peak + 1 and Q[end - 1] > 1e-6:  # drop the next beat's ejection at the window end
        end -= 1
    seg = P[i_peak:end]
    return float(np.max(seg - np.minimum.accumulate(seg))) if seg.size else 0.0


def features(lo, hi):
    out = {}
    for side in ("s", "p"):
        out[f"rebound_{side}"] = rebound(hi, side)
        out[f"rebound_lo_{side}"] = rebound(lo, side)
    for side, P, PC in (("s", "Pas", "Pas_C"), ("p", "Pap", "Pap_C")):
        pp_lo = np.ptp(lo.dense[P])
        out[f"pp_ratio_{side}"] = np.ptp(hi.dense[P]) / pp_lo
        out[f"pp_add_{side}"] = np.ptp(hi.dense[P]) - pp_lo
        out[f"dip_{side}"] = float(np.max(hi.dense[PC] - hi.dense[P]).clip(min=0))
        out[f"peak_shift_{side}"] = float(lo.t_dense[np.argmax(lo.dense[P])] - hi.t_dense[np.argmax(hi.dense[P])])
        out[f"hang_{side}"] = hangout(hi, side)
    out["sbp_diff"] = hi.summaries["sbp"] - lo.summaries["sbp"]
    out["map_diff"] = hi.summaries["map"] - lo.summaries["map"]
    out["sv_diff"] = hi.summaries["sv"] - lo.summaries["sv"]
    return out


GRIDS = {
    "v1": {"wk_s": ([0.02, 0.04, 0.07, 0.12], [30.0, 100.0, 300.0, 1000.0]),
           "wk_p": ([0.05, 0.1, 0.2, 0.4], [30.0, 100.0, 300.0, 1000.0]),
           "lv": [30.0, 100.0, 300.0, 600.0], "lv_pv": []},
    "v2": {"wk_s": ([0.02, 0.03, 0.05, 0.08], [100.0, 300.0, 1000.0, 3000.0]),
           "wk_p": ([0.05, 0.1, 0.2, 0.3], [100.0, 300.0, 1000.0, 3000.0]),
           "lv": [], "lv_pv": [50.0, 100.0, 150.0, 200.0, 300.0]},
}


def configs(grid):
    g = GRIDS[grid]
    for f, tl in itertools.product(*g["wk_s"]):
        yield ("wk_s", f, tl)
    for f, tl in itertools.product(*g["wk_p"]):
        yield ("wk_p", f, tl)
    for lv in g["lv"]:
        yield ("lv", lv, np.nan)
    for lv in g["lv_pv"]:
        yield ("lv_pv", lv, np.nan)


def phi_for(cfg, th):
    fam, a, b = cfg
    if fam == "wk_s":
        z = a * th["Ras"]
        return {"Zc_s": z, "L_s": b * z}
    if fam == "wk_p":
        z = a * th["Rap"]
        return {"Zc_p": z, "L_p": b * z}
    if fam == "lv_pv":
        return {"Lv_pv": a}
    return {"Lv_av": a, "Lv_pv": a}


def run(args):
    i, th, cfgs = args
    lo = simulate(th)
    rows = []
    if not lo.success:
        return rows
    for cfg in cfgs:
        hi = simulate(th, phi_for(cfg, full_theta(th)))
        row = {"theta": i, "family": cfg[0], "a": cfg[1], "b": cfg[2], "ok": hi.success,
               "HR": th["HR"], "Ras": th["Ras"], "Rap": th["Rap"], "Eap": th["Eap"],
               "Cas": th["Cas"], "Emax_RV": th["Emax_RV"]}
        if hi.success:
            row.update(features(lo, hi))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--grid", default="v2", choices=list(GRIDS))
    a = ap.parse_args()
    thetas = sample_theta(np.random.default_rng(123), a.n)
    cfgs = list(configs(a.grid))
    # One task per (theta, family) keeps tasks short and balanced.
    fams = sorted({c[0] for c in cfgs})
    tasks = [(i, th, [c for c in cfgs if c[0] == fam])
             for i, th in enumerate(thetas) for fam in fams]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run, tasks, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), "rows")


if __name__ == "__main__":
    main()
