"""Sweep the septal compliance c_spt (ventricular interdependence) over prior theta.

Reports success, septal volume range, and changes vs low-fi in RV systolic
pressure, LV end-diastolic pressure, SV and mPAP, split by PVR tercile.

Usage: python scripts/vi_sweep.py OUT.csv [--n 300] [--workers 100]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import full_theta, sample_theta  # noqa: E402

CS = [0.02, 0.05, 0.1, 0.2, 0.4]


def run(i):
    th = full_theta(sample_theta(np.random.default_rng(40_000 + i)))
    lo = simulate(th)
    if not lo.success:
        return []
    rows = []
    for c in CS:
        t = time.time()
        r = simulate(th, {"c_spt": c})
        row = {"theta": i, "c": c, "ok": r.success, "sec": time.time() - t, "Rap": th["Rap"],
               "lo_sec": None}
        if r.success:
            v = r.dense["V_spt"]
            row.update(vspt_min=float(v.min()), vspt_max=float(v.max()),
                       d_rv_s=r.summaries["rv_s"] - lo.summaries["rv_s"],
                       rel_rv_s=r.summaries["rv_s"] / lo.summaries["rv_s"] - 1,
                       d_lvedp=r.summaries["lvedp"] - lo.summaries["lvedp"],
                       d_sv=r.summaries["sv"] - lo.summaries["sv"],
                       d_mpap=r.summaries["mpap"] - lo.summaries["mpap"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=100)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(run, range(a.n), chunksize=1) for r in rs])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    df["pvr"] = pd.qcut(df.Rap, 3, labels=["low PVR", "mid PVR", "high PVR"])
    q = lambda p: (lambda x: x.quantile(p))  # noqa: E731
    g = df.groupby("c").agg(ok=("ok", "mean"), sec=("sec", "median"),
                            vspt_lo=("vspt_min", q(.05)), vspt_hi=("vspt_max", q(.95)),
                            rel_rv_s=("rel_rv_s", "median"), d_lvedp=("d_lvedp", "median"),
                            d_lvedp_p90=("d_lvedp", q(.9)), d_sv=("d_sv", "median"), d_mpap=("d_mpap", "median"))
    print(g.round(3).to_string())
    print("\nby PVR tercile (median):")
    print(df.groupby(["c", "pvr"], observed=True)[["vspt_min", "rel_rv_s", "d_lvedp", "d_sv"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
