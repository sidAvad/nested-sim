"""Attribute high-fi effects (failure, SBP/sPA shift, runtime) to phi groups.

For each theta, draws phi at alpha = 1 and runs high-fi with only one group on.
Usage: python scripts/attribute_phi.py [--n 120] [--seeds 53,127] [--workers 200]
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
from nested_sim.params import sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402

GROUPS = {
    "res": ["Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp"],
    "wk": ["Zc_s", "L_s", "Zc_p", "L_p"],
    "lv": ["Lv_av", "Lv_pv"],
    "av": ["k_av_l", "k_av_r"],
    "resp": ["resp_amp", "resp_rate", "resp_phase"],
    "all": None,
}


def run(seed):
    rng = np.random.default_rng(seed)  # same draws as check_hifi.py
    th = sample_theta(rng)
    phi = sample_phi(rng, 1.0, th)
    lo = simulate(th)
    rows = []
    for g, keys in GROUPS.items():
        sub = {k: v for k, v in phi.items() if k in PHI_STRUCT_OFF and (keys is None or k in keys)}
        t = time.time()
        r = simulate(th, sub)
        row = {"seed": seed, "group": g, "ok": r.success, "lo_ok": lo.success, "sec": time.time() - t,
               "msg": r.message}
        if r.success and lo.success:
            row.update(dsbp=r.summaries["sbp"] - lo.summaries["sbp"], dspa=r.summaries["spa"] - lo.summaries["spa"],
                       dmap=r.summaries["map"] - lo.summaries["map"], dsv=r.summaries["sv"] - lo.summaries["sv"])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seeds", default="53,127")
    ap.add_argument("--workers", type=int, default=200)
    a = ap.parse_args()
    seeds = sorted(set(range(a.n)) | {int(s) for s in a.seeds.split(",")})
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(run, seeds, chunksize=1) for r in rs])
    d = df[df.lo_ok]
    print(d.groupby("group").agg(ok=("ok", "mean"), sec_med=("sec", "median"), sec_p95=("sec", lambda x: x.quantile(.95)),
                                 dsbp=("dsbp", "median"), dspa=("dspa", "median"), dmap=("dmap", "median"),
                                 dsv=("dsv", "median")).round(2).to_string())
    print("\nfailures:")
    print(d[~d.ok][["seed", "group", "msg"]].to_string(index=False))


if __name__ == "__main__":
    main()
