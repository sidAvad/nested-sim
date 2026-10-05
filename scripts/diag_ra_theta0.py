"""Which phi group makes the sharp RA swings at theta 0 of plot_phi_spread.py?

For each of the figure's phi draws (seed 1000 * theta_seed + k) runs high-fi with
one phi group at a time and reports the RA range, the maximum |dPra/dt| and its
time, plus state context (tricuspid valve, RA volume) at that time.

Usage: python scripts/diag_ra_theta0.py [--theta-seed 0] [--n-phi 8]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import full_theta, sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402

GROUPS = {
    "none": [],
    "res": ["Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp"],
    "wk": ["Zc_s", "L_s", "Zc_p", "L_p"],
    "lv": ["Lv_av", "Lv_pv"],
    "av": ["k_av_l", "k_av_r"],
    "resp": ["resp_amp", "resp_rate", "resp_phase"],
    "all_but_av": ["Rmv", "Rav", "Rtv", "Rpv", "Rvs", "Rvp", "Zc_s", "L_s", "Zc_p", "L_p",
                   "Lv_av", "Lv_pv", "resp_amp", "resp_rate", "resp_phase"],
    "all": None,
}


def metrics(r):
    t, p = r.t_dense, r.dense["Pra"]
    dp = np.gradient(p, t)
    i = int(np.argmax(np.abs(dp)))
    return {"ra_range": float(np.ptp(p)), "max_slope": float(abs(dp[i])), "t_max_slope": float(t[i]),
            "tv_at": float(r.dense["tv"][i]), "Vra_min": float(r.dense["Vra"].min()),
            "Vrv_max": float(r.dense["Vrv"].max())}


def run(args):
    theta_seed, k = args
    th = full_theta(sample_theta(np.random.default_rng(theta_seed)))
    phi = sample_phi(np.random.default_rng(1000 * theta_seed + k), 1.0, th)
    rows = []
    for g, keys in GROUPS.items():
        sub = {kk: v for kk, v in phi.items() if kk in PHI_STRUCT_OFF and (keys is None or kk in keys)}
        r = simulate(th, sub)
        row = {"phi_draw": k, "group": g, "ok": r.success, "k_av_r": phi["k_av_r"], "Rtv": phi["Rtv"]}
        if r.success:
            row.update(metrics(r))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-seed", type=int, default=0)
    ap.add_argument("--n-phi", type=int, default=8)
    a = ap.parse_args()
    th = full_theta(sample_theta(np.random.default_rng(a.theta_seed)))
    print("theta:", {k: round(v, 3) for k, v in th.items()})
    with ProcessPoolExecutor(a.n_phi) as ex:
        df = pd.DataFrame([r for rs in ex.map(run, [(a.theta_seed, k) for k in range(a.n_phi)]) for r in rs])
    pd.set_option("display.width", 200)
    print(df.groupby("group")[["ra_range", "max_slope"]].agg(["median", "max"]).round(1))
    print(df[df.group.isin(["none", "av", "all", "all_but_av"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
