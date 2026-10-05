"""Evaluate the structural phi prior over prior theta.

Modes:
  zeta  : valve inertance alone at fixed damping ratio zeta_v in {0.3, ..., 1.5}
          (Lv = R^2 / (4 zeta_v^2 Emax), low-fi valve resistances)
  joint : structural phi from sample_phi(rng, 1.0, theta) with the configured
          priors (configs/phi_priors.toml), all families together

Usage: python scripts/prior_proposal.py {zeta,joint} OUT.csv [--n 300] [--workers 250]
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
from nested_sim.phi import PHI_STRUCT_OFF, load_priors, sample_phi  # noqa: E402
from prior_sweep import features  # noqa: E402


def n_flow_peaks(r, ch):
    q = r.dense[ch]
    on = q[1:-1] > 1e-6
    return int(np.sum((q[1:-1] > q[:-2]) & (q[1:-1] >= q[2:]) & on))


def run(args):
    mode, i, th, priors_path = args
    th = full_theta(th)
    lo = simulate(th)
    if not lo.success:
        return []
    if mode == "zeta":
        cases = [(z, {"Lv_av": PHI_STRUCT_OFF["Rav"] ** 2 / (4 * z * z * th["Emax_LV"]),
                      "Lv_pv": PHI_STRUCT_OFF["Rpv"] ** 2 / (4 * z * z * th["Emax_RV"])})
                 for z in (0.3, 0.5, 0.7, 1.0, 1.5)]
    else:
        phi = sample_phi(np.random.default_rng(10_000 + i), 1.0, th, load_priors(priors_path))
        cases = [(np.nan, {k: v for k, v in phi.items() if k in PHI_STRUCT_OFF})]
    rows = []
    for z, phi in cases:
        hi = simulate(th, phi)
        row = {"theta": i, "zeta": z, "ok": hi.success, **{f"phi_{k}": v for k, v in phi.items()},
               "peaks_lo_s": n_flow_peaks(lo, "Qas"), "peaks_lo_p": n_flow_peaks(lo, "Qap")}
        if hi.success:
            row.update(features(lo, hi))
            row.update(peaks_s=n_flow_peaks(hi, "Qas"), peaks_p=n_flow_peaks(hi, "Qap"))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zeta", "joint"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--priors", default=None, help="phi prior TOML (default configs/phi_priors.toml)")
    a = ap.parse_args()
    thetas = sample_theta(np.random.default_rng(123), a.n)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run, [(a.mode, i, th, a.priors) for i, th in enumerate(thetas)], chunksize=1)
                for r in rs]
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), "rows")


if __name__ == "__main__":
    main()
