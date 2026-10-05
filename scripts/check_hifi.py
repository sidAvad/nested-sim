"""Robustness check of the high-fi model over random (theta, phi).

For each theta: low-fi, high-fi at phi off (must equal low-fi), and high-fi at
phi = sample_phi(alpha). Reports success rates, runtimes, nesting error, and an
Lv -> 0 sweep at default theta.

Usage: python scripts/check_hifi.py [--n 200] [--alpha 1.0] [--workers 64]
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import PHI_OFF, SolverSettings, simulate  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402

SETTINGS = SolverSettings(method="LSODA")


def _timed(theta, phi):
    t = time.time()
    r = simulate(theta, phi, SETTINGS)
    return r, time.time() - t


def run_one(args):
    seed, alpha = args
    rng = np.random.default_rng(seed)
    theta = sample_theta(rng)
    phi = {k: v for k, v in sample_phi(rng, alpha, theta).items() if k in PHI_STRUCT_OFF}
    lo, t_lo = _timed(theta, None)
    off, _ = _timed(theta, PHI_OFF)
    hi, t_hi = _timed(theta, phi)
    nest = (max(np.abs(lo.dense[k] - off.dense[k]).max() for k in lo.dense)
            if lo.success and off.success else np.nan)
    return dict(seed=seed, lo=lo.success, off=off.success, hi=hi.success, hi_msg=hi.message,
                t_lo=t_lo, t_hi=t_hi, nest=nest,
                dsbp=hi.summaries["sbp"] - lo.summaries["sbp"] if lo.success and hi.success else np.nan,
                dspa=hi.summaries["spa"] - lo.summaries["spa"] if lo.success and hi.success else np.nan)


def lv_sweep():
    base = simulate({}, None, SETTINGS)
    print("\nLv -> 0 sweep at default theta (both valves), max |diff| vs Lv = 0 branch:")
    for lv in [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1]:
        r, t = _timed({}, {"Lv_av": lv, "Lv_pv": lv})
        if not r.success:
            print(f"  Lv={lv:6.1f}  FAILED: {r.message}")
            continue
        d = {k: np.abs(r.waves201[k] - base.waves201[k]).max() for k in ["Pas", "Pap", "Plv", "Prv"]}
        print(f"  Lv={lv:6.1f}  {t:5.1f}s  " + "  ".join(f"{k} {v:.3e}" for k, v in d.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_one, [(s, a.alpha) for s in range(a.n)]))
    lo = np.array([r["lo"] for r in rows])
    off = np.array([r["off"] for r in rows])
    hi = np.array([r["hi"] for r in rows])
    print(f"n={a.n} alpha={a.alpha}")
    print(f"success: lowfi {lo.mean():.3f}  hifi(off) {off.mean():.3f}  hifi(phi) {hi.mean():.3f}")
    print(f"lowfi ok but hifi failed: {(lo & ~hi).sum()}   hifi ok but lowfi failed: {(~lo & hi).sum()}")
    nest = np.array([r["nest"] for r in rows])
    print(f"nesting max |hifi(off) - lowfi| over all channels: {np.nanmax(nest):.2e}")
    t_lo = np.array([r["t_lo"] for r in rows])
    t_hi = np.array([r["t_hi"] for r in rows])
    print(f"runtime s  lowfi median {np.median(t_lo):.2f} p95 {np.percentile(t_lo, 95):.2f} max {t_lo.max():.2f}")
    print(f"           hifi  median {np.median(t_hi):.2f} p95 {np.percentile(t_hi, 95):.2f} max {t_hi.max():.2f}")
    dsbp = np.array([r["dsbp"] for r in rows])
    dspa = np.array([r["dspa"] for r in rows])
    print(f"hifi - lowfi SBP: median {np.nanmedian(dsbp):+.2f}  IQR [{np.nanpercentile(dsbp, 25):+.2f}, {np.nanpercentile(dsbp, 75):+.2f}]")
    print(f"hifi - lowfi sPA: median {np.nanmedian(dspa):+.2f}  IQR [{np.nanpercentile(dspa, 25):+.2f}, {np.nanpercentile(dspa, 75):+.2f}]")
    for r in rows:
        if r["lo"] and not r["hi"]:
            print(f"  seed {r['seed']}: hifi failed: {r['hi_msg']}")
    lv_sweep()


if __name__ == "__main__":
    main()
