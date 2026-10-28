"""Reconstruction error for the lofihifi methods on hifi_v2strong_test: forward-simulate
each method's posterior mean theta through nested_sim.simulate and compare the predicted
waveform/summary against the actual observed one, in raw physical units (mmHg for
waveforms/map/sbp/dbp, mL for sv -- see load_true). This is a different axis
from the z-score/shrinkage plots (which measure point-estimate accuracy relative to
posterior width) -- reconstruction error measures point-estimate accuracy relative to
the observation itself, independent of how confident the posterior claims to be.

All error numbers are MAE (mean absolute error), not RMSE -- directly interpretable in
the original units. wave_mae_overall is the average of the 4 per-channel waveform MAEs
(Prv/Pra/Pvp/Pap), one summary number for "how far off is the reconstructed waveform."

Test cases are the same subsample eval_pretrained_lofihifi.py used to build each
method's zscore_shrinkage_data.npz (same manifest RNG pick: set_name, n, seed), so
post_mean rows line up by position with no extra bookkeeping.

Usage:
    python scripts/eval_recon_error_lofihifi.py [--n-cases 1000] [--workers 128]
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

HOME = Path.home()
OUT = HOME / "results/nested-sim/pretrained_eval_lofihifi"
OUT_DIR = HOME / "results/nested-sim/pretrained_eval_lofihifi/zscore_overall_6methods"
NESTED_DATA = HOME / "outputs/nested-sim/data"

sys.path.insert(0, str(HOME / "results/cv-dann-sbi/scripts"))
from smc_gibbs_nestedsim import WAVES  # noqa: E402  (also puts nested_sim on sys.path)
from nested_sim import simulate  # noqa: E402
from nested_sim.params import from_binary_names  # noqa: E402
from eval_common import PARAM_KEYS_INFER  # noqa: E402


def _safe_sim_one(args):
    """_sim_one, but some methods' posterior means land outside the simulator's valid
    domain (e.g. a solver-internal log1p(negative) for an unphysical volume/pressure
    combination) and raise instead of returning r.success=False -- that's a real
    reconstruction failure (an invalid point estimate), not a bug in this script, so it's
    counted the same way as a non-converged simulate() call (returns None)."""
    try:
        return _sim_one_raw(args)
    except Exception:
        return None


METHODS = [
    ("npe-gibbs-lofihifi", "Gibbs NPE"),
    ("spin-pathD-lofihifi", "SPIN Dual-flow (Path D)"),
    ("npe-dann-lofihifi", "NPE DANN / WDGRL"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
    ("spin-singleflow-lofihifi", "SPIN single-flow"),
]
SET_NAME = "hifi_v2strong_test"


def manifest_pick(set_name, n, seed):
    man = json.load(open(NESTED_DATA / set_name / "manifest.json"))["index"]
    return sorted(np.random.default_rng(seed).choice(len(man), n, replace=False)), man


def load_true(set_name, man, idx):
    """scal = [Pas.mean(), Pas.max(), Pas.min(), Vlv range] -- i.e. map/sbp/dbp (mmHg)
    and stroke volume (mL; confirmed equal to summaries/sv up to float precision)."""
    e = man[idx]
    with h5py.File(NESTED_DATA / set_name / e["file"], "r") as f:
        g = f[e["group"]]
        hr = float(g["parameters/HR"][()])
        waves = np.stack([g[f"waves/{k}"][:] for k in WAVES]).astype(np.float64)
        pas, vlv = g["waves/Pas"][:].astype(np.float64), g["waves/Vlv"][:].astype(np.float64)
        scal = np.array([pas.mean(), pas.max(), pas.min(), vlv.max() - vlv.min()])
    return hr, waves, scal


def _sim_one_raw(args):
    """Like smc_gibbs_nestedsim._sim_one, but the 4th scalar is raw stroke volume
    (mL), not its log -- matches load_true's raw-units scal so the two sides of the
    MAE comparison are in the same units."""
    theta_binary_row, hr = args
    th = from_binary_names(dict(zip(PARAM_KEYS_INFER, theta_binary_row.tolist())))
    th["HR"] = hr
    r = simulate(th)
    if not r.success:
        return None
    w = np.stack([r.waves201[c] for c in WAVES])
    pas, vlv = r.waves201["Pas"], r.waves201["Vlv"]
    scal = np.array([pas.mean(), pas.max(), pas.min(), vlv.max() - vlv.min()])
    return w, scal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=1000)
    ap.add_argument("--n-build", type=int, default=1000, help="must match --n used to build zscore_shrinkage_data.npz")
    ap.add_argument("--seed", type=int, default=0, help="must match --seed used to build zscore_shrinkage_data.npz")
    ap.add_argument("--workers", type=int, default=128)
    a = ap.parse_args()

    pick, man = manifest_pick(SET_NAME, a.n_build, a.seed)
    pick = pick[: a.n_cases]

    true_data = [load_true(SET_NAME, man, idx) for idx in pick]
    hrs = [d[0] for d in true_data]

    header = ["method", "n_ok", "Prv_mae", "Pra_mae", "Pvp_mae", "Pap_mae", "wave_mae_overall",
              "map_mae", "sbp_mae", "dbp_mae", "sv_mae"]

    # Pass 1: simulate every method's posterior mean, cache per-case (waves, scal, ok) --
    # needed both for each method's own-subset numbers and for the common-subset ones below.
    per_method = {}
    for method, label in METHODS:
        d = np.load(OUT / method / SET_NAME / "zscore_shrinkage_data.npz", allow_pickle=True)
        post_mean = d["post_mean"][: a.n_cases]
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(_safe_sim_one, [(post_mean[i], hrs[i]) for i in range(len(pick))], chunksize=1))
        ok = np.array([r is not None for r in results])
        per_method[method] = (label, results, ok)

    def mae_row(label, results, ok):
        n_ok = int(ok.sum())
        pred_waves = np.stack([r[0] for r, o in zip(results, ok) if o])
        pred_scal = np.stack([r[1] for r, o in zip(results, ok) if o])
        true_waves_ok = np.stack([true_data[i][1] for i in range(len(pick)) if ok[i]])
        wave_mae = np.mean(np.abs(pred_waves - true_waves_ok), axis=(0, 2))  # per channel
        wave_mae_overall = float(wave_mae.mean())
        true_scal_ok = np.stack([true_data[i][2] for i in range(len(pick)) if ok[i]])
        scal_mae = np.mean(np.abs(pred_scal - true_scal_ok), axis=0)
        return [label, n_ok, *wave_mae.tolist(), wave_mae_overall, *scal_mae.tolist()]

    def print_and_write(rows, out_path):
        print(" ".join(f"{h:>16}" for h in header))
        for row in rows:
            print(f"{row[0]:>16} {row[1]:>16} " + " ".join(f"{v:>16.2f}" for v in row[2:]))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(header)
            wtr.writerows([[v if isinstance(v, str) else (int(v) if i == 1 else round(float(v), 2))
                            for i, v in enumerate(row)] for row in rows])
        print("wrote", out_path)

    # Own-subset numbers (each method scored only on its own successfully-simulated cases --
    # the original table; confounded by n_ok varying 529-994 across methods, see common-subset below).
    own_rows = [mae_row(label, results, ok) for label, results, ok in per_method.values()]
    print_and_write(own_rows, OUT_DIR / f"recon_error_{SET_NAME}.csv")

    # Common-subset numbers: restrict every method to the intersection of cases where ALL
    # six methods' posterior means were simulatable -- removes the selection-effect confound
    # where a method with a lower solve-success rate is implicitly scored on an easier subset.
    common_ok = np.all([ok for _, _, ok in per_method.values()], axis=0)
    print(f"\ncommon subset: {int(common_ok.sum())}/{len(pick)} cases where all {len(METHODS)} methods' "
          f"posterior means simulated successfully\n")
    common_rows = [mae_row(label, results, common_ok) for label, results, _ in per_method.values()]
    print_and_write(common_rows, OUT_DIR / f"recon_error_{SET_NAME}_common_subset.csv")


if __name__ == "__main__":
    main()
