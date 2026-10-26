"""Reconstruction error for the lofihifi methods on hifi_v2strong_test: forward-simulate
each method's posterior mean theta through nested_sim.simulate and compare the predicted
waveform/summary against the actual observed one. This is a different axis from the
z-score/shrinkage plots (which measure point-estimate accuracy relative to posterior
width) -- reconstruction error measures point-estimate accuracy relative to the
observation itself, independent of how confident the posterior claims to be.

Two metrics, both forward-simulated once per case per method (nested_sim.simulate,
~1s/call, parallelized via ProcessPoolExecutor):
  - whitened_rmse: RMSE in the same whitened residual space (phi_raw, Gibbs NPE's frozen
    metric M) used by the SMC r2 diagnostic -- one metric shared across all methods, so
    magnitudes are directly comparable (not method-specific normalization).
  - per-channel waveform RMSE (mmHg) and summary-stat error (map/sbp/dbp/logsv) in raw
    physical units, for direct interpretability.

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
NESTED_DATA = HOME / "outputs/nested-sim/data"

sys.path.insert(0, str(HOME / "results/cv-dann-sbi/scripts"))
from smc_gibbs_nestedsim import _sim_one, phi_raw, Metric, resolve_gibbs_ckpt, WAVES  # noqa: E402


def _safe_sim_one(args):
    """_sim_one, but some methods' posterior means land outside the simulator's valid
    domain (e.g. a solver-internal log1p(negative) for an unphysical volume/pressure
    combination) and raise instead of returning r.success=False -- that's a real
    reconstruction failure (an invalid point estimate), not a bug in this script, so it's
    counted the same way as a non-converged simulate() call (returns None)."""
    try:
        return _sim_one(args)
    except Exception:
        return None

METHODS = [
    ("npe-gibbs-lofihifi", "Gibbs NPE"),
    ("spin-pathD-lofihifi", "SPIN v3d Path D"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
    ("spin-pathA-lofihifi", "SPIN v3d Path A"),
]
SET_NAME = "hifi_v2strong_test"
# Metric source for whitening -- the Gibbs NPE checkpoint this metric was fit for; reused
# as-is (not refit) so every method's reconstruction error is measured in the same units.
GIBBS_RUN = "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi_300k"


def manifest_pick(set_name, n, seed):
    man = json.load(open(NESTED_DATA / set_name / "manifest.json"))["index"]
    return sorted(np.random.default_rng(seed).choice(len(man), n, replace=False)), man


def load_true(set_name, man, idx):
    e = man[idx]
    with h5py.File(NESTED_DATA / set_name / e["file"], "r") as f:
        g = f[e["group"]]
        hr = float(g["parameters/HR"][()])
        waves = np.stack([g[f"waves/{k}"][:] for k in WAVES]).astype(np.float64)
        pas, vlv = g["waves/Pas"][:].astype(np.float64), g["waves/Vlv"][:].astype(np.float64)
        scal = np.array([pas.mean(), pas.max(), pas.min(), np.log(max(vlv.max() - vlv.min(), 1e-6))])
    return hr, waves, scal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=1000)
    ap.add_argument("--n-build", type=int, default=1000, help="must match --n used to build zscore_shrinkage_data.npz")
    ap.add_argument("--seed", type=int, default=0, help="must match --seed used to build zscore_shrinkage_data.npz")
    ap.add_argument("--workers", type=int, default=128)
    a = ap.parse_args()

    pick, man = manifest_pick(SET_NAME, a.n_build, a.seed)
    pick = pick[: a.n_cases]

    ckpt_dir = resolve_gibbs_ckpt(GIBBS_RUN)
    meta = json.load(open(ckpt_dir / "gibbs_metric.json"))
    metric = Metric.from_npz(ckpt_dir / "gibbs_metric.npz", meta["ell"])

    true_data = [load_true(SET_NAME, man, idx) for idx in pick]
    hrs = [d[0] for d in true_data]
    phi_obs = phi_raw(np.stack([d[1] for d in true_data]), np.stack([d[2] for d in true_data]), metric)

    header = ["method", "n_ok", "whitened_rmse", "Prv_rmse", "Pra_rmse", "Pvp_rmse", "Pap_rmse",
              "map_err", "sbp_err", "dbp_err", "logsv_err"]
    print(" ".join(f"{h:>14}" for h in header))
    rows = []
    for method, label in METHODS:
        d = np.load(OUT / method / SET_NAME / "zscore_shrinkage_data.npz", allow_pickle=True)
        post_mean = d["post_mean"][: a.n_cases]
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(_safe_sim_one, [(post_mean[i], hrs[i]) for i in range(len(pick))], chunksize=1))
        ok = np.array([r is not None for r in results])
        n_ok = int(ok.sum())
        pred_waves = np.stack([r[0] for r, o in zip(results, ok) if o])
        pred_scal = np.stack([r[1] for r, o in zip(results, ok) if o])
        phi_pred = phi_raw(pred_waves, pred_scal, metric)
        phi_true_ok = phi_obs[ok]
        whitened_rmse = float(np.sqrt(np.mean((phi_pred - phi_true_ok) ** 2)))

        true_waves_ok = np.stack([true_data[i][1] for i in range(len(pick)) if ok[i]])
        wave_rmse = np.sqrt(np.mean((pred_waves - true_waves_ok) ** 2, axis=(0, 2)))

        true_scal_ok = np.stack([true_data[i][2] for i in range(len(pick)) if ok[i]])
        scal_err = np.sqrt(np.mean((pred_scal - true_scal_ok) ** 2, axis=0))

        row = [label, n_ok, whitened_rmse, *wave_rmse.tolist(), *scal_err.tolist()]
        rows.append(row)
        print(f"{label:>14} {n_ok:>14} {whitened_rmse:>14.3f} " +
              " ".join(f"{v:>14.3f}" for v in wave_rmse) + " " +
              " ".join(f"{v:>14.3f}" for v in scal_err))

    out_path = OUT / f"recon_error_{SET_NAME}.csv"
    with open(out_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(header)
        wtr.writerows(rows)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
