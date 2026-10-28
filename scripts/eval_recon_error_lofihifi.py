"""Reconstruction error for the lofihifi methods on hifi_v2strong_test: forward-simulate
each method's posterior mean theta through nested_sim.simulate and compare the predicted
waveform/summary against the actual observed one, in raw physical units (mmHg for
waveforms/map/sbp/dbp, log-units for logsv -- see load_true). This is a different axis
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
OUT_DIR = HOME / "results/nested-sim/zscore_overall_6methods"
NESTED_DATA = HOME / "outputs/nested-sim/data"

sys.path.insert(0, str(HOME / "results/cv-dann-sbi/scripts"))
from smc_gibbs_nestedsim import _sim_one, WAVES  # noqa: E402


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
    """scal = [Pas.mean(), Pas.max(), Pas.min(), log(Vlv range)] -- i.e. map/sbp/dbp
    (mmHg) and a log-space stroke-volume proxy."""
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

    true_data = [load_true(SET_NAME, man, idx) for idx in pick]
    hrs = [d[0] for d in true_data]

    header = ["method", "n_ok", "Prv_mae", "Pra_mae", "Pvp_mae", "Pap_mae", "wave_mae_overall",
              "map_mae", "sbp_mae", "dbp_mae", "logsv_mae"]
    print(" ".join(f"{h:>16}" for h in header))
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

        true_waves_ok = np.stack([true_data[i][1] for i in range(len(pick)) if ok[i]])
        wave_mae = np.mean(np.abs(pred_waves - true_waves_ok), axis=(0, 2))  # per channel
        wave_mae_overall = float(wave_mae.mean())

        true_scal_ok = np.stack([true_data[i][2] for i in range(len(pick)) if ok[i]])
        scal_mae = np.mean(np.abs(pred_scal - true_scal_ok), axis=0)

        row = [label, n_ok, *wave_mae.tolist(), wave_mae_overall, *scal_mae.tolist()]
        rows.append(row)
        print(f"{label:>16} {n_ok:>16} " +
              " ".join(f"{v:>16.2f}" for v in wave_mae) + f" {wave_mae_overall:>16.2f} " +
              " ".join(f"{v:>16.2f}" for v in scal_mae))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"recon_error_{SET_NAME}.csv"
    with open(out_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(header)
        wtr.writerows([[v if isinstance(v, str) else (int(v) if i == 1 else round(float(v), 2))
                        for i, v in enumerate(row)] for row in rows])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
