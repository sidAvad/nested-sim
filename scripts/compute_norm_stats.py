"""Normalisation statistics for a generated training set, in the format of the
cv-* repos' norm_stats.json (cv-sbi-sim/scripts/compute_stats.py): per-parameter
mean/std over sims, per-waveform-channel mean/std over sims and time points
(float64), plus a "scalars" section (map, sbp, dbp, sv, hr) for the loaders'
"zscore" scalar mode.

Uses every sim in --n-files randomly chosen batch files of the set (each batch
is already a random draw from the prior).

Usage: python scripts/compute_norm_stats.py SET_DIR [--n-files 100] [--workers 32]
       writes SET_DIR/norm_stats.json
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

PARAM_KEYS = ["AVD", "Bla", "Blv", "Bra", "Brv", "Cas", "Cvp", "Cvs", "Eap",
              "Eedref_la", "Eedref_lv", "Eedref_ra", "Eedref_rv",
              "Emax_LA", "Emax_LV", "Emax_RA", "Emax_RV",
              "HR", "Rap", "Ras", "Tmax", "Tmax_a", "Vs", "τ", "τ_a"]
WAVE_KEYS_CONT = ["Pap", "Pas", "Pla", "Plv", "Pra", "Prv", "Pvp", "Pvs",
                  "Qap", "Qas", "Qla", "Qlv", "Qra", "Qrv", "Qvp", "Qvs",
                  "Vap", "Vas", "Vla", "Vlv", "Vra", "Vrv", "Vvp", "Vvs"]


def read_file(path):
    with h5py.File(path, "r") as f:
        sims = sorted(k for k in f.keys() if k.startswith("sim_"))
        params = np.array([[float(f[s][f"parameters/{k}"][()]) for k in PARAM_KEYS] for s in sims])
        waves = np.stack([np.stack([f[s][f"waves/{k}"][:] for k in WAVE_KEYS_CONT]) for s in sims]).astype(np.float64)
    return params, waves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--n-files", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.set_dir).expanduser()
    files = sorted(d.glob("batch_*.h5"))
    pick = sorted(np.random.default_rng(a.seed).choice(len(files), min(a.n_files, len(files)), replace=False))
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(read_file, [files[i] for i in pick]))
    params = np.concatenate([p for p, _ in parts])
    waves = np.concatenate([w for _, w in parts])
    pas, vlv = waves[:, WAVE_KEYS_CONT.index("Pas")], waves[:, WAVE_KEYS_CONT.index("Vlv")]
    scal = {"map": pas.mean(1), "sbp": pas.max(1), "dbp": pas.min(1), "sv": vlv.max(1) - vlv.min(1),
            "hr": params[:, PARAM_KEYS.index("HR")]}
    stats = {
        "parameters": {k: {"mean": float(params[:, i].mean()), "std": float(params[:, i].std())}
                       for i, k in enumerate(PARAM_KEYS)},
        "waves": {k: {"mean": float(waves[:, i].mean()), "std": float(waves[:, i].std())}
                  for i, k in enumerate(WAVE_KEYS_CONT)},
        "scalars": {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in scal.items()},
        "source_file": f"{d.name}: {len(pick)} random batch files (seed {a.seed})",
        "n_sims_used": int(params.shape[0]),
    }
    out = d / "norm_stats.json"
    json.dump(stats, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {out} from {params.shape[0]} sims")
    for sec in ("waves", "scalars"):
        print(f"\n{sec}:")
        for k, v in stats[sec].items():
            print(f"  {k:<6} mean {v['mean']:10.3f}  std {v['std']:10.3f}")


if __name__ == "__main__":
    main()
