"""Compare the Python low-fi model with Cv8SimApp reference runs.

Usage: python scripts/compare_binary.py [ref_h5] [--method LSODA]
The reference file comes from running the binary on ref/ref_params.json.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim.lowfi import LowFiModel  # noqa: E402
from nested_sim.params import from_binary_names  # noqa: E402
from nested_sim.solver import SolverSettings, simulate_model  # noqa: E402

CHANNELS = ["Pas", "Pap", "Prv", "Pra", "Pvp", "Plv", "Pla", "Pvs",
            "Vlv", "Vrv", "Vla", "Vra", "Vas", "Vvs", "Vap", "Vvp",
            "Qas", "Qap", "Qlv", "Qrv", "Qla", "Qra", "Qvs", "Qvp"]
SCALARS = ["sv", "sbp", "dbp", "map", "spa", "dpa", "mpap", "pcw", "cvp",
           "rv_s", "rv_d", "rv_m", "lv_s", "lv_d", "lv_m", "rvedp", "lvedp"]


def _run(args):
    theta, method = args
    r = simulate_model(LowFiModel(theta), SolverSettings(method=method))
    return r.success, r.message, r.waves201, r.summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", nargs="?", default="ref/ref_nt200.h5")
    ap.add_argument("--params", default="ref/ref_params.json")
    ap.add_argument("--method", default="LSODA")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    rows = json.load(open(a.params))
    f = h5py.File(a.ref)
    vn = [v.decode() for v in f["var_names"][()]]
    # The binary flags success even when its periodicity check never passed;
    # count those as failures.
    ss_tol = float(f.attrs["sp_steadystate_tol"])
    ok_bin = f["status"]["success"][()].astype(bool) & (f["status"]["steadystate_error"][()] <= 10 * ss_tol)
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_run, [(from_binary_names(r), a.method) for r in rows]))

    wave_err = {c: [] for c in CHANNELS}
    rel_err = {c: [] for c in CHANNELS}
    sc_err = {k: [] for k in SCALARS}
    both = 0
    for i, (ok, msg, w, sm) in enumerate(out):
        if ok != ok_bin[i]:
            print(f"sim {i}: python success={ok} ({msg}), binary success={ok_bin[i]}")
        if not (ok and ok_bin[i]):
            continue
        both += 1
        W = f["waves"][i]
        for c in CHANNELS:
            ref = W[:, vn.index(c)]
            e = np.abs(w[c] - ref).max()
            wave_err[c].append(e)
            if c == "Pas" and e > 0.05:
                print(f"sim {i}: Pas max abs err {e:.3f} (beat window offset?)")
            rel_err[c].append(e / max(np.abs(ref).max(), 1e-12))
        for k in SCALARS:
            sc_err[k].append(abs(sm[k] - f["summaries"][k][i]))
    print(f"\n{both} sims succeeded in both")
    print(f"{'channel':8s} {'max abs err':>12s} {'median':>10s} {'max rel err':>12s}")
    for c in CHANNELS:
        print(f"{c:8s} {np.max(wave_err[c]):12.3e} {np.median(wave_err[c]):10.2e} {np.max(rel_err[c]):12.3e}")
    print(f"\n{'scalar':8s} {'max abs err':>12s} {'median':>10s}")
    for k in SCALARS:
        print(f"{k:8s} {np.max(sc_err[k]):12.3e} {np.median(sc_err[k]):10.2e}")


if __name__ == "__main__":
    main()
