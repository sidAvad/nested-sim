"""Generate the benchmark datasets in the HDF5 + manifest layout of the
Cv8SimApp exports, so the cv-* loaders (dataset.py) read them unchanged.

Sets (sizes from the command line):
  lofi_train                 low-fi, labels used in training
  lofi_test                  low-fi            | same theta, written only when
  hifi_v2_test               high-fi v2        | all three succeed (paired)
  hifi_v2strong_test         high-fi v2-strong |
  hifi_v2_target             high-fi v2, unlabeled target pool (labels stored)
  hifi_v2strong_target       high-fi v2-strong, unlabeled target pool

Layout per set: OUT/<set>/batch_XXXX.h5 with groups sim_XXXXXXX holding
waves/<channel> (24 continuous + 4 valve + t, float32, 201 pts),
summaries/<name>, parameters/<name> (binary names, incl. the fixed ones),
phi/<name> (high-fi only) and attrs (seed, level, alpha); plus
OUT/<set>/manifest.json with "index" [{file, group, seed}] and "config"
(pvar_low/high, simulator, level, priors text, git hash, solver settings).

Theta prior: params.sample_theta_benchmark (uniform, log-uniform for
Emax_RV, Eap, Rap, as the binary simsets). phi: sample_phi at alpha = 1 with the
level's prior file; measurement model off. Each batch file is written
atomically and skipped on rerun, so an interrupted job resumes.

Usage:
  python scripts/make_datasets.py OUT --git-hash HASH [--n-train 990000]
      [--n-test 10000] [--n-target 90000] [--sets all] [--workers 240]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nested_sim import SolverSettings, simulate  # noqa: E402
from nested_sim.params import FIXED, THETA_BOUNDS, sample_theta_benchmark, to_binary_names  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, load_priors, sample_phi  # noqa: E402

WAVES = ["Pap", "Pas", "Pla", "Plv", "Pra", "Prv", "Pvp", "Pvs",
         "Qap", "Qas", "Qla", "Qlv", "Qra", "Qrv", "Qvp", "Qvs",
         "Vap", "Vas", "Vla", "Vlv", "Vra", "Vrv", "Vvp", "Vvs",
         "av", "mv", "pv", "tv"]
LEVELS = {"v2": ROOT / "configs" / "phi_priors_v2.toml",
          "v2strong": ROOT / "configs" / "phi_priors_v2strong.toml"}
SEED_BASE = {"train": 1_000_000_000, "test": 2_000_000_000,
             "target_v2": 3_000_000_000, "target_v2strong": 4_000_000_000}
SETTINGS = SolverSettings(ss_tol=1e-6)
BATCH = 1000


def simulate_one(seed, level):
    """Simulate theta(seed) with low-fi (level None) or high-fi at a level."""
    th = sample_theta_benchmark(np.random.default_rng(seed))
    phi = None
    if level is not None:
        phi = {k: v for k, v in sample_phi(np.random.default_rng([seed, 1]), 1.0, th,
                                           load_priors(LEVELS[level])).items() if k in PHI_STRUCT_OFF}
    r = simulate(th, phi, SETTINGS)
    if not r.success:
        return None
    return {"seed": seed, "theta": th, "phi": phi, "level": level or "lofi",
            "t": r.t201, "waves": {k: r.waves201[k] for k in WAVES}, "summaries": r.summaries}


def job(args):
    kind, seed = args
    if kind == "test":  # paired: all three variants or nothing
        out = [simulate_one(seed, lv) for lv in (None, "v2", "v2strong")]
        return out if all(o is not None for o in out) else None
    level = {"train": None, "target_v2": "v2", "target_v2strong": "v2strong"}[kind]
    return simulate_one(seed, level)


def write_sim(f, gid, rec):
    g = f.create_group(gid)
    g.attrs["seed"] = rec["seed"]
    g.attrs["level"] = rec["level"]
    g.attrs["alpha"] = 0.0 if rec["phi"] is None else 1.0
    g.create_dataset("waves/t", data=np.asarray(rec["t"], np.float32))
    for k, v in rec["waves"].items():
        g.create_dataset(f"waves/{k}", data=np.asarray(v, np.float32))
    for k, v in rec["summaries"].items():
        g.create_dataset(f"summaries/{k}", data=float(v))
    for k, v in to_binary_names({**FIXED, **rec["theta"]}).items():
        g.create_dataset(f"parameters/{k}", data=float(v))
    for k, v in (rec["phi"] or {}).items():
        g.create_dataset(f"phi/{k}", data=float(v))


def write_batch(set_dir, b, recs, first_index):
    tmp = set_dir / f".batch_{b:04d}.h5.tmp"
    with h5py.File(tmp, "w") as f:
        idx = []
        for j, rec in enumerate(recs):
            gid = f"sim_{first_index + j:07d}"
            write_sim(f, gid, rec)
            idx.append({"file": f"batch_{b:04d}.h5", "group": gid, "seed": int(rec["seed"])})
    os.replace(tmp, set_dir / f"batch_{b:04d}.h5")
    json.dump(idx, open(set_dir / f".batch_{b:04d}.index.json", "w"))


def manifest(set_dir, level, simulator, git_hash, n_failed):
    idx = []
    for p in sorted(set_dir.glob(".batch_*.index.json")):
        idx += json.load(open(p))
    priors = LEVELS[level].read_text() if level in LEVELS else None
    cfg = {"pvar_low": to_binary_names({k: lo for k, (lo, hi) in THETA_BOUNDS.items()}),
           "pvar_high": to_binary_names({k: hi for k, (lo, hi) in THETA_BOUNDS.items()}),
           "pfix": to_binary_names(FIXED), "simulator": simulator, "level": level,
           "theta_sampler": "sample_theta_benchmark (log-uniform Emax_RV, Eap, Rap)",
           "alpha": 0.0 if level is None else 1.0, "measurement_model": "off",
           "phi_priors_toml": priors, "git_hash": git_hash,
           "solver": {k: getattr(SETTINGS, k) for k in ("method", "rtol", "atol", "ss_tol", "max_rhs_per_beat")}}
    json.dump({"index": idx, "n_sims": len(idx), "n_failed": n_failed, "config": cfg},
              open(set_dir / "manifest.json", "w"), indent=1)


def run_set(out, name, kind, n, workers, git_hash):
    """Generate n successful sims for one kind ('train', 'target_*', or 'test' = 3 paired sets)."""
    names = ["lofi_test", "hifi_v2_test", "hifi_v2strong_test"] if kind == "test" else [name]
    dirs = [out / s for s in names]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    n_batches = (n + BATCH - 1) // BATCH
    n_failed, t0 = 0, time.time()
    with ProcessPoolExecutor(workers) as ex:
        for b in range(n_batches):
            target = min(BATCH, n - b * BATCH)
            if all((d / f"batch_{b:04d}.h5").exists() for d in dirs):
                continue
            seed_next = SEED_BASE[kind] + b * BATCH * 4  # 4x headroom per batch for failures
            recs = []
            while len(recs) < target:  # oversample to replace failures
                want = int((target - len(recs)) * 1.1) + 8
                seeds = range(seed_next, seed_next + want)
                seed_next += want
                for r in ex.map(job, [(kind, s) for s in seeds], chunksize=4):
                    if r is None:
                        n_failed += 1
                    elif len(recs) < target:
                        recs.append(r)
            for k, d in enumerate(dirs):
                write_batch(d, b, [r[k] for r in recs] if kind == "test" else recs, b * BATCH)
            print(f"[{name}] batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s, failed so far {n_failed})",
                  flush=True)
    for d, s in zip(dirs, names):
        level = None if s.startswith("lofi") else ("v2strong" if "v2strong" in s else "v2")
        manifest(d, level, "lofi" if level is None else "hifi", git_hash, n_failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--git-hash", required=True)
    ap.add_argument("--n-train", type=int, default=990_000)
    ap.add_argument("--n-test", type=int, default=10_000)
    ap.add_argument("--n-target", type=int, default=90_000)
    ap.add_argument("--sets", default="all", help="comma list of: test,target_v2,target_v2strong,train")
    ap.add_argument("--workers", type=int, default=240)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    sets = ["test", "target_v2", "target_v2strong", "train"] if a.sets == "all" else a.sets.split(",")
    for s in sets:
        if s == "test":
            run_set(out, "test", "test", a.n_test, a.workers, a.git_hash)
        elif s == "train":
            run_set(out, "lofi_train", "train", a.n_train, a.workers, a.git_hash)
        else:
            run_set(out, f"hifi_{s.split('_', 1)[1]}_target", s, a.n_target, a.workers, a.git_hash)
    print("done", flush=True)


if __name__ == "__main__":
    main()
