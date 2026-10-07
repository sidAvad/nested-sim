"""Evaluate already-trained cv-dann-sbi / cv-sbi-spin models on the nested test sets.

The test sims play the role of real patients (whose truth is now known for all 24
parameters). Each sim is turned into the models' 809-dim input exactly as
eval_common.load_patients does for a real beat (4 z-scored waveforms + MAP/SBP/DBP
z-scored with the Pas stats, SV / Vlv std, HR z-scored), using the models' own
norm_stats.json, and run through the repos' own loaders and inference functions:

  npe-baseline   cv-dann-sbi exp-npe-baseline_cvdannsbi         x -> encoder -> flow
  npe-noise      cv-dann-sbi exp-npe-noise-baseline_cvdannsbi   x -> encoder -> flow
  spin-pathD     cv-sbi-spin exp-v3d_spin (ep380)  x -> G_sr -> encoder_real -> flow_real
  spin-pathA     cv-sbi-spin exp-v3d_spin (ep380)  x -> G_rs -> encoder_sim -> flow_sim

Per test set it saves z = (post_mean - true)/post_std and
shrinkage = 1 - post_var/prior_var (uniform prior variance, as in
eval_zscore_shrinkage.py) for all 24 parameters, plus acceptance rates.

Run with the method's own venv on palladium, e.g.
  ~/projects/cv-sbi-spin/.venv/bin/python scripts/eval_pretrained.py --method spin-pathD
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HOME = Path.home()
DATA = HOME / "outputs/nested-sim/data"
OUT = HOME / "results/nested-sim/pretrained_eval"
METHODS = {
    "npe-baseline": ("cv-dann-sbi", "exp-npe-baseline_cvdannsbi", None),
    "npe-noise": ("cv-dann-sbi", "exp-npe-noise-baseline_cvdannsbi", None),
    "spin-pathD": ("cv-sbi-spin", "exp-v3d_spin", "20260908-133641_calib_ep380"),
    "spin-pathA": ("cv-sbi-spin", "exp-v3d_spin", "20260908-133641_calib_ep380"),
}
WAVES = ["Prv", "Pra", "Pvp", "Pap"]  # WAVE_KEYS_REAL / WAVE_KEYS_REDUCED order


def build_patients(set_name, n, stats, seed):
    """Test sims as eval_common-style patient dicts (x_avg, gt)."""
    w, p = stats["waves"], stats["parameters"]
    wave_mean = torch.tensor([w[k]["mean"] for k in WAVES], dtype=torch.float32).unsqueeze(1)
    wave_std = torch.tensor([w[k]["std"] for k in WAVES], dtype=torch.float32).unsqueeze(1)
    pas_mean, pas_std = w["Pas"]["mean"], w["Pas"]["std"] + 1e-8
    vlv_std = w["Vlv"]["std"] + 1e-8
    hr_mean, hr_std = p["HR"]["mean"], p["HR"]["std"] + 1e-8
    man = json.load(open(DATA / set_name / "manifest.json"))["index"]
    pick = sorted(np.random.default_rng(seed).choice(len(man), n, replace=False))
    patients, files = [], {}
    for i in pick:
        e = man[i]
        if e["file"] not in files:
            files[e["file"]] = h5py.File(DATA / set_name / e["file"], "r")
        g = files[e["file"]][e["group"]]
        waves = np.stack([g[f"waves/{k}"][:].astype(np.float32) for k in WAVES])
        wt = (torch.from_numpy(waves) - wave_mean) / (wave_std + 1e-8)
        s = {k: float(g[f"summaries/{k}"][()]) for k in ("map", "sbp", "dbp", "sv")}
        hr = float(g["parameters/HR"][()])
        sc = torch.tensor([(s["map"] - pas_mean) / pas_std, (s["sbp"] - pas_mean) / pas_std,
                           (s["dbp"] - pas_mean) / pas_std, s["sv"] / vlv_std, (hr - hr_mean) / hr_std],
                          dtype=torch.float32)
        gt = {k: float(g[f"parameters/{k}"][()]) for k in g["parameters"].keys()}
        patients.append(dict(file=f"{set_name}:{e['seed']}", x_avg=torch.cat([wt.reshape(-1), sc]), gt=gt,
                             seed=e["seed"]))
    for f in files.values():
        f.close()
    return patients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--sets", default="lofi_test,hifi_v2_test,hifi_v2strong_test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    repo, run, ckpt = METHODS[a.method]
    sys.path.insert(0, str(HOME / "results" / repo / "scripts"))
    import eval_common as ec  # noqa: E402  (the repo's own eval helpers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = ec.load_stats(ec.STATS_PATH)
    lo_t, hi_t, _ = ec.load_prior_bounds()
    keys = list(ec.PARAM_KEYS_INFER)

    if repo == "cv-dann-sbi":
        m = ec.load_models(run, device)
        infer = lambda pats: ec.run_posterior_inference(  # noqa: E731
            m["encoder"], m["flow_net"], pats, lo_t, hi_t, a.n_samples, device)
    else:
        import eval_common_v3 as ec3  # noqa: E402
        m = ec3.load_models(run, device, checkpoint_ts=ckpt)
        if a.method == "spin-pathD":
            infer = lambda pats: ec3.run_posterior_inference_sr_direct(  # noqa: E731
                m["encoder_real"], m["flow_real"], m["G_sr"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
        else:
            infer = lambda pats: ec.run_posterior_inference(  # noqa: E731
                m["encoder"], m["flow_net"], m["G_rs"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
    print(f"{a.method}: {run} ckpt={ckpt} stats={ec.STATS_PATH} device={device}", flush=True)

    lo, hi = lo_t.numpy(), hi_t.numpy()
    prior_var = (hi - lo) ** 2 / 12.0
    for set_name in a.sets.split(","):
        pats = build_patients(set_name, a.n, stats, a.seed)
        _, samples, accept = infer(pats)  # samples (N, n_samples, 24), physical units
        post_mean, post_std = samples.mean(1), samples.std(1)
        true = np.array([[pat["gt"][k] for k in keys] for pat in pats])
        out = {"param_keys": np.array(keys), "seeds": np.array([pat["seed"] for pat in pats]),
               "true": true, "post_mean": post_mean, "post_std": post_std, "accept_rate": accept}
        for j, k in enumerate(keys):
            ok = post_std[:, j] > 0
            out[f"{k}_z"] = (post_mean[ok, j] - true[ok, j]) / post_std[ok, j]
            out[f"{k}_shrinkage"] = 1.0 - post_std[ok, j] ** 2 / prior_var[j]
        d = OUT / a.method / set_name
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "zscore_shrinkage_data.npz", **out)
        summary = {k: (float(np.median(out[f"{k}_z"])), float(np.mean(np.abs(out[f"{k}_z"]) > 2)))
                   for k in ("Rap", "Ras", "Cas", "Eap", "Emax_LV", "Vs")}
        print(f"  {set_name}: n={len(pats)} accept median {np.median(accept):.2f} | median z, frac|z|>2: "
              + ", ".join(f"{k} {v[0]:+.2f}/{v[1]:.0%}" for k, v in summary.items()), flush=True)


if __name__ == "__main__":
    main()
