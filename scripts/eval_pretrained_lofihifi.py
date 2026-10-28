"""Evaluate the nested-sim-trained (lofi+hifi) models on the nested test sets.

Same structure/output format as eval_pretrained.py, adapted for the models trained
in this project's lofi/hifi benchmark round (2026-10-13/19) instead of the original
binary-10M-simset-trained models:

  npe-baseline-lofihifi  cv-dann-sbi exp-npe-baseline_nestedsim-lofihifi_cvdannsbi
                         (300k lofi sims, 400ep)              x -> encoder -> flow
  npe-noise-lofihifi     cv-dann-sbi exp-npe-noise-baseline_nestedsim-lofihifi_cvdannsbi_300k400ep
                         (300k lofi sims, 400ep, 11dB noise)  x -> encoder -> flow
  spin-pathD-lofihifi    cv-sbi-spin exp-v3d_spin_nestedsim-lofihifi_300ksims-3kreal-400ep
                         (300k lofi sims, 3k hifi "real", 400ep, final ep400 checkpoint)
                         x -> G_sr -> encoder_real -> flow_real
  spin-pathA-lofihifi    same run/checkpoint              x -> G_rs -> encoder_sim -> flow_sim
  npe-gibbs-lofihifi     cv-dann-sbi exp-gibbs-baseline_nestedsim-lofi_cvdannsbi_300k
                         (300k lofi sims)  [encoder(x), w_embed(log w)] -> flow,
                         at --gibbs-w (default 5.917, this model's own w_hat --
                         NOT the original binary-set model's 5.374)
  spin-singleflow-lofihifi  cv-sbi-spin exp-spin-singleflow_nestedsim-lofihifi_300ksims-3kreal-mixup-400ep
                         (exp/wdgrl-grl single shared-encoder+flow arch, no Path A/D split,
                         300k lofi sims, 3k hifi "real", mixup, 400ep)
                         x -> G_rs -> encoder -> flow (only one inference path)
  npe-dann-lofihifi      cv-dann-sbi exp-dann-wdgrl_nestedsim-lofihifi_cvdannsbi_300ksims-3kreal-400ep
                         (WDGRL DANN, 300k lofi sims, 3k hifi "real", 400ep)
                         x -> encoder -> flow

Two things these models need that the original eval_pretrained.py doesn't provide,
since it was built for binary-set-trained models:
  - norm_stats: these models were trained on nested-sim's own lofi_train/norm_stats.json
    (the binary set's norm_stats.json doesn't fit lofi/hifi's distribution -- see
    nested-sim/README.md's "Datasets" section) -- passed via --stats-path.
  - prior bounds: ec.load_prior_bounds() reads the binary set's own manifest (not
    available on all machines, and wrong prior anyway for lofi/hifi) -- built instead
    directly from nested_sim.params.THETA_BOUNDS (confirmed identical to
    PARAM_KEYS_INFER's 24-key binary naming earlier this session).

Run with the shared venv that has torch+h5py+scipy for all three repos, e.g.
  /home/sa4604/projects/cv-sbi-spin/.venv/bin/python3 scripts/eval_pretrained_lofihifi.py --method spin-pathD-lofihifi
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
OUT = HOME / "results/nested-sim/pretrained_eval_lofihifi"
STATS_PATH = DATA / "lofi_train/norm_stats.json"

METHODS = {
    "npe-baseline-lofihifi": ("cv-dann-sbi", "exp-npe-baseline_nestedsim-lofihifi_cvdannsbi", None),
    "npe-noise-lofihifi": ("cv-dann-sbi", "exp-npe-noise-baseline_nestedsim-lofihifi_cvdannsbi_300k400ep", None),
    "spin-pathD-lofihifi": ("cv-sbi-spin", "exp-v3d_spin_nestedsim-lofihifi_300ksims-3kreal-400ep", None),
    "spin-pathA-lofihifi": ("cv-sbi-spin", "exp-v3d_spin_nestedsim-lofihifi_300ksims-3kreal-400ep", None),
    "npe-gibbs-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi_300k", None),
    # --clamp-info-gap --freeze-scalars rerun (fixes the inverted-gap anomaly found in the
    # uncorrected run above -- see experiments.csv-equivalent discussion this session,
    # 2026-10-20). checkpoint_ts=None resolves to the final untagged epoch-415 folder
    # (resolve_checkpoint_dir excludes the tagged _calib_ep* snapshots by default).
    "spin-pathD-lofihifi-clamped": ("cv-sbi-spin", "exp-v3d_spin_nestedsim-lofihifi_300k3kreal_clamped", None),
    "spin-pathA-lofihifi-clamped": ("cv-sbi-spin", "exp-v3d_spin_nestedsim-lofihifi_300k3kreal_clamped", None),
    "spin-singleflow-lofihifi": ("cv-sbi-spin", "exp-spin-singleflow_nestedsim-lofihifi_300ksims-3kreal-mixup-400ep", None),
    "npe-dann-lofihifi": ("cv-dann-sbi", "exp-dann-wdgrl_nestedsim-lofihifi_cvdannsbi_300ksims-3kreal-400ep", None),
}
WAVES = ["Prv", "Pra", "Pvp", "Pap"]


def load_prior_bounds_nestedsim(keys):
    sys.path.insert(0, str(HOME / "projects/nested-sim"))
    from nested_sim.params import THETA_BOUNDS, to_binary_names  # noqa: E402
    bin_bounds = to_binary_names(THETA_BOUNDS)
    lo_t = torch.tensor([bin_bounds[k][0] for k in keys], dtype=torch.float32)
    hi_t = torch.tensor([bin_bounds[k][1] for k in keys], dtype=torch.float32)
    return lo_t, hi_t


def build_patients(set_name, n, stats, seed):
    """Test sims as eval_common-style patient dicts (x_avg, gt) -- same shape as
    eval_pretrained.py's build_patients, just factored out here to avoid importing
    that script (whose METHODS dict points at the other models' repos/runs)."""
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
    ap.add_argument("--gibbs-w", type=float, default=5.917)
    a = ap.parse_args()

    repo, run, ckpt = METHODS[a.method]
    sys.path.insert(0, str(HOME / "results" / repo / "scripts"))
    import eval_common as ec  # noqa: E402  (the repo's own eval helpers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = ec.load_stats(STATS_PATH)
    keys = list(ec.PARAM_KEYS_INFER)
    lo_t, hi_t = load_prior_bounds_nestedsim(keys)

    if a.method == "npe-gibbs-lofihifi":
        import eval_gibbs_sweep as eg  # noqa: E402
        from train_gibbs_npe import WEmbed  # noqa: E402
        sys.modules["__main__"].WEmbed = WEmbed  # w_embed.pt pickled WEmbed from train_gibbs_npe's __main__
        enc, flow, wemb, _, _ = eg.load_gibbs_models(eg.resolve_gibbs_ckpt(run), device)
        infer = lambda pats: eg.run_posterior_inference_w(  # noqa: E731
            enc, flow, wemb, torch.stack([p["x_avg"] for p in pats]), np.log(a.gibbs_w), lo_t, hi_t,
            a.n_samples, device)
    elif repo == "cv-dann-sbi":
        m = ec.load_models(run, device)
        infer = lambda pats: ec.run_posterior_inference(  # noqa: E731
            m["encoder"], m["flow_net"], pats, lo_t, hi_t, a.n_samples, device)
    elif a.method == "spin-singleflow-lofihifi":
        # exp/wdgrl-grl single shared-encoder+flow arch -- plain v1/v2-style
        # checkpoints (encoder/flow_net/G_rs/G_sr flat under checkpoints/<ts>/),
        # loaded with cv-sbi-spin's own eval_common (not eval_common_v3).
        m = ec.load_models(run, device, checkpoint_ts=ckpt)
        infer = lambda pats: ec.run_posterior_inference(  # noqa: E731
            m["encoder"], m["flow_net"], m["G_rs"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
    else:
        import eval_common_v3 as ec3  # noqa: E402
        m = ec3.load_models(run, device, checkpoint_ts=ckpt)
        if a.method.startswith("spin-pathD"):
            infer = lambda pats: ec3.run_posterior_inference_sr_direct(  # noqa: E731
                m["encoder_real"], m["flow_real"], m["G_sr"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
        else:
            infer = lambda pats: ec.run_posterior_inference(  # noqa: E731
                m["encoder"], m["flow_net"], m["G_rs"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
    print(f"{a.method}: {run} ckpt={ckpt} gibbs_w={a.gibbs_w if 'gibbs' in a.method else None} "
          f"stats={STATS_PATH} device={device}", flush=True)

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
