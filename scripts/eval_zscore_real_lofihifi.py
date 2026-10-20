"""Posterior z-score (Rap/Ras only, real measured PVR/SVR ground truth) for the
nested-sim-trained (lofi+hifi) models, evaluated on the real 802-patient cohort
(/home/sa4604/data/real_data/onebeat_300patients) -- NOT the hifi stand-in, the
genuine real-patient data.

Counterpart to cv-sbi-spin/scripts/eval_zscore_shrinkage{_v3,_patha_v3}.py and
cv-dann-sbi/scripts/eval_{zscore_shrinkage,gibbs_zscore_shrinkage}.py, consolidated
into one script (same --method dispatch as eval_pretrained_lofihifi.py) since all
five methods' real-patient z-score eval is otherwise identical bookkeeping.

shrinkage = 1 - (posterior_var / prior_var)   -- 0 = learned nothing, 1 = collapsed to a point
z         = (posterior_mean - true_theta) / posterior_std

Like eval_pretrained_lofihifi.py, overrides two things the original real-patient
eval scripts assume (binary-set-trained models): stats = nested-sim's own
lofi_train/norm_stats.json, prior bounds = nested_sim.params.THETA_BOUNDS (not
load_prior_bounds(), which reads the binary set's own manifest).

Gibbs NPE is restricted to its own held-out TEST patients (never touched during its
w_hat/metric calibration) -- same rationale/mechanism as eval_gibbs_zscore_shrinkage.py.
The other four methods use all 802, since none of them touch real data during fitting
(SPIN's --real-data was the hifi stand-in, not these real patients).

Usage:
    /home/sa4604/projects/cv-sbi-spin/.venv/bin/python3 eval_zscore_real_lofihifi.py --method spin-pathD-lofihifi
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HOME = Path.home()
DATA = HOME / "outputs/nested-sim/data"
OUT = HOME / "results/nested-sim/pretrained_eval_lofihifi_realpatients"
STATS_PATH = DATA / "lofi_train/norm_stats.json"

METHODS = {
    "npe-baseline-lofihifi": ("cv-dann-sbi", "exp-npe-baseline_nestedsim-lofihifi_cvdannsbi", None),
    "npe-noise-lofihifi": ("cv-dann-sbi", "exp-npe-noise-baseline_nestedsim-lofihifi_cvdannsbi_300k400ep", None),
    "spin-pathD-lofihifi": ("cv-sbi-spin", "exp-v3d_spin_nestedsim-lofihifi_300ksims-3kreal-400ep", None),
    "spin-pathA-lofihifi": ("cv-sbi-spin", "exp-v3d_spin_nestedsim-lofihifi_300ksims-3kreal-400ep", None),
    "npe-gibbs-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi_300k", None),
}


def load_prior_bounds_nestedsim(keys):
    sys.path.insert(0, str(HOME / "projects/nested-sim"))
    from nested_sim.params import THETA_BOUNDS, to_binary_names  # noqa: E402
    bin_bounds = to_binary_names(THETA_BOUNDS)
    lo_t = torch.tensor([bin_bounds[k][0] for k in keys], dtype=torch.float32)
    hi_t = torch.tensor([bin_bounds[k][1] for k in keys], dtype=torch.float32)
    return lo_t, hi_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--gibbs-w", type=float, default=5.917)
    a = ap.parse_args()

    repo, run, ckpt = METHODS[a.method]
    sys.path.insert(0, str(HOME / "results" / repo / "scripts"))
    import eval_common as ec  # noqa: E402  (the repo's own eval helpers -- cv-dann-sbi's for
                              # the three cv-dann-sbi methods, cv-sbi-spin's for the two SPIN ones)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = ec.load_stats(STATS_PATH)
    keys = list(ec.PARAM_KEYS_INFER)
    lo_t, hi_t = load_prior_bounds_nestedsim(keys)
    all_patients, _ = ec.load_patients(stats)

    if a.method == "npe-gibbs-lofihifi":
        import eval_gibbs_sweep as eg  # noqa: E402
        from train_gibbs_npe import WEmbed  # noqa: E402
        sys.modules["__main__"].WEmbed = WEmbed
        ckpt_dir = eg.resolve_gibbs_ckpt(run)
        enc, flow, wemb, metric, meta = eg.load_gibbs_models(ckpt_dir, device)
        test_set = set(meta["test_patients"])
        patients = [p for p in all_patients if p["file"] in test_set]
        print(f"Gibbs NPE: {len(patients)} held-out test patients / {len(test_set)} expected "
              f"(of {len(all_patients)} total; w_hat={meta['w_hat']:.4g}, evaluating at w={a.gibbs_w:.4g})")
        x_avg = torch.stack([p["x_avg"] for p in patients])
        theta_means, samples, accept = eg.run_posterior_inference_w(
            enc, flow, wemb, x_avg, np.log(a.gibbs_w), lo_t, hi_t, a.n_samples, device)
    else:
        patients = all_patients
        print(f"{a.method}: {len(patients)} real patients (all -- never touched during fitting)")
        if repo == "cv-dann-sbi":
            m = ec.load_models(run, device)
            theta_means, samples, accept = ec.run_posterior_inference(
                m["encoder"], m["flow_net"], patients, lo_t, hi_t, a.n_samples, device)
        else:
            import eval_common_v3 as ec3  # noqa: E402
            m = ec3.load_models(run, device, checkpoint_ts=ckpt)
            if a.method == "spin-pathD-lofihifi":
                theta_means, samples, accept = ec3.run_posterior_inference_sr_direct(
                    m["encoder_real"], m["flow_real"], m["G_sr"], m["apply_G"],
                    patients, lo_t, hi_t, a.n_samples, device)
            else:
                theta_means, samples, accept = ec.run_posterior_inference(
                    m["encoder"], m["flow_net"], m["G_rs"], m["apply_G"],
                    patients, lo_t, hi_t, a.n_samples, device)

    idx = {k: keys.index(k) for k in ["Rap", "Ras"]}
    gt = {
        "Rap": np.array([p["gt"].get("Rap", p["gt"].get("PVR", np.nan)) for p in patients]),
        "Ras": np.array([p["gt"].get("Ras", p["gt"].get("SVR", np.nan)) for p in patients]),
    }
    lo_np, hi_np = lo_t.numpy(), hi_t.numpy()

    npz_data = {}
    for key in ["Rap", "Ras"]:
        pi = idx[key]
        prior_var = (hi_np[pi] - lo_np[pi]) ** 2 / 12.0
        post_samples = samples[:, :, pi]
        post_mean = post_samples.mean(axis=1)
        post_std = post_samples.std(axis=1)
        true = gt[key]
        valid = ~np.isnan(true) & (true >= 0) & (post_std > 0)

        shrinkage = 1.0 - (post_std[valid] ** 2 / prior_var)
        z = (post_mean[valid] - true[valid]) / post_std[valid]
        npz_data[f"{key}_shrinkage"] = shrinkage
        npz_data[f"{key}_z"] = z
        print(f"  {key}: n={valid.sum()}  median |z|={np.median(np.abs(z)):.3f}  "
              f"frac |z|>2={(np.abs(z) > 2).mean():.3f}  median shrinkage={np.median(shrinkage):.3f}")

    d = OUT / a.method
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "zscore_shrinkage_data.npz", **npz_data)
    print(f"Saved: {d / 'zscore_shrinkage_data.npz'}")


if __name__ == "__main__":
    main()
