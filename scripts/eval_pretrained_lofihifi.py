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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probscores as ps  # noqa: E402

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
    # Gibbs retrained over log w ~ U[0.05, 50] (sn_ceiling 10, 980k lofi, ep800). The
    # earlier gibbs run's window [0.755, 75.5] excludes both high-fidelity levels'
    # calibrated w, so only this one can be conditioned at them.
    "npe-gibbs-w005to50-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-bnsf800-w005to50", None),
    # the longest-trained Gibbs runs (980k sims, bounded-nsf, sn_ceiling 10). Their window is
    # the earlier [0.755, 75.5], which holds lofi's w=7.5539 but not either high-fidelity
    # level's, so on those sets w is clamped to the window edge -- see clamp_w_to_window.
    "npe-gibbs-1600-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-bnsf1600-nolip", None),
    "npe-gibbs-800nolip-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-bnsf800-nolip", None),
    # repo is the source of eval_common (PARAM_KEYS_INFER, load_stats) only; the MDN itself
    # comes from cv-dann-sbi-gibbs/baselines/sun_routeA via routeA_infer.
    "sun-routeA-lofihifi": ("cv-dann-sbi", "exp-sun-routeA_nestedsim-lofi_mdn-w1024", None),
    # Route A with upstream's own RouteA_NPE.py call-site MDN arch (9 components,
    # 3 layers, batch 128, 500 ep) on the same w1024 pairs.
    "sun-routeA-callsite": ("cv-dann-sbi", "exp-sun-routeA_nestedsim-lofi_mdn-w1024-callsite", None),
    # Fully verbatim arm: 128-wide score net + upstream's Lgmcmc x clamp + call-site MDN.
    "sun-routeA-verbatim": ("cv-dann-sbi", "exp-sun-routeA_nestedsim-lofi_mdn-verbatim128", None),
    # Upstream score-net capacity (128 wide) with the repaired x clamp, isolating the
    # clamp from the capacity fix.
    "sun-routeA-w128fix": ("cv-dann-sbi", "exp-sun-routeA_nestedsim-lofi_mdn-w128-fixedclamp", None),
    # Gibbs with M, ell and w_hat calibrated on the hifi_v2strong stand-in rather than
    # on real patients: w window [0.135, 13.5], d_eff 156, w_hat 1.352.
    "npe-gibbs-v2strongcal-lofihifi": ("cv-dann-sbi", "exp-gibbs-v2strongcal_nestedsim-lofi_cvdannsbi-bnsf800", None),
    # Gibbs with M, ell and w_hat calibrated on the hifi_v2 realistic stand-in:
    # w window [0.186, 18.63], d_eff 156, w_hat 1.863.
    "npe-gibbs-v2realcal-lofihifi": ("cv-dann-sbi", "exp-gibbs-v2realcal_nestedsim-lofi_cvdannsbi-bnsf800", None),
    # Route A's call-site MDN trained on the pre-Langevin bank pairs instead of the
    # synthesized ones: the same architecture with the sampler's decorrelation removed.
    "sun-routeA-cleanpairs": ("cv-dann-sbi", "exp-sun-routeA_nestedsim-lofi_mdn-cleanpairs", None),
    # Flow-expressiveness 2x2: MAF (304,880 flow params) against bounded NSF (3,785,632)
    # at 400 and 800 epochs, all four on the real-patient-calibrated metric (w_hat 7.5539,
    # window [0.755, 75.5]) with the default sn_ceiling 2.0. Not the -nolip BNSF runs,
    # which use sn_ceiling 10 and would confound the flow with the encoder constraint.
    # hifi_v2_test's w = 1.033 sits inside the window, so the realistic level is evaluated
    # unclamped and is directly comparable to the v2realcal family (both at w = 1.033).
    # Only hifi_v2strong's 0.565 is clamped, to the 0.755 lower edge.
    "npe-gibbs-maf400-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-ep400", None),
    "npe-gibbs-maf800-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-ep800", None),
    "npe-gibbs-bnsf400-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-bnsf400", None),
    "npe-gibbs-bnsf800-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-bnsf800", None),
    # Noise-model arms on the v2realcal frozen metric. lr29proj/lr29full add the rank-29
    # low-rank enrichment Sigma_c = I + U diag(a) U^T (projected / full); wpaired keeps
    # Sigma_c = I and only recentres the w window on the theta-paired w = 0.713, so it is
    # the w-only control that makes the two lr29 arms interpretable. Target epoch with
    # --gibbs-epoch; all three carry epoch0100..epoch0800 snapshots.
    "npe-gibbs-lr29proj-lofihifi": ("cv-dann-sbi", "exp-gibbs-v2realcal-lr29proj_nestedsim-lofi_cvdannsbi-bnsf800", None),
    "npe-gibbs-lr29full-lofihifi": ("cv-dann-sbi", "exp-gibbs-v2realcal-lr29full_nestedsim-lofi_cvdannsbi-bnsf800", None),
    "npe-gibbs-wpaired-lofihifi": ("cv-dann-sbi", "exp-gibbs-wpaired_nestedsim-lofi_cvdannsbi-bnsf800", None),
    # Route A's mixture head (133,640 density params) on our CNN encoder, trained with
    # their optimizer settings: head lr 1e-3 against the 1e-4 every other arm uses. The
    # only arm at a different learning rate, so the only one that can separate "the
    # hifi/lofi trajectory is a property of the objective" from "it depends on how fast
    # you optimise". Unbounded support, so draws outside the prior box are rejected.
    "npe-gibbs-mdn-lofihifi": ("cv-dann-sbi", "exp-gibbs-v2realcal-mdn_nestedsim-lofi_cvdannsbi-mdn800", None),
    # Real-patient-calibrated metric (d_eff 180, w_hat 7.554) with the w window narrowed
    # to [0.3, 30] so every eval w below sits unclamped. w03to30 is the window-only
    # control; r29film3e4 adds the rank-32 correction refit on that metric, FiLM
    # conditioning on log w, and lr 3e-4 with a cosine schedule. Target epoch with
    # --gibbs-epoch.
    "npe-gibbs-realptcal-w03to30-lofihifi": ("cv-dann-sbi", "exp-gibbs-realptcal-w03to30_nestedsim-lofi_cvdannsbi-bnsf800", None),
    "npe-gibbs-realptcal-r29film3e4-lofihifi": ("cv-dann-sbi", "exp-gibbs-realptcal-r29film3e4_nestedsim-lofi_cvdannsbi-bnsf800", None),
    # Sigma_c = I under the Sun-et-al-matched protocol (lr 3e-4 + cosine, 400 epochs,
    # patience 20). Patience never fired: best val is the final epoch.
    # The nolip family at 400 epochs. "nolip" is sn-ceiling 10, not absent: the loose
    # ceiling, against 2.0 for the constrained arms. Two timestamps exist; 20261111-090855
    # is the complete one, so pass --ckpt-ts.
    "npe-gibbs-400nolip-lofihifi": ("cv-dann-sbi", "exp-gibbs-baseline_nestedsim-lofi_cvdannsbi-bnsf400-nolip", None),
    "npe-gibbs-realptcal-lr3e4cos400-lofihifi": ("cv-dann-sbi", "exp-gibbs-realptcal-lr3e4cos400_nestedsim-lofi_cvdannsbi-bnsf400", None),
    # rank-32 Sigma_c (cor_r29_full_realpt, unprojected, refit at d_eff 180) on the
    # real-patient metric with the wide window: the one-variable test of the correction.
    "npe-gibbs-realptcal-r32wide-lofihifi": ("cv-dann-sbi", "exp-gibbs-realptcal-r32wide_nestedsim-lofi_cvdannsbi-bnsf800", None),
}

# Per-level Gibbs tolerance, from the truth-anchored calibration (2026-11-16): one lofi
# simulation at the known theta, 499/500 calibration draws per level, metric frozen at
# ell=26.1327, d_eff=180. lofi_test is w=7.5539 by construction -- its noise IS the
# training noise -- while the high-fidelity levels' extra physics makes the achievable
# residual larger and so the tolerance looser.
W_LOFI = 7.5539
W_PER_SET = {"lofi_test": W_LOFI, "hifi_v2_test": 1.033, "hifi_v2strong_test": 0.565}

# Route A's beta is the exponent on the same low-fidelity likelihood that w scales, so the
# two tempering knobs map onto each other as beta = w / W_LOFI: at w = W_LOFI the Gibbs
# posterior IS the lofi likelihood's posterior, i.e. beta = 1. Clipped to the beta grid the
# MDN was amortised over (0.1 to 1.5) -- v2strong's 0.0748 falls below it, so that level is
# evaluated at the grid edge rather than extrapolated, and the clip is reported.
BETA_GRID_LO, BETA_GRID_HI = 0.1, 1.5


def clamp_w_to_window(w, ckpt_dir):
    """Hold w inside the log-w range the run was actually trained over.

    WEmbed standardises log w to [-1, 1] across that range; outside it the embedding is
    extrapolating, which is not a meaningful conditioning value. Returns (w_used, w_asked).
    """
    meta = json.load(open(Path(ckpt_dir) / "gibbs_metric.json"))
    lo, hi = meta.get("w_log_range", [-np.inf, np.inf])
    return float(np.exp(np.clip(np.log(w), lo, hi))), float(w)


def beta_for_set(set_name):
    b = W_PER_SET[set_name] / W_LOFI
    return float(np.clip(b, BETA_GRID_LO, BETA_GRID_HI)), b
WAVES = ["Prv", "Pra", "Pvp", "Pap"]


def load_prior_bounds_nestedsim(keys):
    sys.path.insert(0, str(HOME / "projects/nested-sim"))
    from nested_sim.params import THETA_BOUNDS, to_binary_names  # noqa: E402
    bin_bounds = to_binary_names(THETA_BOUNDS)
    lo_t = torch.tensor([bin_bounds[k][0] for k in keys], dtype=torch.float32)
    hi_t = torch.tensor([bin_bounds[k][1] for k in keys], dtype=torch.float32)
    return lo_t, hi_t


def load_log_dims(keys):
    """Indices of the log-uniform prior dimensions, in PARAM_KEYS_INFER order."""
    sys.path.insert(0, str(HOME / "projects/nested-sim"))
    from nested_sim.params import LOG_SAMPLED, to_binary_names  # noqa: E402
    names = set(to_binary_names({k: 0 for k in LOG_SAMPLED}))
    return sorted(keys.index(k) for k in names)


def to_unit_unclamped(phys, lo, hi, log_mask):
    """Prior box -> unit cube, logs on the log-uniform dims, WITHOUT clamping to [0,1].

    An unbounded flow can place a draw anywhere, including at a non-positive value on a
    log dimension; those are floored at a tiny positive number so the log is finite, and
    land far below 0 in unit coordinates, where the caller's own winsorisation handles
    them. In-box draws are unaffected.
    """
    x = np.where(log_mask, np.log(np.clip(phys, 1e-30, None)), phys)
    lo2 = np.where(log_mask, np.log(lo), lo)
    hi2 = np.where(log_mask, np.log(hi), hi)
    return (x - lo2) / (hi2 - lo2)


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
    ap.add_argument("--gibbs-w", type=float, default=None,
                    help="force one w for every set, overriding the per-level calibrated "
                         "W_PER_SET; use to reproduce an older single-w run")
    ap.add_argument("--routeA-beta", type=float, default=None,
                    help="force one beta for every set, overriding beta = w/W_LOFI")
    ap.add_argument("--ckpt-ts", default=None,
                    help="override the METHODS entry's checkpoint_ts, e.g. "
                         "20261019-121007_calib_ep100, to evaluate a periodic SPIN/NPE "
                         "snapshot instead of the final weights. Gibbs runs use "
                         "--gibbs-epoch instead.")
    ap.add_argument("--gibbs-epoch", type=int, default=None,
                    help="evaluate a periodic checkpoint (e.g. 400) instead of the final "
                         "weights; the epoch subdir carries its own metric and w_embed")
    ap.add_argument("--percase-w", default=None,
                    help="npz from percase_w.py: temper each case by its own "
                         "w_i = d_eff/||r_i||^2 (theta-paired). Gibbs runs use w_i "
                         "clamped to their own training window; Route A uses "
                         "beta_i = w_i/W_LOFI clipped to the amortised grid. Only the "
                         "hifi sets carry a residual, so lofi_test keeps its scalar w.")
    ap.add_argument("--marginalise-w", type=int, default=0, metavar="K",
                    help="instead of plugging in each case's w, draw K values from its "
                         "Gamma(q/2, rate) posterior and pool the samples, which "
                         "marginalises over w. Needs a <level>_rate field in the "
                         "--percase-w npz. 0 = plug in the point estimate.")
    ap.add_argument("--tag", default=None,
                    help="write under <method>__<tag> instead of <method>, so a swept "
                         "w or beta does not overwrite the calibrated run")
    a = ap.parse_args()

    repo, run, ckpt = METHODS[a.method]
    if a.ckpt_ts:
        ckpt = a.ckpt_ts
    # the posterior draws are Monte Carlo, so without this a re-run perturbs every
    # reported number by sampling noise alone
    torch.manual_seed(a.seed)
    sys.path.insert(0, str(HOME / "results" / repo / "scripts"))
    import eval_common as ec  # noqa: E402  (the repo's own eval helpers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stats = ec.load_stats(STATS_PATH)
    keys = list(ec.PARAM_KEYS_INFER)
    lo_t, hi_t = load_prior_bounds_nestedsim(keys)

    pcw = pcw_rate = None
    if a.percase_w:
        d = np.load(a.percase_w)
        pcw = {k: d[k] for k in d.files if k.endswith("_test")}
        pcw_rate = {k[:-5]: d[k] for k in d.files if k.endswith("_test_rate")}
        q_dof = int(d["q"]) if "q" in d.files else int(d["d_eff"])
        print(f"  per-case w from {a.percase_w}: "
              + ", ".join(f"{k} n={len(v)} median {np.median(v):.3f}"
                          for k, v in pcw.items()))
        if a.marginalise_w:
            if not pcw_rate:
                raise SystemExit("--marginalise-w needs <level>_rate in the npz")
            print(f"  marginalising over w: {a.marginalise_w} Gamma({q_dof/2:g}, rate) "
                  f"draws per case (rel. spread {np.sqrt(2/q_dof):.1%})")

    if a.method.startswith("npe-gibbs"):
        import eval_gibbs_sweep as eg  # noqa: E402
        # w_embed.pt pickles whichever conditioner the run used, from train_gibbs_npe's
        # __main__: WEmbed for --w-cond concat, WFiLM for --w-cond film.
        from train_gibbs_npe import WEmbed, WFiLM  # noqa: E402
        for _c in (WEmbed, WFiLM):
            setattr(sys.modules["__main__"], _c.__name__, _c)
        # --flow-arch mdn pickles WPassthrough as its w_embed and a ConditionalMDN as its
        # flow, both from train_gibbs_npe's __main__, so both names must resolve here too
        from mdn_flow import WPassthrough, ConditionalMDN, ResBlock  # noqa: E402
        for _c in (WPassthrough, ConditionalMDN, ResBlock):
            setattr(sys.modules["__main__"], _c.__name__, _c)
        # --ckpt-ts has to be honoured here too, not just on the SPIN paths: a run with
        # relaunches holds the same epoch under several timestamps, only one of which is
        # the completed trajectory, and resolve_gibbs_ckpt returns the LAST of them.
        if ckpt:
            gibbs_ckpt = eg.OUTPUTS_ROOT / run / "checkpoints" / ckpt
            if not gibbs_ckpt.is_dir():
                raise SystemExit(f"--ckpt-ts: no such directory {gibbs_ckpt}")
        else:
            gibbs_ckpt = eg.resolve_gibbs_ckpt(run, a.gibbs_epoch)
        if a.gibbs_epoch:
            gibbs_ckpt = gibbs_ckpt / f"epoch{a.gibbs_epoch:04d}"
            if not (gibbs_ckpt / "flow_net.pt").exists():
                raise SystemExit(f"no flow_net.pt under {gibbs_ckpt}")
        print(f"  gibbs ckpt {gibbs_ckpt}")
        enc, flow, wemb, _, _ = eg.load_gibbs_models(gibbs_ckpt, device)
        w_for = lambda s_: clamp_w_to_window(  # noqa: E731
            a.gibbs_w if a.gibbs_w else W_PER_SET[s_], gibbs_ckpt)

        def logw_for(s_):
            if pcw is None or s_ not in pcw:
                return np.log(w_for(s_)[0])
            if a.marginalise_w and s_ in pcw_rate:
                rng = np.random.default_rng(a.seed)
                w = rng.gamma(q_dof / 2.0, 1.0 / pcw_rate[s_][:, None],
                              size=(len(pcw[s_]), a.marginalise_w))
            else:
                w = np.asarray(pcw[s_])[:, None]
            cl = np.vectorize(lambda v: clamp_w_to_window(float(v), gibbs_ckpt)[0])
            return np.log(cl(w))

        infer = lambda pats, s_: eg.run_posterior_inference_w(  # noqa: E731
            enc, flow, wemb, torch.stack([p["x_avg"] for p in pats]),
            logw_for(s_), lo_t, hi_t, a.n_samples, device)
    elif a.method.startswith("sun-routeA"):
        import routeA_infer as ra  # noqa: E402
        mdn, nth, nx, tf_unit, ra_meta = ra.load_routeA(run, device)
        print(f"  routeA mdn {ra_meta['ckpt_dir']} best_val {ra_meta['best_val']}")
        def beta_arg(s_):
            if pcw is not None and s_ in pcw:
                return np.clip(np.asarray(pcw[s_]) / W_LOFI, BETA_GRID_LO, BETA_GRID_HI)
            return a.routeA_beta if a.routeA_beta else beta_for_set(s_)[0]

        infer = lambda pats, s_: ra.run_posterior_inference_routeA(  # noqa: E731
            mdn, nth, nx, tf_unit, pats, beta_arg(s_), a.n_samples, device)
    elif repo == "cv-dann-sbi":
        m = ec.load_models(run, device)
        infer = lambda pats, s_: ec.run_posterior_inference(  # noqa: E731
            m["encoder"], m["flow_net"], pats, lo_t, hi_t, a.n_samples, device)
    elif a.method == "spin-singleflow-lofihifi":
        # exp/wdgrl-grl single shared-encoder+flow arch -- plain v1/v2-style
        # checkpoints (encoder/flow_net/G_rs/G_sr flat under checkpoints/<ts>/),
        # loaded with cv-sbi-spin's own eval_common (not eval_common_v3).
        m = ec.load_models(run, device, checkpoint_ts=ckpt)
        infer = lambda pats, s_: ec.run_posterior_inference(  # noqa: E731
            m["encoder"], m["flow_net"], m["G_rs"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
    else:
        import eval_common_v3 as ec3  # noqa: E402
        m = ec3.load_models(run, device, checkpoint_ts=ckpt)
        if a.method.startswith("spin-pathD"):
            infer = lambda pats, s_: ec3.run_posterior_inference_sr_direct(  # noqa: E731
                m["encoder_real"], m["flow_real"], m["G_sr"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
        else:
            infer = lambda pats, s_: ec.run_posterior_inference(  # noqa: E731
                m["encoder"], m["flow_net"], m["G_rs"], m["apply_G"], pats, lo_t, hi_t, a.n_samples, device)
    print(f"{a.method}: {run} ckpt={ckpt} stats={STATS_PATH} device={device}", flush=True)

    lo, hi = lo_t.numpy().astype(np.float64), hi_t.numpy().astype(np.float64)
    prior_var = (hi - lo) ** 2 / 12.0
    log_mask = np.zeros(len(keys), dtype=bool)
    log_mask[load_log_dims(keys)] = True
    out_name = a.method if not a.tag else f"{a.method}__{a.tag}"

    for set_name in a.sets.split(","):
        setting = {}
        percase = pcw is not None and set_name in pcw
        if a.method.startswith("npe-gibbs"):
            if percase:
                lw = logw_for(set_name)
                setting["w"] = ("per-case marginalised" if a.marginalise_w
                                else "per-case")
                setting["w_median"] = float(np.exp(np.median(lw)))
                setting["w_draws"] = int(lw.shape[1]) if lw.ndim == 2 else 1
                # compare with a tolerance: clamp_w_to_window round-trips through
                # log/exp, so an exact != fires on almost every in-window value
                setting["w_clamped_frac"] = float(np.mean(
                    [not np.isclose(clamp_w_to_window(float(v), gibbs_ckpt)[0],
                                    float(v), rtol=1e-9)
                     for v in pcw[set_name]]))
            else:
                wu, wa = w_for(set_name)
                setting["w"] = wu
                setting["w_unclamped"] = wa
        if a.method.startswith("sun-routeA"):
            if percase:
                b = beta_arg(set_name)
                setting["beta"] = "per-case"
                setting["beta_median"] = float(np.median(b))
                raw = np.asarray(pcw[set_name]) / W_LOFI
                setting["beta_clipped_frac"] = float(np.mean(
                    (raw < BETA_GRID_LO) | (raw > BETA_GRID_HI)))
            else:
                bc, braw = beta_for_set(set_name)
                setting["beta"] = float(a.routeA_beta if a.routeA_beta else bc)
                setting["beta_unclipped"] = float(braw)
        pats = build_patients(set_name, a.n, stats, a.seed)
        _, samples, accept = infer(pats, set_name)  # (N, n_samples, 24), physical units
        post_mean, post_std = samples.mean(1), samples.std(1)
        true = np.array([[pat["gt"][k] for k in keys] for pat in pats])
        out = {"param_keys": np.array(keys), "seeds": np.array([pat["seed"] for pat in pats]),
               "true": true, "post_mean": post_mean, "post_std": post_std, "accept_rate": accept,
               "setting": np.array(json.dumps(setting))}
        for j, k in enumerate(keys):
            ok = post_std[:, j] > 0
            out[f"{k}_z"] = (post_mean[ok, j] - true[ok, j]) / post_std[ok, j]
            out[f"{k}_shrinkage"] = 1.0 - post_std[ok, j] ** 2 / prior_var[j]

        # Proper scores, in unit-cube coordinates so the 24 parameters are commensurate.
        # The forward map here is deliberately NOT BoxToUnit.forward, which clamps to
        # [eps, 1-eps] and would collapse every out-of-box draw onto the boundary --
        # that mask is exactly what the rejection rate is about, so it has to survive.
        u = to_unit_unclamped(samples, lo, hi, log_mask)
        u_true = to_unit_unclamped(true[:, None, :], lo, hi, log_mask)[:, 0, :]
        in_box = ((u >= 0.0) & (u <= 1.0)).all(axis=2)
        sc = ps.score_set(np.clip(u, -5.0, 6.0), u_true, in_box)
        for j, k in enumerate(keys):
            for nm in ("crps_box", "crps_raw", "is_box", "is_raw", "width_box", "cov_box",
                       "crps_prior", "is_prior"):
                out[f"{k}_{nm}"] = sc[nm][:, j]
        out["n_degenerate"] = sc["n_degenerate"]

        d = OUT / out_name / set_name
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "zscore_shrinkage_data.npz", **out)
        summary = {k: (float(np.median(out[f"{k}_z"])), float(np.mean(np.abs(out[f"{k}_z"]) > 2)))
                   for k in ("Rap", "Ras", "Cas", "Eap", "Emax_LV", "Vs")}
        skill = 1.0 - np.nanmean(sc["crps_box"]) / np.nanmean(sc["crps_prior"])
        skill_is = 1.0 - np.nanmean(sc["is_box"]) / np.nanmean(sc["is_prior"])
        print(f"  {set_name}: n={len(pats)} {setting} accept median {np.median(accept):.3f} "
              f"degenerate {int(sc['n_degenerate'])} | CRPS {np.nanmean(sc['crps_box']):.4f} "
              f"(skill {skill:+.3f})  IS95 {np.nanmean(sc['is_box']):.4f} (skill {skill_is:+.3f})  "
              f"cov95 {np.nanmean(sc['cov_box']):.3f}  width95 {np.nanmean(sc['width_box']):.4f}",
              flush=True)
        print(f"    median z, frac|z|>2: "
              + ", ".join(f"{k} {v[0]:+.2f}/{v[1]:.0%}" for k, v in summary.items()), flush=True)


if __name__ == "__main__":
    main()
