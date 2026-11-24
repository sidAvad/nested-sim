"""CRPS, interval-score and z-score figures for the nested-sim lofi/hifi method comparison.

Three figures over one method list, one fidelity-level palette and one row order, so they
stack and can be read against each other:

  crps_is_<set>  One per fidelity level, in the same two-panel form as cv-dann-sbi's
          crps_smc_vs_flow.py: CRPS violins over every case and parameter on the left,
          and on the right the 95% interval score as an additive bar, width plus
          undercoverage penalty. Both are divided by the prior sd. CRPS is proper on the
          whole posterior rather than its first two moments; the interval split is what
          separates "narrow" from "narrow and wrong".
  zscore  The existing (posterior mean - true)/posterior std figure, reproduced here from
          the same re-scored runs so every panel comes from one set of draws.

Scores are in unit-cube coordinates (see probscores) and computed on the samples that
survive the prior box, i.e. the posterior each method actually deploys; the acceptance rate
that produces them is annotated, since a method cannot be credited for sharpness it only
reaches by throwing most of its draws away.

Usage: python scripts/plot_pretrained_scores_lofihifi.py [--suffix scored] [--metric all]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
# Figures live outside the eval tree: the npz under ROOT are large and are not
# synced to the Mac, so the whole eval tree is a mutagen ignore and anything
# written there would be invisible locally.
OUT_DIR = Path.home() / "results/nested-sim/figures_lofihifi/scores_lofihifi"

METHODS = [
    ("npe-gibbs-w005to50-lofihifi", "Gibbs NPE, ep800, w window [0.05, 50]"),
    ("npe-gibbs-1600-lofihifi", "Gibbs NPE, ep1600, w window [0.76, 75]"),
    ("npe-gibbs-800nolip-lofihifi", "Gibbs NPE, ep800, w window [0.76, 75]"),
    ("npe-gibbs-v2strongcal-lofihifi", "Gibbs NPE, ep400, hifi-calibrated w window [0.14, 13.5]"),
    ("npe-gibbs-lofihifi", "Gibbs NPE, ep100/300k (w=5.92)"),
    ("npe-gibbs-v2strongcal-ep800", "Gibbs NPE, ep800, hifi-calibrated w window [0.14, 13.5]"),
    ("npe-gibbs-v2realcal-lofihifi", "Gibbs NPE, ep800, realistic-calibrated w window [0.19, 18.6]"),
    ("npe-gibbs-v2realcal-ep400", "Gibbs NPE, ep400, realistic-calibrated w window [0.19, 18.6]"),
    ("sun-routeA-lofihifi", "Sun et al. Route A MDN (beta per level: 1/0.14/0.10)"),
    ("sun-routeA-cleanpairs", "Sun et al. Route A MDN, clean pairs (no Langevin), beta=1"),
    ("spin-pathD-lofihifi", "SPIN dual-flow, Path D"),
    ("spin-pathA-lofihifi", "SPIN dual-flow, Path A"),
    ("spin-singleflow-lofihifi", "SPIN single-flow (mixup)"),
    ("npe-dann-lofihifi", "NPE + WDGRL DANN"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
# Per-case-tempering roster. The realistic-calibrated runs are the matched arms for
# these test sets, so the four tempering choices are shown for each of its two
# checkpoints; the real-patient-calibrated runs clamp 34-62% of the theta-subspace w
# and are left out. Route A and the three tempering-free baselines anchor the scale.
# The theta-subspace evals cover the hifi sets only -- lofi_test carries no residual,
# so those rows are absent from the lofi panel by construction.
METHODS_PERCASE = [
    ("pcw-realcal-ep400-global", "Gibbs ep400 realistic-cal, global w"),
    ("pcw-realcal-ep400", "Gibbs ep400, per-case w (full norm)"),
    ("pcwth-realcal-ep400", "Gibbs ep400, per-case w (theta-subspace)"),
    ("pcwthmarg-realcal-ep400", "Gibbs ep400, per-case w (theta-sub, marginalised)"),
    ("pcw-realcal-ep800-global", "Gibbs ep800 realistic-cal, global w"),
    ("pcw-realcal-ep800", "Gibbs ep800, per-case w (full norm)"),
    ("pcwth-realcal-ep800", "Gibbs ep800, per-case w (theta-subspace)"),
    ("pcwthmarg-realcal-ep800", "Gibbs ep800, per-case w (theta-sub, marginalised)"),
    ("pcw-routeA", "Sun et al. Route A MDN (per-case beta)"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("spin-pathD-lofihifi", "SPIN dual-flow, Path D"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
PERCASE_OVERRIDE = {
    "pcw-realcal-ep400-global": ("npe-gibbs-v2realcal-lofihifi", "ep400"),
    "pcw-realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "pcw-ep400"),
    "pcwth-realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "pcwth-ep400"),
    "pcwthmarg-realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "pcwthmarg-ep400"),
    "pcw-realcal-ep800-global": ("npe-gibbs-v2realcal-lofihifi", ""),
    "pcw-realcal-ep800": ("npe-gibbs-v2realcal-lofihifi", "pcw-ep800"),
    "pcwth-realcal-ep800": ("npe-gibbs-v2realcal-lofihifi", "pcwth-ep800"),
    "pcwthmarg-realcal-ep800": ("npe-gibbs-v2realcal-lofihifi", "pcwthmarg-ep800"),
    "pcw-routeA": ("sun-routeA-lofihifi", "pcw"),
}
# Flow-expressiveness roster. MAF (304,880 flow params) against bounded NSF (3,785,632)
# at 400 and 800 epochs, all four sharing the real-patient-calibrated metric
# (w_hat 7.5539, window [0.755, 75.5]) and sn_ceiling 2.0, so the flow is the only thing
# that varies. The two v2realcal checkpoints are the hifi-realistic-calibrated comparison,
# evaluated at the same w = 1.033 on hifi_v2_test, which sits inside both windows; Route A
# and the vanilla NPE anchor the scale.
METHODS_ARCH = [
    ("arch-maf400", "Gibbs MAF ep400, real-patient-cal (305k flow params)"),
    ("arch-maf800", "Gibbs MAF ep800, real-patient-cal (305k flow params)"),
    ("arch-bnsf400", "Gibbs NSF ep400, real-patient-cal (3.79M flow params)"),
    ("arch-bnsf800", "Gibbs NSF ep800, real-patient-cal (3.79M flow params)"),
    ("arch-v2realcal-ep400", "Gibbs NSF ep400, realistic-cal w window [0.19, 18.6]"),
    ("arch-v2realcal-ep800", "Gibbs NSF ep800, realistic-cal w window [0.19, 18.6]"),
    ("sun-routeA-lofihifi", "Sun et al. Route A MDN (beta per level: 1/0.14/0.10)"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
ARCH_OVERRIDE = {
    "arch-maf400": ("npe-gibbs-maf400-lofihifi", "arch2x2"),
    "arch-maf800": ("npe-gibbs-maf800-lofihifi", "arch2x2"),
    "arch-bnsf400": ("npe-gibbs-bnsf400-lofihifi", "arch2x2"),
    "arch-bnsf800": ("npe-gibbs-bnsf800-lofihifi", "arch2x2"),
    "arch-v2realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "ep400"),
    "arch-v2realcal-ep800": ("npe-gibbs-v2realcal-lofihifi", ""),
}
# Noise-model roster. All five Gibbs arms share the same frozen v2realcal metric
# (ell 30.8602, d_eff 156, w_hat 1.86258), so M is held fixed and only the noise model
# and its w window vary: v2realcal is Sigma_c = I on [0.186, 18.6]; wpaired is Sigma_c = I
# recentred on the theta-paired w = 0.713, window [0.071, 7.13], making it the w-only
# control; lr29proj/lr29full add the rank-29 enrichment Sigma_c = I + U diag(a) U^T
# (projected / full), windows [0.134, 13.4] and [0.250, 25.0]. Every one of those windows
# contains both hifi levels' calibrated w (1.033 and 0.565), so unlike the arch roster
# neither high-fidelity level is clamped on any arm here.
METHODS_NOISE = [
    ("noise-v2realcal-ep400", "Gibbs NSF ep400, Sigma_c = I (baseline)"),
    ("noise-v2realcal-ep800", "Gibbs NSF ep800, Sigma_c = I (baseline)"),
    ("noise-wpaired-ep400", "Gibbs NSF ep400, w-recentred on 0.713, Sigma_c = I"),
    ("noise-wpaired-ep800", "Gibbs NSF ep800, w-recentred on 0.713, Sigma_c = I"),
    ("noise-lr29proj-ep400", "Gibbs NSF ep400, rank-29 Sigma_c (projected)"),
    ("noise-lr29proj-ep800", "Gibbs NSF ep800, rank-29 Sigma_c (projected)"),
    ("noise-lr29full-ep400", "Gibbs NSF ep400, rank-29 Sigma_c (full)"),
    ("noise-lr29full-ep800", "Gibbs NSF ep800, rank-29 Sigma_c (full)"),
    ("sun-routeA-lofihifi", "Sun et al. Route A MDN (beta per level: 1/0.14/0.10)"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
NOISE_OVERRIDE = {
    "noise-v2realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "ep400"),
    "noise-v2realcal-ep800": ("npe-gibbs-v2realcal-lofihifi", ""),
    "noise-wpaired-ep400": ("npe-gibbs-wpaired-lofihifi", "ep400"),
    "noise-wpaired-ep800": ("npe-gibbs-wpaired-lofihifi", "ep800"),
    "noise-lr29proj-ep400": ("npe-gibbs-lr29proj-lofihifi", "ep400"),
    "noise-lr29proj-ep800": ("npe-gibbs-lr29proj-lofihifi", "ep800"),
    "noise-lr29full-ep400": ("npe-gibbs-lr29full-lofihifi", "ep400"),
    "noise-lr29full-ep800": ("npe-gibbs-lr29full-lofihifi", "ep800"),
}
# Per-bin-w roster: one row per model, per-bin w only, at the highest epoch evaluated.
# w_i is read off the case's Prv-max bin (edges 34.325 / 61.245 mmHg) and anchored so the
# harmonic mean reproduces that level's calibrated scalar, so the level is unchanged and
# only the between-bin distribution of w differs from a scalar run. Arms carrying a
# low-rank Sigma_c take the table built in their OWN noise geometry (q = r^T Sigma_c^-1 r,
# the _sig_ files); Sigma_c = I arms take the Euclidean one. Both tables are tied to the
# metric that defined phi, so each arm uses the one matching its own metric. Per-bin w is
# defined from a residual, so the lofi panel carries no rows.
METHODS_BINW = [
    ("binw-bnsf800-perbin", "Gibbs NSF ep800, real-pt metric, Sigma_c = I"),
    ("binw-control-perbin", "Gibbs NSF ep600, real-pt metric, window [0.3, 30]"),
    ("binw-film-perbin", "Gibbs NSF ep600, real-pt metric, rank-32 Sigma_c + FiLM"),
    ("binw-lr29full-perbin", "Gibbs NSF ep800, v2realcal metric, rank-29 Sigma_c"),
    ("binw-v2realcal-perbin", "Gibbs NSF ep400, v2realcal metric, Sigma_c = I"),
    ("sun-routeA-lofihifi", "Sun et al. Route A MDN (beta per level: 1/0.14/0.10)"),
    ("spin-pathD-lofihifi", "SPIN dual-flow, Path D"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
BINW_OVERRIDE = {
    "binw-bnsf800-perbin": ("npe-gibbs-bnsf800-lofihifi", "ep800_binw"),
    "binw-control-perbin": ("npe-gibbs-realptcal-w03to30-lofihifi", "ep600_binw"),
    "binw-film-perbin": ("npe-gibbs-realptcal-r29film3e4-lofihifi", "ep600_binwSig"),
    "binw-lr29full-perbin": ("npe-gibbs-lr29full-lofihifi", "ep800_binwSig"),
    "binw-v2realcal-perbin": ("npe-gibbs-v2realcal-lofihifi", "ep400_binw"),
}
SETS = [("lofi_test", "low-fi (no gap)", "#2a78d6"),
        ("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0
ROW_H, GAP = 1.0, 0.35
ANN_X = 0.58          # where the right-hand annotation column starts, in data units


# Rows whose eval was written under a tag other than --suffix. The key is the METHODS
# name; the value is (eval directory stem, tag).
SUFFIX_OVERRIDE = {
    "npe-gibbs-v2strongcal-ep800": ("npe-gibbs-v2strongcal-lofihifi", "ep800"),
    "npe-gibbs-v2realcal-lofihifi": ("npe-gibbs-v2realcal-lofihifi", ""),
    "npe-gibbs-v2realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "ep400"),
    "sun-routeA-cleanpairs": ("sun-routeA-cleanpairs", "beta1"),
}


def load(method, set_name, suffix):
    method, suffix = SUFFIX_OVERRIDE.get(method, (method, suffix))
    d = ROOT / (f"{method}__{suffix}" if suffix else method) / set_name / "zscore_shrinkage_data.npz"
    if not d.exists():
        return None
    return np.load(d, allow_pickle=True)


def pooled(d, field):
    """Stack one per-case field over the 24 parameters -> (n_cases * 24,)."""
    return np.concatenate([d[f"{k}_{field}"] for k in d["param_keys"]])


def setting_str(d):
    try:
        s = json.loads(str(d["setting"]))
    except (KeyError, ValueError):
        return ""
    for name, med, frac in (("w", "w_median", "w_clamped_frac"),
                            ("beta", "beta_median", "beta_clipped_frac")):
        if name not in s:
            continue
        v = s[name]
        if isinstance(v, str):      # per-case tempering: report the median and the
            return (f"{name}~{s[med]:.3g} per-case"   # share pinned to the window edge
                    + (f", {s[frac]:.0%} clamped" if s.get(frac) else ""))
        unc = s.get("w_unclamped" if name == "w" else "beta_unclipped", v)
        return f"{name}={v:g}" + ("*" if abs(unc - v) > 1e-9 else "")
    return ""


def gather(suffix):
    """rows[(method, set)] -> dict of the pooled numbers both bar figures need."""
    rows = {}
    for m, _ in METHODS:
        for s, _, _ in SETS:
            d = load(m, s, suffix)
            if d is None:
                continue
            crps, crps_p = pooled(d, "crps_box"), pooled(d, "crps_prior")
            isc, is_p = pooled(d, "is_box"), pooled(d, "is_prior")
            width = pooled(d, "width_box")
            rows[(m, s)] = {
                "crps": np.nanmean(crps), "crps_skill": 1 - np.nanmean(crps) / np.nanmean(crps_p),
                "crps_raw_skill": 1 - np.nanmean(pooled(d, "crps_raw")) / np.nanmean(crps_p),
                "is": np.nanmean(isc), "is_skill": 1 - np.nanmean(isc) / np.nanmean(is_p),
                "is_prior": np.nanmean(is_p),
                "width": np.nanmean(width), "penalty": np.nanmean(isc) - np.nanmean(width),
                "cov95": np.nanmean(pooled(d, "cov_box")),
                "accept": float(np.median(d["accept_rate"])),
                "degenerate": int(d["n_degenerate"]) if "n_degenerate" in d else 0,
                "medabsz": float(np.median(np.abs(pooled(d, "z")))),
                "setting": setting_str(d),
            }
    return rows


def order_methods(rows, key="crps_skill"):
    """Best on the exaggerated level first; that is the level the comparison is about.
    Methods missing that level fall to the bottom, keeping the declared order."""
    def k(m):
        r = rows.get((m, "hifi_v2strong_test"))
        return (0, -r[key]) if r else (1, 0)
    return sorted([m for m in METHODS if any((m[0], s) in rows for s, _, _ in SETS)],
                  key=lambda ml: k(ml[0]))


def _frame(ax, methods, yticks):
    n = len(methods)
    ax.set_yticks(yticks, [lab for _, lab in methods], fontsize=9)
    ax.set_ylim(n * (ROW_H + GAP) - GAP / 2, -GAP / 2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def _bands(ax, methods):
    for r in range(len(methods)):
        if r % 2 == 0:
            top = r * (ROW_H + GAP)
            ax.axhspan(top - GAP / 2, top + ROW_H + GAP / 2, color=GRID, alpha=0.35, lw=0, zorder=0)


PRIOR_SD = 1.0 / np.sqrt(12.0)    # sd of U[0,1]: the unit-cube prior's own sd
PENALTY_C = "#eda4a4"             # reserved for the undercoverage term, every figure
CRPS_CLIP = 3.0                   # display clip: a handful of cases reach ~20 prior sd and
                                  # would otherwise flatten every violin to a spike


def fig_scores(rows, suffix, set_name, set_label, color):
    """CRPS violins beside the interval score split into width and undercoverage penalty.

    Same two-panel form as crps_smc_vs_flow.py's overall figure, so the two families read
    together: CRPS is divided by the prior sd on the left, and the right bar is additive,
    which is the point of showing it -- it separates "narrow" from "narrow and wrong".
    One figure per fidelity level, every row in that level's colour.
    """
    present = [(m, lab) for m, lab in METHODS if (m, set_name) in rows]
    present.sort(key=lambda ml: rows[(ml[0], set_name)]["crps"])
    n = len(present)
    fig, (axv, axb) = plt.subplots(1, 2, figsize=(15.5, 0.95 * n + 2.6),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    ys = np.arange(n)[::-1]          # best at the top
    xmax, ann = 0.0, []
    for y, (m, lab) in zip(ys, present):
        v = rows[(m, set_name)]
        d = load(m, set_name, suffix)
        c = pooled(d, "crps_box") / PRIOR_SD
        c = c[np.isfinite(c)]
        med, mean = np.median(c), np.mean(c)
        parts = axv.violinplot([np.clip(c, 0, CRPS_CLIP)], positions=[y], vert=False,
                               widths=0.82,
                               showmedians=False, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.75)
        axv.plot([med], [y], marker="|", ms=13, color=INK, mew=1.8, zorder=4)
        axv.plot([min(mean, CRPS_CLIP)], [y], marker="o", ms=5.5, color=INK, zorder=4)
        axv.text(CRPS_CLIP + 0.06, y, f"{med:.2f} / {mean:.2f}", va="center", fontsize=8,
                 color=INK, family="monospace")

        w, pen = v["width"] / PRIOR_SD, v["penalty"] / PRIOR_SD
        axb.barh(y, w, height=0.62, color=color, alpha=0.9, edgecolor="white",
                 linewidth=1.2, zorder=3)
        axb.barh(y, pen, left=w, height=0.62, color=PENALTY_C, hatch="//",
                 edgecolor="white", linewidth=1.2, zorder=3)
        xmax = max(xmax, w + pen)
        axb.text(w + pen + 0.08, y, f"{w + pen:.2f}", va="center", fontsize=8.5, color=INK)
        ann.append((y, f"cov {v['cov95']:4.0%}  accept {v['accept']:4.0%}"
                    + (f"  {v['setting']}" if v["setting"] else "")))

    for ax in (axv, axb):
        ax.set_ylim(-0.7, n - 0.3)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axv.set_yticks(ys, [lab for _, lab in present], fontsize=9.5)
    axb.set_yticks(ys, [])
    ann_x = xmax * 1.20
    for y, txt in ann:
        axb.text(ann_x, y, txt, va="center", fontsize=7.5, color=MUTED,
                 family="monospace", zorder=4)
    axv.set_xlim(0, CRPS_CLIP * 1.33)
    axv.set_xticks(np.arange(0, CRPS_CLIP + 0.01, 0.5))
    # Room for the longest annotation: the per-case rows carry both a median w and a
    # clamped share, which overran the old 1.62 margin and truncated mid-word.
    axb.set_xlim(0, xmax * 1.80)
    axv.axvline(1.0, color=INK, lw=1.0, ls="--", zorder=2)
    axv.text(1.0, n - 0.32, " prior", fontsize=8, color=INK, va="bottom")
    axv.text(CRPS_CLIP + 0.06, n - 0.32, "med / mean", fontsize=7.5, color=MUTED,
             va="bottom", family="monospace")
    axv.set_xlabel(f"CRPS / prior sd   (lower is better; | = median, o = mean; "
                   f"display clipped at {CRPS_CLIP:g})", fontsize=9.5, color=MUTED)
    axb.set_xlabel("mean interval score (level 95%) / prior sd", fontsize=9.5, color=MUTED)
    axv.set_title("CRPS, pooled over 24 parameters", fontsize=10.5, color=INK)
    axb.set_title("Interval score at 95%, split into its two terms", fontsize=10.5, color=INK)
    axb.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.9,
                                      label="width (sharpness)"),
                        plt.Rectangle((0, 0), 1, 1, fc=PENALTY_C, hatch="//",
                                      label="undercoverage penalty")],
               loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle(f"Proper scores vs ground truth -- {set_label}, 1,000 test sims\n"
                 "scored on the posterior each method deploys (draws surviving the prior "
                 "box); unit-cube coordinates",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    return fig, f"crps_is_{set_name}.png"


def fig_zscore(rows, suffix):
    methods = order_methods(rows)
    lane = ROW_H / len(SETS)
    fig, ax = plt.subplots(figsize=(11.5, 1.25 * len(methods) + 2.0))
    _bands(ax, methods)
    yticks = []
    for r, (m, _) in enumerate(methods):
        top = r * (ROW_H + GAP)
        yticks.append(top + ROW_H / 2)
        for j, (s, _, color) in enumerate(SETS):
            d = load(m, s, suffix)
            if d is None:
                continue
            z = pooled(d, "z")
            y = top + (j + 0.5) * lane
            parts = ax.violinplot([np.clip(z, -CLIP, CLIP)], positions=[y], vert=False,
                                  widths=lane * 0.95, showmedians=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(color)
                b.set_edgecolor(color)
                b.set_alpha(0.6)
            parts["cmedians"].set_color(color)
            ax.text(CLIP + 0.4, y, f"{np.median(np.abs(z)):.2f}   {np.mean(np.abs(z) > 2):4.0%}   "
                    f"{np.mean(np.abs(z) < 1.645):4.0%}", va="center", fontsize=7.5,
                    color=color, family="monospace")
    _frame(ax, methods, yticks)
    for x, ls in ((0, "--"), (2, ":"), (-2, ":")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=ls, zorder=1)
    ax.set_xlim(-CLIP - 0.5, CLIP + 6.0)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel(f"z = (posterior mean - true) / posterior std, pooled over all 24 parameters "
                  f"(clipped to +-{CLIP:.0f})", fontsize=9, color=MUTED)
    ax.text(CLIP + 0.4, -GAP / 2 - 0.08, "med|z|  |z|>2  in 90% CI", fontsize=7.5, color=MUTED,
            va="bottom", family="monospace")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6, label=lab) for _, lab, c in SETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=3,
              frameon=False, fontsize=9)
    ax.set_title("Posterior z-scores by model and fidelity level, pooled over all 24 parameters\n"
                 "(same draws as the CRPS and interval-score figures; rows in the same order)",
                 fontsize=11, loc="left", color=INK, pad=20)
    return fig, "zscore_overall_lofihifi.png"


def write_table(rows, path):
    cols = ["method", "set", "setting", "accept", "degenerate", "crps", "crps_skill",
            "crps_raw_skill", "is95", "is_skill", "width95", "penalty95", "cov95", "medabsz"]
    # csv.writer, not manual joins: both the method labels and the per-case setting
    # strings contain commas, so an unquoted table cannot be parsed back reliably.
    out = []
    for m, label in order_methods(rows):
        for s, _, _ in SETS:
            v = rows.get((m, s))
            if v is None:
                continue
            out.append([label, s, v["setting"], f"{v['accept']:.4f}",
                        v["degenerate"], f"{v['crps']:.5f}",
                        f"{v['crps_skill']:.4f}", f"{v['crps_raw_skill']:.4f}",
                        f"{v['is']:.5f}", f"{v['is_skill']:.4f}",
                        f"{v['width']:.5f}", f"{v['penalty']:.5f}",
                        f"{v['cov95']:.4f}", f"{v['medabsz']:.3f}"])
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        wr.writerows(out)
    for r in out:
        print("  %-52s %-20s CRPS %+8s  IS %+8s  cov %6s  width %6s"
              % (r[0][:52], r[1], r[6], r[9], r[12], r[10]))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="scored")
    ap.add_argument("--metric", default="all", choices=["all", "scores", "zscore"])
    ap.add_argument("--roster", default="full",
                    choices=["full", "percase", "arch", "noise", "binw"])
    ap.add_argument("--out-sub", default="",
                    help="subfolder of scores_lofihifi to write into")
    a = ap.parse_args()
    global METHODS, SUFFIX_OVERRIDE, OUT_DIR
    if a.roster == "percase":
        METHODS, SUFFIX_OVERRIDE = METHODS_PERCASE, PERCASE_OVERRIDE
    elif a.roster == "arch":
        METHODS, SUFFIX_OVERRIDE = METHODS_ARCH, ARCH_OVERRIDE
    elif a.roster == "noise":
        METHODS, SUFFIX_OVERRIDE = METHODS_NOISE, NOISE_OVERRIDE
    elif a.roster == "binw":
        METHODS, SUFFIX_OVERRIDE = METHODS_BINW, BINW_OVERRIDE
    if a.out_sub:
        OUT_DIR = OUT_DIR / a.out_sub
    rows = gather(a.suffix)
    if not rows:
        raise SystemExit(f"no scored runs found under {ROOT} with suffix '{a.suffix}'")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    if a.metric in ("all", "scores"):
        for set_name, set_label, color in SETS:
            if not any((m, set_name) in rows for m, _ in METHODS):
                continue
            made.append(fig_scores(rows, a.suffix, set_name, set_label, color))
    if a.metric in ("all", "zscore"):
        made.append(fig_zscore(rows, a.suffix))
    for fig, fname in made:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT_DIR / fname, dpi=130)
        plt.close(fig)
        print("wrote", OUT_DIR / fname)
    write_table(rows, OUT_DIR / "scores_lofihifi.csv")


if __name__ == "__main__":
    main()
