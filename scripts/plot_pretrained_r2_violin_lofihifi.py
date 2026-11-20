"""R^2 of the posterior mean against truth, pooled over the 24 parameters as a violin.

One violin per method: the 24 per-parameter R^2 values, R^2 = 1 - SS_res/SS_tot (the
coefficient of determination, not squared Pearson r -- a posterior mean that collapses
to a near-constant regardless of the true input must score <= 0, and only this form
does that).

R^2 is computed in unit-cube coordinates (prior box -> [0,1], log on the log-uniform
dims), the same mapping the CRPS and interval-score figures use. R^2 is already
invariant to an affine rescaling, so the box mapping alone changes nothing; the log on
Eap/Emax_RV/Rap does, and it is what makes those three commensurate with the rest
rather than dominated by their upper decade.

Usage: python scripts/plot_pretrained_r2_violin_lofihifi.py [--set hifi_v2_test]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HOME = Path.home()
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = HOME / "results/nested-sim/pretrained_eval_lofihifi"
# Figures live outside the eval tree: the npz under ROOT are large and are not
# synced to the Mac, so the whole eval tree is a mutagen ignore and anything
# written there would be invisible locally.
OUT_DIR = Path.home() / "results/nested-sim/figures_lofihifi/scores_lofihifi"

# Same roster and labels as plot_pretrained_scores_lofihifi.py, so the figures line up.
METHODS = [
    ("npe-gibbs-w005to50-lofihifi", "Gibbs NPE, ep800, w window [0.05, 50]"),
    ("npe-gibbs-1600-lofihifi", "Gibbs NPE, ep1600, w window [0.76, 75]"),
    ("npe-gibbs-800nolip-lofihifi", "Gibbs NPE, ep800, w window [0.76, 75]"),
    ("npe-gibbs-v2strongcal-lofihifi", "Gibbs NPE, ep400, hifi-calibrated"),
    ("npe-gibbs-lofihifi", "Gibbs NPE, ep100/300k (w=5.92)"),
    ("npe-gibbs-v2strongcal-ep800", "Gibbs NPE, ep800, hifi-calibrated"),
    ("npe-gibbs-v2realcal-lofihifi", "Gibbs NPE, ep800, realistic-calibrated"),
    ("npe-gibbs-v2realcal-ep400", "Gibbs NPE, ep400, realistic-calibrated"),
    ("sun-routeA-lofihifi", "Sun et al. Route A MDN"),
    ("sun-routeA-cleanpairs", "Route A MDN, clean pairs (no Langevin)"),
    ("spin-pathD-lofihifi", "SPIN dual-flow, Path D"),
    ("spin-pathA-lofihifi", "SPIN dual-flow, Path A"),
    ("spin-singleflow-lofihifi", "SPIN single-flow (mixup)"),
    ("npe-dann-lofihifi", "NPE + WDGRL DANN"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
# Rows whose eval was written under a tag other than --suffix. The key is the METHODS
# name; the value is (eval directory stem, tag).
SUFFIX_OVERRIDE = {
    "npe-gibbs-v2strongcal-ep800": ("npe-gibbs-v2strongcal-lofihifi", "ep800"),
    "npe-gibbs-v2realcal-lofihifi": ("npe-gibbs-v2realcal-lofihifi", ""),
    "npe-gibbs-v2realcal-ep400": ("npe-gibbs-v2realcal-lofihifi", "ep400"),
    "sun-routeA-cleanpairs": ("sun-routeA-cleanpairs", "beta1"),
}
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
SETS = {"lofi_test": ("low-fi (no gap)", "#2a78d6"),
        "hifi_v2_test": ("high-fi realistic", "#eb6834"),
        "hifi_v2strong_test": ("high-fi exaggerated", "#1baf7a")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP_LO = -1.0            # display floor; true values are printed in the margin


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def r2_case(true_u, pred_u):
    """Per-case R^2 against the prior-mean predictor, pooled over the 24 parameters.

    A single case has no across-case variance to form SS_tot from, so the reference is
    the prior itself: SS_tot = d * Var(U[0,1]) = d/12, the error a predictor that
    always returned the prior mean would make in expectation. The result is on the same
    footing as the CRPS and interval-score skill scores, which use the same baseline:
    1 = exact, 0 = no better than the prior, negative = worse.

    Using each case's own empirical spread instead (its 24 true values about their own
    mean) was considered and rejected -- it compares a parameter against the average of
    unrelated parameters, and goes unstable whenever a case sits near the box centre.
    """
    d = true_u.shape[1]
    ss_res = np.sum((true_u - pred_u) ** 2, axis=1)
    return 1.0 - ss_res / (d / 12.0)


def to_unit(phys, lo, hi, log_mask):
    x = np.where(log_mask, np.log(np.clip(phys, 1e-30, None)), phys)
    lo2 = np.where(log_mask, np.log(lo), lo)
    hi2 = np.where(log_mask, np.log(hi), hi)
    return (x - lo2) / (hi2 - lo2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hifi_v2_test", choices=list(SETS))
    ap.add_argument("--suffix", default="scored")
    ap.add_argument("--by", default="param", choices=["param", "case"],
                    help="param: one point per parameter, R^2 across the 1000 cases. "
                         "case: one point per case, R^2 across the 24 parameters "
                         "against the prior-mean predictor.")
    ap.add_argument("--roster", default="full", choices=["full", "percase"])
    ap.add_argument("--out-sub", default="",
                    help="subfolder of scores_lofihifi to write into")
    a = ap.parse_args()
    global METHODS, SUFFIX_OVERRIDE, OUT_DIR
    if a.roster == "percase":
        METHODS, SUFFIX_OVERRIDE = METHODS_PERCASE, PERCASE_OVERRIDE
    if a.out_sub:
        OUT_DIR = OUT_DIR / a.out_sub

    import eval_pretrained_lofihifi as ep  # noqa: E402  (prior box + log dims)

    rows = []
    for m, lab in METHODS:
        stem, suf = SUFFIX_OVERRIDE.get(m, (m, a.suffix))
        f = ROOT / (f"{stem}__{suf}" if suf else stem) / a.set / "zscore_shrinkage_data.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        keys = [str(x) for x in d["param_keys"]]
        lo_t, hi_t = ep.load_prior_bounds_nestedsim(keys)
        lo, hi = lo_t.numpy(), hi_t.numpy()
        log_mask = np.zeros(len(keys), bool)
        log_mask[ep.load_log_dims(keys)] = True
        tu = to_unit(d["true"], lo, hi, log_mask)
        pu = to_unit(d["post_mean"], lo, hi, log_mask)
        vals = (np.array([r2(tu[:, j], pu[:, j]) for j in range(len(keys))])
                if a.by == "param" else r2_case(tu, pu))
        rows.append((lab, vals))
    if not rows:
        raise SystemExit(f"no scored runs for {a.set} under {ROOT}")

    rows.sort(key=lambda t: -np.median(t[1]))
    labs = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    set_label, color = SETS[a.set]

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 2.2))
    pos = np.arange(len(rows))[::-1]
    parts = ax.violinplot([np.clip(v, CLIP_LO, 1.0) for v in vals], positions=pos,
                          vert=False, widths=0.82, showextrema=False, showmedians=False)
    for b in parts["bodies"]:
        b.set_facecolor(color); b.set_alpha(0.55); b.set_edgecolor(color); b.set_linewidth(1.2)
    for p, v in zip(pos, vals):
        vc = np.clip(v, CLIP_LO, 1.0)
        ax.scatter(vc, np.full_like(vc, p), s=9, color=INK, alpha=0.5, zorder=3,
                   linewidths=0)
        ax.plot([np.median(v)] * 2, [p - 0.33, p + 0.33], color=INK, lw=2.0, zorder=4)

    xmax = 1.0
    for p, v in zip(pos, vals):
        ax.text(xmax + 0.03, p, f"{np.median(v):+.2f}", va="center", ha="left",
                fontsize=9, color=INK, family="monospace")
        ax.text(xmax + 0.19, p, f"{np.mean(v):+.2f}", va="center", ha="left",
                fontsize=9, color=MUTED, family="monospace")
        frac = f"{int((v > 0.5).sum()):2d}" if a.by == "param" else f"{np.mean(v > 0):.2f}"
        ax.text(xmax + 0.40, p, frac, va="center", ha="left",
                fontsize=9, color=MUTED, family="monospace")
    ax.text(xmax + 0.03, pos[0] + 0.72, "med", fontsize=9, color=INK, family="monospace")
    ax.text(xmax + 0.19, pos[0] + 0.72, "mean", fontsize=9, color=MUTED, family="monospace")
    ax.text(xmax + 0.40, pos[0] + 0.72, "n>.5" if a.by == "param" else "f>0",
            fontsize=9, color=MUTED, family="monospace")

    ax.axvline(0.0, color=INK, ls="--", lw=1.1)
    ax.text(0.0, pos[0] + 0.72, "no better than\npredicting the mean", fontsize=8,
            color=MUTED, ha="center", va="bottom")
    ax.set_yticks(pos); ax.set_yticklabels(labs, fontsize=9.5)
    ax.set_xlim(CLIP_LO - 0.02, xmax + 0.56)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xticks(np.arange(-1.0, 1.01, 0.25))
    if a.by == "param":
        ax.set_xlabel("R² of posterior mean vs truth, one point per parameter "
                      f"(unit-cube coordinates; display floor {CLIP_LO:g})")
        ax.set_title(f"Per-parameter R² pooled over 24 parameters — {set_label}, "
                     "1,000 test sims", loc="left", fontsize=12)
    else:
        ax.set_xlabel("R² of posterior mean vs truth, one point per test case, pooled "
                      f"over 24 parameters vs the prior-mean predictor "
                      f"(unit-cube; display floor {CLIP_LO:g})")
        ax.set_title(f"Per-case R² pooled over 24 parameters — {set_label}, "
                     "1,000 test sims", loc="left", fontsize=12)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (f"r2_violin_lofihifi_{a.set}.png" if a.by == "param"
                     else f"r2_violin_bycase_lofihifi_{a.set}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, v in rows:
        tail = (f"n>0.5 {int((v > 0.5).sum())}/24" if a.by == "param"
                else f"frac>0 {np.mean(v > 0):.3f}")
        print(f"  {lab:46s} med {np.median(v):+.3f}  mean {np.mean(v):+.3f}  "
              f"min {v.min():+.3f}  {tail}")


if __name__ == "__main__":
    main()
