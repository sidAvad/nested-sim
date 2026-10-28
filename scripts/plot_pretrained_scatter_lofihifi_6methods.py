"""Posterior mean vs ground truth scatter, per parameter, across the 6-method lofihifi
comparison -- complements the pooled z-score violins (plot_pretrained_overall_lofihifi_6methods.py)
and the per-parameter z-score rows (plot_pretrained_rows_lofihifi_6methods.py), which both
normalize error by the posterior's own width. This plot shows absolute point-estimate
accuracy directly (post_mean vs true, in raw parameter units, un-normalized) -- useful
because a model can have a low median|z| while still being numerically far from the
truth (if its posterior is honestly wide), and because the forward map theta->observable
is nonlinear/"sloppy" (some parameter directions barely affect the waveform/scalars, so
small-looking per-parameter z-score errors can still concentrate in the few sensitive
directions that actually drive reconstruction error -- see eval_recon_error_lofihifi.py).

One panel per parameter (group 1 = the 8 most-identifiable, by default -- the ones that
actually matter for reconstruction), one color per method, true value on x, posterior
mean on y, y=x = perfect recovery.

Usage: python scripts/plot_pretrained_scatter_lofihifi_6methods.py [--set hifi_v2strong_test] [--group 1]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
OUT_DIR = ROOT / "zscore_overall_6methods"
METHODS = [("npe-gibbs-lofihifi", "Gibbs NPE", "#2a78d6"),
           ("spin-pathD-lofihifi", "SPIN Dual-flow (Path D)", "#9b59b6"),
           ("npe-dann-lofihifi", "NPE DANN / WDGRL", "#c0392b"),
           ("npe-noise-lofihifi", "NPE + 11dB noise", "#1baf7a"),
           ("npe-baseline-lofihifi", "Vanilla NPE", "#eb6834"),
           ("spin-singleflow-lofihifi", "SPIN single-flow", "#eda100")]
SET_LABELS = {"lofi_test": "low-fi (no gap)", "hifi_v2_test": "high-fi realistic",
              "hifi_v2strong_test": "high-fi exaggerated"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"

GROUPS = {
    1: ["Rap", "Ras", "Cas", "Eap", "Emax_LV", "Vs", "Tmax", "Emax_RV"],
    2: ["τ", "Eedref_rv", "Cvp", "Brv", "Eedref_lv", "Eedref_ra", "Blv", "Cvs"],
    3: ["Emax_LA", "Emax_RA", "τ_a", "AVD", "Bla", "Bra", "Tmax_a", "Eedref_la"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hifi_v2strong_test", choices=list(SET_LABELS))
    ap.add_argument("--group", type=int, default=1, choices=[1, 2, 3])
    a = ap.parse_args()

    keys = GROUPS[a.group]
    data = {m: np.load(ROOT / m / a.set / "zscore_shrinkage_data.npz", allow_pickle=True) for m, _, _ in METHODS}

    cols = 4
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), squeeze=False)
    axes_flat = axes.flat
    for ax in axes_flat[len(keys):]:
        ax.axis("off")

    for j, k in enumerate(keys):
        ax = axes_flat[j]
        param_keys = [str(x) for x in data[METHODS[0][0]]["param_keys"]]
        pi = param_keys.index(k)
        lo_all, hi_all = [], []
        for m, label, color in METHODS:
            d = data[m]
            true = d["true"][:, pi]
            post_mean = d["post_mean"][:, pi]
            ax.scatter(true, post_mean, s=5, color=color, alpha=0.35, edgecolors="none", label=label)
            # Robust (percentile-based) axis range -- a handful of degenerate posterior
            # means (same failure mode that makes some methods' points unsimulatable, see
            # eval_recon_error_lofihifi.py's n_ok) can otherwise stretch min/max far enough
            # to hide the bulk's actual behavior.
            lo_all.append(min(np.percentile(true, 0.5), np.percentile(post_mean, 0.5)))
            hi_all.append(max(np.percentile(true, 99.5), np.percentile(post_mean, 99.5)))
        lo, hi = min(lo_all), max(hi_all)
        pad = 0.04 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.9, ls="--", zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(k, fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.supxlabel("true value", fontsize=10, color=MUTED)
    fig.supylabel("posterior mean", fontsize=10, color=MUTED)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=7, label=lab)
               for _, lab, c in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Posterior mean vs ground truth on {SET_LABELS[a.set]} "
                 f"(group {a.group}, 1,000 sims per method, dashed = perfect recovery)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"scatter_lofihifi_{a.set}_group{a.group}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
