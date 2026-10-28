"""R^2 heatmap for the 6-method lofihifi comparison: one row per parameter, one column
per method, color = R^2 of posterior mean vs true value (coefficient of determination,
R^2 = 1 - SS_res/SS_tot, *not* squared Pearson r -- this matters here specifically
because a posterior mean that collapses to a near-constant value regardless of the true
input (the failure mode plot_pretrained_scatter_lofihifi_6methods.py found for SPIN
Dual-flow's Cas) scores near 0 or negative on R^2, correctly flagging "no better than
predicting the mean", whereas Pearson r^2 only measures linear association and is blind
to that kind of bias.

Replaces plot_pretrained_scatter_lofihifi_6methods.py's overlapping per-parameter scatter
(illegible with 6 methods on one axis) with a single compact summary; use the scatter
script separately to drill into one specific (parameter, method) cell if a value here
looks surprising.

Usage: python scripts/plot_pretrained_r2_heatmap_lofihifi.py [--set hifi_v2strong_test]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
OUT_DIR = ROOT / "zscore_overall_6methods"
METHODS = [("npe-gibbs-lofihifi", "Gibbs NPE"),
           ("spin-pathD-lofihifi", "SPIN Dual-flow\n(Path D)"),
           ("npe-dann-lofihifi", "NPE DANN\n/ WDGRL"),
           ("npe-noise-lofihifi", "NPE + 11dB\nnoise"),
           ("npe-baseline-lofihifi", "Vanilla NPE"),
           ("spin-singleflow-lofihifi", "SPIN\nsingle-flow")]
SET_LABELS = {"lofi_test": "low-fi (no gap)", "hifi_v2_test": "high-fi realistic",
              "hifi_v2strong_test": "high-fi exaggerated"}
INK, MUTED = "#1f1f1e", "#6b6a63"

# Ordered by identifiability (median shrinkage on Gibbs NPE/lofi_test), most to least --
# same grouping as plot_pretrained_rows_lofihifi_6methods.py's GROUPS, concatenated.
PARAM_ORDER = [
    "Rap", "Ras", "Cas", "Eap", "Emax_LV", "Vs", "Tmax", "Emax_RV",
    "τ", "Eedref_rv", "Cvp", "Brv", "Eedref_lv", "Eedref_ra", "Blv", "Cvs",
    "Emax_LA", "Emax_RA", "τ_a", "AVD", "Bla", "Bra", "Tmax_a", "Eedref_la",
]


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hifi_v2strong_test", choices=list(SET_LABELS))
    a = ap.parse_args()

    data = {m: np.load(ROOT / m / a.set / "zscore_shrinkage_data.npz", allow_pickle=True) for m, _ in METHODS}
    param_keys = [str(x) for x in data[METHODS[0][0]]["param_keys"]]
    keys = [k for k in PARAM_ORDER if k in param_keys]

    mat = np.full((len(keys), len(METHODS)), np.nan)
    for i, k in enumerate(keys):
        pi = param_keys.index(k)
        for j, (m, _) in enumerate(METHODS):
            d = data[m]
            mat[i, j] = r2(d["true"][:, pi], d["post_mean"][:, pi])

    fig, ax = plt.subplots(figsize=(1.7 * len(METHODS) + 2.2, 0.38 * len(keys) + 1.8))
    im = ax.imshow(np.clip(mat, -1, 1), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(METHODS)):
            v = mat[i, j]
            txt_color = INK if -0.6 < v < 0.85 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
    ax.set_xticks(range(len(METHODS)), [lab for _, lab in METHODS], fontsize=8, color=MUTED)
    ax.set_yticks(range(len(keys)), keys, fontsize=9, color=INK)
    ax.set_xticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("R² (posterior mean vs true; 1=perfect, 0=no better than predicting\nthe mean, "
                    "<0=worse than predicting the mean)", fontsize=8, color=MUTED)
    ax.set_title(f"Posterior mean R² by parameter and method, {SET_LABELS[a.set]}\n"
                 f"(rows ordered most- to least-identifiable, 1,000 sims per method)",
                 fontsize=11, loc="left", color=INK)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"r2_heatmap_lofihifi_{a.set}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
