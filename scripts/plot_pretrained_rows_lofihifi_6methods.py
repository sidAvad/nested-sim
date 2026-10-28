"""Per-parameter z-score figure for the 6-method nested-sim lofi/hifi comparison --
same row-per-parameter / violin-per-model layout as plot_pretrained_rows_lofihifi.py,
updated to the 6-method set from plot_pretrained_overall_lofihifi_6methods.py (SPIN
Path A dropped, NPE DANN/WDGRL and SPIN single-flow added).

Groups are ordered by identifiability (median posterior shrinkage, measured on
Gibbs NPE / lofi_test -- the best-calibrated model/set, so shrinkage there isn't
confounded by miscalibration): group 1 is the 8 most-identifiable params, group 2
the next 8, group 3 the remaining (least-identifiable) 8.

Usage: python scripts/plot_pretrained_rows_lofihifi_6methods.py [--set hifi_v2strong_test] [--group 1]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
OUT_DIR = Path.home() / "results/nested-sim/zscore_overall_6methods"
METHODS = [("npe-gibbs-lofihifi", "Gibbs NPE", "#2a78d6"),
           ("spin-pathD-lofihifi", "SPIN Dual-flow (Path D)", "#9b59b6"),
           ("npe-dann-lofihifi", "NPE DANN / WDGRL", "#c0392b"),
           ("npe-noise-lofihifi", "NPE + 11dB noise", "#1baf7a"),
           ("npe-baseline-lofihifi", "Vanilla NPE", "#eb6834"),
           ("spin-singleflow-lofihifi", "SPIN single-flow", "#eda100")]
SET_LABELS = {"lofi_test": "low-fi (no gap)", "hifi_v2_test": "high-fi realistic",
              "hifi_v2strong_test": "high-fi exaggerated"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0

GROUPS = {
    1: ["Rap", "Ras", "Cas", "Eap", "Emax_LV", "Vs", "Tmax", "Emax_RV"],
    2: ["τ", "Eedref_rv", "Cvp", "Brv", "Eedref_lv", "Eedref_ra", "Blv", "Cvs"],
    3: ["Emax_LA", "Emax_RA", "τ_a", "AVD", "Bla", "Bra", "Tmax_a", "Eedref_la"],
}
GROUP_LABELS = {1: "most identifiable 8", 2: "middle 8", 3: "least identifiable 8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hifi_v2strong_test", choices=list(SET_LABELS))
    ap.add_argument("--group", type=int, default=1, choices=[1, 2, 3])
    a = ap.parse_args()

    data = {m: np.load(ROOT / m / a.set / "zscore_shrinkage_data.npz", allow_pickle=True) for m, _, _ in METHODS}
    keys = GROUPS[a.group]
    first = METHODS[0][0]
    keys = sorted(keys, key=lambda k: -np.mean(np.abs(data[first][f"{k}_z"]) > 2))

    n = len(keys)
    row_h, gap = 1.0, 0.25
    lane = row_h / len(METHODS)
    fig, ax = plt.subplots(figsize=(12, 0.62 * n + 1.8))
    yticks = []
    for r, k in enumerate(keys):
        top = r * (row_h + gap)
        yticks.append(top + row_h / 2)
        if r % 2 == 0:
            ax.axhspan(top - gap / 2, top + row_h + gap / 2, color=GRID, alpha=0.35, lw=0, zorder=0)
        for j, (m, _, color) in enumerate(METHODS):
            z = data[m][f"{k}_z"]
            y = top + (j + 0.5) * lane
            parts = ax.violinplot([np.clip(z, -CLIP, CLIP)], positions=[y], vert=False, widths=lane * 0.95,
                                  showmedians=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(color)
                b.set_edgecolor(color)
                b.set_alpha(0.6)
            parts["cmedians"].set_color(color)
            sh = np.median(data[m][f"{k}_shrinkage"])
            ax.text(CLIP + 0.4, y, f"{np.median(z):+.1f}  {np.mean(np.abs(z) > 2):.0%}  {sh:.2f}", va="center",
                    fontsize=6.5, color=color)
    for x, ls in ((0, "--"), (2, ":"), (-2, ":")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=ls, zorder=1)
    ax.set_yticks(yticks, keys, fontsize=9)
    ax.set_ylim(n * (row_h + gap) - gap / 2, -gap / 2)  # first row at the top
    ax.set_xlim(-CLIP - 0.5, CLIP + 4.4)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel(f"z = (posterior mean - true) / posterior std   (clipped to ±{CLIP:.0f})", fontsize=9, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(CLIP + 0.4, -gap / 2 - 0.15, "median z  |z|>2  shrinkage", fontsize=7, color=MUTED, va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6, label=lab) for _, lab, c in METHODS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3,
              frameon=False, fontsize=9)
    ax.set_title(f"Posterior z-scores on {SET_LABELS[a.set]} ({GROUP_LABELS[a.group]}, by median shrinkage), "
                 f"one violin per model;\nrows sorted by Gibbs NPE fraction |z| > 2",
                 fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"zscore_rows_lofihifi_{a.set}_group{a.group}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
