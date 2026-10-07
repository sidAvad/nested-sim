"""One-test-set z-score figure: parameters on rows, one horizontal violin per model.

Reads eval_pretrained.py outputs for a single test set (default: the exaggerated
high-fi set) and draws, for every parameter, four stacked horizontal violins (one
per model). Rows are sorted by the fraction |z| > 2 of the first model (worst at
the top). Right-hand annotations: median z / fraction |z| > 2. z is clipped to
+-CLIP for display.

Usage: python scripts/plot_pretrained_rows.py [--set hifi_v2strong_test] [--params all]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval"
METHODS = [("npe-baseline", "Vanilla NPE", "#2a78d6"), ("npe-noise", "NPE + 11dB noise", "#eb6834"),
           ("spin-pathA", "SPIN v3d Path A", "#1baf7a"), ("spin-pathD", "SPIN v3d Path D", "#eda100")]
SET_LABELS = {"lofi_test": "low-fi (no gap)", "hifi_v2_test": "high-fi realistic",
              "hifi_v2strong_test": "high-fi exaggerated"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hifi_v2strong_test", choices=list(SET_LABELS))
    ap.add_argument("--params", default="all")
    a = ap.parse_args()

    data = {m: np.load(ROOT / m / a.set / "zscore_shrinkage_data.npz", allow_pickle=True) for m, _, _ in METHODS}
    keys = list(data[METHODS[0][0]]["param_keys"]) if a.params == "all" else a.params.split(",")
    first = METHODS[0][0]
    keys.sort(key=lambda k: -np.mean(np.abs(data[first][f"{k}_z"]) > 2))

    n = len(keys)
    row_h, gap = 1.0, 0.25
    lane = row_h / len(METHODS)
    fig, ax = plt.subplots(figsize=(11, 0.62 * n + 1.8))
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
            ax.text(CLIP + 0.4, y, f"{np.median(z):+.1f}  {np.mean(np.abs(z) > 2):.0%}", va="center",
                    fontsize=6.5, color=color)
    for x, ls in ((0, "--"), (2, ":"), (-2, ":")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=ls, zorder=1)
    ax.set_yticks(yticks, keys, fontsize=9)
    ax.set_ylim(n * (row_h + gap) - gap / 2, -gap / 2)  # first row at the top
    ax.set_xlim(-CLIP - 0.5, CLIP + 3.2)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel(f"z = (posterior mean - true) / posterior std   (clipped to ±{CLIP:.0f})", fontsize=9, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.text(CLIP + 0.4, -gap / 2 - 0.15, "median z  |z|>2", fontsize=7, color=MUTED, va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6, label=lab) for _, lab, c in METHODS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.035 * 24 / n), ncol=4,
              frameon=False, fontsize=9)
    ax.set_title(f"Posterior z-scores on the {SET_LABELS[a.set]} test set (1,000 sims), one violin per model;\n"
                 f"rows sorted by Vanilla NPE fraction |z| > 2", fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    out = ROOT / f"zscore_rows_{a.set}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
