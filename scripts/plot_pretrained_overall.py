"""Overall z-score figure: one row per model, z pooled over all 24 parameters.

Each row holds three horizontal violins (low-fi control, high-fi realistic,
high-fi exaggerated) of every z-score for that model and test set (24 params x
1,000 sims). Right-hand annotations: median |z|, fraction |z| > 2, and coverage of
the Gaussian-approximate 90% interval (|z| < 1.645; ideal 90%). z clipped to
+-CLIP for display.

Usage: python scripts/plot_pretrained_overall.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval"
METHODS = [("npe-baseline", "Vanilla NPE"), ("npe-noise", "NPE + 11dB noise"),
           ("spin-pathA", "SPIN v3d Path A"), ("spin-pathD", "SPIN v3d Path D")]
SETS = [("lofi_test", "low-fi (no gap)", "#2a78d6"),
        ("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0


def pooled(method, set_name):
    d = np.load(ROOT / method / set_name / "zscore_shrinkage_data.npz", allow_pickle=True)
    return np.concatenate([d[f"{k}_z"] for k in d["param_keys"]])


def main():
    row_h, gap = 1.0, 0.35
    lane = row_h / len(SETS)
    fig, ax = plt.subplots(figsize=(11, 1.25 * len(METHODS) + 1.8))
    yticks = []
    for r, (m, label) in enumerate(METHODS):
        top = r * (row_h + gap)
        yticks.append(top + row_h / 2)
        if r % 2 == 0:
            ax.axhspan(top - gap / 2, top + row_h + gap / 2, color=GRID, alpha=0.35, lw=0, zorder=0)
        for j, (s, _, color) in enumerate(SETS):
            z = pooled(m, s)
            y = top + (j + 0.5) * lane
            parts = ax.violinplot([np.clip(z, -CLIP, CLIP)], positions=[y], vert=False, widths=lane * 0.95,
                                  showmedians=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(color)
                b.set_edgecolor(color)
                b.set_alpha(0.6)
            parts["cmedians"].set_color(color)
            ax.text(CLIP + 0.4, y, f"{np.median(np.abs(z)):.2f}   {np.mean(np.abs(z) > 2):4.0%}   "
                    f"{np.mean(np.abs(z) < 1.645):4.0%}", va="center", fontsize=7.5, color=color,
                    family="monospace")
    for x, ls in ((0, "--"), (2, ":"), (-2, ":")):
        ax.axvline(x, color=MUTED, lw=0.8, ls=ls, zorder=1)
    n = len(METHODS)
    ax.set_yticks(yticks, [lab for _, lab in METHODS], fontsize=10)
    ax.set_ylim(n * (row_h + gap) - gap / 2, -gap / 2)
    ax.set_xlim(-CLIP - 0.5, CLIP + 6.0)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel(f"z = (posterior mean - true) / posterior std, pooled over all 24 parameters   "
                  f"(clipped to ±{CLIP:.0f})", fontsize=9, color=MUTED)
    ax.text(CLIP + 0.4, -gap / 2 - 0.08, "med|z|  |z|>2  in 90% CI", fontsize=7.5, color=MUTED, va="bottom",
            family="monospace")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6, label=lab) for _, lab, c in SETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.14), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Posterior z-scores pooled over all parameters, by model and test set (24 params x 1,000 sims each)",
                 fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    out = ROOT / "zscore_overall.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
