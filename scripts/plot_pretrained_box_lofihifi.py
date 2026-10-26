"""Box-plot variant of plot_pretrained_overall_lofihifi.py's violin figure, for a
readability comparison. Same data, same annotations (med|z|, frac|z|>2, 90% CI
coverage) -- just methods on the x-axis (worst -> best, left to right) with z-score
on the y-axis, and boxes instead of violins.

Usage: python scripts/plot_pretrained_box_lofihifi.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
# Worst -> best, left to right (per user's framing of overall calibration quality).
METHODS = [("spin-pathA-lofihifi", "SPIN v3d\nPath A"),
           ("npe-baseline-lofihifi", "Vanilla\nNPE"),
           ("npe-noise-lofihifi", "NPE +\n11dB noise"),
           ("spin-pathD-lofihifi", "SPIN v3d\nPath D"),
           ("npe-gibbs-lofihifi", "Gibbs\nNPE")]
SETS = [("lofi_test", "low-fi (no gap)", "#2a78d6"),
        ("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0


def pooled(method, set_name):
    d = np.load(ROOT / method / set_name / "zscore_shrinkage_data.npz", allow_pickle=True)
    return np.concatenate([d[f"{k}_z"] for k in d["param_keys"]])


def main():
    n_methods, n_sets = len(METHODS), len(SETS)
    group_w, box_w, gap = 1.0, 0.24, 0.03
    fig, ax = plt.subplots(figsize=(1.9 * n_methods + 2.0, 6.5))

    table = []
    xticks = []
    for i, (m, label) in enumerate(METHODS):
        gx = i * (group_w + 0.4)
        xticks.append(gx + (n_sets - 1) * (box_w + gap) / 2)
        if i % 2 == 0:
            ax.axvspan(gx - 0.2, gx + (n_sets) * (box_w + gap) + 0.17, color=GRID, alpha=0.35, lw=0, zorder=0)
        for j, (s, _, color) in enumerate(SETS):
            z = pooled(m, s)
            d = np.load(ROOT / m / s / "zscore_shrinkage_data.npz", allow_pickle=True)
            sh = np.concatenate([d[f"{k}_shrinkage"] for k in d["param_keys"]])
            med_abs_z = np.median(np.abs(z))
            frac_gt2 = np.mean(np.abs(z) > 2)
            cov90 = np.mean(np.abs(z) < 1.645)
            table.append((label.replace("\n", " "), s, np.median(d["accept_rate"]), med_abs_z, frac_gt2,
                          cov90, np.median(sh)))
            x = gx + j * (box_w + gap)
            bp = ax.boxplot([np.clip(z, -CLIP, CLIP)], positions=[x], widths=box_w * 0.95,
                             showfliers=False, patch_artist=True, medianprops={"color": color, "lw": 1.6},
                             whiskerprops={"color": color, "lw": 1.0}, capprops={"color": color, "lw": 1.0},
                             boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.5})
            ax.text(x, CLIP + 0.5, f"{med_abs_z:.2f}\n{frac_gt2:4.0%}\n{cov90:4.0%}", ha="center", va="bottom",
                    fontsize=6.8, color=color, family="monospace", linespacing=1.3)
    for y, ls in ((0, "--"), (2, ":"), (-2, ":")):
        ax.axhline(y, color=MUTED, lw=0.8, ls=ls, zorder=1)
    ax.set_xticks(xticks, [lab for _, lab in METHODS], fontsize=9.5)
    ax.set_xlim(-0.35, (n_methods - 1) * (group_w + 0.4) + (n_sets - 1) * (box_w + gap) + 0.35)
    ax.set_ylim(-CLIP - 0.5, CLIP + 3.2)
    ax.set_yticks(np.arange(-10, 11, 2))
    ax.set_ylabel(f"z = (posterior mean - true) / posterior std, pooled over all 24 parameters   "
                  f"(clipped to ±{CLIP:.0f})", fontsize=9, color=MUTED)
    ax.text(-0.3, CLIP + 0.5, "med|z|\n|z|>2\nin 90% CI", fontsize=6.8, color=MUTED, va="bottom", ha="right",
            family="monospace", linespacing=1.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.5, label=lab) for _, lab, c in SETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Nested-sim-trained models (lofi 300k sims + hifi_v2strong real-stand-in): posterior\n"
                 "z-scores pooled over all 24 parameters, methods ordered worst -> best, by test set "
                 "(24 params x 1,000 sims per box)",
                 fontsize=11, loc="left", color=INK, pad=10)
    fig.tight_layout()
    out = ROOT / "zscore_overall_lofihifi_boxplot.png"
    fig.savefig(out, dpi=130)
    print("| model | test set | accept | med|z| | |z|>2 | cov90 | med shrinkage |\n|---|---|---|---|---|---|---|")
    for label, s, acc, mz, gt2, cov, sh in table:
        print(f"| {label} | {s} | {acc:.2f} | {mz:.2f} | {gt2:.1%} | {cov:.1%} | {sh:.2f} |")
    print("wrote", out)


if __name__ == "__main__":
    main()
