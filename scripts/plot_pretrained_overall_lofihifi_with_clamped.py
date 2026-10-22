"""Same as plot_pretrained_overall_lofihifi.py, with the --clamp-info-gap
--freeze-scalars-corrected SPIN rerun's Path A/D added as two extra rows, for direct
comparison against the uncorrected run. New file (not overwriting
plot_pretrained_overall_lofihifi.py / zscore_overall_lofihifi.png) so the original
5-row figure stays intact as a historical artifact.

Usage: python scripts/plot_pretrained_overall_lofihifi_with_clamped.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
METHODS = [("npe-gibbs-lofihifi", "Gibbs NPE (300k, w=5.92)"),
           ("npe-baseline-lofihifi", "Vanilla NPE (300k/400ep)"),
           ("npe-noise-lofihifi", "NPE + 11dB noise (300k/400ep)"),
           ("spin-pathA-lofihifi", "SPIN v3d Path A (uncorrected, ep400)"),
           ("spin-pathD-lofihifi", "SPIN v3d Path D (uncorrected, ep400)"),
           ("spin-pathA-lofihifi-clamped", "SPIN v3d Path A (clamp+freeze-scalars, ep415)"),
           ("spin-pathD-lofihifi-clamped", "SPIN v3d Path D (clamp+freeze-scalars, ep415)")]
SETS = [("lofi_test", "low-fi (no gap)", "#2a78d6"),
        ("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0


def pooled(method, set_name):
    d = np.load(ROOT / method / set_name / "zscore_shrinkage_data.npz", allow_pickle=True)
    return np.concatenate([d[f"{k}_z"] for k in d["param_keys"]])


def main():
    methods = METHODS
    row_h, gap = 1.0, 0.35
    lane = row_h / len(SETS)
    fig, ax = plt.subplots(figsize=(11, 1.25 * len(methods) + 1.8))
    table = []
    yticks = []
    for r, (m, label) in enumerate(methods):
        top = r * (row_h + gap)
        yticks.append(top + row_h / 2)
        if r % 2 == 0:
            ax.axhspan(top - gap / 2, top + row_h + gap / 2, color=GRID, alpha=0.35, lw=0, zorder=0)
        for j, (s, _, color) in enumerate(SETS):
            z = pooled(m, s)
            d = np.load(ROOT / m / s / "zscore_shrinkage_data.npz", allow_pickle=True)
            sh = np.concatenate([d[f"{k}_shrinkage"] for k in d["param_keys"]])
            table.append((label, s, np.median(d["accept_rate"]), np.median(np.abs(z)), np.mean(np.abs(z) > 2),
                          np.mean(np.abs(z) < 1.645), np.median(sh)))
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
    n = len(methods)
    ax.set_yticks(yticks, [lab for _, lab in methods], fontsize=10)
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
    ax.set_title("Nested-sim-trained models, incl. SPIN clamp-info-gap/freeze-scalars rerun: posterior\n"
                 "z-scores pooled over all 24 parameters, by model and test set (24 params x 1,000 sims per violin)",
                 fontsize=11, loc="left", color=INK, pad=22)
    fig.tight_layout()
    out = ROOT / "zscore_overall_lofihifi_with_clamped.png"
    fig.savefig(out, dpi=130)
    print("| model | test set | accept | med|z| | |z|>2 | cov90 | med shrinkage |\n|---|---|---|---|---|---|---|")
    for label, s, acc, mz, gt2, cov, sh in table:
        print(f"| {label} | {s} | {acc:.2f} | {mz:.2f} | {gt2:.1%} | {cov:.1%} | {sh:.2f} |")
    print("wrote", out)


if __name__ == "__main__":
    main()
