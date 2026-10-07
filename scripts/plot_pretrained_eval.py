"""Violins of posterior z-scores and shrinkage from eval_pretrained.py outputs.

One panel per parameter; x = method; three violins per method, one per test set
(low-fi control, realistic high-fi, exaggerated high-fi). Annotations: median z and
fraction |z| > 2 (z figure), median shrinkage (shrinkage figure). Also prints a
summary table over all 24 parameters.

Usage: python scripts/plot_pretrained_eval.py [--params Rap,Ras,Cas,Eap,Emax_LV,Vs] [--subset all|in_support]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval"
METHODS = [("npe-baseline", "Vanilla NPE"), ("npe-noise", "NPE + 11dB noise"),
           ("spin-pathA", "SPIN v3d\nPath A"), ("spin-pathD", "SPIN v3d\nPath D")]
SETS = [("lofi_test", "low-fi (no gap)", "#2a78d6"),
        ("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def load(method, set_name):
    p = ROOT / method / set_name / "zscore_shrinkage_data.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def panel_grid(params):
    n = len(params)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def violins(ax, data_fn, ylabel, annot, clip=None):
    width, offset = 0.24, 0.27
    for k, (set_name, _, color) in enumerate(SETS):
        pos, vals = [], []
        for i, (m, _) in enumerate(METHODS):
            v = data_fn(m, set_name)
            if v is None or len(v) == 0:
                continue
            if clip is not None:
                v = np.clip(v, *clip)
            pos.append(i + (k - 1) * offset)
            vals.append(v)
        if not vals:
            continue
        parts = ax.violinplot(vals, positions=pos, widths=width, showmedians=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.55)
        parts["cmedians"].set_color(color)
        for x, v in zip(pos, vals):
            ax.annotate(annot(v), (x, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=6.5, color=color)
    ax.set_xticks(range(len(METHODS)), [lab for _, lab in METHODS], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="Rap,Ras,Cas,Eap,Emax_LV,Vs")
    a = ap.parse_args()
    params = a.params.split(",")

    def getter(suffix, param):
        def f(m, s):
            d = load(m, s)
            return None if d is None else d[f"{param}_{suffix}"]
        return f

    # z-scores
    fig, axes = panel_grid(params)
    for ax, p in zip(axes, params):
        violins(ax, getter("z", p), "z = (post mean - true) / post std",
                lambda v: f"{np.median(v):+.1f}\n{np.mean(np.abs(v) > 2):.0%}", clip=(-10, 10))
        for y, ls in ((0, "--"), (2, ":"), (-2, ":")):
            ax.axhline(y, color=MUTED, lw=0.8, ls=ls)
        ax.set_title(p, fontsize=11, color=INK, loc="left", pad=22)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.55, label=lab) for _, lab, c in SETS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Posterior z-scores of pretrained models on nested test sets (1,000 sims each; z clipped to ±10)\n"
                 "annotations: median z / fraction |z| > 2", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = ROOT / "zscore_violins.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # shrinkage
    fig, axes = panel_grid(params)
    for ax, p in zip(axes, params):
        violins(ax, getter("shrinkage", p), "shrinkage = 1 - post var / prior var",
                lambda v: f"{np.median(v):.2f}", clip=(-1, 1))
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(p, fontsize=11, color=INK, loc="left", pad=14)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Posterior shrinkage of pretrained models on nested test sets (annotation: median)",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = ROOT / "shrinkage_violins.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # summary over all 24 parameters
    print(f"\n{'method':14s} {'set':20s} {'accept':>7s} {'med|z|':>7s} {'|z|>2':>6s} {'cov90':>6s}")
    for m, _ in METHODS:
        for s, _, _ in SETS:
            d = load(m, s)
            if d is None:
                continue
            keys = list(d["param_keys"])
            Z = np.concatenate([d[f"{k}_z"] for k in keys])
            cov = np.mean(np.abs(Z) < 1.645)  # Gaussian-approx 90% interval coverage
            print(f"{m:14s} {s:20s} {np.median(d['accept_rate']):7.2f} {np.median(np.abs(Z)):7.2f} "
                  f"{np.mean(np.abs(Z) > 2):6.1%} {cov:6.1%}")


if __name__ == "__main__":
    main()
