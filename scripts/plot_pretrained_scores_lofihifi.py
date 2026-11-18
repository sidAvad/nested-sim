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
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
OUT_DIR = ROOT / "scores_lofihifi"

METHODS = [
    ("npe-gibbs-w005to50-lofihifi", "Gibbs NPE, ep800, w window [0.05, 50]"),
    ("npe-gibbs-1600-lofihifi", "Gibbs NPE, ep1600, w window [0.76, 75]"),
    ("npe-gibbs-800nolip-lofihifi", "Gibbs NPE, ep800, w window [0.76, 75]"),
    ("npe-gibbs-v2strongcal-lofihifi", "Gibbs NPE, ep400, hifi-calibrated w window [0.14, 13.5]"),
    ("npe-gibbs-lofihifi", "Gibbs NPE, ep100/300k (w=5.92)"),
    ("sun-routeA-lofihifi", "Sun et al. Route A MDN (beta per level: 1/0.14/0.10)"),
    ("spin-pathD-lofihifi", "SPIN dual-flow, Path D"),
    ("spin-pathA-lofihifi", "SPIN dual-flow, Path A"),
    ("spin-singleflow-lofihifi", "SPIN single-flow (mixup)"),
    ("npe-dann-lofihifi", "NPE + WDGRL DANN"),
    ("npe-noise-lofihifi", "NPE + 11dB noise"),
    ("npe-baseline-lofihifi", "Vanilla NPE"),
]
SETS = [("lofi_test", "low-fi (no gap)", "#2a78d6"),
        ("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CLIP = 10.0
ROW_H, GAP = 1.0, 0.35
ANN_X = 0.58          # where the right-hand annotation column starts, in data units


def load(method, set_name, suffix):
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
    if "w" in s:
        clipped = abs(s.get("w_unclamped", s["w"]) - s["w"]) > 1e-9
        return f"w={s['w']:g}" + ("*" if clipped else "")
    if "beta" in s:
        clipped = abs(s.get("beta_unclipped", s["beta"]) - s["beta"]) > 1e-9
        return f"beta={s['beta']:g}" + ("*" if clipped else "")
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
    axb.set_xlim(0, xmax * 1.62)
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
    lines = [",".join(cols)]
    for m, label in order_methods(rows):
        for s, _, _ in SETS:
            v = rows.get((m, s))
            if v is None:
                continue
            lines.append(",".join([label, s, v["setting"], f"{v['accept']:.4f}",
                                   str(v["degenerate"]), f"{v['crps']:.5f}",
                                   f"{v['crps_skill']:.4f}", f"{v['crps_raw_skill']:.4f}",
                                   f"{v['is']:.5f}", f"{v['is_skill']:.4f}",
                                   f"{v['width']:.5f}", f"{v['penalty']:.5f}",
                                   f"{v['cov95']:.4f}", f"{v['medabsz']:.3f}"]))
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="scored")
    ap.add_argument("--metric", default="all", choices=["all", "scores", "zscore"])
    a = ap.parse_args()
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
