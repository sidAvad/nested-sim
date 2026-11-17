"""CRPS, interval-score and z-score figures for the nested-sim lofi/hifi method comparison.

Three figures over one method list, one fidelity-level palette and one row order, so they
stack and can be read against each other:

  crps    CRPS skill, 1 - CRPS/CRPS_prior, pooled over all 24 parameters. CRPS is a proper
          score on the whole posterior, not just its first two moments, so unlike the
          z-score it cannot be gamed by a posterior that is wide in the right places. 1 is
          a point mass on the truth, 0 is the prior, negative is worse than having not
          looked at the data.
  is      The 95% interval score, split into the two things that make it large: the
          interval's width (sharpness) and the miss penalty (calibration). The sum alone
          cannot tell a wide honest posterior from a narrow overconfident one, which is
          the entire question once the likelihood is misspecified.
  zscore  The existing (posterior mean - true)/posterior std figure, reproduced here from
          the same re-scored runs so all three panels come from one set of draws.

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
    ("npe-gibbs-w005to50-lofihifi", "Gibbs NPE (w per level: 7.55/1.03/0.57)"),
    ("npe-gibbs-lofihifi", "Gibbs NPE (single w=5.92, earlier window)"),
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
        return f"w={s['w']:g}"
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


def fig_crps(rows, suffix):
    methods = order_methods(rows)
    lane = ROW_H / len(SETS)
    fig, ax = plt.subplots(figsize=(11.5, 1.25 * len(methods) + 2.0))
    _bands(ax, methods)
    yticks = []
    for r, (m, _) in enumerate(methods):
        top = r * (ROW_H + GAP)
        yticks.append(top + ROW_H / 2)
        for j, (s, _, color) in enumerate(SETS):
            v = rows.get((m, s))
            if v is None:
                continue
            y = top + (j + 0.5) * lane
            ax.barh(y, v["crps_skill"], height=lane * 0.78, color=color, alpha=0.85,
                    edgecolor="white", linewidth=1.2, zorder=3)
            x = v["crps_skill"]
            ax.text(x + (0.008 if x >= 0 else -0.008), y, f"{x:+.3f}", va="center",
                    ha="left" if x >= 0 else "right", fontsize=7.5, color=INK,
                    family="monospace", zorder=4)
            ax.text(0.655, y, f"{v['crps']:.4f}  {v['accept']:5.1%}  {v['setting']:>9}",
                    va="center", fontsize=7.5, color=color, family="monospace",
                    transform=ax.get_yaxis_transform(), zorder=4)
    _frame(ax, methods, yticks)
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlim(-0.32, 0.62)
    ax.set_xticks(np.arange(-0.3, 0.61, 0.1))
    ax.set_xlabel("CRPS skill score,  1 - CRPS / CRPS(prior)   "
                  "(0 = no better than the prior, 1 = point mass on the truth)",
                  fontsize=9, color=MUTED)
    ax.text(0.655, -GAP / 2 - 0.1, "  CRPS  accept   setting", fontsize=7.5, color=MUTED,
            va="bottom", family="monospace", transform=ax.get_yaxis_transform())
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85, label=lab) for _, lab, c in SETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=3,
              frameon=False, fontsize=9)
    ax.set_title("Posterior CRPS by model and fidelity level, pooled over all 24 parameters\n"
                 "(unit-cube coordinates; scored on the draws that survive the prior box; "
                 "rows ordered by the exaggerated level)",
                 fontsize=11, loc="left", color=INK, pad=20)
    return fig, "crps_skill_lofihifi.png"


def fig_is(rows, suffix):
    methods = order_methods(rows)
    lane = ROW_H / len(SETS)
    fig, ax = plt.subplots(figsize=(11.5, 1.25 * len(methods) + 2.0))
    _bands(ax, methods)
    yticks, xmax = [], 0.0
    for r, (m, _) in enumerate(methods):
        top = r * (ROW_H + GAP)
        yticks.append(top + ROW_H / 2)
        for j, (s, _, color) in enumerate(SETS):
            v = rows.get((m, s))
            if v is None:
                continue
            y = top + (j + 0.5) * lane
            # width then penalty: the same hue at two alphas, so the split reads inside one
            # fidelity level's colour rather than introducing a second categorical scale
            ax.barh(y, v["width"], height=lane * 0.78, color=color, alpha=0.40,
                    edgecolor="white", linewidth=1.2, zorder=3)
            ax.barh(y, v["penalty"], left=v["width"], height=lane * 0.78, color=color,
                    alpha=0.95, edgecolor="white", linewidth=1.2, zorder=3)
            xmax = max(xmax, v["is"])
            ax.text(0.70, y, f"{v['is']:6.3f}  {v['cov95']:5.1%}  {v['width']:.3f}",
                    va="center", fontsize=7.5, color=color, family="monospace",
                    transform=ax.get_yaxis_transform(), zorder=4)
    _frame(ax, methods, yticks)
    prior_is = np.nanmean([v["is_prior"] for v in rows.values()])
    ax.axvline(prior_is, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(prior_is, -GAP / 2 - 0.05, " prior", fontsize=7.5, color=INK, va="bottom")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlim(0, max(xmax, prior_is) * 1.62)
    ax.set_xlabel("95% interval score = interval width (pale) + miss penalty (solid);  "
                  "lower is better, prior reference dashed", fontsize=9, color=MUTED)
    ax.text(0.70, -GAP / 2 - 0.1, "   IS95  cov95  width", fontsize=7.5, color=MUTED,
            va="bottom", family="monospace", transform=ax.get_yaxis_transform())
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85, label=lab) for _, lab, c in SETS]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.40, label="width (sharpness)"),
                plt.Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.95, label="penalty (calibration)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=5,
              frameon=False, fontsize=8.5)
    ax.set_title("95% interval score by model and fidelity level, split into sharpness and "
                 "calibration failure\n(nominal coverage 95%; unit-cube coordinates; "
                 "rows ordered as in the CRPS figure)",
                 fontsize=11, loc="left", color=INK, pad=20)
    return fig, "interval_score_lofihifi.png"


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
    ap.add_argument("--metric", default="all", choices=["all", "crps", "is", "zscore"])
    a = ap.parse_args()
    rows = gather(a.suffix)
    if not rows:
        raise SystemExit(f"no scored runs found under {ROOT} with suffix '{a.suffix}'")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {"crps": fig_crps, "is": fig_is, "zscore": fig_zscore}
    for name in (builders if a.metric == "all" else [a.metric]):
        fig, fname = builders[name](rows, a.suffix)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=130)
        plt.close(fig)
        print("wrote", OUT_DIR / fname)
    write_table(rows, OUT_DIR / "scores_lofihifi.csv")


if __name__ == "__main__":
    main()
