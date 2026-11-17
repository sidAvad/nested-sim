"""How the two tempering knobs behave: Gibbs w and Route A beta, swept on the hifi sets.

Both knobs are meant to do the same job -- widen the posterior to account for a likelihood
that is wrong -- so they are drawn on one figure with a shared y-axis, each against its own
knob. The calibrated setting (w from the truth-anchored residual, beta = w / w_lofi) is
marked, which is the point of the figure: whether the value the calibration picks is also
the value the scores pick.

The panels are interval score, 95% interval width and 95% coverage rather than CRPS,
because CRPS turns out to be nearly flat in w (tempering trades width against accuracy at
almost exactly the CRPS-neutral rate) while the interval score is not -- so CRPS alone
cannot tell you whether a tolerance is set well.

Usage: python scripts/plot_w_beta_sweep.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path.home() / "results/nested-sim/pretrained_eval_lofihifi"
OUT_DIR = ROOT / "scores_lofihifi"
SETS = [("hifi_v2_test", "high-fi realistic", "#eb6834"),
        ("hifi_v2strong_test", "high-fi exaggerated", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
CALIB_W = {"hifi_v2_test": 1.033, "hifi_v2strong_test": 0.565}
W_LOFI = 7.5539


def collect(method, prefix, set_name):
    """(knob value, metrics) for every tagged sweep run of one method, sorted by knob."""
    rows = []
    for d in ROOT.glob(f"{method}__{prefix}*"):
        f = d / set_name / "zscore_shrinkage_data.npz"
        if not f.exists():
            continue
        m = re.fullmatch(rf"{re.escape(method)}__{prefix}([0-9.]+)", d.name)
        if not m:
            continue  # e.g. the "__beta1" and "__scored" runs, which are not sweep points
        z = np.load(f, allow_pickle=True)
        keys = z["param_keys"]
        pool = lambda fld: np.concatenate([z[f"{k}_{fld}"] for k in keys])  # noqa: E731
        rows.append((float(m.group(1)), {
            "is": np.nanmean(pool("is_box")),
            "is_skill": 1 - np.nanmean(pool("is_box")) / np.nanmean(pool("is_prior")),
            "width": np.nanmean(pool("width_box")),
            "cov": np.nanmean(pool("cov_box")),
            "crps_skill": 1 - np.nanmean(pool("crps_box")) / np.nanmean(pool("crps_prior")),
        }))
    return sorted(rows)


def panel(ax, series, field, ylabel, calib, logx=True):
    for label, color, rows, cal in series:
        if not rows:
            continue
        x = [r[0] for r in rows]
        y = [r[1][field] for r in rows]
        ax.plot(x, y, "-o", color=color, lw=2.0, ms=6, label=label,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        if cal is not None:
            yi = np.interp(np.log(cal) if logx else cal,
                           np.log(x) if logx else x, y)
            ax.plot([cal], [yi], marker="*", ms=16, color=color, markeredgecolor=INK,
                    markeredgewidth=0.8, zorder=4, linestyle="none")
    if logx:
        ax.set_xscale("log")
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    w_series, b_series = [], []
    for s, label, color in SETS:
        w_series.append((label, color, collect("npe-gibbs-w005to50-lofihifi", "w", s),
                         CALIB_W[s]))
    b_rows = collect("sun-routeA-lofihifi", "b", "hifi_v2strong_test")
    b_series.append(("high-fi exaggerated", "#1baf7a", b_rows,
                     max(CALIB_W["hifi_v2strong_test"] / W_LOFI, 0.1)))

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.5), sharex="col")
    for col, (series, knob) in enumerate([(w_series, "Gibbs tolerance  w"),
                                          (b_series, "Route A tempering  beta")]):
        for row, (field, ylab) in enumerate([("is", "95% interval score"),
                                             ("width", "95% interval width"),
                                             ("cov", "95% coverage")]):
            ax = axes[row][col]
            panel(ax, series, field, ylab, None)
            if field == "cov":
                ax.axhline(0.95, color=INK, lw=1.0, ls="--", zorder=2)
                ax.text(ax.get_xlim()[0], 0.952, " nominal 95%", fontsize=7.5, color=INK,
                        va="bottom")
                ax.set_xlabel(knob, fontsize=9, color=MUTED)
        axes[0][col].legend(frameon=False, fontsize=8.5, loc="best")
    # The beta panels are drawn on the w panels' y-range on purpose. Left to autoscale they
    # span ~0.001 and a flat line reads as a steep one; on a shared range the point of the
    # figure -- that beta traverses nothing w traverses -- is literal rather than numeric.
    for row in range(3):
        axes[row][1].set_ylim(axes[row][0].get_ylim())
    axes[0][0].set_title("Gibbs NPE (w005to50): w is a working knob\n"
                         "star = calibrated w for that level", fontsize=10, loc="left",
                         color=INK)
    axes[0][1].set_title("Sun et al. Route A: beta does nothing\n"
                         "star = calibrated beta (grid edge)", fontsize=10, loc="left",
                         color=INK)
    fig.suptitle("Tempering swept within one trained model, high-fidelity test sets "
                 "(n=500, pooled over 24 parameters)",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tempering_sweep_lofihifi.png"
    fig.savefig(out, dpi=130)

    print("knob,method,set,value,is95,is_skill,width95,cov95,crps_skill")
    for tag, series in (("w", w_series), ("beta", b_series)):
        for label, _, rows, _ in series:
            for v, m in rows:
                print(f"{tag},{'gibbs-w005to50' if tag=='w' else 'sun-routeA'},{label},{v:g},"
                      f"{m['is']:.4f},{m['is_skill']:.4f},{m['width']:.4f},{m['cov']:.4f},"
                      f"{m['crps_skill']:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
