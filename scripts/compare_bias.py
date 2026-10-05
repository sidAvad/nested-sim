"""Compare theta-bias runs (theta_bias.py summary outputs) across prior versions.

Prints per-parameter signed median shift of fit A, median |shift| of A and of the
control B, and their difference, for each run; writes a figure of signed shift
(median, IQR) per parameter and run.

Usage: python scripts/compare_bias.py FIG.png RUN_DIR[:label] [RUN_DIR[:label] ...]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_real_beats import THETA_FIT  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load(run):
    rows = [json.load(open(p)) for p in sorted(Path(run).glob("t*_*.json"))]
    rows = [r for r in rows if r.get("usable")]
    s = {w: {r["i"]: np.asarray(r["u_fit"]) - np.asarray(r["u_true"]) for r in rows if r["which"] == w}
         for w in ("A", "B")}
    loss = {r["i"]: (r["truth_loss"], r["final_loss"]) for r in rows if r["which"] == "A"}
    ids = sorted(set(s["A"]) & set(s["B"]))
    return np.array([s["A"][i] for i in ids]), np.array([s["B"][i] for i in ids]), \
        np.array([loss[i] for i in ids])


def main():
    fig_path = sys.argv[1]
    runs = []
    for arg in sys.argv[2:]:
        path, _, label = arg.partition(":")
        runs.append((label or Path(path).name, *load(path)))

    tab = pd.DataFrame({"param": THETA_FIT})
    for label, SA, SB, L in runs:
        tab[f"{label}_A_signed"] = np.median(SA, 0)
        tab[f"{label}_A_abs"] = np.median(np.abs(SA), 0)
        tab[f"{label}_B_abs"] = np.median(np.abs(SB), 0)
        tab[f"{label}_excess"] = np.median(np.abs(SA) - np.abs(SB), 0)
    last = runs[-1][0]
    tab = tab.sort_values(f"{last}_excess", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.round(3).to_string(index=False))
    for label, SA, SB, L in runs:
        print(f"{label}: n={len(SA)}  loss at theta* {np.median(L[:, 0]):.4f}  after low-fi refit "
              f"{np.median(L[:, 1]):.4f}  | parameters with excess |shift| > 0.02: "
              f"{int((np.median(np.abs(SA) - np.abs(SB), 0) > 0.02).sum())}, > 0.05: "
              f"{int((np.median(np.abs(SA) - np.abs(SB), 0) > 0.05).sum())}")
    tab.to_csv(Path(fig_path).with_suffix(".csv"), index=False)

    order = list(tab.param)
    idx = [THETA_FIT.index(p) for p in order]
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(order) + 1.6))
    y = np.arange(len(order))
    n = len(runs)
    for k, (label, SA, SB, L) in enumerate(runs):
        dy = (k - (n - 1) / 2) * 0.22
        D = SA[:, idx]
        ax.hlines(y + dy, np.quantile(D, .25, 0), np.quantile(D, .75, 0), color=COLORS[k], lw=2, alpha=0.55)
        ax.plot(np.median(D, 0), y + dy, "o", color=COLORS[k], ms=5, label=label)
    floor = np.median(np.abs(runs[0][2][:, idx]), 0)
    ax.barh(y, 2 * floor, left=-floor, height=0.8, color="#e6e5df", zorder=0,
            label="control floor (median |shift| of B)")
    ax.axvline(0, color="#6b6a63", lw=1)
    ax.set_yticks(y, order, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color="#e6e5df")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("signed shift of low-fi best fit to high-fi data (fraction of prior range), median and IQR",
                  fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=2)
    ax.set_title("Theta bias by prior version (sorted by excess |shift| in the last version)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    print("wrote", fig_path)


if __name__ == "__main__":
    main()
