"""T7 visual check: low-fi vs high-fi arterial waveforms for a few theta.

Per row (one theta, one phi draw at the given alpha): systemic pressures,
pulmonary pressures, aortic and pulmonic valve flow. Ventricular pressure is
drawn thin for context. The hangout interval (pulmonic flow end minus Prv/Pap
crossover) is reported in the title.

Usage: python scripts/plot_hifi_vs_lowfi.py OUT.png [--alpha 1.0] [--n 3]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import SolverSettings, simulate  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402

LOW, HIGH = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def hangout(r):
    """ms from Prv falling below Pap to the end of pulmonic forward flow (first ejection)."""
    t, prv, pap, q = r.t_dense, r.dense["Prv"], r.dense["Pap"], r.dense["Qap"]
    i_peak = int(np.argmax(q))
    after = np.where(q[i_peak:] <= 1e-6)[0]
    if after.size == 0:
        return np.nan
    i_end = i_peak + after[0]
    cross = np.where((prv[i_peak:i_end] >= pap[i_peak:i_end]) & (prv[i_peak + 1:i_end + 1] < pap[i_peak + 1:i_end + 1]))[0]
    return t[i_end] - t[i_peak + cross[0] + 1] if cross.size else 0.0


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    s = SolverSettings(method="LSODA")
    rng = np.random.default_rng(a.seed)
    thetas = [{}] + sample_theta(rng, a.n - 1)
    fig, axes = plt.subplots(a.n, 4, figsize=(16, 3.2 * a.n), squeeze=False)
    for i, th in enumerate(thetas):
        phi = {k: v for k, v in sample_phi(rng, a.alpha, th).items() if k in PHI_STRUCT_OFF}
        lo, hi = simulate(th, None, s), simulate(th, phi, s)
        if not (lo.success and hi.success):
            axes[i, 0].set_title(f"theta {i}: failed ({lo.message} / {hi.message})", fontsize=9)
            continue
        panels = [("Pas", "Plv", "Systemic: Pas (prox)", "mmHg"),
                  ("Pap", "Prv", "Pulmonary: Pap (prox)", "mmHg"),
                  ("Qas", None, "Aortic valve flow", "mL/ms"),
                  ("Qap", None, "Pulmonic valve flow", "mL/ms")]
        for j, (ch, ctx, title, unit) in enumerate(panels):
            ax = axes[i, j]
            style(ax)
            if ctx:
                ax.plot(lo.t_dense, lo.dense[ctx], color=LOW, lw=0.8, alpha=0.35)
                ax.plot(hi.t_dense, hi.dense[ctx], color=HIGH, lw=0.8, alpha=0.35)
            ax.plot(lo.t_dense, lo.dense[ch], color=LOW, lw=2, label="low-fi")
            ax.plot(hi.t_dense, hi.dense[ch], color=HIGH, lw=2, label="high-fi")
            if ch in ("Pas", "Pap"):
                lim = [v for r in (lo, hi) for v in (r.dense[ch].min(), r.dense[ch].max())]
                pad = 0.25 * (max(lim) - min(lim))
                ax.set_ylim(min(lim) - pad, max(lim) + pad)
            t = title
            if ch == "Qap":
                t += f"\nhangout: low {hangout(lo):.0f} ms, high {hangout(hi):.0f} ms"
            if j == 0:
                t = f"theta {i} (HR {60000 / lo.T:.0f})\n" + t
            ax.set_title(t, fontsize=9, color=INK, loc="left")
            ax.set_ylabel(unit, fontsize=8, color=MUTED)
            if i == a.n - 1:
                ax.set_xlabel("ms from PV_OPEN", fontsize=8, color=MUTED)
        if i == 0:
            axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
        print(f"theta {i}: phi=" + ", ".join(f"{k}={v:.3g}" for k, v in phi.items()))
    fig.suptitle(f"Low-fi vs high-fi (alpha={a.alpha}); thin lines: ventricular pressure",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
