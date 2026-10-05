"""Low-fi vs high-fi cath lab waveforms: one row per theta, several phi per theta.

Each panel shows the low-fi beat (thick) and n_phi high-fi beats for the same
theta (thin), each with structural phi drawn from the prior at the given alpha
(measurement model off). The spread of the thin lines is what phi alone does to
one "patient".

Usage: python scripts/plot_phi_spread.py OUT.png [--n-theta 4] [--n-phi 8] [--alpha 1.0]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import full_theta, sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402

CH = [("Prv", "RV pressure"), ("Pra", "RA pressure"), ("Pap", "PA pressure"), ("Pvp", "Pulm. venous pressure")]
LOW, HIGH = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def run(args):
    theta_seed, phi_seed, alpha = args
    th = sample_theta(np.random.default_rng(theta_seed))
    if phi_seed is None:
        r = simulate(th)
    else:
        phi = sample_phi(np.random.default_rng(phi_seed), alpha, th)
        r = simulate(th, {k: v for k, v in phi.items() if k in PHI_STRUCT_OFF})
    if not r.success:
        return None
    return {"t": r.t201, **{c: r.waves201[c] for c, _ in CH}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-theta", type=int, default=4)
    ap.add_argument("--n-phi", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--theta-seeds", default="0,2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    seeds = [int(s) for s in a.theta_seeds.split(",")][: a.n_theta]

    jobs = [(s, None, a.alpha) for s in seeds] + [(s, 1000 * s + k, a.alpha) for s in seeds for k in range(a.n_phi)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(zip([(s, p) for s, p, _ in jobs], ex.map(run, jobs)))

    fig, axes = plt.subplots(len(seeds), 4, figsize=(16, 3.1 * len(seeds)), squeeze=False)
    for i, s in enumerate(seeds):
        th = full_theta(sample_theta(np.random.default_rng(s)))
        lo = res[(s, None)]
        his = [res[(s, 1000 * s + k)] for k in range(a.n_phi)]
        n_ok = sum(h is not None for h in his)
        for j, (c, label) in enumerate(CH):
            ax = axes[i, j]
            for k, h in enumerate(his):
                if h is not None:
                    ax.plot(h["t"], h[c], color=HIGH, lw=1.0, alpha=0.55,
                            label=f"high-fi ({n_ok} phi draws)" if k == 0 else None)
            if lo is not None:
                ax.plot(lo["t"], lo[c], color=LOW, lw=2.4, label="low-fi")
            ax.grid(True, color=GRID, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            title = label
            if j == 0:
                title = (f"theta {i}: HR {th['HR']:.0f}, Rap {th['Rap']:.0f}, Ras {th['Ras']:.0f}, "
                         f"Emax_RV {th['Emax_RV']:.2f}\n" + label)
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("mmHg", fontsize=8, color=MUTED)
            if i == len(seeds) - 1:
                ax.set_xlabel("ms from PV_OPEN", fontsize=8, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"Cath lab waveforms: low-fi vs high-fi with {a.n_phi} structural phi draws per theta "
                 f"(alpha={a.alpha}, measurement off)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
