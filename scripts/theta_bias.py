"""Theta-bias test: how far do low-fi best-fit parameters move from the truth
when the data come from high-fi?

For each target theta* (prior draw, low-fi converges) with one structural phi
(sample_phi, alpha = 1, respiration included, measurement off):
  A  low-fi fitted to the high-fi beat at (theta*, phi)   (misspecified)
  B  low-fi fitted to the low-fi beat at theta*           (control floor)
Both fits are CMA-ES over theta in the same bounded / log-scaled unit cube as
fit_real_beats.py, HR fixed to the truth, started from the same point: theta*
plus a small random offset (START_JITTER of each prior range, seeded per
target). Starting exactly at theta* would make B degenerate (loss 0 there), so
the offset lets B measure the real recovery floor (optimizer + non-identifiable
directions); theta* itself is never offered as a candidate.

Observation = the encoder's cath lab input: Prv, Pra, Pvp, Pap (201 pts) and
MAP, SBP, DBP, SV. Normalisation follows the encoder's default ("legacy"):
each waveform channel divided by its population std (configs/encoder_norm_stats.json),
MAP/SBP/DBP by the Pas waveform std, SV by the Vlv waveform std.

Loss (--loss):
  block   mean of 8 blocks: 4 waveform blocks (mean squared normalised error
          over 201 pts) + 4 scalar blocks (squared normalised error)
  wave    mean of the 4 waveform blocks
  scalar  mean of the 4 scalar blocks

Each (target, fit) writes its own JSON (rerun skips finished ones); `summary`
collects them into a table and figure.

Usage:
  python scripts/theta_bias.py run OUT_DIR [--loss block] [--n 200] [--workers 240]
  python scripts/theta_bias.py summary OUT_DIR FIG.png
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_real_beats import THETA_FIT, theta_from_u, u_from_theta  # noqa: E402

from nested_sim import simulate  # noqa: E402
from nested_sim.params import full_theta, sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, load_priors, sample_phi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WAVES = ["Prv", "Pra", "Pvp", "Pap"]
SCALARS = ["map", "sbp", "dbp", "sv"]
FAIL_LOSS = 1e3
PRIORS_PATH = None  # set by run(); None = configs/phi_priors.toml
SIGMA0, POPSIZE, MAXITER = 0.05, 24, 30
START_JITTER = 0.03

_stats = json.load(open(ROOT / "configs" / "encoder_norm_stats.json"))["waves"]
SCALE = {**{c: _stats[c]["std"] for c in WAVES},
         "map": _stats["Pas"]["std"], "sbp": _stats["Pas"]["std"], "dbp": _stats["Pas"]["std"],
         "sv": _stats["Vlv"]["std"]}


def observe(r):
    return {**{c: np.asarray(r.waves201[c]) for c in WAVES}, **{k: r.summaries[k] for k in SCALARS}}


def loss_fn(obs, target, mode):
    blocks = []
    if mode in ("block", "wave"):
        blocks += [float(np.mean(((obs[c] - target[c]) / SCALE[c]) ** 2)) for c in WAVES]
    if mode in ("block", "scalar"):
        blocks += [((obs[k] - target[k]) / SCALE[k]) ** 2 for k in SCALARS]
    return float(np.mean(blocks))


def sim_loss(theta, target, mode):
    try:
        r = simulate(theta)
    except Exception:  # noqa: BLE001 - any blow-up counts as a failed sim
        return FAIL_LOSS
    return loss_fn(observe(r), target, mode) if r.success else FAIL_LOSS


def make_target(i):
    """theta*, phi and both target observations for target index i (None if unusable)."""
    rng = np.random.default_rng(20_000 + i)
    th = full_theta(sample_theta(rng))
    phi = {k: v for k, v in sample_phi(rng, 1.0, th, load_priors(PRIORS_PATH)).items() if k in PHI_STRUCT_OFF}
    lo, hi = simulate(th), simulate(th, phi)
    if not (lo.success and hi.success):
        return None
    return th, phi, observe(lo), observe(hi)


def fit_one(args):
    global PRIORS_PATH
    i, which, mode, out, PRIORS_PATH = args
    path = Path(out) / f"t{i:03d}_{which}.json"
    if path.exists():
        return json.load(open(path))
    t = make_target(i)
    if t is None:
        res = {"i": i, "which": which, "usable": False}
        json.dump(res, open(path, "w"))
        return res
    th, phi, obs_lo, obs_hi = t
    target = obs_hi if which == "A" else obs_lo
    hr = th["HR"]
    u_true = u_from_theta(th)
    u0 = np.clip(u_true + START_JITTER * np.random.default_rng(30_000 + i).standard_normal(u_true.size), 0, 1)
    truth_loss = sim_loss(th, target, mode)
    start_loss = sim_loss(theta_from_u(u0, hr), target, mode)
    best = (start_loss, u0)
    es = cma.CMAEvolutionStrategy(u0, SIGMA0, {"bounds": [0, 1], "popsize": POPSIZE, "maxiter": MAXITER,
                                               "seed": 1 + i, "verbose": -9})
    while not es.stop():
        X = es.ask()
        L = [sim_loss(theta_from_u(x, hr), target, mode) for x in X]
        es.tell(X, L)
        j = int(np.argmin(L))
        if L[j] < best[0]:
            best = (L[j], np.asarray(X[j]))
    res = {"i": i, "which": which, "usable": True, "mode": mode, "theta_true": th, "phi": phi,
           "truth_loss": truth_loss, "start_loss": start_loss, "final_loss": best[0],
           "u_true": list(u_true), "u_start": list(u0), "u_fit": list(best[1]),
           "theta_fit": theta_from_u(best[1], hr)}
    json.dump(res, open(path, "w"), default=float)
    return res


def run(out, mode, n, workers, priors):
    out.mkdir(parents=True, exist_ok=True)
    src = Path(priors) if priors else ROOT / "configs" / "phi_priors.toml"
    (out / "phi_priors_used.toml").write_text(src.read_text())
    tasks = [(i, w, mode, str(out), str(src)) for i in range(n) for w in ("A", "B")]
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(fit_one, tasks, chunksize=1)):
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(tasks)} fits done", flush=True)
    print("all fits done", flush=True)


def summary(out, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.load(open(p)) for p in sorted(out.glob("t*_*.json"))]
    rows = [r for r in rows if r.get("usable")]
    shift = {w: {} for w in ("A", "B")}
    loss = {w: {} for w in ("A", "B")}
    for r in rows:
        shift[r["which"]][r["i"]] = np.asarray(r["u_fit"]) - np.asarray(r["u_true"])
        loss[r["which"]][r["i"]] = (r["truth_loss"], r["start_loss"], r["final_loss"])
    ids = sorted(set(shift["A"]) & set(shift["B"]))
    SA = np.array([shift["A"][i] for i in ids])
    SB = np.array([shift["B"][i] for i in ids])
    tab = pd.DataFrame({
        "param": THETA_FIT,
        "A_signed_median": np.median(SA, 0), "A_abs_median": np.median(np.abs(SA), 0),
        "B_signed_median": np.median(SB, 0), "B_abs_median": np.median(np.abs(SB), 0),
        "excess_abs_median": np.median(np.abs(SA) - np.abs(SB), 0),
        "A_signed_q25": np.quantile(SA, .25, 0), "A_signed_q75": np.quantile(SA, .75, 0),
    }).sort_values("excess_abs_median", ascending=False)
    tab.to_csv(out / "summary_params.csv", index=False)
    LA = np.array([loss["A"][i] for i in ids])
    LB = np.array([loss["B"][i] for i in ids])
    print(f"{len(ids)} paired targets")
    print(f"median loss  at theta*: A {np.median(LA[:, 0]):.4f}   at start: A {np.median(LA[:, 1]):.4f}, "
          f"B {np.median(LB[:, 1]):.4f}   final: A {np.median(LA[:, 2]):.4f}, B {np.median(LB[:, 2]):.5f}")
    print("shift = theta_fit - theta*, in prior-range units (log-scaled where the prior is)")
    with pd.option_context("display.width", 200):
        print(tab.round(3).to_string(index=False))

    order = list(tab.param)
    idx = [THETA_FIT.index(p) for p in order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * len(order) + 1.5), sharey=True)
    y = np.arange(len(order))
    for ax, data, title in ((axes[0], (SA, SB), "signed shift (median, IQR)"),
                            (axes[1], (np.abs(SA), np.abs(SB)), "|shift| (median, IQR)")):
        for k, (D, color, lab, dy) in enumerate(((data[0], "#eb6834", "A: low-fi fit to high-fi data", -0.15),
                                                 (data[1], "#2a78d6", "B: low-fi fit to low-fi data", 0.15))):
            med = np.median(D[:, idx], 0)
            lo, hi = np.quantile(D[:, idx], .25, 0), np.quantile(D[:, idx], .75, 0)
            ax.hlines(y + dy, lo, hi, color=color, lw=2, alpha=0.6)
            ax.plot(med, y + dy, "o", color=color, ms=6, label=lab)
        ax.axvline(0, color="#6b6a63", lw=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", color="#e6e5df")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("fraction of prior range", fontsize=9)
    axes[0].set_yticks(y, order, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Theta bias of low-fi best fits ({len(ids)} targets, loss={rows[0]['mode']}); "
                 "sorted by excess |shift|", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    print("wrote", fig_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "summary"])
    ap.add_argument("out")
    ap.add_argument("fig", nargs="?")
    ap.add_argument("--loss", default="block", choices=["block", "wave", "scalar"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=240)
    ap.add_argument("--priors", default=None, help="phi prior TOML (default configs/phi_priors.toml)")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    if a.stage == "run":
        run(out, a.loss, a.n, a.workers, a.priors)
    else:
        summary(out, a.fig)


if __name__ == "__main__":
    main()
