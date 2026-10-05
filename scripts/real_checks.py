"""Data checks that inform high-fi additions.

1. Wedge: harmonic content and amplitude of real Pvp vs simulated Pvp and Pla
   (low-fi, prior theta), incl. Pla through a 1st-order lag (wedge transmission).
2. Within-patient beat-to-beat level shifts: correlation across channels
   (simultaneous recording -> correlated via respiration; sequential -> not).
3. Non-periodicity: (P(T) - P(0)) / peak-to-peak, real vs sim.

Usage: python scripts/real_checks.py [--n 300] [--workers 200]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402
from real_vs_sim import CH, load_real  # noqa: E402


def harmonics(x):
    x = np.asarray(x)[:-1]
    s = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return s[1:6] / s[1:].sum()


def lag1(x, dt, tau):
    """Periodic steady-state 1st-order low-pass with time constant tau (ms)."""
    if tau <= 0:
        return np.array(x)
    p = np.asarray(x[:-1], float)
    a = np.exp(-dt / tau)
    y = p[0]
    for _ in range(int(np.ceil(10 * tau / (dt * len(p)))) + 2):
        out = np.empty_like(p)
        for i, v in enumerate(p):
            y = a * y + (1 - a) * v
            out[i] = y
    return np.append(out, out[0])


def sim_one(i):
    th = sample_theta(np.random.default_rng(5000 + i))
    r = simulate(th)
    if not r.success:
        return None
    dt = r.T / 200
    out = {"Pvp": r.waves201["Pvp"], "Pla": r.waves201["Pla"]}
    for tau in (20.0, 50.0):
        out[f"Pla_lag{tau:.0f}"] = lag1(r.waves201["Pla"], dt, tau)
    out["prv"] = r.waves201["Prv"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=200)
    a = ap.parse_args()
    df, W = load_real()

    with ProcessPoolExecutor(a.workers) as ex:
        sims = [s for s in ex.map(sim_one, range(a.n), chunksize=1) if s is not None]

    print("1. Wedge: harmonic power fractions h1..h5 (median), share of beats with h2 > h1, median peak-to-peak")
    rows = {"real Pvp": [w[3] for w in W]}
    for k in ("Pvp", "Pla", "Pla_lag20", "Pla_lag50"):
        rows[f"sim {k}"] = [s[k] for s in sims]
    for name, xs in rows.items():
        H = np.array([harmonics(x) for x in xs])
        ptp = np.median([np.ptp(x) for x in xs])
        print(f"  {name:14s} h1..h5 {np.round(np.median(H, axis=0), 3)}  h2>h1 {np.mean(H[:, 1] > H[:, 0]):.2f}  ptp {ptp:.1f} mmHg")

    print("\n2. Within-patient beat-to-beat level shifts: correlation across channels")
    devs = []
    for _, idx in df.groupby("id").groups.items():
        if len(idx) < 4:
            continue
        m = W[list(idx)].mean(axis=2)  # (beats, 4) mean level per beat
        devs.append(m - m.mean(axis=0))
    D = np.concatenate(devs)
    C = np.corrcoef(D.T)
    print(pd.DataFrame(C, index=CH, columns=CH).round(2).to_string())
    print(f"  ({len(D)} beats from {len(devs)} patients)")

    print("\n3. Non-periodicity |P(T) - P(0)| / peak-to-peak, median [p90]")
    for j, c in enumerate(CH):
        r = np.abs(W[:, j, -1] - W[:, j, 0]) / np.ptp(W[:, j], axis=1)
        print(f"  real {c}: {np.median(r):.3f} [{np.quantile(r, .9):.3f}]")
    r = np.array([abs(s["prv"][-1] - s["prv"][0]) / np.ptp(s["prv"]) for s in sims])
    print(f"  sim  Prv: {np.median(r):.3f} [{np.quantile(r, .9):.3f}]  (steady state; window edge only)")


if __name__ == "__main__":
    main()
