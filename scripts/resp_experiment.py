"""Calibrate the respiratory amplitude against real beat-to-beat statistics.

Uses the theta that are nearest neighbours of real beats (from
scripts/wedge_matched.py output), simulates whole breaths (simulate_breath) at a
few amplitudes, and computes per theta ("patient") the same statistics as for
the real data: rms level shift and rms shape deviation of beats around the
patient's mean beat, start/end mismatch relative to peak-to-peak, and the
cross-channel correlation of level shifts.

Usage: python scripts/resp_experiment.py MATCHED_DIR OUT_DIR [--n-theta 200]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate_breath  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402
from real_vs_sim import CH, load_real  # noqa: E402
from wedge_matched import MATCH  # noqa: E402

AMPS = [1.0, 2.5, 5.0, 10.0]
RATE = 15.0


def patient_stats(W):
    """W: (beats, 4, 201). Per-channel rms level shift, rms shape deviation."""
    m = W.mean(axis=0)
    dev = W - m
    dc = dev.mean(axis=2, keepdims=True)
    out = {}
    for j, c in enumerate(CH):
        out[f"{c}_dc"] = float(np.sqrt(np.mean(dc[:, j] ** 2)))
        out[f"{c}_shape"] = float(np.sqrt(np.mean((dev - dc)[:, j] ** 2)))
        out[f"{c}_endstart"] = float(np.median(np.abs(W[:, j, -1] - W[:, j, 0]) / np.ptp(W[:, j], axis=1)))
    return out, dc[:, :, 0]


def run(args):
    i, amp = args
    th = sample_theta(np.random.default_rng(9000 + i))
    beats = simulate_breath(th, {"resp_amp": amp, "resp_rate": RATE})
    if not all(b.success for b in beats):
        return None
    W = np.array([[b.waves201[c] for c in CH] for b in beats])
    st, dc = patient_stats(W)
    return {"theta": i, "amp": amp, "n_beats": len(beats), **st}, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matched")
    ap.add_argument("out")
    ap.add_argument("--n-theta", type=int, default=200)
    ap.add_argument("--workers", type=int, default=250)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    real_df, W = load_real()
    real_df["hr"] = 60000.0 / real_df["T"]
    sims = pd.read_csv(Path(a.matched) / "sims.csv")
    mu, sd = real_df[MATCH].mean(), real_df[MATCH].std()
    S = ((sims[MATCH] - mu) / sd).to_numpy()
    R = ((real_df[MATCH] - mu) / sd).to_numpy()
    nn = np.argmin(np.sqrt(((R[:, None] - S[None]) ** 2).sum(-1)), axis=1)
    counts = pd.Series(sims.theta.to_numpy()[nn]).value_counts()
    thetas = counts.index[:a.n_theta].tolist()  # most-matched theta first
    print(f"{len(counts)} distinct matched theta; using {len(thetas)}")

    rows, dcs = [], {amp: [] for amp in AMPS}
    for pid, idx in real_df.groupby("id").groups.items():
        if len(idx) >= 4:
            st, dc = patient_stats(W[list(idx)])
            rows.append({"amp": "real", **st})
            dcs.setdefault("real", []).append(dc)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run, [(i, amp) for amp in AMPS for i in thetas], chunksize=1):
            if r is not None:
                rows.append(r[0])
                dcs[r[0]["amp"]].append(r[1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "resp_stats.csv", index=False)

    cols = [f"{c}_{m}" for m in ("dc", "shape", "endstart") for c in CH]
    tab = df.groupby(df.amp.astype(str))[cols].median().T
    tab = tab[["real"] + [str(x) for x in AMPS]]
    print("\nMedian over patients / theta (mmHg; endstart = |P(T) - P(0)| / ptp):")
    print(tab.round(3).to_string())
    print("\nbeats per breath (sim):", df[df.amp != "real"].n_beats.describe()[["min", "50%", "max"]].to_dict())
    print("\nCorrelation of beat level shifts across channels (Pra-Pvp, Prv-Pap, Pra-Prv):")
    for k, v in dcs.items():
        D = np.concatenate(v)
        C = np.corrcoef(D.T)
        print(f"  {str(k):5s} Pra-Pvp {C[1, 3]:+.2f}  Prv-Pap {C[0, 2]:+.2f}  Pra-Prv {C[1, 0]:+.2f}")


if __name__ == "__main__":
    main()
