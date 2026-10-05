"""Compare real cath lab beats with simulated beats to inform the measurement model.

Datasets (201-pt beats of Prv, Pra, Pap, Pvp):
  real      ~/data/real_data/onebeat_300patients
  lofi      low-fi on prior theta
  hifi      high-fi, structural phi at alpha = 1 (measurement off)

Metrics per beat and channel:
  rough   rms residual after Savitzky-Golay smoothing (11 pts, cubic), mmHg
  rough_rel   rough / peak-to-peak
  hf8, hf15   fraction of AC power in beat harmonics >= 8 / >= 15
  upstroke_lag  time of max dP/dt of Pap minus that of Prv, ms
Real only: within-patient beat-to-beat variability; integer fraction of map/sbp/dbp.

Usage: python scripts/real_vs_sim.py OUT_DIR [--n 400] [--workers 200]
"""

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402

REAL_DIR = Path.home() / "data/real_data/onebeat_300patients"
CH = ["Prv", "Pra", "Pap", "Pvp"]
COLORS = {"real": "#2a78d6", "lofi": "#eb6834", "hifi": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def load_real():
    man = json.load(open(REAL_DIR / "manifest.json"))
    rows, waves = [], []
    for e in man["index"]:
        with h5py.File(REAL_DIR / e["file"]) as f:
            g = f[e["group"]]
            waves.append(np.stack([g["waves"][c][()] for c in CH]))
            r = {"id": e["id"], "cohort": e["covariates"]["cohort"], "T": g["waves"]["t"][()][-1]}
            r.update({k: g["summaries"][k][()] for k in g["summaries"]})
            rows.append(r)
    return pd.DataFrame(rows), np.array(waves)


def sim_one(i):
    rng = np.random.default_rng(1000 + i)
    th = sample_theta(rng)
    phi = {k: v for k, v in sample_phi(rng, 1.0, th).items() if k in PHI_STRUCT_OFF}
    lo, hi = simulate(th), simulate(th, phi)
    if not (lo.success and hi.success):
        return None
    return {"T": lo.T, "lofi": np.stack([lo.waves201[c] for c in CH]),
            "hifi": np.stack([hi.waves201[c] for c in CH])}


def beat_metrics(w, T):
    """w: (4, 201) one beat. Returns dict of metrics per channel."""
    out = {}
    dt = T / 200.0
    x = w[:, :-1]  # one period, drop duplicated endpoint
    for j, c in enumerate(CH):
        s = savgol_filter(w[j], 11, 3, mode="interp")
        rough = float(np.sqrt(np.mean((w[j] - s) ** 2)))
        ptp = float(np.ptp(w[j]))
        spec = np.abs(np.fft.rfft(x[j] - x[j].mean())) ** 2
        tot = spec[1:].sum()
        out[f"{c}_rough"] = rough
        out[f"{c}_rough_rel"] = rough / max(ptp, 1e-9)
        out[f"{c}_hf8"] = float(spec[8:].sum() / tot) if tot > 0 else np.nan
        out[f"{c}_hf15"] = float(spec[15:].sum() / tot) if tot > 0 else np.nan
    dprv = np.gradient(w[0], dt)
    dpap = np.gradient(w[2], dt)
    out["upstroke_lag"] = float((np.argmax(dpap) - np.argmax(dprv)) * dt)
    return out


def spectrum(w):
    x = w[:, :-1]
    s = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
    return s[:, 1:41] / s[:, 1:].sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    real_df, real_w = load_real()
    cache = out / "sims.npz"
    if cache.exists():
        z = np.load(cache)
        sims = {k: z[k] for k in z.files}
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            res = [r for r in ex.map(sim_one, range(a.n), chunksize=1) if r is not None]
        sims = {k: np.array([r[k] for r in res]) for k in res[0]}
        np.savez(cache, **sims)
    print(f"real beats {len(real_w)} ({real_df.id.nunique()} patients); sims {len(sims['T'])}")

    sets = {"real": (real_w, real_df["T"].to_numpy())}
    for k in ("lofi", "hifi"):
        sets[k] = (sims[k], sims["T"])
    metrics = {k: pd.DataFrame([beat_metrics(w, T) for w, T in zip(W, Ts)]) for k, (W, Ts) in sets.items()}

    print("\nMedian [p90] per dataset")
    cols = [f"{c}_{m}" for m in ("rough", "rough_rel", "hf8", "hf15") for c in CH] + ["upstroke_lag"]
    tab = pd.DataFrame({k: [f"{df[c].median():.3g} [{df[c].quantile(.9):.3g}]" for c in cols]
                        for k, df in metrics.items()}, index=cols)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(tab)
    tab.to_csv(out / "metrics_summary.csv")

    # Within-patient beat-to-beat variability (real): rms deviation from the
    # patient's mean beat, split into mean shift (DC) and shape.
    print("\nReal within-patient beat-to-beat variability (patients with >= 4 beats), mmHg")
    rows = []
    for pid, idx in real_df.groupby("id").groups.items():
        if len(idx) < 4:
            continue
        W = real_w[list(idx)]
        m = W.mean(axis=0)
        dev = W - m
        dc = dev.mean(axis=2, keepdims=True)
        rows.append({**{f"{c}_dc": float(np.sqrt(np.mean(dc[:, j] ** 2))) for j, c in enumerate(CH)},
                     **{f"{c}_shape": float(np.sqrt(np.mean((dev - dc)[:, j] ** 2))) for j, c in enumerate(CH)},
                     **{f"{c}_ptp": float(np.ptp(m[j])) for j, c in enumerate(CH)},
                     "T_cv": float(real_df.loc[list(idx), "T"].std() / real_df.loc[list(idx), "T"].mean())})
    bb = pd.DataFrame(rows)
    print(bb.describe(percentiles=[.5, .9]).loc[["count", "50%", "90%"]].round(2).to_string())

    print("\nmap/sbp/dbp integer fraction:",
          {k: float(np.mean(np.isclose(real_df[k], np.round(real_df[k])))) for k in ("map", "sbp", "dbp", "sv")})

    # Figure: example beats + median spectra.
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    pids = [p for p, g in real_df.groupby("id") if len(g) >= 6][:3]
    for j, c in enumerate(CH):
        ax = axes[0, j]
        for k, pid in enumerate(pids):
            for i in real_df.index[real_df.id == pid]:
                t = np.linspace(0, real_df.at[i, "T"], 201)
                ax.plot(t, real_w[i, j], color=COLORS["real"], lw=1, alpha=0.4 + 0.2 * k)
        ax.set_title(f"Real {c}: {len(pids)} patients, all beats", fontsize=9, loc="left", color=INK)
        ax = axes[1, j]
        for name in ("lofi", "hifi"):
            W, Ts = sets[name]
            for i in range(3):
                ax.plot(np.linspace(0, Ts[i], 201), W[i, j], color=COLORS[name], lw=1.2,
                        label=name if i == 0 else None)
        ax.set_title(f"Sim {c}: 3 theta", fontsize=9, loc="left", color=INK)
        ax = axes[2, j]
        for name, (W, _) in sets.items():
            med = np.median(np.stack([spectrum(w[j:j + 1])[0] for w in W]), axis=0)
            ax.semilogy(np.arange(1, 41), med, color=COLORS[name],
                        lw=2 if name == "real" else 1.4, label=name)
        ax.set_title(f"{c}: median power fraction per beat harmonic", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("harmonic of heart rate", fontsize=8, color=MUTED)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[2, 0].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[:2].flat:
        ax.set_xlabel("ms", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "real_vs_sim.png", dpi=120)
    print("wrote", out / "real_vs_sim.png")


if __name__ == "__main__":
    main()
