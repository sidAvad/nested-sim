"""Population-matched version of the wedge experiment.

Simulates many prior theta at a few (k_av, tau_wedge) settings, matches each real
beat to its nearest simulated theta on standardised low-fi summaries
(cvp, pcw, mpap, spa, dpa, HR), and compares Pra / wedge shape features of the
matched sims with the real beats.

Usage: python scripts/wedge_matched.py OUT_DIR [--n 1500] [--workers 250]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.hifi import apply_wedge  # noqa: E402
from nested_sim.params import sample_theta  # noqa: E402
from nested_sim.phi import full_phi  # noqa: E402
from real_checks import harmonics  # noqa: E402
from real_vs_sim import load_real  # noqa: E402

MATCH = ["cvp", "pcw", "mpap", "spa", "dpa", "hr"]
CONFIGS = [(0.0, 0.0), (0.0, 50.0), (0.3, 0.0), (0.3, 50.0), (0.5, 100.0)]


def feats(r, prefix):
    out = {}
    for ch, name in (("Pra", "ra"), ("Pvp", "wedge")):
        h = harmonics(r.waves201[ch])
        out[f"{prefix}{name}_double"] = float(h[1] > h[0])
        out[f"{prefix}{name}_ptp"] = float(np.ptp(r.waves201[ch]))
    return out


def run(i):
    th = sample_theta(np.random.default_rng(9000 + i))
    lo = simulate(th)
    if not lo.success:
        return None
    row = {"theta": i, **{k: lo.summaries[k] for k in MATCH if k != "hr"}, "hr": th["HR"]}
    row.update(feats(lo, "lo_"))
    for k, tau in CONFIGS[1:]:
        r = simulate(th, {"k_av_l": k, "k_av_r": k}) if k else simulate(th, {})
        if not r.success:
            return None
        r = apply_wedge(r, full_phi({"w_wedge": 1.0, "tau_wedge": tau, "k_av_l": k, "k_av_r": k}))
        row.update(feats(r, f"k{k}_t{tau:.0f}_"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=250)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    real_df, W = load_real()
    real_df["hr"] = 60000.0 / real_df["T"]
    for j, ch, name in ((1, "Pra", "ra"), (3, "Pvp", "wedge")):
        H = np.array([harmonics(w[j]) for w in W])
        real_df[f"{name}_double"] = H[:, 1] > H[:, 0]
        real_df[f"{name}_ptp"] = np.ptp(W[:, j], axis=1)

    with ProcessPoolExecutor(a.workers) as ex:
        sims = pd.DataFrame([r for r in ex.map(run, range(a.n), chunksize=1) if r is not None])
    sims.to_csv(out / "sims.csv", index=False)

    mu, sd = real_df[MATCH].mean(), real_df[MATCH].std()
    S = ((sims[MATCH] - mu) / sd).to_numpy()
    R = ((real_df[MATCH] - mu) / sd).to_numpy()
    d = np.sqrt(((R[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    nn = np.argsort(d, axis=1)[:, :5]  # 5 nearest sims per real beat
    print(f"{len(sims)} sims; median NN distance (std units) {np.median(d[np.arange(len(R))[:, None], nn]):.2f}")
    matched = sims.iloc[nn.ravel()]

    print("\nreal:          RA double {:.2f}  RA ptp {:.1f} | wedge double {:.2f}  wedge ptp {:.1f}".format(
        real_df.ra_double.mean(), real_df.ra_ptp.median(), real_df.wedge_double.mean(), real_df.wedge_ptp.median()))
    labels = [("lo_", "low-fi (k=0, Pvp vessel)")] + [(f"k{k}_t{t:.0f}_", f"k={k}, wedge tau={t:.0f}") for k, t in CONFIGS[1:]]
    for p, lab in labels:
        print("{:26s} RA double {:.2f}  RA ptp {:.1f} | wedge double {:.2f}  wedge ptp {:.1f}".format(
            lab, matched[p + "ra_double"].mean(), matched[p + "ra_ptp"].median(),
            matched[p + "wedge_double"].mean(), matched[p + "wedge_ptp"].median()))


if __name__ == "__main__":
    main()
