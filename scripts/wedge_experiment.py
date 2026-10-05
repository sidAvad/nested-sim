"""Can AV-plane coupling + a wedge observation reproduce the real Pra / Pvp shape?

Sweeps k_av (both atria) and the wedge lag tau (w_wedge = 1) over prior theta and
reports, per config, the share of beats whose 2nd harmonic dominates (double
hump: a + v waves) and the median peak-to-peak, for Pra and the Pvp channel.
Real targets are printed first.

Usage: python scripts/wedge_experiment.py OUT.csv [--n 300] [--workers 250]
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
from real_vs_sim import CH, load_real  # noqa: E402

KS = [0.0, 0.1, 0.2, 0.3, 0.5]
TAUS = [0.0, 20.0, 50.0]


def run(i):
    th = sample_theta(np.random.default_rng(7000 + i))
    rows = []
    for k in KS:
        base = simulate(th, {"k_av_l": k, "k_av_r": k})
        if not base.success:
            rows.append({"theta": i, "k": k, "ok": False})
            continue
        dense0 = dict(base.dense)
        for tau in TAUS:
            base.dense = dict(dense0)
            r = apply_wedge(base, full_phi({"w_wedge": 1.0, "tau_wedge": tau, "k_av_l": k, "k_av_r": k}))
            row = {"theta": i, "k": k, "tau": tau, "ok": True}
            for ch in ("Pra", "Pvp", "Pla"):
                h = harmonics(r.waves201[ch])
                row[f"{ch}_h1"], row[f"{ch}_h2"] = h[0], h[1]
                row[f"{ch}_ptp"] = float(np.ptp(r.waves201[ch]))
            row["pcw"] = r.summaries["pcw"]
            row["sv"] = r.summaries["sv"]
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=250)
    a = ap.parse_args()

    _, W = load_real()
    print("real targets:")
    for j, c in ((1, "Pra"), (3, "Pvp")):
        H = np.array([harmonics(w[j]) for w in W])
        print(f"  {c}: double-hump share {np.mean(H[:, 1] > H[:, 0]):.2f}  median ptp {np.median(np.ptp(W[:, j], axis=1)):.1f}")

    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame([r for rs in ex.map(run, range(a.n), chunksize=1) for r in rs])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nsuccess by k: {df.groupby('k').ok.mean().round(3).to_dict()}")
    d = df[df.ok]
    g = d.groupby(["k", "tau"]).apply(lambda x: pd.Series({
        "Pra_double": np.mean(x.Pra_h2 > x.Pra_h1), "Pra_ptp": x.Pra_ptp.median(),
        "wedge_double": np.mean(x.Pvp_h2 > x.Pvp_h1), "wedge_ptp": x.Pvp_ptp.median(),
        "Pla_double": np.mean(x.Pla_h2 > x.Pla_h1), "pcw_med": x.pcw.median()}), include_groups=False)
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
