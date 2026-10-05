"""Is high-fi closer to the real systemic pressures (and RA shape) than low-fi?

Simulates prior theta with low-fi and with high-fi (structural phi at alpha = 1).
For each real beat, takes the 5 nearest sims of EACH model on standardised
(cvp, pcw, mpap, spa, dpa, hr, map) and compares the matched sims' SBP, DBP,
pulse pressure and RA double-hump share (2nd harmonic dominant) with the real
values. --priors selects the phi prior file.

Usage: python scripts/matched_systemic.py OUT_DIR [--n 1500] [--workers 250]
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
from nested_sim.phi import PHI_STRUCT_OFF, sample_phi  # noqa: E402
from real_checks import harmonics  # noqa: E402
from real_vs_sim import load_real  # noqa: E402
from nested_sim.phi import load_priors  # noqa: E402

MATCH = ["cvp", "pcw", "mpap", "spa", "dpa", "hr", "map"]
SUMM = ["cvp", "pcw", "mpap", "spa", "dpa", "map", "sbp", "dbp"]


def run(args):
    i, priors_path = args
    rng = np.random.default_rng(11000 + i)
    th = sample_theta(rng)
    phi = {k: v for k, v in sample_phi(rng, 1.0, th, load_priors(priors_path)).items() if k in PHI_STRUCT_OFF}
    out = []
    for model, r in (("lofi", simulate(th)), ("hifi", simulate(th, phi))):
        if r.success:
            h = harmonics(r.waves201["Pra"])
            out.append({"model": model, "theta": i, "hr": th["HR"], "ra_double": float(h[1] > h[0]),
                        **{k: r.summaries[k] for k in SUMM}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--priors", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    real, W = load_real()
    real["hr"] = 60000.0 / real["T"]
    real["pp"] = real.sbp - real.dbp
    real["ra_double"] = [float(harmonics(w[1])[1] > harmonics(w[1])[0]) for w in W]
    with ProcessPoolExecutor(a.workers) as ex:
        sims = pd.DataFrame([r for rs in ex.map(run, [(i, a.priors) for i in range(a.n)], chunksize=1) for r in rs])
    sims["pp"] = sims.sbp - sims.dbp
    sims.to_csv(out / "sims.csv", index=False)

    mu, sd = real[MATCH].mean(), real[MATCH].std()
    R = ((real[MATCH] - mu) / sd).to_numpy()
    print(f"real: SBP {real.sbp.median():.0f}  DBP {real.dbp.median():.0f}  PP {real.pp.median():.0f}  "
          f"(p10-p90 PP {real.pp.quantile(.1):.0f}-{real.pp.quantile(.9):.0f})  RA double {real.ra_double.mean():.2f}")
    for model in ("lofi", "hifi"):
        s = sims[sims.model == model].reset_index(drop=True)
        S = ((s[MATCH] - mu) / sd).to_numpy()
        d = np.sqrt(((R[:, None] - S[None]) ** 2).sum(-1))
        nn = np.argsort(d, axis=1)[:, :5]
        m = s.iloc[nn.ravel()]
        err = (m.pp.to_numpy().reshape(-1, 5).mean(1) - real.pp.to_numpy())
        print(f"{model}: matched SBP {m.sbp.median():.0f}  DBP {m.dbp.median():.0f}  PP {m.pp.median():.0f}  "
              f"(p10-p90 PP {m.pp.quantile(.1):.0f}-{m.pp.quantile(.9):.0f})  "
              f"per-beat PP error: median {np.median(err):+.1f}, median |err| {np.median(np.abs(err)):.1f}  "
              f"RA double {m.ra_double.mean():.2f}  [match dist {np.median(d[np.arange(len(R))[:, None], nn]):.2f}]")


if __name__ == "__main__":
    main()
