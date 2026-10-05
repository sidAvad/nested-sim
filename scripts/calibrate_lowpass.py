"""How much low-pass filtering brings simulated spectra to the real roll-off?

Applies the catheter filter (periodic, steady state) to the cached 201-pt
high-fi beats from real_vs_sim.py for a grid of (fn, zeta) and compares the
median normalised harmonic spectrum with the real one (mean squared log10
distance over harmonics 2..20).

Usage: python scripts/calibrate_lowpass.py RESULTS_DIR (the real_vs_sim output dir)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim.measurement import catheter_filter  # noqa: E402
from real_vs_sim import CH, load_real  # noqa: E402

K = np.arange(2, 21)


def med_spec(W):
    x = W[:, :, :-1]
    s = np.abs(np.fft.rfft(x - x.mean(axis=2, keepdims=True), axis=2)) ** 2
    s = s[:, :, 1:41] / s[:, :, 1:].sum(axis=2, keepdims=True)
    return np.median(s, axis=0)  # (4, 40)


def dist(a, b):
    return np.mean((np.log10(a[:, K - 1]) - np.log10(b[:, K - 1])) ** 2, axis=1)


def main():
    d = Path(sys.argv[1])
    z = np.load(d / "sims.npz")
    W, T = z["hifi"], z["T"]
    _, real_w = load_real()
    real = med_spec(real_w)
    base = dist(med_spec(W), real)
    print("channels:", CH)
    print(f"unfiltered hifi            dist={np.round(base, 3)}  mean={base.mean():.3f}")
    for zeta in (0.7, 1.0):
        for fn in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0):
            F = np.empty_like(W)
            for i in range(len(W)):
                dt = T[i] / 200.0
                for j in range(4):
                    F[i, j] = catheter_filter(W[i, j], dt, fn, zeta)
            dd = dist(med_spec(F), real)
            print(f"fn={fn:5.1f} Hz zeta={zeta:.1f}  dist={np.round(dd, 3)}  mean={dd.mean():.3f}")


if __name__ == "__main__":
    main()
