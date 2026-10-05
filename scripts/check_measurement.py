"""Checks for the measurement layer.

1. With phi off, measure() is the identity (exact).
2. catheter_filter matches the analog response H(jw) on sinusoids (gain, phase)
   and has unit DC gain.
3. Figure: raw vs measured catheter channels at the corners of the fn/zeta range.

Usage: python scripts/check_measurement.py OUT.png
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import PHI_OFF, SolverSettings, simulate  # noqa: E402
from nested_sim.measurement import CATHETER_CHANNELS, catheter_filter, measure  # noqa: E402

RAW, MEAS = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def check_identity(sim):
    out = measure(sim, PHI_OFF, np.random.default_rng(0))
    worst = max(np.abs(out["dense"][k] - sim.dense[k]).max() for k in sim.dense)
    w201 = max(np.abs(out["waves201"][k] - sim.waves201[k]).max() for k in sim.waves201)
    sc = max(abs(out["scalars"][k] - sim.summaries[k]) for k in ("map", "sbp", "dbp", "sv"))
    print(f"[1] identity with phi off: dense {worst:.1e}, 201-pt {w201:.1e}, scalars {sc:.1e}")
    assert worst == 0 and w201 == 0 and sc == 0


def check_response():
    T, n = 800.0, 2000
    dt = T / n
    t = np.arange(n + 1) * dt
    print("[2] digital vs analog catheter response (sinusoid at f_signal; max abs error vs |H| sin(wt + arg H))")
    worst = 0.0
    for fn, z in [(10.0, 0.15), (10.0, 0.6), (25.0, 0.15), (25.0, 0.6)]:
        wn = 2 * math.pi * fn / 1000
        for k in [1, 5, 8, 12, 20]:  # harmonics of the beat
            w = 2 * math.pi * k / T
            x = np.sin(w * t)
            y = catheter_filter(x, dt, fn, z)
            H = wn ** 2 / (-(w ** 2) + 2j * z * wn * w + wn ** 2)
            ref = np.abs(H) * np.sin(w * t + np.angle(H))
            err = np.abs(y - ref).max()
            worst = max(worst, err / max(np.abs(H), 1e-12))
            print(f"    fn={fn:4.0f} zeta={z:4.2f} f={k / T * 1000:5.2f} Hz: |H|={np.abs(H):6.3f}  max err {err:.1e}")
        dc = catheter_filter(np.full(n + 1, 7.0), dt, fn, z)
        assert np.allclose(dc, 7.0, atol=1e-9), "DC gain != 1"
    print(f"    worst relative error {worst:.2e}")


def figure(sim, out):
    cases = [(10.0, 0.15, "fn 10 Hz, zeta 0.15"), (25.0, 0.6, "fn 25 Hz, zeta 0.6"),
             (15.0, 0.3, "fn 15 Hz, zeta 0.3, offset +2, noise 0.5")]
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 3 * len(cases)), squeeze=False)
    for i, (fn, z, label) in enumerate(cases):
        phi = {"fn_cath": fn, "zeta_cath": z}
        if i == 2:
            phi.update(offset_cath=2.0, sigma_cath=0.5)
        m = measure(sim, phi, np.random.default_rng(1))
        for j, ch in enumerate(CATHETER_CHANNELS):
            ax = axes[i, j]
            ax.plot(sim.t_dense, sim.dense[ch], color=RAW, lw=2, label="simulated")
            ax.plot(sim.t201, m["waves201"][ch], color=MEAS, lw=1.5, marker="o", ms=2.5,
                    label="measured (201 pt)")
            ax.grid(True, color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title((label + "\n" if j == 0 else "\n") + ch, fontsize=9, color=INK, loc="left")
            if i == len(cases) - 1:
                ax.set_xlabel("ms from PV_OPEN", fontsize=8, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Catheter measurement model at default theta (low-fi beat)", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("[3] wrote", out)


def main():
    sim = simulate({}, None, SolverSettings(method="LSODA"))
    check_identity(sim)
    check_response()
    figure(sim, sys.argv[1])


if __name__ == "__main__":
    main()
