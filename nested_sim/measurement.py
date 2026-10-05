"""Measurement layer for the 4 cath lab waveforms (Prv, Pra, Pap, Pvp).

One catheter, so these are shared across the 4 channels:
  * 2nd-order low-pass H(s) = wn^2 / (s^2 + 2 zeta wn s + wn^2), wn = 2 pi fn_cath,
    applied to the periodic dense beat (tiled, filtered to steady state, last beat
    kept);
  * zero offset `offset_cath` (mmHg);
  * additive white noise, std `sigma_cath` (mmHg), independent per channel/sample.
All other channels (Pas, volumes, flows, ...) are left exactly as simulated.

Scalars are not modelled separately: as in Cv8SimApp they are the summaries of
the (measured) 201-point waveforms, so cath-derived scalars (spa, dpa, mpap, pcw,
cvp, rv_*) carry the catheter effects and sbp/dbp/map/sv are exact.

The 201-point beat stays on the true-valve PV_OPEN grid of the simulation, so
catheter lag appears as a phase shift, as it would in real recordings. With
phi at PHI_MEAS_OFF every output equals the simulation's exactly.
"""

import math

import numpy as np
from scipy import signal

from .phi import PHI_MEAS_OFF, full_phi
from .solver import summaries

CATHETER_CHANNELS = ("Prv", "Pra", "Pap", "Pvp")
MIN_TILES = 5
SETTLE_TIME_CONSTANTS = 12.0  # tiles cover at least this many 1/(zeta*wn)


def catheter_filter(x, dt, fn_hz, zeta):
    """Steady-state response of the catheter to one periodic beat.

    x: dense beat sampled at spacing dt (ms), including both endpoints
    (x[-1] is the start of the next beat). Returns an array like x.
    """
    if math.isinf(fn_hz):
        return np.array(x, float)
    wn = 2.0 * math.pi * fn_hz / 1000.0  # rad/ms
    period = np.asarray(x[:-1], float)
    n = period.size
    T = n * dt
    tiles = max(MIN_TILES, math.ceil(SETTLE_TIME_CONSTANTS / (zeta * wn) / T) + 1)
    # Bilinear transform, prewarped so the digital resonance sits exactly at fn.
    fs = 1.0 / dt
    wp = 2.0 * fs * math.tan(wn / (2.0 * fs))
    num, den = signal.bilinear([wp * wp], [1.0, 2.0 * zeta * wp, wp * wp], fs=fs)
    zi = signal.lfilter_zi(num, den) * period[0]
    y, _ = signal.lfilter(num, den, np.tile(period, tiles), zi=zi)
    last = y[-n:]
    return np.append(last, last[0])


def measure(sim, phi=None, rng=None):
    """Apply the measurement model to a successful SimResult.

    Returns dict with 'dense' and 'waves201' (all channels; cath lab channels
    measured, the rest unchanged) and 'scalars' (summaries of waves201, plus hr).
    """
    if not sim.success:
        raise ValueError("cannot measure a failed simulation")
    ph = full_phi(phi)
    m = {k: ph[k] for k in PHI_MEAS_OFF}
    rng = np.random.default_rng() if rng is None else rng

    dt = sim.t_dense[1] - sim.t_dense[0]
    dense = dict(sim.dense)
    for ch in CATHETER_CHANNELS:
        x = catheter_filter(sim.dense[ch], dt, m["fn_cath"], m["zeta_cath"])
        x = x + m["offset_cath"] + m["sigma_cath"] * rng.standard_normal(x.size)
        dense[ch] = x

    factor = (len(sim.t_dense) - 1) // (len(sim.t201) - 1)
    waves201 = {k: v[::factor] for k, v in dense.items()}
    scalars = summaries(waves201)
    scalars["hr"] = 60000.0 / sim.T
    return {"dense": dense, "waves201": waves201, "scalars": scalars}
