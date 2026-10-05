"""Low-fi Python port vs the Cv8SimApp binary reference runs (ref/)."""

import json

import h5py
import numpy as np
import pytest
from conftest import ROOT, SETTINGS

from nested_sim import simulate
from nested_sim.params import from_binary_names, full_theta

REF = ROOT / "ref" / "ref_nt200.h5"
PARAMS = ROOT / "ref" / "ref_params.json"
# Beats the binary starts one grid step later (high HR, incomplete relaxation);
# its cycle/summary code (cv8.jl) is not available.
WINDOW_OFFSET = {10, 17}
SCALARS = ["sv", "sbp", "dbp", "spa", "dpa", "rv_s", "rv_d", "lv_s", "lv_d"]

pytestmark = pytest.mark.skipif(not REF.exists(), reason="binary reference runs not present")


def converged_reference_ids():
    with h5py.File(REF) as f:
        tol = float(f.attrs["sp_steadystate_tol"])
        ok = f["status"]["success"][()].astype(bool) & (f["status"]["steadystate_error"][()] <= 10 * tol)
    return [int(i) for i in np.flatnonzero(ok)[:24] if i not in WINDOW_OFFSET]


def off_cycle_reset(r, avd, eps=1e-6):
    """Mask of 201-grid points not exactly on an activation reset (atrial at
    t = 0 mod T, ventricular at t = AVD mod T). With incomplete relaxation the
    activation jumps there, and which side a grid point lands on depends on the
    last bit of t mod T, so those points are not comparable."""
    t = r.t0 + r.t201
    keep = np.ones(t.size, bool)
    for shift in (0.0, avd):
        ph = np.mod(t - shift, r.T)
        keep &= np.minimum(ph, r.T - ph) > eps
    return keep


@pytest.mark.parametrize("i", converged_reference_ids() if REF.exists() else [], ids=lambda i: f"sim{i}")
def test_lowfi_matches_binary(i):
    rows = json.load(open(PARAMS))
    theta = from_binary_names(rows[i])
    r = simulate(theta, settings=SETTINGS)
    assert r.success, r.message
    with h5py.File(REF) as f:
        vn = [v.decode() for v in f["var_names"][()]]
        W = f["waves"][i]
        for k in SCALARS:
            assert r.summaries[k] == pytest.approx(float(f["summaries"][k][i]), abs=1e-3), k
    keep = off_cycle_reset(r, full_theta(theta)["AVD"])
    assert keep.sum() >= 199
    for ch in ("Pas", "Pap", "Prv", "Pra", "Pvp", "Vlv"):
        ref = W[:, vn.index(ch)]
        assert np.max(np.abs(r.waves201[ch] - ref)[keep]) < 5e-3 * max(1.0, np.ptp(ref)), ch
