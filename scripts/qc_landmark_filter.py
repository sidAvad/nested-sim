"""QC filter for nested-sim's lofi/hifi beat cuts: flags sims whose reported
PV_OPEN landmark isn't actually near a Prv-Pap zero crossing.

Background: the solver's output-beat convention (nested_sim/solver.py) windows
each sim to start at the first grid point at/after the Prv-Pap upward zero
crossing (PV_OPEN). Comparing against stored Cv8SimApp binary reference runs
(scripts/compare_binary.py) showed the vast majority of sims match the binary
almost exactly, but a small tail disagree -- and checking the landmark
directly (this script) confirms some of those are genuine misdetections (the
reported t=0 isn't near any zero crossing of Prv-Pap at all), not just
solver-vs-solver numerical drift. Root cause not isolated further; most likely
the valve-gap signal crosses zero more than once near the landmark in some
theta/phi regions (more common under the stronger hifi_v2strong phi prior)
and the "first crossing" logic picks up the wrong one.

Metric: norm_gap0 = (Prv[0] - Pap[0]) at the reported beat start, divided by
that beat's own Prv-Pap peak-to-peak range. A correctly-landmarked beat has
this near 0 (median ~0.01-0.02 across all existing sets, consistent with
ordinary T/200 grid-quantization noise); a misdetected one can be O(1) or more.

Threshold: norm_gap0 > 0.2, chosen 2026-10-12 -- comfortably above the p99 of
every existing set (0.11-0.39) while well below where genuine misdetections
sit (most flagged sims are 1-5x, i.e. the point is on the opposite side of the
beat from where Prv-Pap actually crosses zero). Not re-derived from a formal
criterion; a round number clear of the noise floor.

For the paired test sets (lofi_test/hifi_v2_test/hifi_v2strong_test, same
theta across all three), if ANY one of a theta-matched triplet fails, the
whole triplet is marked for exclusion, since the cross-domain comparison
needs all three members to be valid.

Usage:
    python qc_landmark_filter.py             # check + write qc_report.json per set
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/sa4604/outputs/nested-sim/data")
THRESHOLD = 0.2
SETS = ["lofi_train", "lofi_test", "hifi_v2_test", "hifi_v2strong_test",
        "hifi_v2_target", "hifi_v2strong_target"]
TEST_TRIPLET = ["lofi_test", "hifi_v2_test", "hifi_v2strong_test"]


def norm_gap0(d, bf, gid):
    with h5py.File(d / bf, "r") as f:
        g = f[gid]
        prv = g["waves"]["Prv"][()]
        pap = g["waves"]["Pap"][()]
    gap = prv - pap
    scale = max(np.ptp(gap), 1e-6)
    return abs(gap[0] / scale)


def check_set(name):
    d = ROOT / name
    man = json.load(open(d / "manifest.json"))
    index = man["index"]  # [{file, group, seed}]
    vals = np.empty(len(index))
    t0 = time.time()
    cur_file, cur_h5 = None, None
    for i, e in enumerate(index):
        if e["file"] != cur_file:
            if cur_h5 is not None:
                cur_h5.close()
            cur_h5 = h5py.File(d / e["file"], "r")
            cur_file = e["file"]
        g = cur_h5[e["group"]]
        prv = g["waves"]["Prv"][()]
        pap = g["waves"]["Pap"][()]
        gap = prv - pap
        scale = max(np.ptp(gap), 1e-6)
        vals[i] = abs(gap[0] / scale)
    if cur_h5 is not None:
        cur_h5.close()
    dt = time.time() - t0
    fail = vals > THRESHOLD
    print(f"{name}: n={len(vals)}  median={np.median(vals):.4f}  p99={np.percentile(vals,99):.4f}  "
          f"max={vals.max():.4f}  n_fail(>{THRESHOLD})={fail.sum()} ({fail.mean():.4%})  ({dt:.1f}s)")
    return index, vals, fail


def main():
    results = {}
    for name in SETS:
        index, vals, fail = check_set(name)
        results[name] = {"index": index, "vals": vals, "fail": fail}

    # Propagate triplet failures across lofi_test/hifi_v2_test/hifi_v2strong_test
    # (same theta/seed across all three -- see make_datasets.py's paired "test" kind).
    n = len(results["lofi_test"]["fail"])
    assert all(len(results[s]["fail"]) == n for s in TEST_TRIPLET), "triplet sets must be same length/order"
    triplet_fail = np.zeros(n, dtype=bool)
    for s in TEST_TRIPLET:
        triplet_fail |= results[s]["fail"]
    for s in TEST_TRIPLET:
        results[s]["fail_propagated"] = triplet_fail
    print(f"\ntest triplet: {triplet_fail.sum()} / {n} theta's excluded after propagating "
          f"across lofi/hifi_v2/hifi_v2strong ({triplet_fail.mean():.4%})")

    for name in SETS:
        r = results[name]
        fail = r.get("fail_propagated", r["fail"])
        out = ROOT / name / "qc_report.json"
        json.dump({
            "method": "landmark check: |Prv[0]-Pap[0]| / ptp(Prv-Pap), threshold on this sim's own beat",
            "threshold": THRESHOLD,
            "threshold_date": "2026-10-12",
            "n_total": int(len(fail)),
            "n_excluded": int(fail.sum()),
            "frac_excluded": float(fail.mean()),
            "excluded_groups": [r["index"][i]["file"] + "/" + r["index"][i]["group"]
                                for i in np.where(fail)[0]],
        }, open(out, "w"), indent=1)
        print(f"wrote {out}  ({fail.sum()} excluded)")


if __name__ == "__main__":
    main()
