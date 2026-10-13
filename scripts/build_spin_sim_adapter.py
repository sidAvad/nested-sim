"""Filesystem adapter: nested-sim's lofi_train/lofi_test -> cv-sbi-spin's
train_spin.py --sim-data-root layout (dataset.py's ReducedCVDataset expects
manifest_train.json + manifest_test.json directly under the root, plus
train/ and test/ subdirs of the referenced HDF5 files).

train/ and test/ are plain symlinks to lofi_train/lofi_test (zero-copy).
manifest_train.json reuses the already-QC-filtered index built for the Gibbs
NPE retrain (lofi_train_for_gibbs/manifest_train.json) -- lofi_train isn't
part of qc_landmark_filter.py's test-triplet propagation, so that index is
already the correct single-set one, just copied here under a run-appropriate
directory name.

manifest_test.json is NOT copied from lofi_test/qc_report.json -- that report
stores the TRIPLET-propagated exclusion (cross-referencing hifi_v2_test/
hifi_v2strong_test, ~183/10,000), which is the wrong mask for this adapter's
single-domain (lofi-only) --n-val-sims diagnostic. This script recomputes the
single-set landmark check directly on lofi_test, reusing qc_landmark_filter's
norm_gap0 metric/threshold (0.2).

Usage:
    python build_spin_sim_adapter.py
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qc_landmark_filter import THRESHOLD  # noqa: E402

ROOT = Path("/home/sa4604/outputs/nested-sim/data")
OUT = ROOT / "lofi_for_spin"


def check_single_set(name):
    d = ROOT / name
    man = json.load(open(d / "manifest.json"))
    index = man["index"]
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
    fail = vals > THRESHOLD
    print(f"{name}: n={len(vals)}  n_fail(>{THRESHOLD})={fail.sum()} ({fail.mean():.4%})  "
          f"({time.time()-t0:.1f}s)")
    return index, fail


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for link_name, target in [("train", "../lofi_train"), ("test", "../lofi_test")]:
        link = OUT / link_name
        if link.is_symlink() or link.exists():
            print(f"{link} already exists, leaving as-is")
        else:
            link.symlink_to(target)
            print(f"symlinked {link} -> {target}")

    gibbs_manifest = ROOT / "lofi_train_for_gibbs" / "manifest_train.json"
    train_man = json.load(open(gibbs_manifest))
    with open(OUT / "manifest_train.json", "w") as f:
        json.dump(train_man, f)
    print(f"wrote manifest_train.json ({len(train_man['index'])} entries, "
          f"copied from {gibbs_manifest})")

    index, fail = check_single_set("lofi_test")
    keep = [index[i] for i in range(len(index)) if not fail[i]]
    with open(OUT / "manifest_test.json", "w") as f:
        json.dump({"index": keep}, f)
    print(f"wrote manifest_test.json ({len(keep)} / {len(index)} entries, "
          f"{fail.sum()} excluded by single-set landmark check)")


if __name__ == "__main__":
    main()
