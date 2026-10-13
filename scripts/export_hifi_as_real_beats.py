"""Filesystem adapter: nested-sim's hifi_v2strong_target (manifest.json +
batch_XXXX.h5, many sims per file) -> cv-sbi-spin's train_spin.py --real-data
layout (dataset.py's RealBeatsDataset has no manifest/index support -- it
globs *.h5 directly in the given directory and reads the first beat_* group
per file, i.e. genuinely one file per patient). This is a real file-splitting
operation, not just a directory/symlink trick.

QC-excluded sims (hifi_v2strong_target/qc_report.json's excluded_groups --
this set isn't part of qc_landmark_filter.py's test-triplet propagation, so
this report is already the correct single-set exclusion mask) are skipped.

Per surviving sim, writes one file <out>/sim_<global index>.h5 with a single
beat_0 group:
  waves/{Prv,Pra,Pvp,Pap}   <- copied unchanged (RealBeatsDataset's WAVE_KEYS_REDUCED)
  summaries/{map,sbp,dbp,sv} <- copied unchanged (same field names on both sides)
  parameters/HR             <- copied unchanged
  parameters/PVR            <- source parameters/Rap (real-patient naming for the
                                same quantity; RealBeatsDataset reads PVR/SVR, not
                                the sim-side Rap/Ras binary names)
  parameters/SVR            <- source parameters/Ras

Writes are atomic (.tmp + os.replace) and skip any output file that already
exists, so an interrupted run is resumable without redoing completed work.

Source batch files are opened once each (surviving entries are grouped by
source file first), not reopened per-sim.

Usage:
    python export_hifi_as_real_beats.py [--target-set hifi_v2strong_target]
        [--out <dir>] [--limit N]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import h5py

ROOT = Path("/home/sa4604/outputs/nested-sim/data")

WAVE_KEYS = ["Prv", "Pra", "Pvp", "Pap"]
SUMMARY_KEYS = ["map", "sbp", "dbp", "sv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-set", default="hifi_v2strong_target")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src_dir = ROOT / args.target_set
    out_dir = Path(args.out) if args.out else ROOT / f"{args.target_set}_for_spin_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(src_dir / "manifest.json"))
    index = manifest["index"]

    qc = json.load(open(src_dir / "qc_report.json"))
    excluded = set(qc["excluded_groups"])
    print(f"{args.target_set}: {len(index)} total, {len(excluded)} QC-excluded")

    surviving = [(gi, e) for gi, e in enumerate(index)
                 if f"{e['file']}/{e['group']}" not in excluded]
    if args.limit is not None:
        surviving = surviving[:args.limit]
    print(f"processing {len(surviving)} surviving sims -> {out_dir}")

    by_file = defaultdict(list)
    for gi, e in surviving:
        by_file[e["file"]].append((gi, e))

    n_written, n_skipped = 0, 0
    for fname, entries in by_file.items():
        with h5py.File(src_dir / fname, "r") as fh:
            for gi, e in entries:
                dst = out_dir / f"sim_{gi}.h5"
                if dst.exists():
                    n_skipped += 1
                    continue
                g = fh[e["group"]]
                tmp = out_dir / f"sim_{gi}.h5.tmp"
                with h5py.File(tmp, "w") as out:
                    beat = out.create_group("beat_0")
                    waves = beat.create_group("waves")
                    for k in WAVE_KEYS:
                        waves.create_dataset(k, data=g[f"waves/{k}"][:])
                    summaries = beat.create_group("summaries")
                    for k in SUMMARY_KEYS:
                        summaries.create_dataset(k, data=g[f"summaries/{k}"][()])
                    params = beat.create_group("parameters")
                    params.create_dataset("HR", data=g["parameters/HR"][()])
                    params.create_dataset("PVR", data=g["parameters/Rap"][()])
                    params.create_dataset("SVR", data=g["parameters/Ras"][()])
                os.replace(tmp, dst)
                n_written += 1
        if (n_written + n_skipped) % 5000 < len(entries):
            print(f"  {n_written + n_skipped}/{len(surviving)} "
                  f"(written={n_written}, skipped_existing={n_skipped})")

    print(f"done: written={n_written} skipped_existing={n_skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
