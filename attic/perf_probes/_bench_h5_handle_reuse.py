#!/usr/bin/env python
"""Timing + bit-identity evidence for the load_fragment_arrays handle reuse.

Run it once with the pristine `dataframe.py` and once with the change, writing
a JSON to a different path each time, then compare with `--diff a.json b.json`.
The comparison is over per-region, per-field sha256 digests of the RAW ARRAY
BYTES plus dtype, so a dtype change, a reordering or a single flipped mantissa
bit all show up. A spot-check of a few numbers would not catch those.

The h5 is a real 1.19 GB sample on EFS; `load_fragment_arrays` is called on the
SAME frame and the SAME regions in both runs, so the only difference between
the two JSONs can be the code under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fragments_h5 import FragmentsH5  # noqa: E402

from fragmentomics_tools.dataframe import SampleAndRegionDataFrame  # noqa: E402

H5 = ("/efs/analytics/nathanboley/biomarker-projects/data_cache/DC4-16709/"
      "9c95b98289ff163dfd16f6554adb401a-142-AC-124123-Lib1_DC4-16709_S3"
      ".hg38.fragments.h5")
N_REGIONS = 300
REGION_LEN = 2048
STEP = 100_000          # spread the 300 tiles over 30Mb rather than tiling
START = 1_000_000       # contiguously, which would flatter the h5 chunk cache
MIN_MAPQ = 10
MAX_FRAG_LEN = 511
REPS = 3


def make_srdf():
    starts = [START + i * STEP for i in range(N_REGIONS)]
    return SampleAndRegionDataFrame(
        pd.DataFrame({
            "contig": ["chr1"] * N_REGIONS,
            "start": starts,
            "stop": [s + REGION_LEN for s in starts],
            "sample_id": ["s1"] * N_REGIONS,
            "frag_h5": [H5] * N_REGIONS,
        }),
        ref="hg38",
    )


def digest_one(fa):
    """Per-field sha256 of the exact bytes, keyed by field name."""
    out = {"region": repr(fa.region), "n_frags": int(fa.n_frags),
           "max_frag_len": int(fa.max_frag_len),
           "is_flipped": bool(fa.is_flipped)}
    for key, val in sorted(fa.init_kwargs.items()):
        if key == "region":
            continue
        if val is None:
            out[key] = None
        elif isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val)
            out[key] = f"{arr.dtype.str}:{arr.shape}:" + hashlib.sha256(
                arr.tobytes()).hexdigest()[:32]
        else:
            out[key] = repr(val)
    return out


def digest_all(arrays):
    return [digest_one(fa) for fa in arrays]


class OpenCounter:
    """Count FragmentsH5 opens/closes IN THIS PROCESS."""

    def __init__(self):
        self.opens = 0
        self.closes = 0
        self._init = FragmentsH5.__init__
        self._close = FragmentsH5.close

    def __enter__(self):
        real_init, real_close, me = self._init, self._close, self

        def counting_init(obj, *a, **k):
            real_init(obj, *a, **k)
            me.opens += 1

        def counting_close(obj, *a, **k):
            me.closes += 1
            return real_close(obj, *a, **k)

        FragmentsH5.__init__ = counting_init
        FragmentsH5.close = counting_close
        return self

    def __exit__(self, *exc):
        FragmentsH5.__init__ = self._init
        FragmentsH5.close = self._close


def fds_on(path):
    real = os.path.realpath(path)
    n = 0
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.path.join("/proc/self/fd", name)) == real:
                n += 1
        except OSError:
            pass
    return n


def run(label, n_workers):
    srdf = make_srdf()
    times, opens, closes = [], [], []
    last = None
    for _ in range(REPS):
        with OpenCounter() as c:
            t0 = time.perf_counter()
            rv = srdf.load_fragment_arrays(
                n_workers=n_workers, verbose=0,
                min_mapq=MIN_MAPQ, max_frag_len=MAX_FRAG_LEN,
            )
            times.append(time.perf_counter() - t0)
        opens.append(c.opens)
        closes.append(c.closes)
        last = list(rv["fragment_array"]) if hasattr(rv, "columns") else list(rv)
    leaked = fds_on(H5)
    print(f"  {label:<18} times={[round(t, 2) for t in times]}  "
          f"median={np.median(times):6.2f}s  parent_opens={opens}  "
          f"parent_closes={closes}  fds_still_open_on_h5={leaked}")
    return {
        "label": label,
        "n_workers": n_workers,
        "times_s": times,
        "median_s": float(np.median(times)),
        "parent_opens": opens,
        "parent_closes": closes,
        "fds_still_open_on_h5": leaked,
        "total_frags": int(sum(fa.n_frags for fa in last)),
        "digests": digest_all(last),
    }


def flatten(obj, prefix=""):
    """Flatten to leaf paths so a diff names the exact field that moved."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def do_diff(path_a, path_b):
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    rc = 0
    for cfg in sorted(set(a["runs"]) | set(b["runs"])):
        if cfg not in a["runs"] or cfg not in b["runs"]:
            print(f"{cfg}: present in only one run -- cannot compare")
            rc = 1
            continue
        fa = dict(flatten(a["runs"][cfg]["digests"]))
        fb = dict(flatten(b["runs"][cfg]["digests"]))
        keys = set(fa) | set(fb)
        bad = [k for k in sorted(keys) if fa.get(k, "<missing>") != fb.get(k, "<missing>")]
        print(f"{cfg}: {len(keys)} leaf fields compared, {len(bad)} differ "
              f"(total_frags {a['runs'][cfg]['total_frags']} vs "
              f"{b['runs'][cfg]['total_frags']})")
        for k in bad[:20]:
            print(f"    {k}: {fa.get(k, '<missing>')} != {fb.get(k, '<missing>')}")
        if bad:
            rc = 1
        print(f"    timing: median {a['runs'][cfg]['median_s']:.2f}s -> "
              f"{b['runs'][cfg]['median_s']:.2f}s  "
              f"({a['runs'][cfg]['median_s'] / b['runs'][cfg]['median_s']:.2f}x)")
        print(f"    parent opens: {a['runs'][cfg]['parent_opens']} -> "
              f"{b['runs'][cfg]['parent_opens']}")
    print("IDENTICAL" if rc == 0 else "DIFFERENCES FOUND")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2)
    args = ap.parse_args()

    if args.diff:
        return do_diff(*args.diff)

    print(f"h5 = {H5}")
    print(f"{N_REGIONS} regions of {REGION_LEN}bp, step {STEP}, "
          f"{REPS} reps each")
    runs = {}
    for label, nw in (("n_workers=1", 1), ("n_workers=8", 8)):
        runs[label] = run(label, nw)
    payload = {
        "h5": H5, "n_regions": N_REGIONS, "region_len": REGION_LEN,
        "step": STEP, "start": START, "min_mapq": MIN_MAPQ,
        "max_frag_len": MAX_FRAG_LEN, "reps": REPS, "runs": runs,
    }
    if args.out:
        with open(args.out, "w") as fp:
            json.dump(payload, fp, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
