#!/usr/bin/env python
"""Measure per-sample cut-site hexamer counts C(h), N(h) and r(h), and write them.

    python scripts/measure_cut_site_hexamers.py \
        --sample-id RD-56804 \
        --fragments-h5 /efs/.../RD-56804-Lib1.hg38.fragments.h5 \
        --region-bed  <main checkout>/data/region_sets/quiet_v2_pad1200_repeats_removed_tile1536.bed \
        --fasta       /efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz \
        --out-dir /efs/.../cut_site_measure_out

This replaces the retired ``scripts/count_cut_site_hexamers.py`` (owner
decision 189, moved to ``attic/``). That script carried its own counting
rule -- containment admission, its own band/encoder machinery. This one holds
NO counting rule of its own: it calls ``simulator.measure.measure_sample``,
the same function ``scripts/run_cut_site_simulator.py`` calls, and writes the
result.

Two differences from the retired tables, so they are NOT comparable:

- Admission here is **start-in-region** (``measure``'s rule), not the retired
  script's containment rule (owner-accepted divergence).
- **The 16 length-band split is dropped.** The retired script wrote each
  table per fragment-length band; ``measure`` counts each table over all
  admitted lengths ``[L_MIN, L_MAX]`` at once. ``f(L)`` is written instead,
  as ``fl_counts``.

``measure_sample`` owns the stage order (``f(L)`` before ``N(h)``, ``N(h)``
from the rdf, not the srdf); see its docstring.

Numeric accumulation in this repo is always float64 (see CLAUDE.md);
``measure`` already does this throughout, and nothing here adds a float32
buffer.

Prefer ``/efs`` for ``--out-dir``, not ``/home`` -- see
``run_cut_site_simulator.py``'s module docstring on why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time

import numpy as np

# scripts/ is not a package and the repo is not pip-installed, so the repo
# root has to go on sys.path before the first-party imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fragmentomics_tools.dataframe import RegionDataFrame  # noqa: E402

from background_model.constants import (  # noqa: E402
    HEX_HALF, KMER, L_MAX, L_MIN, N_LENGTHS, NHEX,
)
from background_model.simulator.measure import (  # noqa: E402
    TABLE_NAMES,
    UNIFORM_BLOCK_SIZE,
    measure_sample,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample-id", required=True)
    p.add_argument("--fragments-h5", required=True)
    p.add_argument("--region-bed", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--out-dir", required=True,
                   help="output directory; prefer /efs, not /home")
    p.add_argument("--n-regions", type=int, default=None,
                   help="take only the first N regions (smoke runs)")
    p.add_argument("--ref", default="hg38", help="reference name for from_bed")
    p.add_argument("--min-mapq", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=None,
                   help="processes for every parallel stage; default every "
                        "CPU, 1 runs in-process")
    return p.parse_args(argv)


def _git_sha_and_dirty():
    """``(sha, dirty)`` for the repo at ``_REPO_ROOT``; ``(None, None)`` if git
    is unavailable -- it is not on PATH in the batch container."""
    try:
        sha = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            # Untracked files do not change what ran, so they do not count.
            ["git", "-C", _REPO_ROOT, "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return sha, dirty
    except Exception:
        return None, None


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(final_path, write_fn):
    """Write via a temp name in the same dir, then ``os.replace``.

    So a killed run never leaves a truncated artifact under the final name.
    ``write_fn`` receives an open binary file handle, not a path --
    ``np.savez_compressed`` silently appends ``.npz`` to a bare path lacking
    that suffix, which would break the rename below.
    """
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        write_fn(fh)
    os.replace(tmp_path, final_path)


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.sample_id)
    t0 = time.time()

    rdf = RegionDataFrame.from_bed(args.region_bed, ref=args.ref)
    if args.n_regions is not None:
        rdf = rdf.iloc[: args.n_regions]
    print(f"[regions] {len(rdf)} from {os.path.basename(args.region_bed)}")

    m = measure_sample(
        rdf, args.sample_id, args.fragments_h5, args.fasta,
        min_mapq=args.min_mapq, n_workers=args.n_workers, verbose=False,
        log=print,
    )
    C, region_counts, stats, srdf = m.C, m.region_counts, m.stats, m.srdf
    fl, N, n_meta, r = m.fl, m.N, m.n_meta, m.r

    region_index = np.asarray(srdf["region_index"], dtype=np.int64)
    assert len(region_index) == len(region_counts) == len(rdf), (
        f"length mismatch: region_index={len(region_index)}, "
        f"region_counts={len(region_counts)}, rdf={len(rdf)}"
    )

    npz_path = f"{stem}.cut_site_measure.npz"

    def _write_npz(fh):
        arrays = {}
        for table in TABLE_NAMES:
            arrays[f"C_{table}"] = C[table].astype(np.int64)
            arrays[f"r_{table}"] = r[table].astype(np.float64)
        arrays["N_start"] = N["start"].astype(np.int64)
        arrays["N_end"] = N["end"].astype(np.float64)
        arrays["fl_counts"] = fl.counts.astype(np.int64)
        arrays["fl_min_fl"] = np.int64(fl.min_fl)
        arrays["region_counts"] = region_counts.astype(np.int64)
        arrays["region_index"] = region_index
        np.savez_compressed(fh, **arrays)

    _atomic_write(npz_path, _write_npz)

    git_sha, git_dirty = _git_sha_and_dirty()
    json_path = f"{stem}.cut_site_measure.json"
    # Size and mtime, not a hash: the h5 runs to GBs, and these suffice to
    # tell a rebuilt input from the one that was measured.
    h5_stat = os.stat(args.fragments_h5)
    meta = dict(
        format_version=1,
        sample_id=args.sample_id,
        fragments_h5=os.path.abspath(args.fragments_h5),
        fragments_h5_size=h5_stat.st_size,
        fragments_h5_mtime=h5_stat.st_mtime,
        region_bed=os.path.abspath(args.region_bed),
        fasta=os.path.abspath(args.fasta),
        region_bed_sha256=_sha256_of(args.region_bed),
        n_regions=len(rdf),
        n_regions_limit=args.n_regions,
        ref=args.ref,
        min_mapq=args.min_mapq,
        # propensities' floor on N: cells with N <= this get r = 0.
        min_expected=m.min_expected,
        # The count actually used: None means every CPU, as in the driver.
        n_workers=(args.n_workers if args.n_workers is not None
                   else multiprocessing.cpu_count()),
        admission="start-in-region",
        git_sha=git_sha,
        git_dirty=git_dirty,
        constants=dict(
            L_MIN=L_MIN, L_MAX=L_MAX, N_LENGTHS=N_LENGTHS, KMER=KMER,
            HEX_HALF=HEX_HALF, NHEX=NHEX, TABLE_NAMES=list(TABLE_NAMES),
            UNIFORM_BLOCK_SIZE=UNIFORM_BLOCK_SIZE,
        ),
        count_stats=stats,
        n_meta=n_meta,
        fl_min=fl.min_fl,
        fl_max=fl.max_fl,
        fl_n=int(fl.counts.sum()),
        npz=os.path.basename(npz_path),
        elapsed_s=round(time.time() - t0, 2),
    )

    def _write_json(fh):
        fh.write(json.dumps(meta, indent=2, default=str).encode())

    _atomic_write(json_path, _write_json)

    print(f"[done]    {meta['elapsed_s']}s, npz -> {npz_path}, json -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
