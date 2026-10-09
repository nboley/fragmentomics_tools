#!/usr/bin/env python
"""Run the cut-site simulator end to end: measure parameters, draw, build an h5.

    python scripts/run_cut_site_simulator.py \
        --sample-id RD-56804 \
        --fragments-h5 /efs/.../RD-56804-Lib1.hg38.fragments.h5 \
        --region-bed  <main checkout>/data/region_sets/quiet_v2_pad1200_repeats_removed_tile1536.bed \
        --fasta       /efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz \
        --n-regions 10 --seed 1234 --out-dir /efs/.../sim_out

`docs/pending/simulator_spec.md` is the authority on what this computes. This
script is only the wiring; it holds no rules of its own.

Promoted from a throwaway `/tmp` script on 2026-10-07. The first end-to-end run
lived only in `/tmp`, so it was not reproducible from a checkout -- the same gap
as the module having no tests.


The four stages, and why the order is forced
--------------------------------------------
Stages 1, 2 and 4 of the spec share ONE pass over the h5, which is why
``count_sample`` returns the frame it built rather than discarding it:

    srdf ---> count_srdf            ---> C(h), region_counts   (stages 1, 4)
         \--> FragmentLengthDist.from_srdf ---> f(L)           (stage 2)
                             f(L) ---> uniform_hexamer_counts ---> N(h)  (stage 3)
                         C(h), N(h) ---> propensities ---> r(h)

**f(L) must exist before N(h).** ``uniform_hexamer_counts`` takes ``f(L)`` as an
argument because its end-position expectation is f(L)-weighted. So stage 3
cannot run in parallel with stage 2.

**``uniform_hexamer_counts`` takes the rdf, NOT the srdf.** It refuses a frame
carrying fragment arrays, because the uniform counts are sequence-only and
region-set-scoped. That is why both are passed around below.


Costs worth knowing before running at scale
-------------------------------------------
- ``build_fragments_h5`` reads the reference for every 10 Mbp chunk of every
  contig holding at least one fragment, to compute per-fragment GC, so its cost
  scales with genome covered, not with fragment count. **It runs serially
  unless given** ``num_processes`` (its Pool is used only for values other than
  None and 1), so this script forwards ``--n-workers`` -- resolving None to
  every CPU, the same rule ``parallel_apply`` uses for the counting step.
  Measured on 322 fragments on chr1, warm cache: serial 54.4 s, 16 processes
  12.4 s. The 1 m 47 s once quoted here was a cold, serial build.
- ``pysam.tabix_index`` CONSUMES the plain BED. After it returns, only
  ``.bed.gz`` and ``.bed.gz.tbi`` remain. Do not plan to re-read or hash the
  plain file afterwards.
- ``--n-workers`` reaches every parallel stage: the fragment pass, N(h), the
  draw and the h5 build. The first full run (66,649 regions, 905 s) spent
  ~11.5 min in N(h) and the draw while both were still single-process (owner
  decision 166 parallelised them). Per-stage wall-clock times are printed and
  recorded in ``run.json`` under ``stage_seconds``.
- **The worker count does not change the output.** Same seed, any
  ``--n-workers``: byte-identical BED, sidecar and stats. Each region draws
  from ``default_rng([seed, region_index])`` and every reduction is grouped
  independently of the worker count; see ``region_rng``. What DOES change the
  draw is ``--n-regions``: r(h) and f(L) are re-estimated from the regions
  kept, so a k-region run is not a subset of the full run.


Conventions this script does not get to choose
----------------------------------------------
- ``p_plus = 0.5`` by construction, not by fitting (owner, 2026-10-07). A
  fragment is double-stranded and has no intrinsic orientation; the strand label
  records which of its two ends became read 1, and adapter ligation is
  symmetric. There is nothing to measure. Exposed as a flag only so a
  deliberate sensitivity check is possible.
- ``--seed`` is REQUIRED, and must lie in ``[0, 2**32)``. The seed is the
  output's only provenance, and an unseeded run cannot be reproduced.
  ``simulate_fragments_to_bed`` likewise refuses to run without one.
- Write output to ``/efs``, not ``/home``. ``/home`` writability in the batch
  container is disputed and the standing practice is to treat it as read-only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import sys
import time

import numpy as np

# scripts/ is not a package and the repo is not pip-installed, so the repo root
# has to go on sys.path before the first-party imports. Without this the script
# only runs from the repo root with PYTHONPATH already set.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fragmentomics_tools.dataframe import RegionDataFrame  # noqa: E402

from background_model.cut_site_stats import (  # noqa: E402
    FragmentLengthDist,
    TABLE_NAMES,
    count_sample,
    propensities,
    uniform_hexamer_counts,
)
from background_model.simulator.draw import (  # noqa: E402
    simulate_fragments_to_bed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample-id", required=True,
                   help="sample label, recorded in the run metadata")
    p.add_argument("--fragments-h5", required=True,
                   help="REAL fragments h5 -- the source of C(h), f(L) and the "
                        "per-region counts")
    p.add_argument("--region-bed", required=True,
                   help="region set. Note these live in the MAIN checkout's "
                        "gitignored data/region_sets/, not in a worktree")
    p.add_argument("--fasta", required=True,
                   help="reference FASTA, bgzipped and faidx-indexed")
    p.add_argument("--out-dir", required=True,
                   help="output directory; prefer /efs, not /home")
    p.add_argument("--seed", type=int, required=True,
                   help="REQUIRED. The output's only provenance")
    p.add_argument("--n-regions", type=int, default=None,
                   help="take only the first N regions (smoke runs)")
    p.add_argument("--ref", default="hg38", help="reference name for from_bed")
    p.add_argument("--min-mapq", type=int, default=10)
    p.add_argument("--p-plus", type=float, default=0.5,
                   help="0.5 by construction; see the module docstring before "
                        "changing it")
    p.add_argument("--n-workers", type=int, default=None,
                   help="processes for every parallel stage; default every "
                        "CPU, 1 runs in-process. Does not change the output")
    p.add_argument("--skip-h5", action="store_true",
                   help="stop after the BED. Useful because build_fragments_h5 "
                        "walks the whole contig and dominates a small run")
    return p.parse_args(argv)


@contextlib.contextmanager
def _stage(name, seconds):
    """Announce a stage, then record its wall-clock time in ``seconds``.

    The start line is the point: a long stage otherwise leaves the log silent
    for minutes, indistinguishable from a hang.
    """
    print(f"[{name}] start")
    t = time.perf_counter()
    yield
    seconds[name] = round(time.perf_counter() - t, 2)
    print(f"[{name}] {seconds[name]} s")


def main(argv=None):
    # Line-buffered even when redirected to a file: a block-buffered log sits
    # empty through a long stage and looks frozen.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.sample_id)
    # The count actually used, for run.json and the h5 build. None means every
    # CPU, which is how parallel_apply resolves it in each parallel stage.
    n_workers = (args.n_workers if args.n_workers is not None
                 else multiprocessing.cpu_count())
    t0 = time.time()
    seconds = {}

    with _stage("regions", seconds):
        rdf = RegionDataFrame.from_bed(args.region_bed, ref=args.ref)
        if args.n_regions is not None:
            rdf = rdf.iloc[: args.n_regions]
    print(f"[regions] {len(rdf)} from {os.path.basename(args.region_bed)}")

    # Stages 1, 2 and 4 -- one pass. The frame is returned, not discarded,
    # because f(L), the padded sequences and the region coordinates all live on
    # it and rebuilding it costs a second fetch plus a second FASTA walk.
    with _stage("count_sample", seconds):
        C, region_counts, stats, srdf = count_sample(
            rdf, args.sample_id, args.fragments_h5, args.fasta,
            min_mapq=args.min_mapq, n_workers=args.n_workers, verbose=False,
        )
    print(f"[C(h)]    {stats}")
    print(f"[counts]  per-region min/median/max = "
          f"{region_counts.min()}/{int(np.median(region_counts))}/{region_counts.max()}")

    with _stage("f(L)", seconds):
        fl = FragmentLengthDist.from_srdf(srdf)
    print(f"[f(L)]    support [{fl.min_fl}, {fl.max_fl}], "
          f"n={int(fl.counts.sum())}")

    # Stage 3. Takes the RDF, not the srdf, and needs f(L) to already exist.
    with _stage("N(h)", seconds):
        N, n_meta = uniform_hexamer_counts(
            rdf, args.fasta, fl, n_workers=args.n_workers, verbose=False,
        )
    print(f"[N(h)]    {n_meta}")

    r = propensities(C, N)
    for name in TABLE_NAMES:
        nz = int((r[name] > 0).sum())
        print(f"[r(h)]    {name:<10} nonzero {nz:4d}/4096  max {r[name].max():.4f}")

    bed_path = f"{stem}.bed"
    # seed and sample_id go into the p sidecar's header: the seed is the
    # output's only provenance, so a sidecar separated from run.json must still
    # say which run produced it.
    with _stage("draw", seconds):
        emit = simulate_fragments_to_bed(
            srdf, bed_path, r=r, fl=fl, region_counts=region_counts,
            seed=args.seed, p_plus=args.p_plus, sample_id=args.sample_id,
            n_workers=args.n_workers,
        )
    print(f"[emit]    {emit}")
    # No short-draw branch: dead starts and duplicates are redrawn, and an
    # infeasible request raises inside sample_region, so n_drawn == n_requested
    # whenever this line is reached.

    run = dict(
        sample_id=args.sample_id, seed=args.seed,
        # Seeds before decision 166 used one shared stream, so the same seed
        # gives different fragments across that change; this records which.
        rng_scheme="default_rng([seed, region_index])",
        n_workers=n_workers, n_regions=len(rdf),
        region_bed=os.path.abspath(args.region_bed),
        fasta=os.path.abspath(args.fasta),
        fragments_h5=os.path.abspath(args.fragments_h5),
        min_mapq=args.min_mapq, p_plus=args.p_plus,
        count_stats=stats, emit_stats=emit, n_meta=n_meta,
        fl_min=fl.min_fl, fl_max=fl.max_fl, fl_n=int(fl.counts.sum()),
    )

    if not args.skip_h5:
        import pysam
        from fragments_h5.fragments_h5 import FragmentsH5, build_fragments_h5

        # CONSUMES bed_path -- only .bed.gz and .bed.gz.tbi survive this call.
        gz = pysam.tabix_index(bed_path, preset="bed", force=True)
        out_h5 = f"{stem}.fragments.h5"
        if os.path.exists(out_h5):
            os.remove(out_h5)
        print(f"[h5]      building with {n_workers} process(es)")
        with _stage("h5", seconds):
            build_fragments_h5(gz, out_h5, fasta_filename=args.fasta,
                               num_processes=n_workers)

        with FragmentsH5(out_h5) as f:
            total = int(f.fragment_length_counts.sum())
        print(f"[h5]      {out_h5} holds {total} fragments "
              f"(wrote {emit['n_rows_written']})")
        run.update(fragments_h5_out=out_h5, h5_total_fragments=total,
                   bed_gz=gz)
        # The ingest does NOT dedup (an earlier version of this comment said it
        # did; it never has). Duplicates are redrawn at emission instead, so the
        # h5 should hold exactly the rows written. A mismatch is unexpected.
        if total != emit["n_rows_written"]:
            print(f"[h5]      WARNING h5 holds {total} but {emit['n_rows_written']} "
                  f"rows were written -- unexpected; duplicates are redrawn at "
                  f"emission and the ingest does not dedup")

    run["stage_seconds"] = seconds
    run["elapsed_s"] = round(time.time() - t0, 1)
    meta_path = f"{stem}.run.json"
    with open(meta_path, "w") as fh:
        json.dump(run, fh, indent=2, default=str)
    print(f"[stages]  {seconds}")
    print(f"[done]    {run['elapsed_s']}s, metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
