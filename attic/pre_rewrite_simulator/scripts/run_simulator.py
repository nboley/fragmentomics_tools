#!/usr/bin/env python
"""End-to-end driver for the cfDNA fragment simulator (Steps 1-6).

Wires together the pieces built in Phases 1-3 of
``docs/pending/simulator_and_fragment_nll.md``:

  Step 1-2  ``simulator.capture.fit_and_build``      -> predict_lut, marginal_fl
  Step 3    ``simulator.precompute.precompute_region`` -> hex_fwd/hex_rc/cum_gc/valid
  Step 4    ``simulator.weights.build_region_weights`` -> normalised w over Omega
  Step 5    ``simulator.sampler.draw_fragments_for_region``
  Step 6    ``simulator.emit`` -> sorted/bgzipped/tabixed BED, fragment h5, manifest

**Scope (owner decision 79).** The deliverable is the per-sample fragment ``h5``
plus the manifest needed to reconstruct the weights.  This driver does NOT build
a zarr store, choose a scoring domain ``D``, or compute anchors / ``W_D`` /
``% captured`` — those belong to the model agent.

The driver also verifies, and reports the measured value of, the invariants the
output is only useful if it satisfies:

  1. ``Sum_Omega w = 1`` per region, each strand marginal exactly 1/2.
  2. ``generative_domain_size(region_len)`` equals the number of non-zero weights.
  3. The fragment h5 is NON-EMPTY when read back at ``min_mapq=10``
     (the ``-1 >= 10`` trap: an unknown MAPQ reads back as -1 and every
     fragment is silently filtered).
  4. ``w`` reconstructed from the MANIFEST ALONE matches the original.
  5. The emitted fragment count matches ``target x n_regions``.
  6. Both strands and both ``L`` parities survive into the h5.

Environment
-----------
Run under ``/home/nathanboley/miniconda3/envs/biomarker_env/bin/python`` with
``PYTHONPATH=/home/nathanboley/src/biomarker`` on the path — ``flgc.model`` is a
RUNTIME dependency of Step 1.  ``bgzip`` / ``tabix`` / ``build-fragments-h5``
must be on ``PATH``; they ship in that env's ``bin/``.

Example
-------
    export PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH
    PYTHONPATH=/home/nathanboley/src/biomarker \\
      python scripts/run_simulator.py --n-regions 300 \\
        --out-dir /efs/analytics/nathanboley/background_model/sim_smoke
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time

import numpy as np
from numpy.random import SeedSequence

# Pin THIS FILE's repo at sys.path[0] rather than appending. `python
# scripts/run_simulator.py` puts scripts/ on sys.path[0], so `background_model`
# is not importable at all; and running the script from another directory (a
# /tmp scratch dir, say) would otherwise let a DIFFERENT checkout win the
# import. Both failures are silent-ish and have cost time on this project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.simulator.capture import fit_and_build  # noqa: E402
from background_model.simulator.emit import (  # noqa: E402
    _hash_file,  # the repo's own sha256 helper; duplicating it here would be a
                 # second implementation of the provenance hash.
    build_fragments_h5,
    load_manifest,
    sort_bgzip_tabix,
    write_bed,
    write_manifest,
)
from background_model.simulator.precompute import precompute_region  # noqa: E402
from background_model.simulator.sampler import (  # noqa: E402
    draw_fragments_for_region,
    target_count_for_region,
)
from background_model.simulator.weights import (  # noqa: E402
    L_MAX,
    L_MIN,
    MAX_FL_HALF,
    NHEX,
    HexamerTables,
    build_region_weights,
    generative_domain_size,
)

DEFAULT_REGION_SET = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "quiet_v2_pad1200_repeats_removed_tile2560.bed"
)
DEFAULT_FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
DEFAULT_OUT_DIR = "/efs/analytics/nathanboley/background_model/sim_smoke"
DEFAULT_SAMPLE = "RD-56670"


def build_hexamer_tables(seed: int, dynamic_range: float) -> HexamerTables:
    """Four synthetic log-normal hexamer tables (Layer 1).

    ``dynamic_range`` is the target p95/p5 spread, so
    ``sigma = log(dynamic_range) / (2 * z_0.95)``.  This mirrors
    ``scripts/sim_fragments.py::build_w6``, which is not importable (the
    scripts directory is not a package and ``sim_fragments`` pulls in the whole
    old simulator at import time).  The divergence is harmless here because the
    manifest stores the REALISED tables, not the recipe (design doc, Step 6:
    "store the realised tables, not the recipe") — nothing downstream depends
    on this function being reproducible.

    The four tables are drawn from ONE generator in sequence, so they are
    independent of each other: the tables are untied by design.
    """
    rng = np.random.default_rng(seed)
    sigma = np.log(dynamic_range) / (2 * 1.6448536269514722)  # z_0.95
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0.0, sigma, size=NHEX))
        tables.append(w / w.max())
    return HexamerTables(*tables)


def load_regions(bed_path: str, n_regions: int | None):
    """Load the region set through the sanctioned entry point.

    ``RegionDataFrame.from_bed`` is what ``CLAUDE.md`` mandates for BED loading;
    hand-rolling the parse is the divergence trap it exists to prevent.
    """
    from fragmentomics_tools.dataframe import RegionDataFrame

    rdf = RegionDataFrame.from_bed(bed_path, ref="hg38")
    if n_regions is not None:
        rdf = rdf.iloc[:n_regions]
    return rdf


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB (ru_maxrss is KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def git_commit_sha(repo_dir: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ── multiprocessing worker ────────────────────────────────────────────

_worker_state = None  # set by _init_worker


class _WorkerState:
    """Per-worker mutable state, initialised once per process."""
    __slots__ = ('fasta', 'hex_tables', 'marginal_fl', 'predict_lut',
                 'region_len', 'target', 'expected_domain')


def _init_worker(fasta_path, hex_tables, marginal_fl, predict_lut,
                 region_len, target, expected_domain):
    """Pool initializer: open a fasta handle and cache shared arrays."""
    global _worker_state
    import pysam
    ws = _WorkerState()
    ws.fasta = pysam.FastaFile(fasta_path)
    ws.hex_tables = hex_tables
    ws.marginal_fl = marginal_fl
    ws.predict_lut = predict_lut
    ws.region_len = region_len
    ws.target = target
    ws.expected_domain = expected_domain
    _worker_state = ws


def _process_one_region(task):
    """Precompute, build weights, check invariants, draw fragments.

    Called from Pool.imap (multiprocessing) or directly (single-process).
    Returns a dict with fragment arrays and invariant statistics.
    """
    contig, gstart, gstop, child_seed, keep_w = task
    ws = _worker_state

    rng = np.random.default_rng(child_seed)

    pc = precompute_region(contig, gstart, gstop, "", fasta=ws.fasta,
                           pad=MAX_FL_HALF)

    rw = build_region_weights(
        hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
        hex_tables=ws.hex_tables, marginal_fl=ws.marginal_fl,
        predict_lut=ws.predict_lut, region_len=ws.region_len, valid=pc.valid,
        pad=MAX_FL_HALF,
    )

    # Invariant stats
    s_plus = float(rw.w_plus.sum())
    s_minus = float(rw.w_minus.sum())

    starts, stops, strands = draw_fragments_for_region(
        hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
        valid=pc.valid, hex_tables=ws.hex_tables, marginal_fl=ws.marginal_fl,
        predict_lut=ws.predict_lut, region_len=ws.region_len,
        n_fragments=ws.target, rng=rng, region_weights=rw,
        pad=MAX_FL_HALF,
    )

    result = {
        'contig': contig, 'gstart': gstart, 'gstop': gstop,
        'starts': starts, 'stops': stops, 'strands': strands,
        'total_err': abs(1.0 - (s_plus + s_minus)),
        'plus_err': abs(0.5 - s_plus),
        'minus_err': abs(0.5 - s_minus),
        'n_nonzero': int(np.count_nonzero(rw.w_plus) + np.count_nonzero(rw.w_minus)),
        'model_L': rw.w_plus.sum(axis=0) + rw.w_minus.sum(axis=0),
    }

    if keep_w:
        result['pc'] = pc
        result['w_plus'] = rw.w_plus.copy()
        result['w_minus'] = rw.w_minus.copy()

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region-set-bed", default=DEFAULT_REGION_SET)
    ap.add_argument(
        "--region-set-name", default=None,
        help="Defaults to the BED's basename without the .bed suffix.",
    )
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--sample", default=DEFAULT_SAMPLE)
    ap.add_argument("--n-regions", type=int, default=300)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument(
        "--w6-seed", type=int, default=42,
        help="Seed for the synthetic Layer 1 hexamer tables.",
    )
    ap.add_argument("--w6-dynamic-range", type=float, default=4.0)
    ap.add_argument(
        "--n-roundtrip", type=int, default=5,
        help="How many regions to retain w for, to check the manifest round trip.",
    )
    ap.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes.  Default: cpu_count().  "
             "Use 1 to disable multiprocessing.",
    )
    args = ap.parse_args()
    if args.workers is None:
        args.workers = os.cpu_count() or 1

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    region_set_name = args.region_set_name or os.path.basename(
        args.region_set_bed
    ).removesuffix(".bed")

    print(f"driver      : {os.path.abspath(__file__)}")
    print(f"repo        : {repo_dir}")
    print(f"commit      : {git_commit_sha(repo_dir)}")
    print(f"python      : {sys.executable}")
    print(f"out dir     : {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── regions ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    rdf = load_regions(args.region_set_bed, args.n_regions)
    region_lens = (rdf.stop - rdf.start).unique()
    if len(region_lens) != 1:
        raise SystemExit(
            f"Region set has non-uniform lengths {sorted(region_lens)[:5]}; "
            f"the Layer 1 target count is defined per region_len."
        )
    region_len = int(region_lens[0])
    n_regions = len(rdf)
    target = target_count_for_region(region_len)
    print(
        f"regions     : {n_regions} of {region_set_name}, region_len={region_len}, "
        f"target={target}/region  ({time.perf_counter() - t0:.1f}s to load)"
    )

    # ── provenance hashes ────────────────────────────────────────────────
    t0 = time.perf_counter()
    region_set_hash = _hash_file(args.region_set_bed)
    t_rs = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference_hash = _hash_file(args.fasta)
    t_ref = time.perf_counter() - t0
    print(f"hashes      : region set {t_rs:.1f}s, reference {t_ref:.1f}s")

    # ── Steps 1-2: capture surface + marginal_fl ─────────────────────────
    t0 = time.perf_counter()
    gcfl_path = os.path.join(args.out_dir, f"{args.sample}.gcfl_model.json")
    predict_lut, marginal_fl = fit_and_build(args.sample, save_path=gcfl_path)
    t_fit = time.perf_counter() - t0
    print(
        f"capture fit : {t_fit:.1f}s  predict_lut{predict_lut.shape} "
        f"[{predict_lut.min():.4f}, {predict_lut.max():.4f}]  "
        f"marginal_fl sum={marginal_fl.sum():.12f}"
    )

    hex_tables = build_hexamer_tables(args.w6_seed, args.w6_dynamic_range)

    # ── Steps 3-5: per region (parallel, per-region seeding) ────────────
    bed_path = os.path.join(args.out_dir, f"{args.sample}.bed")
    if os.path.exists(bed_path):
        os.remove(bed_path)  # write_bed appends

    expected_domain = generative_domain_size(region_len)

    # Per-region seeding: each region gets an independent child seed derived
    # from the master seed.  This makes the output independent of the number
    # of workers and their assignment — a property worth having in its own
    # right, and the precondition for both parallelism and vectorisation.
    # Outputs are NOT bit-identical to the old single-stream code (commit
    # c7848e5) but the distribution is unchanged: each fragment is drawn
    # independently from P(s) · P(c5|s) · P(L|c5,s).
    master_ss = SeedSequence(args.seed)
    child_seeds = master_ss.spawn(n_regions)

    tasks = []
    for i, row in enumerate(rdf.itertuples(index=False)):
        tasks.append((
            row.contig, int(row.start), int(row.stop),
            child_seeds[i],
            i < args.n_roundtrip,   # keep_w flag
        ))

    n_workers = min(args.workers, n_regions)
    init_args = (args.fasta, hex_tables, marginal_fl, predict_lut,
                 region_len, target, expected_domain)
    print(f"workers     : {n_workers}")

    worst_total_err = 0.0
    worst_plus_err = 0.0
    worst_minus_err = 0.0
    worst_total_region = "every region exact"
    min_nonzero = None
    min_nonzero_region = None
    n_regions_domain_exact = 0
    kept_w = []  # (region_key, precompute, w_plus, w_minus) for the round trip
    # Exact model-implied length marginal, accumulated over every region.
    model_L = np.zeros(L_MAX - L_MIN + 1, dtype=np.float64)

    t_loop0 = time.perf_counter()
    per_region_counts = {}
    regions = []

    if n_workers > 1:
        pool = mp.Pool(n_workers, initializer=_init_worker, initargs=init_args)
        results_iter = pool.imap(_process_one_region, tasks, chunksize=64)
    else:
        _init_worker(*init_args)
        results_iter = map(_process_one_region, tasks)

    for i, result in enumerate(results_iter):
        contig = result['contig']
        gstart = result['gstart']
        gstop = result['gstop']
        key = f"{contig}:{gstart}-{gstop}"
        regions.append((contig, gstart, gstop))

        write_bed(bed_path, contig, gstart,
                  result['starts'], result['stops'], result['strands'])
        per_region_counts[key] = int(len(result['starts']))

        total_err = result['total_err']
        if total_err > worst_total_err:
            worst_total_err, worst_total_region = total_err, key
        worst_plus_err = max(worst_plus_err, result['plus_err'])
        worst_minus_err = max(worst_minus_err, result['minus_err'])

        n_nonzero = result['n_nonzero']
        if n_nonzero == expected_domain:
            n_regions_domain_exact += 1
        if min_nonzero is None or n_nonzero < min_nonzero:
            min_nonzero, min_nonzero_region = n_nonzero, key

        model_L += result['model_L']

        if 'w_plus' in result:
            kept_w.append((key, result['pc'], result['w_plus'], result['w_minus']))

        if (i + 1) % 500 == 0:
            el = time.perf_counter() - t_loop0
            print(
                f"  region {i + 1}/{n_regions}  {el:.1f}s elapsed  "
                f"{1000 * el / (i + 1):.1f} ms/region"
            )

    if n_workers > 1:
        pool.close()
        pool.join()
    else:
        _worker_state.fasta.close()

    t_loop = time.perf_counter() - t_loop0
    n_emitted = sum(per_region_counts.values())
    print(
        f"loop        : {t_loop:.1f}s total, {1000 * t_loop / n_regions:.1f} ms/region"
    )
    print(f"emitted     : {n_emitted} fragments into {bed_path}")

    # ── Step 6: sort/bgzip/tabix, h5, manifest ───────────────────────────
    t0 = time.perf_counter()
    bgz_path = sort_bgzip_tabix(bed_path)
    t_bgz = time.perf_counter() - t0

    h5_path = os.path.join(args.out_dir, f"{args.sample}.fragments.h5")
    if os.path.exists(h5_path):
        os.remove(h5_path)
    t0 = time.perf_counter()
    build_fragments_h5(bgz_path, h5_path, args.fasta)
    t_h5 = time.perf_counter() - t0

    manifest_path = os.path.join(args.out_dir, f"{args.sample}.manifest.json")
    t0 = time.perf_counter()
    write_manifest(
        manifest_path,
        hex_tables=hex_tables,
        predict_lut=predict_lut,
        marginal_fl=marginal_fl,
        region_set_name=region_set_name,
        region_set_hash=region_set_hash,
        reference_name=os.path.basename(args.fasta),
        reference_hash=reference_hash,
        region_len=region_len,
        # Recorded so the round trip reconstructs w from the MANIFEST rather
        # than from this module's MAX_FL_HALF. Previously pad was supplied from
        # the constant on BOTH sides of the round trip, so that check agreed by
        # construction instead of by verification.
        pad=MAX_FL_HALF,
        per_region_counts=per_region_counts,
        rng_seed=args.seed,
        commit_sha=git_commit_sha(repo_dir),
        # Names THIS file so load_manifest can check that the code which drew
        # the fragments is the code present at load time. Resolves to a git
        # blob sha only while this driver is committed; an uncommitted driver
        # records None and the check is skipped rather than silently passed.
        simulator_script=os.path.abspath(__file__),
    )
    t_manifest = time.perf_counter() - t0
    print(
        f"emit        : bgzip+tabix {t_bgz:.1f}s, build-fragments-h5 {t_h5:.1f}s, "
        f"manifest {t_manifest:.1f}s"
    )

    sizes = {
        "bed.gz": os.path.getsize(bgz_path),
        "bed.gz.tbi": os.path.getsize(bgz_path + ".tbi"),
        "fragments.h5": os.path.getsize(h5_path),
        "manifest.json": os.path.getsize(manifest_path),
        "gcfl_model.json": os.path.getsize(gcfl_path),
    }

    # ── read back through production ─────────────────────────────────────
    from fragments_h5 import FragmentsH5
    from fragmentomics_tools.fragment_array import RegionFragmentArray
    from fragmentomics_tools.region import Region

    t0 = time.perf_counter()
    n_readback = 0
    n_contained = 0
    L_hist = np.zeros(L_MAX + 2, dtype=np.int64)
    n_plus = n_minus = 0
    with FragmentsH5(h5_path, cache_pointers=False) as fh5:
        for contig, gstart, gstop in regions:
            region = Region(contig, gstart, gstop, strand=None)
            rfa = RegionFragmentArray.from_fragments_h5(fh5, region, min_mapq=10)
            n_readback += rfa.n_frags
            # Tiles are contiguous, so an overlap-based fetch can return a
            # neighbour's fragments.  Regions are disjoint and every emitted
            # fragment lies wholly inside its own region, so containment
            # counts each fragment exactly once.
            g_starts = gstart + rfa.starts_0
            g_stops = gstart + rfa.stops_0
            inside = (g_starts >= gstart) & (g_stops <= gstop)
            n_contained += int(inside.sum())
            lengths = rfa.lengths[inside]
            L_hist += np.bincount(lengths, minlength=L_MAX + 2)[: L_MAX + 2]
            strands_rb = np.asarray(rfa.fragment_strands)[inside]
            n_plus += int((strands_rb == "+").sum())
            n_minus += int((strands_rb == "-").sum())
    t_readback = time.perf_counter() - t0

    # ── manifest round trip: rebuild w from the manifest ALONE ───────────
    # Paths are passed so the provenance checks actually RUN; load_manifest
    # raises ManifestVerificationIncomplete if a mandatory check (reference,
    # region_set) cannot run because its path is absent.
    loaded = load_manifest(
        manifest_path,
        fasta_path=args.fasta,
        region_set_path=args.region_set_bed,
    )
    worst_rt = 0.0
    worst_rt_region = "every region bit-identical"
    for key, pc, w_plus, w_minus in kept_w:
        rw2 = build_region_weights(
            hex_fwd=pc.hex_fwd,
            hex_rc=pc.hex_rc,
            cum_gc=pc.cum_gc,
            hex_tables=loaded["hex_tables"],
            marginal_fl=loaded["marginal_fl"],
            predict_lut=loaded["predict_lut"],
            region_len=loaded["region_len"],
            valid=pc.valid,
            # From the MANIFEST, not MAX_FL_HALF. Taking it from the constant
            # made this round trip agree by construction: both sides used the
            # same in-process value, so a manifest that recorded the wrong
            # geometry -- or none at all -- would still round-trip perfectly.
            pad=loaded["pad"],
        )
        d = max(
            float(np.abs(rw2.w_plus - w_plus).max()),
            float(np.abs(rw2.w_minus - w_minus).max()),
        )
        if d > worst_rt:
            worst_rt, worst_rt_region = d, key

    # ── report ───────────────────────────────────────────────────────────
    L_vals = np.arange(L_MIN, L_MAX + 1)
    realised = L_hist[L_MIN : L_MAX + 1].astype(np.float64)
    realised_p = realised / realised.sum() if realised.sum() else realised
    tv = 0.5 * float(np.abs(realised_p - marginal_fl).sum())
    model_L_p = model_L / model_L.sum()
    tv_model = 0.5 * float(np.abs(realised_p - model_L_p).sum())
    mean_realised = float((L_vals * realised_p).sum())
    mean_marginal = float((L_vals * marginal_fl).sum())
    mean_model = float((L_vals * model_L_p).sum())
    odd = L_vals % 2 == 1
    n_odd = int(realised[odd].sum())
    n_even = int(realised[~odd].sum())
    out_of_band = int(L_hist[L_MAX])  # L = 180 is drawn but out of FL_BANDS

    print()
    print("=" * 72)
    print("ASSERTIONS (measured values)")
    print("=" * 72)
    print(
        f"1. worst |1 - sum w|             = {worst_total_err:.3e}  ({worst_total_region})\n"
        f"   worst |0.5 - sum w_plus|      = {worst_plus_err:.3e}\n"
        f"   worst |0.5 - sum w_minus|     = {worst_minus_err:.3e}"
    )
    print(
        f"2. generative_domain_size({region_len})  = {expected_domain}\n"
        f"   min non-zero weight count     = {min_nonzero}  ({min_nonzero_region})\n"
        f"   regions with exact match      = {n_regions_domain_exact}/{n_regions}"
    )
    print(
        f"3. h5 readback at min_mapq=10    = {n_readback} fragments "
        f"(overlap), {n_contained} contained"
    )
    print(
        f"4. worst |w_manifest - w|        = {worst_rt:.3e} over "
        f"{len(kept_w)} regions ({worst_rt_region})\n"
        # NB: an entry like "simulator_script:unresolvable" means that check
        # did NOT run (e.g. no git on Batch), not that it passed.
        f"   provenance checks that ran    = {loaded['verified'] or 'NONE'}"
    )
    print(
        f"5. emitted / expected            = {n_emitted} / {target * n_regions}\n"
        f"   contained in h5               = {n_contained}"
    )
    print(
        f"6. strands in h5                 = {n_plus} plus, {n_minus} minus\n"
        f"   L parities in h5              = {n_odd} odd, {n_even} even\n"
        f"   L == {L_MAX} (out of band)      = {out_of_band}"
    )
    print()
    print("MEASUREMENTS")
    print(
        f"  wall clock/region              = {1000 * t_loop / n_regions:.1f} ms  "
        f"({n_workers} workers)\n"
        f"  ESTIMATE for 66,649 regions    = "
        f"{66649 * t_loop / n_regions / 60:.1f} min"
    )
    for k, v in sizes.items():
        print(f"  size {k:<16}= {v / 1e6:.3f} MB")
    print(f"  h5 readback                    = {t_readback:.1f}s")
    print(f"  peak RSS                       = {peak_rss_mb():.0f} MiB")
    print(
        f"  realised FL vs marginal_fl     = TV {tv:.4f}, "
        f"mean {mean_realised:.2f} vs {mean_marginal:.2f}\n"
        f"  realised FL vs w's L-marginal  = TV {tv_model:.4f}, "
        f"mean {mean_realised:.2f} vs {mean_model:.2f}"
    )

    summary = {
        "sample": args.sample,
        "region_set_name": region_set_name,
        "n_regions": n_regions,
        "region_len": region_len,
        "target_per_region": target,
        "worst_abs_err_total_weight": worst_total_err,
        "worst_abs_err_plus_marginal": worst_plus_err,
        "worst_abs_err_minus_marginal": worst_minus_err,
        "generative_domain_size": expected_domain,
        "min_nonzero_weight_count": min_nonzero,
        "n_regions_domain_exact": n_regions_domain_exact,
        "n_emitted": n_emitted,
        "n_readback_overlap": n_readback,
        "n_readback_contained": n_contained,
        "worst_manifest_roundtrip_abs_diff": worst_rt,
        # An "…:unresolvable" entry means that check did NOT run.
        "provenance_checks_run": loaded["verified"],
        "n_plus": n_plus,
        "n_minus": n_minus,
        "n_odd_L": n_odd,
        "n_even_L": n_even,
        "n_L_equals_max": out_of_band,
        "ms_per_region": 1000 * t_loop / n_regions,
        "n_workers": n_workers,
        "seconds_capture_fit": t_fit,
        "seconds_bgzip_tabix": t_bgz,
        "seconds_build_fragments_h5": t_h5,
        "seconds_h5_readback": t_readback,
        "peak_rss_mib": peak_rss_mb(),
        "fl_total_variation_vs_marginal_fl": tv,
        "fl_total_variation_vs_w_L_marginal": tv_model,
        "fl_mean_realised": mean_realised,
        "fl_mean_marginal": mean_marginal,
        "fl_mean_w_L_marginal": mean_model,
        "sizes_bytes": sizes,
    }
    summary_path = os.path.join(args.out_dir, f"{args.sample}.smoke_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary written to {summary_path}")

    # ── hard failures ────────────────────────────────────────────────────
    failures = []
    if worst_total_err > 1e-12:
        failures.append(f"sum w deviates by {worst_total_err:.3e} (> 1e-12)")
    if max(worst_plus_err, worst_minus_err) > 1e-12:
        failures.append(
            f"strand marginal deviates by "
            f"{max(worst_plus_err, worst_minus_err):.3e} (> 1e-12)"
        )
    if n_contained == 0:
        failures.append(
            "h5 read back EMPTY at min_mapq=10 — the -1 >= 10 trap"
        )
    if worst_rt > 1e-12:
        failures.append(
            f"manifest round trip differs by {worst_rt:.3e} (> 1e-12)"
        )
    # Only assert the mandatory checks; "simulator_script:unresolvable"
    # may appear in verified and means that check did NOT run.
    for check in ("reference", "region_set"):
        if check not in loaded["verified"]:
            failures.append(
                f"manifest provenance check '{check}' did not run — the "
                f"hash is recorded but unenforced"
            )
    if n_emitted != target * n_regions:
        failures.append(f"emitted {n_emitted} != {target * n_regions}")
    if n_contained != n_emitted:
        failures.append(
            f"h5 contains {n_contained} fragments, emitted {n_emitted}"
        )
    if n_plus == 0 or n_minus == 0:
        failures.append(f"missing a strand: {n_plus} plus, {n_minus} minus")
    if n_odd == 0 or n_even == 0:
        failures.append(f"missing an L parity: {n_odd} odd, {n_even} even")

    if failures:
        print()
        print("FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print()
    print("All assertions hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
