#!/usr/bin/env python
"""Paired region-level delta between two sanctioned per-fragment TSVs (D4).

Both NLLs come from S2 (``reduce_split_nll.reduce_splits`` ->
``score_fragments.region_weighted_mult_nll``) with ``return_terms=True``, on
the same store, split and min_N, so the two term vectors cover the same
regions in the same order (checked).  ``delta = candidate - reference`` is the
difference of the two region-weighted NLLs; negative means the candidate is
better.  Two SEs of that delta:

  paired_se      sd(d) / sqrt(n), d = per-region term difference
  bootstrap_se   sd of mean(d) over B region resamples, fixed seed
                 (design §6.2: B=2000)

Spatially abutting tiles make both slightly anti-conservative (design §6.2).

Usage:
    python scripts/cut_site_model/baselines/paired_delta.py --store S.zarr \\
        --candidate-tsv N0.tsv --reference-tsv per_fragment_old_prior.tsv \\
        --splits val test --output N0_vs_old_prior.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

from scripts.cut_site_model.baselines.reduce_split_nll import (  # noqa: E402
    _sha256,
    reduce_splits,
)


def paired_delta(candidate_tsv, reference_tsv, store_path, splits, min_N=10,
                 n_boot=2000, seed=0):
    cand = reduce_splits(candidate_tsv, store_path, splits, min_N, return_terms=True)
    ref = reduce_splits(reference_tsv, store_path, splits, min_N, return_terms=True)
    out = {}
    for name in splits:
        c, r = cand[name], ref[name]
        if not np.array_equal(c["term_tiles"], r["term_tiles"]):
            raise AssertionError(f"{name}: the two TSVs reduce over different regions")
        d = c["terms"] - r["terms"]
        n = len(d)
        idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
        boot = d[idx].mean(axis=1, dtype=np.float64)
        out[name] = dict(
            candidate_nll=c["region_weighted_nll"],
            reference_nll=r["region_weighted_nll"],
            delta=c["region_weighted_nll"] - r["region_weighted_nll"],
            paired_se=float(np.std(d, ddof=1) / np.sqrt(n)),
            bootstrap_se=float(np.std(boot, ddof=1)),
            sd_region_diff=float(np.std(d, ddof=1)),
            frac_regions_candidate_better=float(np.mean(d < 0)),
            n_regions=n, n_boot=n_boot, seed=seed,
        )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True)
    p.add_argument("--candidate-tsv", required=True)
    p.add_argument("--reference-tsv", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--min-N", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    res = paired_delta(args.candidate_tsv, args.reference_tsv, args.store,
                       args.splits, args.min_N, args.n_boot, args.seed)
    doc = dict(
        store=os.path.abspath(args.store), min_N=args.min_N,
        candidate_tsv=os.path.abspath(args.candidate_tsv),
        candidate_tsv_sha256=_sha256(args.candidate_tsv),
        reference_tsv=os.path.abspath(args.reference_tsv),
        reference_tsv_sha256=_sha256(args.reference_tsv),
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        splits=res,
    )
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
