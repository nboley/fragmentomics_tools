#!/usr/bin/env python
"""Gate G0 of the baseline-priors design (Phase 0, at bad981d model code).

Reads S2 output (reduce_split_nll.py JSON) and checks, each to |d| < 1e-5:

  1. uniform: every scored fragment has log_prob = -ln|D|, so every region
     term is ln|D| = ln(tile_size * n_L * 2); and S2's mean is ln|D|.
  2. old prior (optional, real store only): S2 reproduces the RETIRED
     12.146835 (val) / 12.142214 (test).  This checks the REDUCTION; those
     numbers are not an anchor.
  3. unif+p_L: S2 equals the closed form.  Under constant hexamer tables
     log_softmax cancels them, so p(m, L, s) = p_L(L) / (tile_size * 2) and
     the region term is -mean(log p_L[L_f]) + ln(tile_size * 2).

The closed form takes fragment admission from CutSiteFragmentDataset
(train_mode=False), never from the TSV, so it does not share a population
with the value it checks.  It is a cross-check of ONE closed form, not an
evaluator: every reported NLL comes from score_fragments + S2.

1e-5, not 1e-6: the forward is float32, and the prior TSV's own normaliser
check already reads max |logsumexp| ~ 8e-7.

Exit status 1 if any check fails.
"""

import argparse
import json
import os
import sys

import numpy as np
import zarr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

from scripts.cut_site_model.baselines.reduce_split_nll import read_per_fragment  # noqa: E402
from background_model.cut_site_model.dataset import CutSiteFragmentDataset  # noqa: E402
from background_model.cut_site_model.store import CutSiteStoreConfig  # noqa: E402
from scripts.cut_site_model.score_fragments import SPLITS, load_model  # noqa: E402

TOL = 1e-5
# HISTORICAL: stage1 v4 tables at m+L-1, real store.  RETIRED by owner
# rulings 5-6; reproducible only at bad981d model code.
RETIRED_OLD_PRIOR = {"val": 12.146835, "test": 12.142214}


def closed_form_unif_pL(store_path, p_L_npz, split, min_N):
    root = zarr.open_group(store_path, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    model = load_model(cfg, p_L_npz, zero_correction=True)
    ds = CutSiteFragmentDataset(
        store_path=store_path, model_input_size=model.model_input_size,
        model_L_min=cfg.L_min, model_L_max=cfg.L_max, split=SPLITS[split][1],
        min_N=min_N, train_mode=False, preload=True,
    )
    log_p_L = np.log(np.maximum(np.asarray(np.load(p_L_npz)["p_L"],
                                           dtype=np.float64), 1e-30))
    log_strand_pos = np.log(float(cfg.tile_size * 2))
    terms = np.empty(len(ds), dtype=np.float64)
    n_frags = 0
    for k in range(len(ds)):
        idx = ds[k]["frag_L_idx"].numpy()
        n_frags += len(idx)
        # An empty region contributes 0 but counts, as in _step.
        terms[k] = 0.0 if len(idx) == 0 else (
            -log_p_L[idx].mean(dtype=np.float64) + log_strand_pos)
    return float(terms.mean(dtype=np.float64)), len(ds), n_frags


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True)
    p.add_argument("--uniform-tsv", required=True,
                   help="per-fragment TSV; its .nll.json must exist")
    p.add_argument("--unif-pL-tsv", required=True)
    p.add_argument("--p-L-from", required=True,
                   help=".npz whose p_L built the unif+p_L tables")
    p.add_argument("--old-prior-tsv", help="real store only (check 2)")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--min-N", type=int, default=10)
    args = p.parse_args(argv)

    root = zarr.open_group(args.store, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    ln_D = float(np.log(float(cfg.tile_size * (cfg.L_max - cfg.L_min + 1) * 2)))

    def s2(tsv):
        with open(tsv + ".nll.json") as f:
            doc = json.load(f)
        if os.path.abspath(doc["store"]) != os.path.abspath(args.store):
            raise ValueError(f"{tsv}: S2 json is for store {doc['store']}")
        return doc["splits"]

    checks = []

    def check(name, got, want):
        d = got - want
        checks.append((name, got, want, d, abs(d) < TOL))

    # 1. uniform, per fragment and per split.
    pf = read_per_fragment(args.uniform_tsv, args.store)
    uni = s2(args.uniform_tsv)
    tile_of_row = np.repeat(np.arange(len(pf["split_arr"])), np.diff(pf["indptr"]))
    for split in args.splits:
        rows = pf["scored"] & (pf["split_arr"][tile_of_row] == SPLITS[split][0])
        worst = float(np.max(np.abs(pf["log_prob"][rows] + ln_D)))
        checks.append((f"uniform {split}: max |log_prob + ln|D|| over "
                       f"{int(rows.sum())} fragments", worst, 0.0, worst, worst < TOL))
        check(f"uniform {split}: S2 mean", uni[split]["region_weighted_nll"], ln_D)

    # 2. old prior reproduces the RETIRED values (a reduction check).
    if args.old_prior_tsv:
        old = s2(args.old_prior_tsv)
        for split in args.splits:
            check(f"HISTORICAL old prior {split} (RETIRED value)",
                  old[split]["region_weighted_nll"], RETIRED_OLD_PRIOR[split])

    # 3. unif+p_L vs closed form.
    upl = s2(args.unif_pL_tsv)
    for split in args.splits:
        cf, n_reg, n_frags = closed_form_unif_pL(args.store, args.p_L_from,
                                                 split, args.min_N)
        if n_reg != upl[split]["n_regions"]:
            raise AssertionError(f"{split}: closed form has {n_reg} regions, "
                                 f"S2 {upl[split]['n_regions']}")
        check(f"unif+p_L {split}: S2 vs closed form ({n_reg} regions, "
              f"{n_frags} fragments)", upl[split]["region_weighted_nll"], cf)

    print(f"store {args.store}\nln|D| = {ln_D:.9f}\n")
    for name, got, want, d, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got:.9f} want "
              f"{want:.9f} delta {d:+.3e}")
    sys.exit(0 if all(c[-1] for c in checks) else 1)


if __name__ == "__main__":
    main()
