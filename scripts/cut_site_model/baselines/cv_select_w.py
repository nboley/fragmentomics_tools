#!/usr/bin/env python
"""Choose N0's shrinkage weight ``w`` by k-fold cross-validation on TRAIN (P0b.2).

Design: docs/pending/cut_site_baseline_priors_design.md §3.5, §5 (P0b.2), §6.2.
Hyperparameters are selected on train-CV ONLY; val confirms, test is read once.

Two subcommands:

  folds   partition the store's TRAIN tiles into k folds by region, fixed seed.
          Writes fold{i}_heldout.npy and fold{i}_fit.npy (store tile indices)
          and folds.json.  The fit set is the counting population of the
          other k-1 folds (every tile, min_N=0, as count_tables.py counts);
          the held-out set is scored, and reduced at the trainer's min_N.
  curve   reduce every (fold, w) TSV through S2 -- reduce_split_nll.reduce_splits,
          i.e. score_fragments.region_weighted_mult_nll over the held-out
          fold's regions -- and select w* = argmin of the CV score.

The CV score for a ``w`` is the n_regions-weighted mean of its k fold
values, which is the region-weighted NLL over every TRAIN region, each scored
under tables fitted without its own fold.  The paired SE of ``w`` against
``w*`` uses the per-region terms (``return_terms``) over the same regions.
Spatially abutting tiles make region-level CV slightly anti-conservative
(design §6.2).

Expected layout for ``curve``: ``<cv-dir>/fold{i}/w{w}.tsv``, each scored with
``score_fragments.py --splits train --tiles-npy <cv-dir>/fold{i}_heldout.npy``.

Usage:
    python scripts/cut_site_model/baselines/cv_select_w.py folds \\
        --store S.zarr --k 5 --seed 20261010 --outdir cv/
    python scripts/cut_site_model/baselines/cv_select_w.py curve \\
        --store S.zarr --cv-dir cv/ --grid 0 16 64 256 1024 4096 16384 \\
        --output cv/cv_curve.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import zarr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

from scripts.cut_site_model.baselines.reduce_split_nll import (  # noqa: E402
    _sha256,
    reduce_splits,
)
from scripts.cut_site_model.score_fragments import SPLITS  # noqa: E402


def make_folds(store_path, k, seed):
    """``[(fit_tiles, heldout_tiles)] * k`` over the store's TRAIN tiles.

    Every TRAIN tile lands in exactly one held-out set; each fit set is the
    complement within TRAIN.  Both sorted, as store tile indices.
    """
    split_arr = np.asarray(zarr.open_group(store_path, mode="r")["tiles/split"][:])
    train = np.nonzero(split_arr == SPLITS["train"][0])[0]
    if len(train) < k:
        raise ValueError(f"{len(train)} TRAIN tiles cannot make {k} folds")
    perm = np.random.default_rng(seed).permutation(train)
    heldout = [np.sort(perm[i::k]) for i in range(k)]
    return [(np.setdiff1d(train, h), h) for h in heldout]


def w_tag(w):
    """File-name tag for a grid value: 64.0 -> 'w64'."""
    return f"w{int(w)}" if float(w).is_integer() else f"w{w:g}"


def cv_curve(store_path, cv_dir, grid, k, min_N=10):
    """Per-fold and pooled CV NLL for each ``w``, plus the selection."""
    folds = [np.load(os.path.join(cv_dir, f"fold{i}_heldout.npy")) for i in range(k)]
    rows, terms = [], {}
    for w in grid:
        fold_nll, fold_n, fold_terms, fold_tiles, n_empty = [], [], [], [], 0
        for i, held in enumerate(folds):
            tsv = os.path.join(cv_dir, f"fold{i}", f"{w_tag(w)}.tsv")
            r = reduce_splits(tsv, store_path, ["train"], min_N, tiles=held,
                              return_terms=True)["train"]
            n_empty += r["n_regions_no_scored_fragment"]
            fold_nll.append(r["region_weighted_nll"])
            fold_n.append(r["n_regions"])
            fold_terms.append(r["terms"])
            fold_tiles.append(r["term_tiles"])
        n = np.asarray(fold_n, dtype=np.float64)
        cv = float(np.sum(np.asarray(fold_nll) * n, dtype=np.float64) / n.sum())
        terms[w] = (np.concatenate(fold_terms), np.concatenate(fold_tiles))
        rows.append(dict(w=float(w), cv_nll=cv, fold_nll=fold_nll,
                         fold_n_regions=fold_n, n_regions_no_scored_fragment=n_empty))

    # Pre-registered rule: w* = argmin of the CV score.  Nothing else.
    best = min(rows, key=lambda r: r["cv_nll"])
    w_star = best["w"]
    t_star, tiles_star = terms[w_star]
    for r in rows:
        t, tiles = terms[r["w"]]
        if not np.array_equal(tiles, tiles_star):
            raise AssertionError("fold region sets differ across w")
        d = t - t_star
        r["delta_vs_w_star"] = r["cv_nll"] - best["cv_nll"]
        r["paired_se_vs_w_star"] = float(np.std(d, ddof=1) / np.sqrt(len(d)))
        r["fold_delta_vs_w_star"] = [a - b for a, b in zip(r["fold_nll"], best["fold_nll"])]
    grid_sorted = sorted(float(w) for w in grid)
    return dict(
        rows=rows, w_star=w_star, cv_nll_at_w_star=best["cv_nll"],
        w_star_on_grid_boundary=w_star in (grid_sorted[0], grid_sorted[-1]),
        n_regions=int(sum(best["fold_n_regions"])), k=k, min_N=min_N,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--store", required=True)
    f.add_argument("--k", type=int, default=5)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--outdir", required=True)
    c = sub.add_parser("curve")
    c.add_argument("--store", required=True)
    c.add_argument("--cv-dir", required=True)
    c.add_argument("--grid", type=float, nargs="+", required=True)
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--min-N", type=int, default=10)
    c.add_argument("--output", required=True)
    c.add_argument("--w-star-file", default=None,
                   help="also write w* alone, for a shell driver to read")
    args = p.parse_args(argv)

    root = zarr.open_group(args.store, mode="r")
    if args.cmd == "folds":
        os.makedirs(args.outdir, exist_ok=True)
        meta = dict(store=os.path.abspath(args.store),
                    config_hash=str(root.attrs["config_hash"]),
                    k=args.k, seed=args.seed, folds=[])
        for i, (fit, held) in enumerate(make_folds(args.store, args.k, args.seed)):
            paths = {}
            for name, arr in (("fit", fit), ("heldout", held)):
                path = os.path.join(args.outdir, f"fold{i}_{name}.npy")
                np.save(path, arr.astype(np.int64))
                paths[name] = dict(path=os.path.abspath(path), n_tiles=len(arr),
                                   sha256=_sha256(path))
            meta["folds"].append(paths)
        with open(os.path.join(args.outdir, "folds.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(json.dumps(meta, indent=2))
        return

    out = cv_curve(args.store, args.cv_dir, args.grid, args.k, args.min_N)
    out.update(store=os.path.abspath(args.store),
               config_hash=str(root.attrs["config_hash"]),
               cv_dir=os.path.abspath(args.cv_dir),
               date=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    if args.w_star_file:
        with open(args.w_star_file, "w") as fh:
            fh.write(f"{out['w_star']:g}\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
