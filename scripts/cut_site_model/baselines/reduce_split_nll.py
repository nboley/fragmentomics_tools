#!/usr/bin/env python
"""Region-weighted NLL per split from a sanctioned per-fragment TSV (seam S2).

`score_fragments.py --reproduce-val-mult-nll` reduces only the VAL split
(it hardcodes split="val").  This wrapper closes that gap for any split,
test included, WITHOUT a second evaluator:

  * the per-fragment values are the scorer's own output (its TSV);
  * the region population is CutSiteFragmentDataset(split, min_N,
    train_mode=False), as the trainer builds it;
  * the reduction IS score_fragments.region_weighted_mult_nll, imported and
    called.  Nothing here restates it.

The TSV holds only the store rows of the tiles the scorer processed, in store
order.  They are placed back at their store row indices, and every row is
checked against the store's own (start, stop, strand, split) before use, so a
TSV from another store or a reordered TSV raises instead of mis-joining.

A split is reduced only if EVERY one of its tiles was processed.  A region
the scorer never visited has no scored rows, and region_weighted_mult_nll
counts such a region as 0 in the mean, so a partial TSV would silently
deflate the number.

Usage:
    python scripts/cut_site_model/baselines/reduce_split_nll.py \\
        --per-fragment-tsv out/per_fragment_prior.tsv --store S.zarr \\
        --splits val test --min-N 10 --output out/prior_nll.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import zarr

# Bootstrap: background_model is not an installed package
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

import background_model  # noqa: E402

# CWD trap: a background_model/ at the cwd shadows _REPO even with sys.path
# pinned (sys.path[0] == '' for `python -c`).  Refuse to run foreign code.
if not os.path.abspath(background_model.__file__).startswith(_REPO + os.sep):
    raise ImportError(
        f"background_model resolved to {background_model.__file__}, not under "
        f"{_REPO}; cd elsewhere (e.g. /tmp) and set PYTHONPATH={_REPO}"
    )

from background_model.cut_site_model.dataset import CutSiteFragmentDataset  # noqa: E402
from background_model.cut_site_model.model import CutSiteFragmentModel  # noqa: E402
from background_model.cut_site_model.store import CutSiteStoreConfig  # noqa: E402
from scripts.cut_site_model.score_fragments import (  # noqa: E402
    SPLITS,
    STRAND_CHARS,
    region_weighted_mult_nll,
    selected_tiles,
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_per_fragment(tsv_path: str, store_path: str) -> Dict[str, np.ndarray]:
    """Per-fragment TSV -> arrays over ALL store rows, as score_store returns.

    Returns `log_prob` (float64, NaN where not scored), `scored` (bool) and
    `processed_tiles` (bool per tile).
    """
    root = zarr.open_group(store_path, mode="r")
    indptr = np.asarray(root["fragments/indptr"][:])
    split_arr = np.asarray(root["tiles/split"][:])
    n_tiles = len(split_arr)
    if len(indptr) != n_tiles + 1:
        raise ValueError("assumes a single-sample store (S == 1)")
    nnz = int(indptr[-1])
    tile_of_row = np.repeat(np.arange(n_tiles), np.diff(indptr))

    df = pd.read_csv(tsv_path, sep="\t", dtype={"contig": str, "split": str,
                                                "strand": str, "scored": str})
    code_of = {name: code for name, (code, _) in SPLITS.items()}
    tsv_codes = df["split"].map(code_of)
    if tsv_codes.isna().any():
        raise ValueError(f"unknown split names in {tsv_path}")

    # The scorer writes every row of every processed tile; recover the tile
    # set with the scorer's own selection rule, then check the TSV is exactly
    # those rows.  Summaries written before --tiles-npy existed lack the key.
    with open(tsv_path + ".summary.json") as f:
        summary = json.load(f)
    tiles_npy = summary.get("tiles_npy")
    processed_tiles = selected_tiles(
        split_arr, summary["splits"],
        None if tiles_npy is None else np.load(tiles_npy),
        summary["limit_regions"],
    )
    rows = np.nonzero(processed_tiles[tile_of_row])[0]
    if len(rows) != len(df):
        raise ValueError(f"TSV has {len(df)} rows; the store's processed "
                         f"tiles hold {len(rows)}")

    tile_start = np.asarray(root["tiles/start"][:]).astype(np.int64)
    start = (tile_start[tile_of_row[rows]]
             + np.asarray(root["fragments/start"][:]).astype(np.int64)[rows])
    stop = start + np.asarray(root["fragments/length"][:]).astype(np.int64)[rows]
    strand = STRAND_CHARS[np.asarray(root["fragments/strand"][:])[rows]]
    if not (np.array_equal(df["start"].to_numpy(np.int64), start)
            and np.array_equal(df["stop"].to_numpy(np.int64), stop)
            and np.array_equal(df["strand"].to_numpy(), strand)
            and np.array_equal(tsv_codes.to_numpy(np.int64),
                               split_arr[tile_of_row[rows]].astype(np.int64))):
        raise ValueError("TSV rows do not match the store rows in store order")

    log_prob = np.full(nnz, np.nan, dtype=np.float64)
    scored = np.zeros(nnz, dtype=bool)
    log_prob[rows] = df["log_prob"].to_numpy(np.float64)
    scored[rows] = df["scored"].to_numpy() == "true"
    return {"log_prob": log_prob, "scored": scored,
            "processed_tiles": processed_tiles, "indptr": indptr,
            "split_arr": split_arr, "summary": summary}


def reduce_splits(
    tsv_path: str,
    store_path: str,
    splits: Sequence[str],
    min_N: int,
    tiles: Optional[Sequence[int]] = None,
    return_terms: bool = False,
) -> Dict[str, object]:
    """region_weighted_mult_nll for each split, from the scorer's TSV.

    `tiles` restricts each split's region population to those store tile
    indices -- a cross-validation fold's held-out regions -- and then only
    those tiles need to have been scored.  `return_terms` is passed through.
    """
    pf = read_per_fragment(tsv_path, store_path)
    root = zarr.open_group(store_path, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    # The model's input geometry depends only on tile_size, so a geometry-only
    # model gives it.  Not load_model on the TSV's own tables: the loader now
    # refuses pre-m+L tables, and the per-fragment values are already in the
    # TSV, so the tables are never read here -- loading them would only make
    # the Phase-0 old-prior TSV unreducible.
    model_input_size = CutSiteFragmentModel(
        tile_size=cfg.tile_size, L_min=cfg.L_min, L_max=cfg.L_max,
        N1=4, K=4, N2=4, H=4,
    ).model_input_size
    in_fold = (np.ones(len(pf["split_arr"]), dtype=bool) if tiles is None
               else np.isin(np.arange(len(pf["split_arr"])), np.asarray(tiles)))

    out: Dict[str, object] = {}
    for name in splits:
        code, ds_split = SPLITS[name]
        if not pf["processed_tiles"][(pf["split_arr"] == code) & in_fold].all():
            raise ValueError(f"split {name!r}: not every tile was scored")
        ds = CutSiteFragmentDataset(
            store_path=store_path, model_input_size=model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max, split=ds_split,
            min_N=min_N, train_mode=False, preload=False,
        )
        if tiles is not None:
            ds.index = [(s, t) for s, t in ds.index if in_fold[t]]
        out[name] = region_weighted_mult_nll(
            pf["log_prob"], pf["scored"], pf["indptr"], ds,
            return_terms=return_terms)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-fragment-tsv", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=list(SPLITS))
    p.add_argument("--min-N", type=int, default=10,
                   help="min_N the trainer used (region population)")
    p.add_argument("--tiles-npy", default=None,
                   help=".npy of store tile indices: reduce over only those "
                        "regions of each split (a cross-validation fold)")
    p.add_argument("--output", help="JSON path (default: <tsv>.nll.json)")
    args = p.parse_args(argv)

    root = zarr.open_group(args.store, mode="r")
    tiles = None if args.tiles_npy is None else np.load(args.tiles_npy)
    result = reduce_splits(args.per_fragment_tsv, args.store, args.splits,
                           args.min_N, tiles)
    with open(args.per_fragment_tsv + ".summary.json") as f:
        summary = json.load(f)
    doc = {
        "per_fragment_tsv": os.path.abspath(args.per_fragment_tsv),
        "per_fragment_tsv_sha256": _sha256(args.per_fragment_tsv),
        "store": os.path.abspath(args.store),
        "config_hash": root.attrs["config_hash"],
        "hex_tables": summary["hex_tables"],
        "hex_tables_sha256": _sha256(summary["hex_tables"]),
        "checkpoint": summary["checkpoint"],
        "zero_correction": summary["zero_correction"],
        "min_N": args.min_N,
        "tiles_npy": args.tiles_npy and os.path.abspath(args.tiles_npy),
        "tiles_npy_sha256": args.tiles_npy and _sha256(args.tiles_npy),
        "background_model": os.path.abspath(background_model.__file__),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "splits": result,
    }
    out = args.output or args.per_fragment_tsv + ".nll.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
