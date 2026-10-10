#!/usr/bin/env python
"""Per-fragment log-probabilities under the cut-site fragment model.

Emits one tab-separated row per fragment in the cut-site STORE (not per
fragment in the source h5: the store has already applied the l_target
start-admission, the [L_min, L_max] length filter and (start, stop) dedup).

Columns:
    contig start stop strand   absolute genomic coordinates, as in the h5
                               (start = tiles/start + region-local start,
                               stop = start + length, i.e. half-open)
    log_prob     ln p(m, L, s | tile, N) over the full domain
                 |D| = tile_size * n_L * 2 (319,488 at tile 1024, L 25-180)
    log_prob_ml  ln p(m, L | tile, N) = logsumexp over the two strand cells;
                 domain tile_size * n_L.  log_prob_ml - log_prob =
                 log(1 + p_other_strand / p_this_strand) >= 0; it is ~ln 2
                 for a strand-symmetric model but has no upper bound.
    split        train / val / test, from the store's tiles/split
    scored       true iff the fragment is admitted to the CENTRED crop

Row order: store order, i.e. tiles in store order and, within a tile, the
store's CSR order.  Deterministic for a given store.

Coverage.  The model scores a tile_size crop inside each l_target region.
With train_mode=False the crop is centred, so a fragment whose start lies in
the l_target margin outside that crop has NO probability under this
geometry.  Those rows are emitted with log_prob = NaN and scored = false.
They are not dropped and no value is invented for them.  Scoring them at
other crop offsets is deliberately NOT done: each crop is its own
multinomial with its own N, so rows would carry different conditioning.

Reuse, not reimplementation.  Fragments and their (m, L_idx, s) come from
CutSiteFragmentDataset itself (train_mode=False, so j = 0), joined back to
store rows through the dataset's `frag_store_idx`.  Log-probabilities come
from CutSiteFragmentModel.domain_log_probs, the same method `_step` uses,
gathered with the same flat index m * (n_L * 2) + L_idx * 2 + s.

min_N.  Regions are scored at min_N=1 so that no region is excluded.  The
model was TRAINED at min_N=10; that only changes which regions entered the
loss, not the per-region multinomial, so scoring the extra regions is valid.
`--reproduce-val-mult-nll` re-applies min_N=10 to check against the trainer.

Precision.  The forward runs in float32, as at --precision 32.  Every value
taken out of the model is converted to float64 before any accumulation.

Usage:
    python scripts/score_cut_site_fragments.py --store S.zarr \\
        --hex-tables hex.npz --zero-correction --output out.tsv \\
        --device cpu --limit-regions 20
    python scripts/score_cut_site_fragments.py --store S.zarr \\
        --hex-tables hex.npz --checkpoint best.ckpt --output out.tsv \\
        --device cuda --batch-size 16 --reproduce-val-mult-nll
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

# Bootstrap: background_model is not an installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from background_model.cut_site_model.dataset import (  # noqa: E402
    CutSiteFragmentDataset,
    cut_site_collate_fn,
)
from background_model.cut_site_model.model import CutSiteFragmentModel  # noqa: E402
from background_model.cut_site_model.store import CutSiteStoreConfig  # noqa: E402
from background_model_core import calculate_start_and_stop_from_jitter  # noqa: E402

log = logging.getLogger(__name__)

COLUMNS = ("contig", "start", "stop", "strand", "log_prob", "log_prob_ml",
           "split", "scored")

# Output split name -> (store tiles/split code, CutSiteFragmentDataset split).
# The builder writes code 2 for its test fraction; the dataset calls that
# code "heldout_inactive".
SPLITS = {"train": (0, "train"), "val": (1, "val"), "test": (2, "heldout_inactive")}
STRAND_CHARS = np.array(["+", "-"])   # store strand 0 = '+', 1 = '-'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(
    cfg: CutSiteStoreConfig,
    hex_tables: str,
    checkpoint: Optional[str] = None,
    zero_correction: bool = False,
) -> CutSiteFragmentModel:
    """Build the model from a checkpoint, or prior-only (f_theta == 0)."""
    if (checkpoint is None) == (not zero_correction):
        raise ValueError("give exactly one of checkpoint / zero_correction")
    if checkpoint is not None:
        # hex_tables_path / freeze_hex are excluded from saved hparams, so the
        # tables must be supplied to create the slots; the state dict then
        # overwrites them with the checkpoint's (possibly trained) values.
        model = CutSiteFragmentModel.load_from_checkpoint(
            checkpoint, map_location="cpu",
            hex_tables_path=hex_tables, freeze_hex=True,
        )
    else:
        # Same architecture as scripts/train_cut_site.py.  The stem weights
        # are irrelevant: logit = (hid * w_out).sum(-1) + len_bias, so zeroing
        # both makes the logits exactly log_p0.
        torch.manual_seed(0)
        model = CutSiteFragmentModel(
            L_min=cfg.L_min, L_max=cfg.L_max, tile_size=cfg.tile_size,
            N1=128, K=128, N2=32, H=32, L_chunk=32,
            hex_tables_path=hex_tables, freeze_hex=True,
        )
        with torch.no_grad():
            model.w_out.zero_()
            model.len_bias.zero_()

    hp = model.hparams
    if (hp.tile_size, hp.L_min, hp.L_max) != (cfg.tile_size, cfg.L_min, cfg.L_max):
        raise ValueError(
            f"model (tile {hp.tile_size}, L [{hp.L_min}, {hp.L_max}]) does not "
            f"match store (tile {cfg.tile_size}, L [{cfg.L_min}, {cfg.L_max}])"
        )
    if not model._use_hex_baseline:
        raise ValueError("model has no hex prior; refusing to score")
    return model.eval()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def selected_tiles(
    split_arr: np.ndarray,
    splits: Sequence[str],
    tiles: Optional[Sequence[int]] = None,
    limit_regions: Optional[int] = None,
) -> np.ndarray:
    """Boolean mask over store tiles: the tiles `score_store` processes.

    The ONE selection rule, shared with `reduce_split_nll.read_per_fragment`,
    which must recover the processed tile set from the summary JSON: split
    membership, then the `tiles` restriction, then the `limit_regions` cut.
    """
    selected = np.isin(split_arr, [SPLITS[s][0] for s in splits])
    if tiles is not None:
        tiles = np.asarray(tiles, dtype=np.int64)
        if tiles.size and (tiles.min() < 0 or tiles.max() >= len(split_arr)):
            raise ValueError("tile index outside the store")
        selected &= np.isin(np.arange(len(split_arr)), tiles)
    if limit_regions is not None:
        keep = np.nonzero(selected)[0][:limit_regions]
        selected = np.zeros(len(split_arr), dtype=bool)
        selected[keep] = True
    return selected


class _Items(Dataset):
    """Flat view over (dataset, item index) pairs, in store tile order."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        ds, i = self.pairs[k]
        return ds[i]


def _collate(batch):
    out = cut_site_collate_fn(batch)
    out["frag_store_idx"] = [b["frag_store_idx"] for b in batch]
    return out


def score_store(
    model: CutSiteFragmentModel,
    store_path: str,
    splits: Sequence[str] = tuple(SPLITS),
    limit_regions: Optional[int] = None,
    batch_size: int = 1,
    device: str = "cpu",
    workers: int = 0,
    tiles: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Score every store fragment in the selected tiles.

    Tiles are the store tiles whose split is in `splits`, restricted to the
    store tile indices in `tiles` if given (a cross-validation fold), in
    store order, truncated to the first `limit_regions`.

    Returns arrays over ALL store rows (nnz):
        log_prob, log_prob_ml   float64, NaN where not scored
        scored                  bool, admitted to the centred crop
        processed               bool, row belongs to a selected tile
    plus `max_abs_logsumexp`, the worst |logsumexp| of a region's full-domain
    log-probs (normalisation check, every scored region).
    """
    root = zarr.open_group(store_path, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    indptr = np.asarray(root["fragments/indptr"][:])
    split_arr = np.asarray(root["tiles/split"][:])
    n_tiles = len(split_arr)
    if len(indptr) != n_tiles + 1:
        raise ValueError("scorer assumes a single-sample store (S == 1)")
    nnz = int(indptr[-1])

    selected = selected_tiles(split_arr, splits, tiles, limit_regions)

    pairs = []
    for name in splits:
        ds = CutSiteFragmentDataset(
            store_path=store_path, model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max,
            split=SPLITS[name][1], min_N=1, train_mode=False, preload=True,
        )
        pairs.extend((ds, i) for i, (_, t) in enumerate(ds.index) if selected[t])
    pairs.sort(key=lambda p: p[0].index[p[1]][1])

    # Store columns, used only to CHECK the dataset's frag_store_idx join.
    st_start = np.asarray(root["fragments/start"][:]).astype(np.int64)
    st_length = np.asarray(root["fragments/length"][:]).astype(np.int64)
    st_strand = np.asarray(root["fragments/strand"][:]).astype(np.int64)
    # Same call, same args as CutSiteFragmentDataset.__getitem__ at j = 0.
    crop_start, _ = calculate_start_and_stop_from_jitter(
        cfg.l_target, cfg.tile_size, 0
    )

    log_prob = np.full(nnz, np.nan, dtype=np.float64)
    log_prob_ml = np.full(nnz, np.nan, dtype=np.float64)
    scored = np.zeros(nnz, dtype=bool)
    max_abs_lse = 0.0

    model = model.to(device)
    loader = DataLoader(_Items(pairs), batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=_collate)
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            log_probs, _, _ = model.domain_log_probs(batch["x"].to(device))
            B, P, n_L, _ = log_probs.shape
            flat = log_probs.reshape(B, P * n_L * 2)
            lse = torch.logsumexp(flat.double(), dim=-1)
            max_abs_lse = max(max_abs_lse, float(lse.abs().max()))
            for b in range(B):
                m = batch["frag_m"][b]
                L_idx = batch["frag_L_idx"][b]
                s = batch["frag_s"][b]
                idx = batch["frag_store_idx"][b].numpy()
                if len(idx) != int(batch["frag_N"][b]):
                    raise AssertionError("frag_store_idx length != frag_N")
                # The dataset's join must reproduce the store rows exactly.
                if not (np.array_equal(st_start[idx], m.numpy() + crop_start)
                        and np.array_equal(st_length[idx] - cfg.L_min, L_idx.numpy())
                        and np.array_equal(st_strand[idx], s.numpy())):
                    raise AssertionError("frag_store_idx does not match store rows")
                # Same flat index as _step.
                cell = (m * (n_L * 2) + L_idx * 2).to(device)
                both = torch.stack(
                    [flat[b, cell], flat[b, cell + 1]], dim=-1
                ).double().cpu().numpy()                        # (n, 2)
                log_prob[idx] = both[np.arange(len(idx)), s.numpy()]
                log_prob_ml[idx] = np.logaddexp(both[:, 0], both[:, 1])
                scored[idx] = True
            n_done += B
            if n_done % max(1, 50 * batch_size) < B:
                log.info("scored %d/%d regions (%.1fs)", n_done, len(pairs),
                         time.time() - t0)

    tile_of_row = np.repeat(np.arange(n_tiles), np.diff(indptr))
    return {
        "log_prob": log_prob,
        "log_prob_ml": log_prob_ml,
        "scored": scored,
        "processed": selected[tile_of_row],
        "n_regions": np.int64(len(pairs)),
        "max_abs_logsumexp": np.float64(max_abs_lse),
    }


def build_table(store_path: str, result: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Rows for every processed store fragment, in store order."""
    root = zarr.open_group(store_path, mode="r")
    indptr = np.asarray(root["fragments/indptr"][:])
    tile_of_row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    contig = np.asarray(root["tiles/contig"][:])
    tile_start = np.asarray(root["tiles/start"][:]).astype(np.int64)
    split_arr = np.asarray(root["tiles/split"][:])
    split_names = np.empty(int(split_arr.max()) + 1, dtype=object)
    for name, (code, _) in SPLITS.items():
        if code < len(split_names):
            split_names[code] = name

    start = tile_start[tile_of_row] + np.asarray(root["fragments/start"][:]).astype(np.int64)
    stop = start + np.asarray(root["fragments/length"][:]).astype(np.int64)
    strand = STRAND_CHARS[np.asarray(root["fragments/strand"][:])]
    keep = result["processed"]
    return pd.DataFrame({
        "contig": contig[tile_of_row][keep],
        "start": start[keep],
        "stop": stop[keep],
        "strand": strand[keep],
        "log_prob": result["log_prob"][keep],
        "log_prob_ml": result["log_prob_ml"][keep],
        "split": split_names[split_arr[tile_of_row]][keep],
        "scored": result["scored"][keep],
    }, columns=list(COLUMNS))


def write_table(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    out["scored"] = np.where(out["scored"], "true", "false")
    # %.17g round-trips float64 exactly.
    out.to_csv(path, sep="\t", index=False, na_rep="NaN", float_format="%.17g")


# ---------------------------------------------------------------------------
# Acceptance check: reproduce the trainer's val_mult_nll
# ---------------------------------------------------------------------------


def region_weighted_mult_nll(
    log_prob: np.ndarray,
    scored: np.ndarray,
    indptr: np.ndarray,
    dataset: CutSiteFragmentDataset,
    return_terms: bool = False,
) -> Dict[str, float]:
    """The trainer's REGION-weighted estimator, from per-fragment values.

    `_step` computes per region -sum(lp) / clamp(N, min=1), and Lightning
    averages that over regions (batch-size-weighted epoch mean).  The region
    population is `dataset.index`, so pass a dataset built with the trainer's
    split and min_N.  A region whose crop admits no fragment contributes 0 to
    the mean but still counts in the denominator, exactly as in `_step`.
    Accumulates in float64.

    `return_terms=True` adds the per-region terms (float64, in
    `dataset.index` order) as `terms` and their store tile indices as
    `term_tiles`, for paired comparisons.  The other entries are unchanged.
    """
    terms = np.empty(len(dataset.index), dtype=np.float64)
    n_empty = 0
    for k, (s, t) in enumerate(dataset.index):
        u = s * dataset.n_tiles + t
        lo, hi = int(indptr[u]), int(indptr[u + 1])
        lp = log_prob[lo:hi][scored[lo:hi]]
        if np.isnan(lp).any():
            raise ValueError(f"tile {t} has a scored row with NaN log_prob")
        n_empty += len(lp) == 0
        terms[k] = -lp.sum(dtype=np.float64) / max(len(lp), 1)
    out = {
        "region_weighted_nll": float(terms.mean(dtype=np.float64)),
        "n_regions": len(terms),
        "n_regions_no_scored_fragment": int(n_empty),
    }
    if return_terms:
        out["terms"] = terms
        out["term_tiles"] = np.array([t for _, t in dataset.index], dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True)
    p.add_argument("--hex-tables", required=True,
                   help="Stage-1 hex tables .npz (the model's prior)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="trained .ckpt")
    g.add_argument("--zero-correction", action="store_true",
                   help="prior-only model: w_out=0, len_bias=0, logits = log_p0")
    p.add_argument("--output", required=True, help="output .tsv")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--limit-regions", type=int, default=None,
                   help="score only the first N selected tiles (smoke test)")
    p.add_argument("--tiles-npy", default=None,
                   help=".npy of store tile indices; score only those tiles of "
                        "--splits (a cross-validation fold)")
    p.add_argument("--reproduce-val-mult-nll", action="store_true",
                   help="also report the trainer's region-weighted val_mult_nll "
                        "recomputed from the per-fragment output")
    p.add_argument("--trained-min-N", type=int, default=10,
                   help="min_N the trainer used, for --reproduce-val-mult-nll")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    root = zarr.open_group(args.store, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    model = load_model(cfg, args.hex_tables, args.checkpoint, args.zero_correction)

    tiles = None if args.tiles_npy is None else np.load(args.tiles_npy)
    result = score_store(model, args.store, args.splits, args.limit_regions,
                         args.batch_size, args.device, args.workers, tiles)
    df = build_table(args.store, result)
    write_table(df, args.output)

    nnz = len(result["scored"])
    complete = (args.limit_regions is None and tiles is None
                and set(args.splits) == set(SPLITS))
    summary = {
        "store": os.path.abspath(args.store),
        "config_hash": root.attrs["config_hash"],
        "checkpoint": args.checkpoint and os.path.abspath(args.checkpoint),
        "zero_correction": args.zero_correction,
        "hex_tables": os.path.abspath(args.hex_tables),
        "splits": args.splits,
        "limit_regions": args.limit_regions,
        "tiles_npy": args.tiles_npy and os.path.abspath(args.tiles_npy),
        "n_regions_scored": int(result["n_regions"]),
        "rows_written": len(df),
        "rows_scored": int(df["scored"].sum()),
        "rows_unscored": int((~df["scored"]).sum()),
        "store_fragments": nnz,
        "max_abs_logsumexp": float(result["max_abs_logsumexp"]),
    }
    if complete and len(df) != nnz:
        raise AssertionError(f"wrote {len(df)} rows for {nnz} store fragments")

    if args.reproduce_val_mult_nll:
        if (args.limit_regions is not None or tiles is not None
                or "val" not in args.splits):
            p.error("--reproduce-val-mult-nll needs every val tile scored")
        val_ds = CutSiteFragmentDataset(
            store_path=args.store, model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max, split="val",
            min_N=args.trained_min_N, train_mode=False, preload=False,
        )
        summary["val_mult_nll"] = region_weighted_mult_nll(
            result["log_prob"], result["scored"],
            np.asarray(root["fragments/indptr"][:]), val_ds,
        )

    with open(args.output + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
