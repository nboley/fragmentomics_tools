#!/usr/bin/env python
"""S3 counter: the four hexamer cut-site tables and p_L for the cut-site prior.

The SINGLE producer of the ``--hex-tables`` .npz after owner rulings 5 and 6
(docs/pending/cut_site_baseline_priors_design.md §0.4, §4.3).
``stage1_hex_baseline.py`` is retired as a producer, not edited: it registers
the 3' cut at m+L-1 and smooths with a category error, and the model's loader
now refuses any .npz that lacks the registration key this script writes.

Two subcommands:

  count   one pass over a split's centred crop -> counts .npz
          (C per table, N_start, N_end, length counts) + provenance sidecar
  tables  counts .npz + estimator -> the model's .npz + provenance sidecar

Estimators (ruling 5).  ``N(h)`` is a PER-HEXAMER opportunity count, so
``C(h)/N(h)`` is a rate, and the shrinkage is ``(C + w*q) / (N + w)`` with
``q`` on the same cuts-per-occurrence scale; each table is then normalised to
sum 1 (the loader does not renormalise).

  raw         w = 0: the clean rate C/N
  shrink      w = --w, q = the table's pooled rate sum(C)/sum(N) unless --q
  stage1_ref  w = 4096, q = 1/4096: (C + 1) / (N + 4096), which is
              stage1_hex_baseline.py's formula exactly.  The decomposition
              row (N0_ref) that separates the estimator change from the
              registration change.  Not a candidate.

Geometry and registration are the MODEL's, not restated here: fragment
admission and the one-hot sequence come from ``CutSiteFragmentDataset``
(train_mode=False, so j=0, the centred crop), the hexamer encoding from
``model._onehot_to_hex_indices``, and the cut positions from
``model.fragment_cut_bases`` / ``model.hex_pos_of_tile_base`` -- the same two
functions ``compute_log_p0`` reads through.  So the table producer and the
table reader cannot be keyed one way and read another, which is the bug
recorded at ``model.py`` beside ``CUT_OFFSET_IN_KMER``.

Denominators are store-backed rather than taken from
``simulator.measure.uniform_hexamer_counts``, deliberately (design Q20): that
driver fetches sequence from a FASTA, while the model reads ``tiles/seq``, and
a denominator counted on one sequence and applied to another changes values,
not shapes, so nothing would flag it.  Its geometry ([gstart, gstop) plus
flanks) is not our centred crop, and it does not honour ``tiles/mask``.  The
pure pieces ARE reused: ``FragmentLengthDist`` and ``fl_end_weight``, and
``tests/test_cut_site_count_tables.py`` checks this counter against
``measure.propensities`` on a shared fixture, exhaustively over all 4096
hexamers.

Usage:
    python scripts/cut_site_model/baselines/count_tables.py count \\
        --store quiet_tile1536_v3.zarr --split train --output counts_train.npz
    python scripts/cut_site_model/baselines/count_tables.py tables \\
        --counts counts_train.npz --estimator stage1_ref --output N0_ref.npz
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import zarr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

from background_model.constants import NHEX  # noqa: E402
from background_model.cut_site_model.dataset import CutSiteFragmentDataset  # noqa: E402
from background_model.cut_site_model.model import (  # noqa: E402
    HEX_K,
    CutSiteFragmentModel,
    _onehot_to_hex_indices,
    fragment_cut_bases,
    hex_pos_of_tile_base,
    hex_registration_record,
)
from background_model.cut_site_model.store import CutSiteStoreConfig  # noqa: E402
from background_model.simulator.measure import (  # noqa: E402
    FragmentLengthDist,
    fl_end_weight,
)
from background_model_core import calculate_start_and_stop_from_jitter  # noqa: E402

# Which strand and which END of the fragment each table counts.  ONE
# definition, used both to tally C and to pick the denominator, so a table's
# numerator and denominator cannot come from different ends.  It mirrors the
# gathers in compute_log_p0: start_fwd/stop_rev at the left cut,
# stop_fwd/start_rev at the right cut; tests/test_cut_site_count_tables.py
# checks it against those gathers.  Strand code 0 is '+'.
TABLE_ENDS = {
    "start_fwd": (0, "left"),
    "stop_fwd": (0, "right"),
    "start_rev": (1, "right"),
    "stop_rev": (1, "left"),
}
TABLES = tuple(TABLE_ENDS)

# A left cut is a uniform start, weight 1 per position; a right cut at tile
# base q is reachable from every (m, L) with m + L at q, so its expectation
# carries fl_end_weight's ramp.
DENOMINATOR = {"left": "N_start", "right": "N_end"}

ESTIMATORS = ("raw", "shrink", "stage1_ref")


def _git_state():
    try:
        sha = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", _REPO, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def tile_hexamers(x, n_bases):
    """``(hex, ok)`` for tile bases ``0 .. n_bases-1`` of one dataset item's ``x``.

    ``hex`` is in the MODEL's encoding, read at the model's position.  ``ok``
    is False where the 6-mer holds a non-ACGT base: ``one_hot_encode_sequences``
    encodes N as 0.25 in every channel, and ``_onehot_to_hex_indices`` then
    reads it as A, so the index alone cannot tell.
    """
    hex_idx = _onehot_to_hex_indices(x[None])[0].numpy()
    base_ok = (x.max(dim=0).values == 1).numpy()
    win_ok = np.lib.stride_tricks.sliding_window_view(base_ok, HEX_K).all(axis=-1)
    pos = hex_pos_of_tile_base(np.arange(n_bases))
    return hex_idx[pos], win_ok[pos]


def cut_base_window(tile_size, L_max):
    """Tile bases ``0 .. n-1`` a cut can occupy: left cuts in ``[0, P)``, right
    cuts up to the far 3' cut of an ``L_max`` fragment starting at ``P-1``."""
    return int(fragment_cut_bases(tile_size - 1, L_max)[1]) + 1


def count_split(store_path, split="train", min_N=0, verbose=True):
    """C, N_start, N_end and the length counts over one split's centred crop.

    A fragment counts only when BOTH its cut windows are valid and unmasked
    (as ``simulator.measure.cut_site_hexamers``), and the same fragments make
    ``length_counts``.  ``length_counts_all`` keeps every admitted fragment,
    which is the population stage1's ``p_L`` used, so the two differ by
    exactly the dropped fragments.

    ``min_N`` defaults to 0, i.e. every tile of the split: that is the
    population the stage1 tables were counted on (599,412 TRAIN fragments on
    the real store), not the evaluator's min_N=10 subset.
    """
    root = zarr.open_group(store_path, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    P, L_min, L_max = cfg.tile_size, cfg.L_min, cfg.L_max
    n_L = L_max - L_min + 1

    # A throwaway model, for its geometry and its constructor guards: the
    # counter refuses whatever geometry the model refuses.
    model_input_size = CutSiteFragmentModel(
        tile_size=P, L_min=L_min, L_max=L_max, N1=4, K=4, N2=4, H=4,
    ).model_input_size
    ds = CutSiteFragmentDataset(
        store_path=store_path, model_input_size=model_input_size,
        model_L_min=L_min, model_L_max=L_max, split=split,
        min_N=min_N, train_mode=False, preload=True,
    )

    n_bases = cut_base_window(P, L_max)

    # tiles/mask spans the l_target window, of which the tile is the centred
    # crop; a right cut can sit past the window's end, where the store has
    # sequence but no mask, and there the mask does not apply.
    mask_all = np.asarray(root["tiles/mask"][:]).astype(bool)
    crop_start, _ = calculate_start_and_stop_from_jitter(cfg.l_target, P, 0)
    n_in_mask = min(n_bases, mask_all.shape[1] - crop_start)

    C = {t: np.zeros(NHEX, dtype=np.int64) for t in TABLES}
    length_counts = np.zeros(n_L, dtype=np.int64)
    length_counts_all = np.zeros(n_L, dtype=np.int64)
    # occ[q, h]: tiles whose valid, unmasked base q carries hexamer h.  Flat
    # codes are buffered and bincounted in batches, int64 throughout.
    occ = np.zeros(n_bases * NHEX, dtype=np.int64)
    codes = []
    n_dropped = 0

    for i in range(len(ds)):
        _s, t = ds.index[i]
        item = ds[i]
        hx, ok = tile_hexamers(item["x"], n_bases)
        mask = np.ones(n_bases, dtype=bool)
        mask[:n_in_mask] = mask_all[t, crop_start:crop_start + n_in_mask]
        if not np.array_equal(mask[:P], item["mask"].numpy()):
            raise AssertionError(f"tile {t}: mask crop disagrees with the dataset's")
        ok &= mask

        q = np.nonzero(ok)[0]
        codes.append(q * NHEX + hx[q])
        if len(codes) >= 1000:
            occ += np.bincount(np.concatenate(codes), minlength=occ.size)
            codes = []

        m = item["frag_m"].numpy()
        L = item["frag_L_idx"].numpy() + L_min
        strand = item["frag_s"].numpy()
        if not np.isin(strand, (0, 1)).all():
            raise ValueError(f"tile {t}: strand codes outside {{0, 1}}")
        length_counts_all += np.bincount(L - L_min, minlength=n_L)
        ends = dict(zip(("left", "right"), fragment_cut_bases(m, L)))
        keep = ok[ends["left"]] & ok[ends["right"]]
        n_dropped += int((~keep).sum())
        length_counts += np.bincount(L[keep] - L_min, minlength=n_L)
        for name, (s_code, end) in TABLE_ENDS.items():
            sel = keep & (strand == s_code)
            C[name] += np.bincount(hx[ends[end][sel]], minlength=NHEX)

        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(ds)} tiles", flush=True)

    if codes:
        occ += np.bincount(np.concatenate(codes), minlength=occ.size)
    occ = occ.reshape(n_bases, NHEX)

    fl = FragmentLengthDist(length_counts, L_min)
    # fl_end_weight's index is the cut base's region-local coordinate, which
    # for us is the tile base q: w(q) is the f(L) mass of lengths with a start
    # m = q - L in [0, P).  Position-indexed, so no hexamer crosses here.
    w_end = fl_end_weight(n_bases, P, fl)
    N_start = occ[:P].sum(axis=0)
    N_end = (w_end[:, None] * occ).sum(axis=0, dtype=np.float64)

    sha, dirty = _git_state()
    out = dict(C)
    out.update(
        N_start=N_start, N_end=N_end, length_counts=length_counts,
        length_counts_all=length_counts_all,
        tile_size=np.int64(P), L_min=np.int64(L_min), L_max=np.int64(L_max),
    )
    meta = dict(
        store=os.path.abspath(store_path), split=split, min_N=int(min_N),
        crop="centred", config_hash=str(root.attrs["config_hash"]),
        n_tiles=len(ds), n_fragments_admitted=int(length_counts_all.sum()),
        n_fragments_counted=int(length_counts.sum()),
        n_dropped_invalid_or_masked=n_dropped,
        # Reviewer F-01: a consumer of these counts cannot see from the arrays
        # alone that the mask stops short.  `tiles/mask` spans the l_target
        # window, but a right cut reaches up to L_max past the tile, so the
        # last `n_bases - n_in_mask` cut positions are counted UNMASKED.
        # Harmless while the mask is all-True, silently wrong if it is not —
        # so it is recorded rather than left to the source comment.
        mask_honoured_bases=int(n_in_mask),
        cut_base_window=int(n_bases),
        mask_unhonoured_bases=int(n_bases - n_in_mask),
        mask_all_true=bool(mask_all.all()),
        git_sha=sha, git_dirty=dirty,
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    return out, meta


def rates(C, N, w, q):
    """``(C + w*q) / (N + w)``, float64.  ``w = 0`` is the clean rate ``C/N``,
    with 0 where ``N == 0``, as ``simulator.measure.propensities``."""
    C = np.asarray(C, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    num, den = C + w * q, N + w
    r = np.zeros(NHEX, dtype=np.float64)
    ok = den > 0
    r[ok] = num[ok] / den[ok]
    return r


def build_tables(counts, estimator, w=None, q=None):
    """Normalised tables from ``count_split``'s counts.  Returns ``(tables, params)``."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, not {estimator!r}")
    if estimator == "stage1_ref":
        if w is not None or q is not None:
            raise ValueError("stage1_ref fixes w and q; do not pass them")
        w, q = float(NHEX), 1.0 / NHEX
    elif estimator == "raw":
        if w is not None or q is not None:
            raise ValueError("raw is w = 0; do not pass w or q")
        w = 0.0
    elif w is None or w < 0:
        raise ValueError("shrink needs --w >= 0")

    tables, params = {}, {}
    for name, (_s, end) in TABLE_ENDS.items():
        C, N = counts[name], counts[DENOMINATOR[end]]
        q_t = (float(np.sum(C, dtype=np.float64) / np.sum(N, dtype=np.float64))
               if q is None else float(q))
        r = rates(C, N, float(w), q_t)
        total = r.sum(dtype=np.float64)
        if not total > 0:
            raise ValueError(f"{name}: every rate is 0, nothing to normalise")
        tables[name] = r / total
        params[name] = dict(w=float(w), q=q_t, denominator=DENOMINATOR[end])
    return tables, params


def p_L_from(counts):
    lc = np.asarray(counts["length_counts"], dtype=np.float64)
    return lc / lc.sum(dtype=np.float64)


def _write_sidecar(path, meta):
    with open(path + ".provenance.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("count", help="count one split's centred crop")
    c.add_argument("--store", required=True)
    c.add_argument("--split", default="train",
                   choices=["train", "val", "heldout_inactive"])
    c.add_argument("--min-N", type=int, default=0)
    c.add_argument("--output", required=True)
    t = sub.add_parser("tables", help="counts -> the model's .npz")
    t.add_argument("--counts", required=True)
    t.add_argument("--estimator", required=True, choices=ESTIMATORS)
    t.add_argument("--w", type=float, default=None)
    t.add_argument("--q", type=float, default=None,
                   help="backoff rate; default each table's pooled sum(C)/sum(N)")
    t.add_argument("--output", required=True)
    args = p.parse_args(argv)

    if args.cmd == "count":
        counts, meta = count_split(args.store, args.split, args.min_N)
        np.savez(args.output, **counts)
        _write_sidecar(args.output, meta)
        print(json.dumps(meta, indent=2))
        return

    counts = np.load(args.counts)
    with open(args.counts + ".provenance.json") as f:
        count_meta = json.load(f)
    tables, params = build_tables(counts, args.estimator, args.w, args.q)
    out = dict(tables)
    out.update(p_L=p_L_from(counts), k=np.int64(HEX_K),
               L_min=np.int64(counts["L_min"]), L_max=np.int64(counts["L_max"]),
               **hex_registration_record())
    np.savez(args.output, **out)
    sha, dirty = _git_state()
    _write_sidecar(args.output, dict(
        counts=os.path.abspath(args.counts), counts_provenance=count_meta,
        estimator=args.estimator, tables=params, git_sha=sha, git_dirty=dirty,
        date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
