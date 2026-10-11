#!/usr/bin/env python
"""Oracle NLL of a simulated cut-site store, on the model's own evaluation domain.

    cd /tmp && PYTHONPATH=<repo> python <repo>/scripts/simulator/cut_site_oracle_nll.py \\
        --sim-run-dir /efs/.../sim_full_20261009_070c187 \\
        --store /efs/.../model_output_sim_full_20261009/sim_full.zarr \\
        --out-dir /efs/.../oracle_nll/sim_full \\
        --compare model=/efs/.../per_fragment_log_prob.tsv

The ceiling the model's capture fraction divides by.  Evaluation only: the
script reads the generator's internals, and nothing it writes may feed the
model (the model must never use the generating f(L) or hexamer counts as a
prior).  It lives under ``scripts/simulator/`` because the import boundary
(``test_model_code_never_imports_simulator``) forbids model code from importing
``background_model.simulator``; the dependency runs this way only.


What is scored, and how it is reduced
-------------------------------------
The model scores, per region, a multinomial over the CENTRED crop's domain
``tile_size x n_L x 2`` (starts in the crop, ``L`` in ``[L_min, L_max]``
inclusive, both strands), and its columns are REGION-weighted means over the
split's regions with ``min_N`` stored fragments.  The oracle is the generator's
own probability conditioned on that domain::

    p_cond = p / Z_r,    Z_r = P(a draw from region r lands in the domain)

so ``-log p_cond = -log p + log Z_r`` (``docs/pending/training_analysis.md``
§11, "The published oracle number is not directly comparable").

Reuse, not restatement:

* the per-fragment population is ``CutSiteFragmentDataset``'s rule:
  ``admit_fragments`` at ``crop_origin(l_target, tile_size, 0)``;
* the TSV is ``score_fragments.build_table`` / ``write_table``, so it has the
  scorer's columns and row order;
* the region population and the reduction are
  ``reduce_split_nll.reduce_splits``, i.e. ``CutSiteFragmentDataset(split,
  min_N, train_mode=False)`` and ``score_fragments.region_weighted_mult_nll``.
  Nothing here restates either.

``p`` is the sidecar's (the generator's own record of each draw).  ``r(h)`` and
``f(L)`` are not stored by the simulator, so they are re-measured with
``simulator.measure.measure_sample`` on the run's recorded inputs, and the
re-measurement is CHECKED: its count stats must equal ``run.json``'s, and the
``p`` it implies must equal the sidecar's ``p`` for every fragment.


Two normalisers
---------------
``exact``        ``Z_r`` from the generator: the strand-weighted start mass in
                 the crop.  The length factor sums to 1 per start because
                 ``f``'s support lies inside ``[L_min, L_max]`` (checked), so
                 only the start distribution is needed -- but over EVERY start
                 in the region, not only the drawn ones, which is why the
                 sidecar alone does not suffice.
``count_ratio``  ``Z_hat_r = n_scored_r / n_stored_r``, the region's own
                 fraction of draws landing in the crop.  This is how the
                 published anchors were made (``5eee866``: "Z_r estimated per
                 region"); it is kept so they can be reproduced.  ``log Z_hat``
                 is Jensen-biased low at ~36 draws, and reuses the draws it
                 scores.

``strand_start_probs`` restates the start and length distributions of
``simulator.draw.sample_region``, which returns ``p`` for drawn fragments only
and exposes no per-start distribution.  Changing the simulator was out of scope
(this is an additive measurement), so the restatement is pinned instead: it must
reproduce ``sample_region``'s ``probs`` bit for bit
(``tests/test_cut_site_oracle_nll.py``), and every run re-checks it against the
sidecar.

Not modelled, by owner decision (``simulator_spec.md``, Settled, "Duplicates
are redrawn"): the sidecar's ``p`` is the first-draw marginal, while duplicate
redraws make the draws slightly non-iid.  The oracle inherits that.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import zarr
from numpy.lib.stride_tricks import sliding_window_view

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import background_model  # noqa: E402

# CWD trap, as in reduce_split_nll: a background_model/ at the cwd shadows _REPO.
if not os.path.abspath(background_model.__file__).startswith(_REPO + os.sep):
    raise ImportError(
        f"background_model resolved to {background_model.__file__}, not under "
        f"{_REPO}; cd elsewhere (e.g. /tmp) and set PYTHONPATH={_REPO}"
    )

from fragmentomics_tools.dataframe import DataFrameBase  # noqa: E402

from background_model.cut_site_model.dataset import (  # noqa: E402
    admit_fragments,
    crop_origin,
)
from background_model.cut_site_model.store import CutSiteStoreConfig  # noqa: E402
from background_model.hexamers import hexamer_indices  # noqa: E402
from scripts.cut_site_model.baselines.reduce_split_nll import (  # noqa: E402
    read_per_fragment,
    reduce_splits,
)
from scripts.cut_site_model.score_fragments import (  # noqa: E402
    SPLITS,
    STRAND_CHARS,
    build_table,
    write_table,
)

NORMALISERS = ("exact", "count_ratio")

# The sidecar's p must reproduce to this relative tolerance: exactly.  It checks
# the re-measured r(h)/f(L) and the restated formula together, and both are
# bit-identical to the generator's (all 241,078 rows of the 10pct run).
P_RTOL = 0.0


# ---------------------------------------------------------------------------
# The generator's distribution
# ---------------------------------------------------------------------------


def strand_start_probs(sequence, region_len: int, *, r, fl):
    """Per strand, the generator's start and length distributions.

    Returns ``{is_plus: (start_probs, W_s, t_s)}``: ``start_probs[i] = P(i | s)``
    over ``[0, region_len)``, and ``P(L | i, s) = W_s[i, L - fl.min_fl] /
    t_s[i]``.  A RESTATEMENT of ``simulator.draw.sample_region``, kept in its
    operation order so the two agree bit for bit; see the module docstring.

    A strand with no live start raises.  ``sample_region`` skips such a strand
    and emits fewer fragments, so its draws would not follow ``P(s) = p_plus``
    and no oracle built on that assumption would be the generator's.
    """
    seq = np.frombuffer(bytes(sequence).upper(), dtype=np.uint8)
    fwd, rc, valid = hexamer_indices(seq)
    n_L = fl.max_fl - fl.min_fl + 1
    pos = np.arange(region_len)
    end_span = np.arange(fl.min_fl, region_len + fl.max_fl)
    out = {}
    for is_plus in (True, False):
        track = fwd if is_plus else rc
        s_tab = r["start_fwd"] if is_plus else r["end_rev"]
        e_tab = r["end_fwd"] if is_plus else r["start_rev"]
        e_end = e_tab[track[end_span]] * valid[end_span]
        W_s = np.multiply(sliding_window_view(e_end, n_L),
                          fl.densities[None, :], order="C")
        t_s = W_s.sum(axis=1, dtype=np.float64)
        a_s_live = s_tab[track[pos]] * valid[pos] * (t_s > 0)
        tot = a_s_live.sum(dtype=np.float64)
        if tot <= 0:
            raise ValueError(f"strand {'+' if is_plus else '-'} has no live "
                             f"start; the generator would not draw it")
        out[is_plus] = (a_s_live / tot, W_s, t_s)
    return out


def region_oracle(sequence, region_len: int, starts_0, lengths, is_plus, *,
                  r, fl, p_plus: float, crop_start: int, tile_size: int) -> dict:
    """``Z`` and the per-fragment ``p`` for one region.

    ``Z`` is the generator's mass on starts in ``[crop_start, crop_start +
    tile_size)``, all lengths, both strands.  ``p`` is each given fragment's
    probability, formed as ``sample_region`` forms it; ``p_other`` is the same
    ``(start, length)`` cell on the other strand, for ``log_prob_ml``.
    """
    if not 0 <= crop_start <= crop_start + tile_size <= region_len:
        raise ValueError(f"crop [{crop_start}, {crop_start + tile_size}) is "
                         f"outside the region [0, {region_len})")
    dist = strand_start_probs(sequence, region_len, r=r, fl=fl)
    p_strand = {True: np.float64(p_plus), False: np.float64(1.0 - p_plus)}
    crop = slice(crop_start, crop_start + tile_size)
    Z = sum(p_strand[s] * dist[s][0][crop].sum(dtype=np.float64)
            for s in (True, False))

    starts_0 = np.asarray(starts_0, dtype=np.int64)
    l_idx = np.asarray(lengths, dtype=np.int64) - fl.min_fl
    is_plus = np.asarray(is_plus, dtype=bool)

    def cell_p(s, rows):
        start_probs, W_s, t_s = dist[s]
        i, li = starts_0[rows], l_idx[rows]
        return p_strand[s] * start_probs[i] * (W_s[i, li] / t_s[i])

    p = np.empty(len(starts_0), dtype=np.float64)
    p_other = np.empty(len(starts_0), dtype=np.float64)
    for s in (True, False):
        rows = is_plus == s
        p[rows] = cell_p(s, rows)
        p_other[rows] = cell_p(not s, rows)
    return {"Z": np.float64(Z), "p": p, "p_other": p_other}


# ---------------------------------------------------------------------------
# Store-side assembly
# ---------------------------------------------------------------------------


def store_rows(store_path: str) -> Dict[str, np.ndarray]:
    """The store's per-tile and per-fragment columns, fragments in store order."""
    root = zarr.open_group(store_path, mode="r")
    cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    indptr = np.asarray(root["fragments/indptr"][:]).astype(np.int64)
    n_tiles = len(root["tiles/split"])
    if len(indptr) != n_tiles + 1:
        raise ValueError("assumes a single-sample store (S == 1)")
    return {
        "cfg": cfg,
        "config_hash": root.attrs["config_hash"],
        "indptr": indptr,
        "tile_of_row": np.repeat(np.arange(n_tiles), np.diff(indptr)),
        "contig": np.asarray(root["tiles/contig"][:]).astype(str),
        "tile_start": np.asarray(root["tiles/start"][:]).astype(np.int64),
        "tile_stop": np.asarray(root["tiles/stop"][:]).astype(np.int64),
        "split": np.asarray(root["tiles/split"][:]),
        "start": np.asarray(root["fragments/start"][:]).astype(np.int64),
        "length": np.asarray(root["fragments/length"][:]).astype(np.int64),
        "strand": np.asarray(root["fragments/strand"][:]).astype(np.int64),
    }


def tile_sequences(st: Dict[str, np.ndarray], regions: pd.DataFrame) -> list:
    """The generator's padded sequence for each store tile, in tile order.

    ``regions`` has ``contig``, ``start``, ``stop``, ``sequence`` (the
    simulator's frame).  Store tiles and regions must be the same set, each
    tile exactly one region of length ``l_target``: the store's fragment
    starts are then region-local in the generator's own frame.
    """
    key = list(zip(regions["contig"].astype(str), regions["start"].astype(np.int64),
                   regions["stop"].astype(np.int64)))
    row_of = {k: i for i, k in enumerate(key)}
    if len(row_of) != len(key):
        raise ValueError("duplicate regions in the simulator's region set")
    tiles = list(zip(st["contig"], st["tile_start"].tolist(), st["tile_stop"].tolist()))
    missing = [t for t in tiles if t not in row_of]
    if missing or len(tiles) != len(key):
        raise ValueError(f"store tiles and simulator regions differ: "
                         f"{len(tiles)} tiles, {len(key)} regions, "
                         f"{len(missing)} tiles not simulated (first {missing[:3]})")
    l_target = st["cfg"].l_target
    if not np.all(st["tile_stop"] - st["tile_start"] == l_target):
        raise ValueError(f"a store tile is not l_target={l_target} long")
    seqs = regions["sequence"].to_numpy()
    return [seqs[row_of[t]] for t in tiles]


def sidecar_p(st: Dict[str, np.ndarray], sidecar: pd.DataFrame) -> np.ndarray:
    """The sidecar's ``p`` for every store row, joined on (contig, start,
    stop, strand).  Every store row and every sidecar row must match once."""
    rows = pd.DataFrame({
        "contig": st["contig"][st["tile_of_row"]],
        "start": st["tile_start"][st["tile_of_row"]] + st["start"],
        "strand": STRAND_CHARS[st["strand"]],
    })
    rows["stop"] = rows["start"] + st["length"]
    rows["row"] = np.arange(len(rows))
    m = rows.merge(sidecar, on=["contig", "start", "stop", "strand"],
                   how="outer", validate="1:1", indicator=True)
    bad = (m["_merge"] != "both").sum()
    if bad:
        raise ValueError(f"{bad} rows do not join between store and sidecar "
                         f"({len(rows)} store rows, {len(sidecar)} sidecar rows)")
    p = np.empty(len(rows), dtype=np.float64)
    p[m["row"].to_numpy(np.int64)] = m["p"].to_numpy(np.float64)
    return p


def oracle_arrays(st, sequences: Sequence, p_side: np.ndarray, *, r, fl,
                  p_plus: float, n_workers=None) -> Dict[str, np.ndarray]:
    """Per store row: the population mask, and ``log_prob`` under each
    normaliser.  Per tile: ``Z`` exact and count-ratio.  Also the p check."""
    cfg = st["cfg"]
    if not (cfg.L_min <= fl.min_fl and fl.max_fl <= cfg.L_max):
        raise ValueError(
            f"f(L) support [{fl.min_fl}, {fl.max_fl}] is not inside the model's "
            f"[{cfg.L_min}, {cfg.L_max}]; Z_r would need the length factor")
    crop_start = crop_origin(cfg.l_target, cfg.tile_size, 0)
    _, scored = admit_fragments(st["start"], st["length"], crop_start,
                                cfg.tile_size, cfg.L_min, cfg.L_max)

    indptr = st["indptr"]
    work = DataFrameBase(pd.DataFrame({
        "sequence": list(sequences),
        "lo": indptr[:-1], "hi": indptr[1:],
    }))
    res = work.parallel_apply(
        lambda row: region_oracle(
            row["sequence"], cfg.l_target,
            st["start"][row["lo"]:row["hi"]], st["length"][row["lo"]:row["hi"]],
            st["strand"][row["lo"]:row["hi"]] == 0,
            r=r, fl=fl, p_plus=p_plus, crop_start=crop_start,
            tile_size=cfg.tile_size,
        ),
        n_workers=n_workers, verbose=False,
    )
    Z = res["Z"].to_numpy(np.float64)
    p_re = np.concatenate(res["p"].tolist())
    p_other = np.concatenate(res["p_other"].tolist())

    rel = np.abs(p_re - p_side) / p_side
    n_stored = np.diff(indptr)
    n_scored = np.bincount(st["tile_of_row"], weights=scored,
                           minlength=len(n_stored)).astype(np.int64)
    Z_hat = n_scored / np.maximum(n_stored, 1)

    tile = st["tile_of_row"]
    log_p = np.log(p_side)
    log_p_ml = np.log(p_side + p_other)
    out = {"scored": scored, "Z_exact": Z, "Z_count_ratio": Z_hat,
           "n_stored": n_stored, "n_scored": n_scored,
           "p_check": {"n": int(len(p_side)),
                       "n_bitwise_equal": int((p_re == p_side).sum()),
                       "max_rel_diff": float(rel.max()) if len(rel) else 0.0}}
    # A tile with no scored row has Z_hat = 0; it contributes no term, so its
    # normaliser is never read, and 1 keeps the log finite.
    for name, z in (("exact", Z), ("count_ratio", np.where(Z_hat > 0, Z_hat, 1.0))):
        lp = np.where(scored, log_p - np.log(z)[tile], np.nan)
        lp_ml = np.where(scored, log_p_ml - np.log(z)[tile], np.nan)
        out[name] = {"log_prob": lp, "log_prob_ml": lp_ml}
    return out


def write_and_reduce(store_path: str, arrays, name: str, out_dir: str,
                     min_N: int, splits=tuple(SPLITS)) -> Dict[str, dict]:
    """Write ``name``'s per-fragment TSV in the scorer's format, then reduce
    it with ``reduce_splits``.  Returns the per-split result with terms."""
    nnz = len(arrays["scored"])
    result = {"log_prob": arrays[name]["log_prob"],
              "log_prob_ml": arrays[name]["log_prob_ml"],
              "scored": arrays["scored"],
              "processed": np.ones(nnz, dtype=bool)}
    tsv = os.path.join(out_dir, f"per_fragment_oracle_{name}.tsv")
    write_table(build_table(store_path, result), tsv)
    # The keys reduce_splits reads to recover the processed tile set.
    with open(tsv + ".summary.json", "w") as f:
        json.dump({"store": os.path.abspath(store_path), "normaliser": name,
                   "splits": list(splits), "limit_regions": None,
                   "tiles_npy": None, "rows_written": nnz,
                   "rows_scored": int(arrays["scored"].sum())}, f, indent=2)
    return reduce_splits(tsv, store_path, list(splits), min_N, return_terms=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", _REPO, *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_sidecar(path: str):
    """``(header, frame)``; the header is ``{"seed": ..., "sample": ...}``."""
    with gzip.open(path, "rt") as f:
        first = f.readline()
    if not first.startswith("# "):
        raise ValueError(f"{path}: no provenance header")
    header = dict(kv.split("=", 1) for kv in first[2:].split())
    # round_trip: pandas' default float parser is NOT exact, and misreads ~28%
    # of the %.17g values by one ulp (measured on the 10pct sidecar).
    df = pd.read_csv(path, sep="\t", skiprows=1, float_precision="round_trip",
                     dtype={"contig": str, "strand": str, "start": np.int64,
                            "stop": np.int64, "p": np.float64})
    return header, df


def _strip_terms(res: Dict[str, dict]) -> Dict[str, dict]:
    return {s: {k: v for k, v in d.items() if k not in ("terms", "term_tiles")}
            for s, d in res.items()}


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-run-dir", required=True,
                   help="simulator output dir holding <sample>.run.json and "
                        "<sample>.p.tsv.gz")
    p.add_argument("--sample-id", default="RD-56804")
    p.add_argument("--store", required=True,
                   help="the cut-site store built from that run's h5")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-N", type=int, default=10,
                   help="min_N the trainer used (region population)")
    p.add_argument("--ref", default="hg38",
                   help="reference name for from_bed; the simulator's default")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--compare", action="append", default=[],
                   metavar="NAME=TSV",
                   help="a scorer per-fragment TSV for the same store, reduced "
                        "alongside; its region population and scored rows "
                        "must equal the oracle's.  NAME 'prior' adds capture "
                        "vs prior")
    args = p.parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    run_json = os.path.join(args.sim_run_dir, f"{args.sample_id}.run.json")
    with open(run_json) as f:
        run = json.load(f)
    side_path = run["emit_stats"]["p_sidecar"]
    header, sidecar = read_sidecar(side_path)
    if (int(header["seed"]) != run["seed"]
            or header.get("sample") != run["sample_id"]):
        raise ValueError(f"sidecar header {header} does not match {run_json}")

    st = store_rows(args.store)
    root = zarr.open_group(args.store, mode="r")
    if os.path.realpath(root.attrs["source_h5"]) != os.path.realpath(
            run["fragments_h5_out"]):
        # A copy is acceptable only if byte-identical.
        if _sha256(root.attrs["source_h5"]) != _sha256(run["fragments_h5_out"]):
            raise ValueError("store source_h5 is not the simulator's output h5")
    print(f"[store]   {args.store}: {len(st['tile_start'])} tiles, "
          f"{len(st['start'])} fragments")

    # Late import: the measure stack is heavy and only main needs it.
    from fragmentomics_tools.dataframe import RegionDataFrame
    from background_model.simulator.measure import measure_sample

    rdf = RegionDataFrame.from_bed(run["region_bed"], ref=args.ref)
    t = time.time()
    m = measure_sample(rdf, run["sample_id"], run["fragments_h5"], run["fasta"],
                       min_mapq=run["min_mapq"], n_workers=args.n_workers,
                       verbose=False)
    print(f"[measure] {time.time() - t:.1f} s")
    measured = {"count_stats": m.stats, "n_meta": m.n_meta,
                "fl_min": m.fl.min_fl, "fl_max": m.fl.max_fl,
                "fl_n": int(m.fl.counts.sum())}
    recorded = {k: run[k] for k in measured}
    if json.loads(json.dumps(measured, default=float)) != recorded:
        raise ValueError(f"re-measurement differs from run.json:\n"
                         f"  measured {measured}\n  recorded {recorded}")

    seqs = tile_sequences(st, m.srdf)
    p_side = sidecar_p(st, sidecar)
    t = time.time()
    arrays = oracle_arrays(st, seqs, p_side, r=m.r, fl=m.fl,
                           p_plus=run["p_plus"], n_workers=args.n_workers)
    print(f"[oracle]  {time.time() - t:.1f} s, p check {arrays['p_check']}")
    if arrays["p_check"]["max_rel_diff"] > P_RTOL:
        raise ValueError(f"recomputed p does not reproduce the sidecar: "
                         f"{arrays['p_check']}")

    results = {}
    for name in NORMALISERS:
        results[name] = write_and_reduce(args.store, arrays, name, args.out_dir,
                                         args.min_N)
        print(f"[{name}] " + ", ".join(
            f"{s} {d['region_weighted_nll']!r} ({d['n_regions']})"
            for s, d in results[name].items()))

    cfg = st["cfg"]
    uniform = float(np.log(np.float64(cfg.tile_size * (cfg.L_max - cfg.L_min + 1) * 2)))
    compares = {}
    for spec in args.compare:
        cname, tsv = spec.split("=", 1)
        res = reduce_splits(tsv, args.store, list(SPLITS), args.min_N,
                            return_terms=True)
        pf = read_per_fragment(tsv, args.store)
        if not np.array_equal(pf["scored"], arrays["scored"]):
            raise ValueError(f"{cname}: scored rows differ from the oracle's")
        for s in SPLITS:
            if not np.array_equal(res[s]["term_tiles"],
                                  results["exact"][s]["term_tiles"]):
                raise ValueError(f"{cname}: {s} region population differs")
        compares[cname] = {"tsv": os.path.abspath(tsv),
                           "tsv_sha256": _sha256(tsv), "splits": res}

    table = {}
    for s in SPLITS:
        row = {"n_regions": results["exact"][s]["n_regions"],
               "uniform": uniform}
        for name in NORMALISERS:
            row[f"oracle_{name}"] = results[name][s]["region_weighted_nll"]
        for cname, c in compares.items():
            row[cname] = c["splits"][s]["region_weighted_nll"]
        for cname in compares:
            for name in NORMALISERS:
                row[f"{cname}_captured_vs_uniform_{name}"] = (
                    (uniform - row[cname]) / (uniform - row[f"oracle_{name}"]))
                if "prior" in compares and cname != "prior":
                    row[f"{cname}_captured_vs_prior_{name}"] = (
                        (row["prior"] - row[cname])
                        / (row["prior"] - row[f"oracle_{name}"]))
        table[s] = row

    regions = pd.DataFrame({
        "tile": np.arange(len(st["tile_start"])), "contig": st["contig"],
        "start": st["tile_start"], "stop": st["tile_stop"],
        "split_code": st["split"], "n_stored": arrays["n_stored"],
        "n_scored": arrays["n_scored"], "Z_exact": arrays["Z_exact"],
        "Z_count_ratio": arrays["Z_count_ratio"],
    })
    regions.to_csv(os.path.join(args.out_dir, "oracle_regions.tsv"), sep="\t",
                   index=False, float_format="%.17g")

    doc = {
        "argv": sys.argv,
        "git_head": _git("rev-parse", "HEAD"),
        "script_dirty": bool(_git("status", "--porcelain", "--",
                                  os.path.abspath(__file__))),
        "script_sha256": _sha256(os.path.abspath(__file__)),
        "background_model": os.path.abspath(background_model.__file__),
        "numpy": np.__version__,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "store": os.path.abspath(args.store),
        "config_hash": st["config_hash"],
        "sim_run_json": os.path.abspath(run_json),
        "sim_run_json_sha256": _sha256(run_json),
        "sidecar": side_path, "sidecar_sha256": _sha256(side_path),
        "sidecar_header": header,
        "published_oracle_nll_fragment_pooled": run["emit_stats"]["oracle_nll"],
        "min_N": args.min_N, "p_plus": run["p_plus"],
        "measurement_matches_run_json": True,
        "p_check": arrays["p_check"],
        "Z_exact_summary": {
            "min": float(arrays["Z_exact"].min()),
            "max": float(arrays["Z_exact"].max()),
            "mean": float(arrays["Z_exact"].mean(dtype=np.float64))},
        "oracle": {name: _strip_terms(results[name]) for name in NORMALISERS},
        "compare": {c: {**v, "splits": _strip_terms(v["splits"])}
                    for c, v in compares.items()},
        "table": table,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "oracle_nll.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps({"table": table, "p_check": arrays["p_check"]}, indent=2))


if __name__ == "__main__":
    main()
