"""BED emission and manifest — Step 6 of the simulator.

Writes one sorted, bgzipped, tabix-indexed 8-column BED per sample, then
(optionally) calls ``build-fragments-h5`` to produce the fragment h5.

The manifest carries the three factor arrays (hexamer tables as DataFrames
keyed by the 6-mer string, the ``(L, gc_bin)`` predict LUT, and
``marginal_fl``) plus provenance fields.  Together these are sufficient to
reconstruct ``w`` via ``build_region_weights`` — the load-bearing property
of the output.

Hexamer table format (owner decision 81): DataFrames keyed by the hexamer
STRING, not bare 4096-element arrays ordered by an implicit integer code.
This makes the k-mer ordering a private detail on each side; a mismatched
table fails to join rather than silently misaligning.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from background_model.simulator.precompute import KMER, hexamer_indices
from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    NHEX,
    HexamerTables,
)

# ── hexamer vocabulary ───────────────────────────────────────────────────
# Reuses the approach from scripts/count_cut_site_hexamers.py::hexamer_vocabulary
# (design doc decision 81: "reuse its vocabulary helper").

def hexamer_vocabulary() -> np.ndarray:
    """``vocab[i]`` is the 6-mer whose forward index is ``i``, as ``S6`` bytes.

    Derived from ``hexamer_indices`` rather than by reimplementing its encoding
    backwards.  See ``scripts/count_cut_site_hexamers.py::hexamer_vocabulary``
    for the canonical version and reasoning.
    """
    grid = np.indices((4,) * KMER).reshape(KMER, -1).T
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)[grid].astype(np.uint8)

    fwd, _rc, valid = hexamer_indices(letters.reshape(-1))
    starts = np.arange(0, NHEX * KMER, KMER)
    idx = fwd[starts]
    assert valid[starts].all()
    assert np.unique(idx).size == NHEX, "hexamer index is not a bijection"

    strings = np.frombuffer(letters.tobytes(), dtype=f"S{KMER}")
    vocab = np.empty(NHEX, dtype=f"S{KMER}")
    vocab[idx] = strings
    return vocab


# Module-level cache (computed once).
_VOCAB: Optional[np.ndarray] = None


def _get_vocab() -> np.ndarray:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = hexamer_vocabulary()
    return _VOCAB


# ── hexamer tables <-> DataFrames ────────────────────────────────────────

def hex_table_to_dataframe(table: np.ndarray, table_name: str) -> pd.DataFrame:
    """Convert a 4096-element weight array to a DataFrame keyed by hexamer string.

    Parameters
    ----------
    table : ndarray, shape (4096,)
        Weight values indexed by the integer hexamer code.
    table_name : str
        One of ``"start_fwd"``, ``"end_fwd"``, ``"start_rev"``, ``"end_rev"``.

    Returns
    -------
    DataFrame with columns ``["hexamer", "weight"]``, 4096 rows.
    """
    vocab = _get_vocab()
    return pd.DataFrame({
        "hexamer": [v.decode() for v in vocab],
        "weight": table,
    })


def dataframe_to_hex_table(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct a 4096-element weight array from a hexamer-keyed DataFrame.

    The join is on the hexamer STRING — if the DataFrame was produced under a
    different k-mer convention, the join will misalign and weights will be wrong,
    which is exactly what the string key exists to make detectable (a bare array
    would silently misalign).
    """
    vocab = _get_vocab()
    vocab_strs = [v.decode() for v in vocab]
    idx_map = {s: i for i, s in enumerate(vocab_strs)}

    table = np.zeros(NHEX, dtype=np.float64)
    for _, row in df.iterrows():
        hexamer = row["hexamer"]
        if hexamer not in idx_map:
            raise ValueError(
                f"Unknown hexamer {hexamer!r} in DataFrame — not in the "
                f"vocabulary derived from hexamer_indices"
            )
        table[idx_map[hexamer]] = row["weight"]
    return table


def hex_tables_to_dict(
    hex_tables: HexamerTables,
) -> Dict[str, pd.DataFrame]:
    """Convert all four hexamer tables to a dict of DataFrames."""
    return {
        name: hex_table_to_dataframe(getattr(hex_tables, name), name)
        for name in HexamerTables._fields
    }


def dict_to_hex_tables(d: Dict[str, pd.DataFrame]) -> HexamerTables:
    """Reconstruct ``HexamerTables`` from a dict of DataFrames."""
    return HexamerTables(**{
        name: dataframe_to_hex_table(d[name])
        for name in HexamerTables._fields
    })


# ── BED emission ─────────────────────────────────────────────────────────

_MAPQ = 60  # All simulated reads get MAPQ 60.


def write_bed(
    bed_path: str,
    contig: str,
    gstart: int,
    starts: np.ndarray,
    stops: np.ndarray,
    strands: np.ndarray,
) -> None:
    """Append fragment records to a BED file (unsorted, uncompressed).

    8-column BED format:
    ``contig  start  stop  name  score  strand  mapq1  mapq2``

    The ``start``/``stop`` are region-local; ``gstart`` is added to convert
    to genomic coordinates.
    """
    with open(bed_path, "a") as f:
        for i in range(len(starts)):
            gs = gstart + int(starts[i])
            ge = gstart + int(stops[i])
            strand = str(strands[i])
            f.write(
                f"{contig}\t{gs}\t{ge}\t.\t0\t{strand}\t{_MAPQ}\t{_MAPQ}\n"
            )


def sort_bgzip_tabix(
    bed_path: str,
    bgzip_path: Optional[str] = None,
) -> str:
    """Sort a BED file, bgzip it, and create a tabix index.

    Parameters
    ----------
    bed_path : str
        Path to the unsorted BED file.
    bgzip_path : str, optional
        Output path for the bgzipped file.  Defaults to ``bed_path + ".gz"``.

    Returns
    -------
    str
        Path to the bgzipped file.
    """
    if bgzip_path is None:
        bgzip_path = bed_path + ".gz"

    sorted_path = bed_path + ".sorted"
    # sort by contig (lexicographic) then by start (numeric)
    subprocess.run(
        ["sort", "-k1,1", "-k2,2n", bed_path, "-o", sorted_path],
        check=True,
    )
    # bgzip
    with open(bgzip_path, "wb") as out_f:
        subprocess.run(
            ["bgzip", "-c", sorted_path],
            stdout=out_f,
            check=True,
        )
    # tabix
    subprocess.run(
        ["tabix", "-p", "bed", bgzip_path],
        check=True,
    )
    # clean up intermediates
    os.remove(bed_path)
    os.remove(sorted_path)
    return bgzip_path


# ── manifest ─────────────────────────────────────────────────────────────

def _hash_file(path: str) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    """SHA-256 hex digest of an array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def write_manifest(
    manifest_path: str,
    *,
    hex_tables: HexamerTables,
    predict_lut: np.ndarray,
    marginal_fl: np.ndarray,
    region_set_name: str,
    region_set_hash: str,
    reference_name: str,
    reference_hash: str,
    region_len: int,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    fl_bands: Sequence[Tuple[int, int]] = ((25, 110), (110, 180)),
    per_region_counts: Dict[str, int] | None = None,
    rng_seed: int | None = None,
    commit_sha: str | None = None,
) -> None:
    """Write the simulation manifest to a JSON file.

    The manifest carries the realised factor arrays (not the recipe) plus
    provenance fields, so that ``w`` can be reconstructed from the manifest
    alone via ``build_region_weights``.

    Hexamer tables are stored as DataFrames keyed by the hexamer string
    (owner decision 81).
    """
    # Convert hex tables to JSON-serialisable dicts
    hex_dfs = hex_tables_to_dict(hex_tables)
    hex_json = {
        name: df.to_dict(orient="list")
        for name, df in hex_dfs.items()
    }

    manifest = {
        "version": 1,
        "hex_tables": hex_json,
        "predict_lut": predict_lut.tolist(),
        "marginal_fl": marginal_fl.tolist(),
        "region_set_name": region_set_name,
        "region_set_hash": region_set_hash,
        "reference_name": reference_name,
        "reference_hash": reference_hash,
        "region_len": region_len,
        "l_min": l_min,
        "l_max": l_max,
        "fl_bands": [list(b) for b in fl_bands],
        "per_region_counts": per_region_counts,
        "rng_seed": rng_seed,
        "commit_sha": commit_sha,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load a simulation manifest and reconstruct the factor arrays.

    Returns a dict with keys:
    - ``hex_tables``: ``HexamerTables`` (reconstructed from the string-keyed DFs)
    - ``predict_lut``: ndarray, shape (N_LENGTHS, N_GC_BINS)
    - ``marginal_fl``: ndarray, shape (N_LENGTHS,)
    - plus all provenance fields from the manifest.
    """
    with open(manifest_path) as f:
        raw = json.load(f)

    # Reconstruct hex tables from string-keyed DataFrames
    hex_dfs = {
        name: pd.DataFrame(raw["hex_tables"][name])
        for name in HexamerTables._fields
    }
    hex_tables = dict_to_hex_tables(hex_dfs)

    predict_lut = np.array(raw["predict_lut"], dtype=np.float64)
    marginal_fl = np.array(raw["marginal_fl"], dtype=np.float64)

    return {
        "hex_tables": hex_tables,
        "predict_lut": predict_lut,
        "marginal_fl": marginal_fl,
        "region_set_name": raw["region_set_name"],
        "region_set_hash": raw["region_set_hash"],
        "reference_name": raw["reference_name"],
        "reference_hash": raw["reference_hash"],
        "region_len": raw["region_len"],
        "l_min": raw["l_min"],
        "l_max": raw["l_max"],
        "fl_bands": [tuple(b) for b in raw["fl_bands"]],
        "per_region_counts": raw.get("per_region_counts"),
        "rng_seed": raw.get("rng_seed"),
        "commit_sha": raw.get("commit_sha"),
    }


def build_fragments_h5(
    bed_gz_path: str,
    h5_path: str,
    fasta_path: str,
) -> None:
    """Call ``build-fragments-h5`` to produce a fragment h5 from a bgzipped BED.

    This delegates to the production tool, which computes GC from the FASTA
    (not from the BED).  GC in the h5 therefore comes from the real reference
    through production code — the simulator does not emit GC.
    """
    subprocess.run(
        [
            "build-fragments-h5",
            bed_gz_path,
            h5_path,
            "--fasta", fasta_path,
        ],
        check=True,
    )
