"""PlumbingConfig — frozen configuration + content-addressed hash.

The config hash uniquely identifies the *data* that Phase A produces (geometry,
filters, sample draw parameters).  Phase-B-only parameters (region_fracs,
min_total_fragments, min_N) are EXCLUDED from the hash so that threshold sweeps
don't force a full re-preprocess.  They are recorded in store attrs and versioned
by split_version.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from typing import Dict, Tuple

# Track constants duplicated here to avoid importing background_model_core
# (which requires lightning/torch).  These MUST match background_model_core.py.
_STRANDS = ("+", "-")
_FL_BANDS_DEFAULT = ((40, 65), (120, 175))
_COVERAGE_TYPES = ("first", "last", "midpoint")
_N_DEFAULT_TRACKS = len(_STRANDS) * len(_FL_BANDS_DEFAULT) * len(_COVERAGE_TYPES)  # 12

# ── geometry constants (from design §0) ──────────────────────────────────

TILE = 16_384
JITTER = 128
RF_BUDGET = 2_048
L_TARGET = TILE + 2 * JITTER          # 16_640 — stored counts/mask extent
L_SEQ = TILE + 2 * (JITTER + RF_BUDGET)  # 20_736 — stored sequence extent
C = _N_DEFAULT_TRACKS                   # 12

# ── fields excluded from the config hash ─────────────────────────────────

_HASH_EXCLUDED_FIELDS = frozenset({
    "region_fracs",
    "min_total_fragments",
    "min_N",
})


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fai_md5(fasta_path: str) -> str:
    """Hash the .fai index as a proxy for the FASTA content identity."""
    fai = fasta_path + ".fai"
    if not os.path.exists(fai):
        fai = fasta_path.replace(".fa.gz", ".fa.fai")
        if not os.path.exists(fai):
            raise FileNotFoundError(
                f"Cannot find .fai for {fasta_path}; tried {fasta_path}.fai"
            )
    return _file_md5(fai)


@dataclass(frozen=True)
class PlumbingConfig:
    """Frozen configuration for the background-model data pipeline.

    All file paths are replaced by content hashes in the config hash so that
    the hash is path-independent (moving files doesn't invalidate stores).
    """

    # ── inputs ────────────────────────────────────────────────────────────
    sample_sheet: str              # TSV path
    region_beds: Dict[str, str] = field(default_factory=dict)
    blacklist_bed: str = ""
    fasta: str = ""

    # ── geometry ──────────────────────────────────────────────────────────
    tile_size: int = TILE
    jitter: int = JITTER
    rf_budget: int = RF_BUDGET
    fl_bands: Tuple[Tuple[int, int], ...] = _FL_BANDS_DEFAULT

    # ── counting ──────────────────────────────────────────────────────────
    min_mapq: int = 10
    dedup: bool = True
    blacklist_expansion: int = 120

    # ── sample draw (affects which Phase A shards exist → HASHED) ─────────
    seed: int = 1337
    n_train_samples: int = 40
    n_heldout_samples: int = 10

    # ── Phase-B-only / consumer params → NOT hashed ──────────────────────
    region_fracs: Tuple[float, ...] = (0.8, 0.1, 0.1)
    min_total_fragments: int = 20_000_000
    min_N: int = 50

    # ── derived geometry (read-only) ─────────────────────────────────────

    @property
    def l_target(self) -> int:
        return self.tile_size + 2 * self.jitter

    @property
    def l_seq(self) -> int:
        return self.tile_size + 2 * (self.jitter + self.rf_budget)

    @property
    def max_frag_len(self) -> int:
        """Exclusive upper bound on fragment length (matches subset_fragment_lengths '<' semantics)."""
        return max(hi for _lo, hi in self.fl_bands)

    # ── hashing ──────────────────────────────────────────────────────────

    def _canonical_dict(self, *, with_content_hashes: bool = True) -> dict:
        """Dict of hash-included fields, file paths replaced by content hashes."""
        d = {}
        for f in fields(self):
            if f.name in _HASH_EXCLUDED_FIELDS:
                continue
            val = getattr(self, f.name)

            if with_content_hashes:
                if f.name == "sample_sheet" and val:
                    val = _file_md5(val)
                elif f.name == "blacklist_bed" and val:
                    val = _file_md5(val)
                elif f.name == "fasta" and val:
                    val = _fai_md5(val)
                elif f.name == "region_beds" and val:
                    val = {k: _file_md5(v) for k, v in sorted(val.items())}

            # Normalize tuples to lists for JSON
            if isinstance(val, tuple):
                val = _tuple_to_list(val)

            d[f.name] = val
        return d

    def canonical_json(self, *, with_content_hashes: bool = True) -> str:
        return json.dumps(
            self._canonical_dict(with_content_hashes=with_content_hashes),
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_hash(self, *, with_content_hashes: bool = True) -> str:
        return hashlib.sha256(
            self.canonical_json(with_content_hashes=with_content_hashes).encode()
        ).hexdigest()

    def config_hash8(self, **kwargs) -> str:
        return self.config_hash(**kwargs)[:8]

    def store_name(self, **kwargs) -> str:
        return f"bg_store_{self.config_hash8(**kwargs)}.zarr"

    # ── full config JSON (for store attrs / sidecar) ─────────────────────

    def full_config_json(self) -> str:
        """Complete config including Phase-B params (for store attrs)."""
        d = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, tuple):
                val = _tuple_to_list(val)
            d[f.name] = val
        return json.dumps(d, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PlumbingConfig":
        d = json.loads(json_str)
        # Convert lists back to tuples for frozen fields
        if "fl_bands" in d:
            d["fl_bands"] = tuple(tuple(b) for b in d["fl_bands"])
        if "region_fracs" in d:
            d["region_fracs"] = tuple(d["region_fracs"])
        if "region_beds" in d:
            d["region_beds"] = dict(d["region_beds"])
        return cls(**d)


def _tuple_to_list(val):
    if isinstance(val, tuple):
        return [_tuple_to_list(v) for v in val]
    return val


def verify_config_drift(store_path: str, config: PlumbingConfig) -> None:
    """Verify that a store's config matches the given config.

    Raises ValueError on mismatch (drift rule: never silently append to a
    store whose config has changed).
    """
    import zarr

    store = zarr.open_group(store_path, mode="r")
    stored_hash = store.attrs["config_hash"]
    current_hash = config.config_hash()
    if stored_hash != current_hash:
        raise ValueError(
            f"Config drift detected: store hash {stored_hash!r} != "
            f"current hash {current_hash!r}. Build a new store."
        )
