"""Cut-site fragment store — schema, validation, and CSR access.

A separate zarr store from the 12-track store.  It carries per sample × region
the fragment list (the multinomial target) and two band-free endpoint count
tracks (the NB ancillary targets).

Store layout (zarr v2):
    /attrs: config_json, config_hash, created_utc, split_version, max_len, ...
    /tiles/{contig, start, stop, strand, region_id, split, seq, mask}
    /samples/{library, role, total_fragments}
    /fragments/ (CSR over flat index u = s*T + t)
        indptr   (S*T + 1,)  int64
        start    (nnz,)      uint16   # region-LOCAL left endpoint p
        length   (nnz,)      uint16   # fragment length L
        strand   (nnz,)      uint8    # 0 = +, 1 = -
    /endpoints/ (CSR over flat index u = s*T + t)
        indptr   (S*T + 1,)  int64
        pos      (nnz2,)     uint16
        track    (nnz2,)     uint8    # 0 = first (left), 1 = last (right)
        data     (nnz2,)     uint16
    /totals/N  (S, T, 2)    uint32
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import zarr

from background_model.tracks import FL_BANDS
from background_model.config import check_fl_bands


# ---------------------------------------------------------------------------
# Config for the cut-site store (separate from PlumbingConfig)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutSiteStoreConfig:
    """Geometry + build params for the cut-site fragment store.

    Hashed separately from PlumbingConfig so that adding max_len does NOT
    rehash the 12-track production store.
    """
    tile_size: int = 2048
    jitter: int = 256
    rf_budget: int = 133     # WINDOW_HALF + ceil(conv_margin/2) = 128 + 5 = 133
                             # l_seq must be >= l_in + 2*jitter; rf_budget
                             # provides the (l_seq - l_target)/2 headroom.
    max_len: int = 256
    seed: int = 42
    n_train_samples: int = 1
    fl_bands: Tuple[Tuple[int, int], ...] = FL_BANDS

    @property
    def l_target(self) -> int:
        return self.tile_size + 2 * self.jitter

    @property
    def l_seq(self) -> int:
        return self.tile_size + 2 * (self.jitter + self.rf_budget)

    def config_hash(self) -> str:
        d = asdict(self)
        d["fl_bands"] = [list(b) for b in d["fl_bands"]]
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        d = asdict(self)
        d["fl_bands"] = [list(b) for b in d["fl_bands"]]
        return json.dumps(d, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CutSiteStoreConfig":
        d = json.loads(s)
        d["fl_bands"] = tuple(tuple(b) for b in d["fl_bands"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Store open / validate
# ---------------------------------------------------------------------------


def open_cut_site_store(
    store_path: str,
    config: Optional[CutSiteStoreConfig] = None,
    mode: str = "r",
) -> zarr.Group:
    """Open a cut-site store; optionally verify config drift and fl_bands."""
    root = zarr.open_group(store_path, mode=mode)
    if config is not None:
        stored_hash = root.attrs["config_hash"]
        current_hash = config.config_hash()
        if stored_hash != current_hash:
            raise ValueError(
                f"Config drift: store hash {stored_hash!r} != "
                f"current hash {current_hash!r}. Build a new store."
            )
    # Always check fl_bands
    recorded_fl = json.loads(root.attrs["config_json"]).get("fl_bands")
    check_fl_bands(recorded_fl)
    return root


def validate_max_len(store_root: zarr.Group, model_max_len: int) -> None:
    """Assert model's max_len matches the store's.  Fail loudly on mismatch."""
    cfg = json.loads(store_root.attrs["config_json"])
    store_max_len = cfg["max_len"]
    if store_max_len != model_max_len:
        raise ValueError(
            f"max_len mismatch: model expects {model_max_len} but store has "
            f"{store_max_len}. The store holds a different fragment population."
        )


# ---------------------------------------------------------------------------
# Fragment CSR access
# ---------------------------------------------------------------------------


def fragment_slice(
    root: zarr.Group,
    sample_idx: int,
    tile_idx: int,
    n_tiles: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (start, length, strand) arrays for one (sample, tile)."""
    u = sample_idx * n_tiles + tile_idx
    indptr = root["fragments/indptr"]
    lo = int(indptr[u])
    hi = int(indptr[u + 1])
    if lo == hi:
        return (
            np.empty(0, np.uint16),
            np.empty(0, np.uint16),
            np.empty(0, np.uint8),
        )
    start = np.asarray(root["fragments/start"][lo:hi])
    length = np.asarray(root["fragments/length"][lo:hi])
    strand = np.asarray(root["fragments/strand"][lo:hi])
    return start, length, strand


def endpoint_slice(
    root: zarr.Group,
    sample_idx: int,
    tile_idx: int,
    n_tiles: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, track, data) arrays for one (sample, tile)."""
    u = sample_idx * n_tiles + tile_idx
    indptr = root["endpoints/indptr"]
    lo = int(indptr[u])
    hi = int(indptr[u + 1])
    if lo == hi:
        return (
            np.empty(0, np.uint16),
            np.empty(0, np.uint8),
            np.empty(0, np.uint16),
        )
    pos = np.asarray(root["endpoints/pos"][lo:hi])
    track = np.asarray(root["endpoints/track"][lo:hi])
    data = np.asarray(root["endpoints/data"][lo:hi])
    return pos, track, data


def densify_endpoints(
    pos: np.ndarray,
    track: np.ndarray,
    data: np.ndarray,
    length: int,
) -> np.ndarray:
    """Densify endpoint CSR triples into (2, length) float32 array."""
    y = np.zeros((2, length), dtype=np.float32)
    if len(pos) > 0:
        np.add.at(y, (track, pos), data.astype(np.float32))
    return y


def recompute_endpoints_from_fragments(
    frag_start: np.ndarray,
    frag_length: np.ndarray,
    length: int,
) -> np.ndarray:
    """Recompute the 2 endpoint tracks from a fragment list.

    first[p] += 1 for each fragment; last[p+L-1] += 1.
    Returns (2, length) float32.
    """
    y = np.zeros((2, length), dtype=np.float32)
    if len(frag_start) == 0:
        return y
    np.add.at(y[0], frag_start.astype(np.intp), 1.0)
    right = (frag_start + frag_length - 1).astype(np.intp)
    valid = right < length
    np.add.at(y[1], right[valid], 1.0)
    return y
