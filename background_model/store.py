"""Zarr store layout, open/validate, and CSR access helpers.

Store layout (zarr v2 format):
    /attrs: config_json, config_hash, created_utc, split_version, ...
    /tiles/{contig, start, stop, strand, region_id, split, seq, mask}
    /samples/{library, seqrun, endo_category, h5_path, role, total_fragments}
    /counts/{indptr, pos, track, data}   — CSR over flat index u = s*T + t
    /totals/N  (S, T, C)
"""

import datetime

import numpy as np
import zarr

from background_model.config import PlumbingConfig, C, L_SEQ, L_TARGET, TILE

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])


def _create_array(group, name, overwrite=False, **kwargs):
    """Create an array in a zarr group, compatible with zarr 2.x and 3.x.

    `overwrite` is handled uniformly by deleting any existing member first,
    since zarr 3's ``create_array`` and zarr 2's ``create_dataset`` differ in
    their native support for the ``overwrite`` keyword.
    """
    if overwrite and name in group:
        del group[name]
    if _ZARR_MAJOR >= 3:
        return group.create_array(name, **kwargs)
    else:
        return group.create_dataset(name, **kwargs)


def _get_version_info():
    """Collect version strings for store attrs."""
    from background_model.train import _git_sha
    try:
        git_sha = _git_sha()
    except Exception:
        git_sha = "unknown"
    try:
        import fragmentomics_tools
        ft_version = getattr(fragmentomics_tools, "__version__", "unknown")
    except Exception:
        ft_version = "unknown"
    return git_sha, ft_version


def create_store(
    store_path: str,
    config: PlumbingConfig,
    n_tiles: int,
    n_samples: int,
    nnz: int,
) -> zarr.Group:
    """Create a new zarr v2 store with the canonical layout.

    Returns the open zarr Group (mode='w').
    """
    open_kwargs = {"mode": "w"}
    if _ZARR_MAJOR >= 3:
        open_kwargs["zarr_format"] = 2
    root = zarr.open_group(store_path, **open_kwargs)
    git_sha, ft_version = _get_version_info()

    root.attrs.update({
        "config_json": config.full_config_json(),
        "config_hash": config.config_hash(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "zarr_format_note": f"zarr v2, zarr-python {zarr.__version__}",
        "code_version": git_sha,
        "fragmentomics_tools_version": ft_version,
        "split_version": 0,
    })

    S, T = n_samples, n_tiles

    # ── /tiles/ ──────────────────────────────────────────────────────────
    tiles = root.create_group("tiles")
    _create_array(tiles,"contig", shape=(T,), dtype="<U32", chunks=(T,))
    _create_array(tiles,"start", shape=(T,), dtype="int64", chunks=(T,))
    _create_array(tiles,"stop", shape=(T,), dtype="int64", chunks=(T,))
    _create_array(tiles,"strand", shape=(T,), dtype="<U1", chunks=(T,))
    _create_array(tiles,"region_id", shape=(T,), dtype="<U64", chunks=(T,))
    _create_array(tiles,"split", shape=(T,), dtype="uint8", chunks=(T,))
    _create_array(tiles,"seq", shape=(T, config.l_seq), dtype="uint8", chunks=(64, config.l_seq))
    _create_array(tiles,"mask", shape=(T, config.l_target), dtype="bool", chunks=(64, config.l_target))

    # ── /samples/ ────────────────────────────────────────────────────────
    samples = root.create_group("samples")
    _create_array(samples,"library", shape=(S,), dtype="<U64", chunks=(S,))
    _create_array(samples,"seqrun", shape=(S,), dtype="<U64", chunks=(S,))
    _create_array(samples,"endo_category", shape=(S,), dtype="<U64", chunks=(S,))
    _create_array(samples,"h5_path", shape=(S,), dtype="<U256", chunks=(S,))
    _create_array(samples,"role", shape=(S,), dtype="uint8", chunks=(S,))
    _create_array(samples,"total_fragments", shape=(S,), dtype="uint64", chunks=(S,))

    # ── /counts/ (CSR) ───────────────────────────────────────────────────
    counts = root.create_group("counts")
    _create_array(counts,"indptr", shape=(S * T + 1,), dtype="int64", chunks=(1 << 20,))
    _create_array(counts,"pos", shape=(nnz,), dtype="uint16", chunks=(1 << 20,))
    _create_array(counts,"track", shape=(nnz,), dtype="uint8", chunks=(1 << 20,))
    _create_array(counts,"data", shape=(nnz,), dtype="uint16", chunks=(1 << 20,))

    # ── /totals/ ─────────────────────────────────────────────────────────
    totals = root.create_group("totals")
    _create_array(totals,"N", shape=(S, T, C), dtype="uint32", chunks=(S, T, C))

    return root


def open_store(store_path: str, config: PlumbingConfig = None, mode: str = "r") -> zarr.Group:
    """Open an existing store; optionally verify config drift."""
    root = zarr.open_group(store_path, mode=mode)
    if config is not None:
        stored_hash = root.attrs["config_hash"]
        current_hash = config.config_hash()
        if stored_hash != current_hash:
            raise ValueError(
                f"Config drift: store hash {stored_hash!r} != "
                f"current hash {current_hash!r}. Build a new store."
            )
    return root


def increment_split_version(root: zarr.Group) -> int:
    """Increment split_version attr; return the new value."""
    v = root.attrs.get("split_version", 0) + 1
    root.attrs["split_version"] = v
    return v


def record_phase_b_params(root: zarr.Group, config: PlumbingConfig) -> None:
    """Write the applied Phase-B params into attrs."""
    root.attrs["applied_region_fracs"] = list(config.region_fracs)
    root.attrs["applied_min_total_fragments"] = config.min_total_fragments


# ── CSR access ───────────────────────────────────────────────────────────

def csr_slice(root: zarr.Group, sample_idx: int, tile_idx: int, n_tiles: int):
    """Return (pos, track, data) arrays for one (sample, tile) pair."""
    u = sample_idx * n_tiles + tile_idx
    indptr = root["counts/indptr"]
    lo = int(indptr[u])
    hi = int(indptr[u + 1])
    if lo == hi:
        return np.empty(0, "uint16"), np.empty(0, "uint8"), np.empty(0, "uint16")
    pos = np.asarray(root["counts/pos"][lo:hi])
    track = np.asarray(root["counts/track"][lo:hi])
    data = np.asarray(root["counts/data"][lo:hi])
    return pos, track, data


def densify_counts(
    pos: np.ndarray,
    track: np.ndarray,
    data: np.ndarray,
    n_tracks: int = C,
    length: int = L_TARGET,
) -> np.ndarray:
    """Densify CSR triples into (C, L_TARGET) float32 array."""
    y = np.zeros((n_tracks, length), dtype=np.float32)
    if len(pos) > 0:
        np.add.at(y, (track, pos), data.astype(np.float32))
    return y


def compute_N_for_tile(
    y_dense: np.ndarray,
    mask: np.ndarray,
    tile_size: int = TILE,
    l_target: int = L_TARGET,
) -> np.ndarray:
    """Compute per-track totals over the CENTER TILE (unmasked positions only).

    y_dense: (C, L_TARGET); mask: (L_TARGET,) bool (True=valid).
    Returns (C,) uint32.
    """
    margin = (l_target - tile_size) // 2
    center_slice = slice(margin, margin + tile_size)
    center_y = y_dense[:, center_slice]
    center_mask = mask[center_slice]
    return (center_y * center_mask[None, :]).sum(axis=1).astype(np.uint32)
