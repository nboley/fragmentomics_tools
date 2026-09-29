"""CutSiteFragmentDataset — torch Dataset over a cut-site fragment store.

Yields one region at a time (S=1 currently).  The item dict contains the
one-hot sequence, fragment coordinates, endpoint NB targets, and a mask.

Centre-based fragment inclusion: a fragment survives the crop iff its centre
m = p + L//2 lands inside [0, tile_size).  This gives |D_L| = P for every
in-band L, eliminating length bias.  Do NOT filter by "both endpoints in
crop" — that creates length bias (see design doc §Crop).

Shared-jitter coupling (S > 1 only): all samples of a region share one
jitter offset.  At S=1 this is moot.  Document preserved for future reference.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from background_model.config import check_fl_bands
from background_model.cut_site_model import L_MAX_INBAND, L_MIN_INBAND, N_L, WINDOW_HALF
from background_model.cut_site_store import (
    CutSiteStoreConfig,
    densify_endpoints,
    validate_max_len,
)
from background_model_core import calculate_start_and_stop_from_jitter, jitter_matrix
from fragmentomics_tools.region import one_hot_encode_sequences

# Split and role codes (same as 12-track dataset)
_SPLIT_CODES = {"train": 0, "val": 1, "heldout_inactive": 2, "positive_control": 3}
_ROLE_CODES = {"train": 0, "heldout": 1, "dropped_low_depth": 2}


class CutSiteFragmentDataset(Dataset):
    """Dataset over a cut-site fragment store.

    Each item is one (sample, region) pair yielding a dict with:
        x         : (4, l_in)         float32  one-hot sequence
        frag_m    : (n_frag,)         int64    centre positions in crop coords
        frag_L_idx: (n_frag,)         int64    L index (0-based, L - L_MIN)
        frag_s    : (n_frag,)         int64    strand (0 or 1)
        frag_N    : scalar int64               total fragment count
        endpoints : (2, tile_size)    float32  band-free NB targets
        mask      : (tile_size,)      bool     valid positions
    """

    def __init__(
        self,
        store_path: str,
        model_input_size: int,
        model_max_len: int,
        split: str,
        sample_role: str = "train",
        min_N: int = 10,
        train_mode: bool = True,
        jitter: Optional[int] = None,
        seed: Optional[int] = None,
        preload: bool = True,
    ):
        if split not in _SPLIT_CODES:
            raise ValueError(f"split must be one of {sorted(_SPLIT_CODES)}")
        if sample_role not in _ROLE_CODES:
            raise ValueError(f"sample_role must be one of {sorted(_ROLE_CODES)}")

        self.store_path = store_path
        self.model_input_size = int(model_input_size)
        self.model_max_len = int(model_max_len)
        self.min_N = int(min_N)
        self.train_mode = bool(train_mode)
        self._seed = seed

        root = zarr.open_group(store_path, mode="r")

        # ── config / guards ──────────────────────────────────────────────
        self.config = CutSiteStoreConfig.from_json(root.attrs["config_json"])
        self.config_hash = root.attrs["config_hash"]
        self.split_version = root.attrs["split_version"]
        validate_max_len(root, model_max_len)
        recorded_fl = json.loads(root.attrs["config_json"]).get("fl_bands")
        check_fl_bands(recorded_fl)

        # ── geometry ─────────────────────────────────────────────────────
        self.tile_size = self.config.tile_size
        self.l_target = self.config.l_target
        self.l_seq = self.config.l_seq
        self.jitter = self.config.jitter if jitter is None else int(jitter)

        # Parity checks
        for name, val in (
            ("tile_size", self.tile_size),
            ("l_target", self.l_target),
            ("l_seq", self.l_seq),
            ("model_input_size", self.model_input_size),
        ):
            if val % 2 != 0:
                raise AssertionError(f"{name}={val} must be even")
        if self.l_seq < self.model_input_size + 2 * self.jitter:
            raise AssertionError(
                f"l_seq={self.l_seq} too small for model_input_size="
                f"{self.model_input_size} with jitter={self.jitter}"
            )

        # ── build (sample, tile) index ───────────────────────────────────
        split_arr = root["tiles/split"][:]      # (T,)
        role_arr = root["samples/role"][:]       # (S,)
        N_totals = root["totals/N"][:]           # (S, T, 2)
        self.n_tiles = N_totals.shape[1]
        self.n_samples = N_totals.shape[0]

        split_code = _SPLIT_CODES[split]
        role_code = _ROLE_CODES[sample_role]
        sample_ok = np.nonzero(role_arr == role_code)[0]
        tile_ok = np.nonzero(split_arr == split_code)[0]

        # Filter on total fragment count per (sample, region).
        # Fragment count = indptr[u+1] - indptr[u]
        frag_indptr = root["fragments/indptr"][:]  # (S*T + 1,)
        index = []
        for s in sample_ok:
            for t in tile_ok:
                u = int(s) * self.n_tiles + int(t)
                n_frags = int(frag_indptr[u + 1] - frag_indptr[u])
                if n_frags >= self.min_N:
                    index.append((int(s), int(t)))
        self.index = index

        # ── worker-safe state ────────────────────────────────────────────
        self._root = None
        self._root_pid = None
        self._rng = None

        # ── optional preload ─────────────────────────────────────────────
        self._preloaded = False
        if preload:
            self._preload(root)

    def _preload(self, root):
        """Read all arrays into resident numpy arrays (COW fork sharing)."""
        self._mem_frag_indptr = np.asarray(root["fragments/indptr"][:])
        self._mem_frag_start = np.asarray(root["fragments/start"][:])
        self._mem_frag_length = np.asarray(root["fragments/length"][:])
        self._mem_frag_strand = np.asarray(root["fragments/strand"][:])
        self._mem_ep_indptr = np.asarray(root["endpoints/indptr"][:])
        self._mem_ep_pos = np.asarray(root["endpoints/pos"][:])
        self._mem_ep_track = np.asarray(root["endpoints/track"][:])
        self._mem_ep_data = np.asarray(root["endpoints/data"][:])
        self._mem_mask = np.asarray(root["tiles/mask"][:])
        self._mem_seq = np.asarray(root["tiles/seq"][:])
        self._preloaded = True

    def _get_root(self):
        pid = os.getpid()
        if self._root is None or self._root_pid != pid:
            self._root = zarr.open_group(self.store_path, mode="r")
            self._root_pid = pid
            base = 0x9E3779B9 if self._seed is None else int(self._seed)
            self._rng = np.random.default_rng([base, pid])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s, t = self.index[i]
        u = s * self.n_tiles + t

        # ── Load raw arrays ──────────────────────────────────────────────
        if self._preloaded:
            flo = int(self._mem_frag_indptr[u])
            fhi = int(self._mem_frag_indptr[u + 1])
            frag_start = self._mem_frag_start[flo:fhi].copy()
            frag_length = self._mem_frag_length[flo:fhi].copy()
            frag_strand = self._mem_frag_strand[flo:fhi].copy()

            elo = int(self._mem_ep_indptr[u])
            ehi = int(self._mem_ep_indptr[u + 1])
            ep_pos = self._mem_ep_pos[elo:ehi]
            ep_track = self._mem_ep_track[elo:ehi]
            ep_data = self._mem_ep_data[elo:ehi]

            mask_full = self._mem_mask[t]
            seq_full = self._mem_seq[t]
        else:
            self._get_root()
            root = self._root
            frag_indptr = root["fragments/indptr"]
            flo = int(frag_indptr[u])
            fhi = int(frag_indptr[u + 1])
            frag_start = np.asarray(root["fragments/start"][flo:fhi])
            frag_length = np.asarray(root["fragments/length"][flo:fhi])
            frag_strand = np.asarray(root["fragments/strand"][flo:fhi])

            ep_indptr = root["endpoints/indptr"]
            elo = int(ep_indptr[u])
            ehi = int(ep_indptr[u + 1])
            ep_pos = np.asarray(root["endpoints/pos"][elo:ehi])
            ep_track = np.asarray(root["endpoints/track"][elo:ehi])
            ep_data = np.asarray(root["endpoints/data"][elo:ehi])

            mask_full = np.asarray(root["tiles/mask"][t]).astype(bool)
            seq_full = np.asarray(root["tiles/seq"][t]).astype(np.uint8)

        # ── Jitter offset ────────────────────────────────────────────────
        if self.train_mode:
            if self._rng is None:
                base = 0x9E3779B9 if self._seed is None else int(self._seed)
                self._rng = np.random.default_rng([base, os.getpid()])
            j = int(self._rng.integers(-self.jitter, self.jitter + 1))
        else:
            j = 0

        # ── Crop: concentric crops with shared jitter offset ─────────────
        # Sequence → l_in (model_input_size + 5, rounded up to even for
        # jitter_matrix's matched-parity requirement).  The model trims
        # any extra position from the rounding.
        l_in = self.model_input_size + 5
        l_in_even = l_in + (l_in % 2)  # round up to even
        x_tokens = jitter_matrix(seq_full, j, l_in_even)

        # Mask and endpoints → tile_size
        mask = jitter_matrix(mask_full, j, self.tile_size)

        # Endpoint tracks: densify over l_target, then crop to tile_size
        ep_dense = densify_endpoints(ep_pos, ep_track, ep_data, self.l_target)
        ep_cropped = jitter_matrix(ep_dense, j, self.tile_size)
        # Zero masked positions (same policy as 12-track dataset)
        ep_cropped = ep_cropped * mask[None, :]

        # ── Fragment coordinates: centre-based crop filter ───────────────
        # Crop origin in l_target coordinates
        crop_start, _ = calculate_start_and_stop_from_jitter(
            self.l_target, self.tile_size, j
        )

        # Re-base fragment starts to crop coordinates
        frag_p_crop = frag_start.astype(np.int64) - crop_start

        # Centre of each fragment in crop coordinates
        frag_centre = frag_p_crop + frag_length.astype(np.int64) // 2

        # Centre-based filter: m in [0, tile_size)
        keep = (frag_centre >= 0) & (frag_centre < self.tile_size)

        # Also filter to in-band lengths
        keep &= (frag_length >= L_MIN_INBAND) & (frag_length <= L_MAX_INBAND)

        frag_m = frag_centre[keep]
        frag_L = frag_length[keep].astype(np.int64)
        frag_s = frag_strand[keep].astype(np.int64)
        frag_L_idx = frag_L - L_MIN_INBAND  # 0-based index into n_L

        # ── One-hot encode ───────────────────────────────────────────────
        x_tokens = np.ascontiguousarray(x_tokens, dtype=np.uint8)
        onehot = one_hot_encode_sequences([x_tokens.tobytes()])[0]
        x = np.ascontiguousarray(onehot.T, dtype=np.float32)  # (4, l_in_even)

        return {
            "x": torch.from_numpy(x),
            "frag_m": torch.from_numpy(frag_m.astype(np.int64)),
            "frag_L_idx": torch.from_numpy(frag_L_idx.astype(np.int64)),
            "frag_s": torch.from_numpy(frag_s.astype(np.int64)),
            "frag_N": torch.tensor(len(frag_m), dtype=torch.int64),
            "endpoints": torch.from_numpy(
                np.ascontiguousarray(ep_cropped, dtype=np.float32)
            ),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask, dtype=bool)
            ),
        }


def cut_site_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Custom collate for variable-length fragment lists.

    Fixed-shape tensors (x, endpoints, mask) are stacked normally.
    Variable-length fragment tensors (frag_m, frag_L_idx, frag_s) are kept
    as lists of per-sample tensors.
    """
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "frag_m": [b["frag_m"] for b in batch],
        "frag_L_idx": [b["frag_L_idx"] for b in batch],
        "frag_s": [b["frag_s"] for b in batch],
        "frag_N": torch.stack([b["frag_N"] for b in batch]),
        "endpoints": torch.stack([b["endpoints"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }
