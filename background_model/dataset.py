"""BackgroundTileDataset — torch Dataset over a preprocessed zarr store.

Yields ``(x, y, mask)`` per (sample, tile):
    x    : (4, model_input_size)  float32  one-hot sequence
    y    : (C, TILE)              float32  that sample's per-track counts
    mask : (TILE,)                bool     valid (non-blacklist, in-contig)

The store carries counts/mask over the L_TARGET extent (TILE + 2*jitter) and
sequence over the L_SEQ extent (TILE + 2*(jitter + rf_budget)); all three share
the tile center.  At load time a jitter offset is drawn (train) and applied to
all three via ``jitter_matrix`` so they stay center-aligned, then an optional
reverse-complement augmentation is applied (train only).

Reproducibility contract: ``config_hash`` and ``split_version`` are exposed as
attributes; every training run MUST log the pair (a Phase B re-run bumps
``split_version`` and invalidates artifacts trained against the prior split).

Worker safety: the zarr store handle is opened lazily per worker PID (never
shared across a fork), reads are read-only, and there are NO torch CUDA calls
anywhere in this module.
"""

import os
from typing import Optional

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from background_model.config import PlumbingConfig
from background_model_core import (
    DEFAULT_OUTPUT_TRACKS,
    jitter_matrix,
    reverse_complement_track_permutation,
)
from fragmentomics_tools.region import one_hot_encode_sequences

# split codes written by Phase B (§1 store layout)
_SPLIT_CODES = {
    "train": 0,
    "val": 1,
    "heldout_inactive": 2,
    "positive_control": 3,
}
# sample role codes (§6)
_ROLE_CODES = {
    "train": 0,
    "heldout": 1,
    "dropped_low_depth": 2,
}


def _build_complement_lut() -> np.ndarray:
    """256-entry ASCII complement LUT (A<->T, C<->G; case-preserving; N->N)."""
    lut = np.arange(256, dtype=np.uint8)
    pairs = {
        ord("A"): ord("T"), ord("T"): ord("A"),
        ord("C"): ord("G"), ord("G"): ord("C"),
        ord("a"): ord("t"), ord("t"): ord("a"),
        ord("c"): ord("g"), ord("g"): ord("c"),
    }
    for k, v in pairs.items():
        lut[k] = v
    return lut


_COMPLEMENT_LUT = _build_complement_lut()


class BackgroundTileDataset(Dataset):
    """torch Dataset over a preprocessed zarr store.

    Reproducibility of the augmentation stream: ``seed`` fixes index/split
    selection deterministically (the (sample, tile) index is built in
    ``__init__`` independent of any RNG). The per-item jitter/RC augmentation
    stream, however, is keyed on ``[seed, worker_pid]`` and lazily seeded per
    worker PID in ``_get_root`` — so it is NOT reproducible across runs (the
    PID set differs run to run). Determinism holds only within ``train_mode``
    off, where jitter=0 and RC is disabled.
    """

    def __init__(
        self,
        store_path: str,
        model_input_size: int,
        split: str,
        sample_role: str,
        min_N: int = 50,
        train_mode: bool = True,
        rc_prob: float = 0.5,
        jitter: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if split not in _SPLIT_CODES:
            raise ValueError(f"split must be one of {sorted(_SPLIT_CODES)} (got {split!r})")
        if sample_role not in _ROLE_CODES:
            raise ValueError(
                f"sample_role must be one of {sorted(_ROLE_CODES)} (got {sample_role!r})"
            )

        self.store_path = store_path
        self.model_input_size = int(model_input_size)
        self.split = split
        self.sample_role = sample_role
        self.min_N = int(min_N)
        self.train_mode = bool(train_mode)
        self.rc_prob = float(rc_prob)
        self._seed = seed

        root = zarr.open_group(store_path, mode="r")

        # ── reproducibility contract ─────────────────────────────────────
        self.config = PlumbingConfig.from_json(root.attrs["config_json"])
        self.config_hash = root.attrs["config_hash"]
        self.split_version = root.attrs["split_version"]

        # ── geometry (from the store's own config) ───────────────────────
        self.tile_size = self.config.tile_size
        self.l_target = self.config.l_target
        self.l_seq = self.config.l_seq
        self.jitter = self.config.jitter if jitter is None else int(jitter)

        # Fail LOUDLY on bad geometry (§0): all-even parity + RF budget.
        for name, val in (
            ("tile_size", self.tile_size),
            ("l_target", self.l_target),
            ("l_seq", self.l_seq),
            ("model_input_size", self.model_input_size),
        ):
            if val % 2 != 0:
                raise AssertionError(
                    f"{name}={val} must be even (jitter_matrix same-parity requirement)"
                )
        if self.l_seq < self.model_input_size + 2 * self.jitter:
            raise AssertionError(
                f"l_seq={self.l_seq} too small for model_input_size="
                f"{self.model_input_size} with jitter={self.jitter} "
                f"(need >= {self.model_input_size + 2 * self.jitter}); "
                "the model's receptive field exceeds RF_BUDGET."
            )
        if self.l_target < self.tile_size + 2 * self.jitter:
            raise AssertionError(
                f"l_target={self.l_target} too small for tile_size={self.tile_size} "
                f"with jitter={self.jitter} (need >= {self.tile_size + 2 * self.jitter})"
            )

        # ── track set / RC permutation ───────────────────────────────────
        N = root["totals/N"][:]                 # (S, T, C)
        C_store = N.shape[2]
        self.output_tracks = list(DEFAULT_OUTPUT_TRACKS)
        if len(self.output_tracks) != C_store:
            raise AssertionError(
                f"store has C={C_store} tracks but DEFAULT_OUTPUT_TRACKS has "
                f"{len(self.output_tracks)}"
            )
        self.n_tracks = C_store
        self.rc_perm = np.asarray(
            reverse_complement_track_permutation(self.output_tracks), dtype=np.int64
        )

        # ── build (sample, tile) index (§5) ──────────────────────────────
        split_arr = root["tiles/split"][:]       # (T,)
        role_arr = root["samples/role"][:]        # (S,)
        self.n_tiles = N.shape[1]
        split_code = _SPLIT_CODES[split]
        role_code = _ROLE_CODES[sample_role]

        sample_ok = np.nonzero(role_arr == role_code)[0]
        tile_ok = np.nonzero(split_arr == split_code)[0]
        n_min = N.min(axis=2)                     # (S, T): ALL tracks must pass
        index = []
        for s in sample_ok:
            for t in tile_ok:
                if n_min[s, t] >= self.min_N:
                    index.append((int(s), int(t)))
        self.index = index

        # per-PID lazy zarr handle (worker safety); NOT set from __init__.
        self._root = None
        self._root_pid = None
        self._rng = None
        # cached per-PID array handles + resident indptr (see _get_root)
        self._arrays = None
        self._indptr = None

    # ── worker-safe lazy handle ──────────────────────────────────────────

    def _get_root(self):
        pid = os.getpid()
        if self._root is None or self._root_pid != pid:
            self._root = zarr.open_group(self.store_path, mode="r")
            self._root_pid = pid
            base = 0x9E3779B9 if self._seed is None else int(self._seed)
            self._rng = np.random.default_rng([base, pid])
            # Cache the array handles once per process. Each root["..."]
            # lookup re-instantiates a zarr Array and re-reads its .zarray
            # metadata; measured on EFS that was ~6 metadata opens per
            # __getitem__ and 61% of per-item cost. indptr is ~2 MB, so it
            # is read fully resident and lo/hi become pure numpy.
            self._arrays = {
                k: self._root[k]
                for k in ("counts/pos", "counts/track", "counts/data",
                          "tiles/mask", "tiles/seq")
            }
            self._indptr = np.asarray(self._root["counts/indptr"][:])
        return self._root

    def __len__(self) -> int:
        return len(self.index)

    # ── core transform (crop + optional RC); pure numpy, testable ────────

    def _transform(self, y_full, mask_full, seq_full, j: int, do_rc: bool):
        # Same jitter offset applied to all three -> shared center (asserted
        # even parity at init).
        y = np.array(jitter_matrix(y_full, j, self.tile_size), dtype=np.float32)
        m = np.array(jitter_matrix(mask_full, j, self.tile_size), dtype=bool)
        x_tokens = np.ascontiguousarray(
            jitter_matrix(seq_full, j, self.model_input_size)
        )

        if do_rc:
            x_tokens = _COMPLEMENT_LUT[x_tokens][::-1]
            y = y[self.rc_perm][:, ::-1]
            m = m[::-1]

        # Zero targets at masked positions (model-boundary policy). Applied
        # AFTER the jitter crop + RC flip so y and m share the same frame; the
        # mask itself is unchanged. The frozen BackgroundModel._prepare_mask
        # asserts targets are zero wherever the mask is invalid — a blacklist-
        # SPANNING fragment survives mask_overlapping_fragments and can deposit
        # an endpoint inside the expanded masked zone, so stray counts would
        # otherwise corrupt N = target.sum in every loss. The zarr store keeps
        # RAW unzeroed counts; zeroing is a Dataset-boundary policy only.
        y = y * m

        x_tokens = np.ascontiguousarray(x_tokens, dtype=np.uint8)
        # encoder returns (N, L, 4); [0].T -> (4, L).  Input must be bytes.
        onehot = one_hot_encode_sequences([x_tokens.tobytes()])[0]
        x = np.ascontiguousarray(onehot.T, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        m = np.ascontiguousarray(m, dtype=bool)
        return x, y, m

    def __getitem__(self, i: int):
        self._get_root()
        arrays = self._arrays
        s, t = self.index[i]
        u = s * self.n_tiles + t

        lo = int(self._indptr[u])
        hi = int(self._indptr[u + 1])
        if hi > lo:
            pos = np.asarray(arrays["counts/pos"][lo:hi])
            track = np.asarray(arrays["counts/track"][lo:hi])
            data = np.asarray(arrays["counts/data"][lo:hi])
        else:
            pos = np.empty(0, np.uint16)
            track = np.empty(0, np.uint8)
            data = np.empty(0, np.uint16)

        y_full = np.zeros((self.n_tracks, self.l_target), dtype=np.float32)
        if len(pos):
            np.add.at(y_full, (track, pos), data.astype(np.float32))
        mask_full = np.asarray(arrays["tiles/mask"][t]).astype(bool)
        seq_full = np.asarray(arrays["tiles/seq"][t]).astype(np.uint8)

        if self.train_mode:
            j = int(self._rng.integers(-self.jitter, self.jitter + 1))
            do_rc = bool(self._rng.random() < self.rc_prob)
        else:
            j = 0
            do_rc = False

        x, y, m = self._transform(y_full, mask_full, seq_full, j, do_rc)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )
