"""CutSiteFragmentDataset — torch Dataset over a cut-site fragment store.

Yields one region at a time (S=1 currently).  The item dict contains the
one-hot sequence, fragment coordinates, endpoint NB targets, and a mask.

Start-based fragment admission: a fragment survives the crop iff its START
(the stored ``fragments/start``, the lower coordinate) lies in
[0, tile_size) of the crop and its length is in [L_min, L_max].  The end
may lie past the crop edge.  This gives |D_L| = P for every in-band L.
``frag_m`` is the START position, not the centre.  The rule lives in
``admit_fragments``; ``scripts/cut_site_model/stage1_hex_baseline.py``
calls the same function so that the hex prior and the model see one rule.

The stage-1 hex tables are built over a population that differs from this
loader's in two deliberate ways: centred crop only, and no ``min_N``
(``compute_hex_tables`` docstring has the reasons).  A change here to
admission, crop geometry or ``min_N`` semantics must be reflected there, and
that list kept true.

Every sample in the store is used.  There is no sample-role filter: older
stores carry a ``samples/role`` array, and nothing reads it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, get_worker_info

from background_model.cut_site_model.store import (
    SPLIT_CODES,
    CutSiteStoreConfig,
    densify_endpoints,
    validate_length_range,
)
from background_model_core import calculate_start_and_stop_from_jitter, jitter_matrix
from fragmentomics_tools.region import one_hot_encode_sequences

# Jitter-stream key when no seed is given.  Kept from the PID-keyed version.
_DEFAULT_JITTER_SEED = 0x9E3779B9


def crop_origin(l_target: int, tile_size: int, j: int) -> int:
    """Offset of the ``tile_size`` crop inside the ``l_target`` frame at
    jitter ``j``.  Store fragment starts minus this value are crop coords."""
    crop_start, _ = calculate_start_and_stop_from_jitter(l_target, tile_size, j)
    return crop_start


def admit_fragments(
    frag_start: np.ndarray,
    frag_length: np.ndarray,
    crop_start: int,
    tile_size: int,
    L_min: int,
    L_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Start-based admission of one (sample, region)'s stored fragments.

    Returns ``(p_crop, keep)``: every start re-based to crop coordinates
    (int64), and the boolean mask of admitted rows.  A row is admitted iff
    ``0 <= p_crop < tile_size`` and ``L_min <= length <= L_max`` (both
    inclusive; ``L_max`` is NOT half-open).  Only the start is tested; the
    end may lie past the crop edge.

    This is the ONE implementation of the rule.  ``CutSiteFragmentDataset``
    and ``stage1_hex_baseline.compute_hex_tables`` both call it; a second
    copy is how the stage-1 prior once drifted from the model's population.
    """
    p_crop = frag_start.astype(np.int64) - crop_start
    keep = (p_crop >= 0) & (p_crop < tile_size)
    keep &= (frag_length >= L_min) & (frag_length <= L_max)
    return p_crop, keep


class CutSiteFragmentDataset(Dataset):
    """Dataset over a cut-site fragment store.

    Each item is one (sample, region) pair yielding a dict with:
        x         : (4, l_in)         float32  one-hot sequence
        frag_m    : (n_frag,)         int64    start positions in crop coords
        frag_L_idx: (n_frag,)         int64    L index (0-based, L - L_MIN)
        frag_s    : (n_frag,)         int64    strand (0 or 1)
        frag_N    : scalar int64               number of fragments ADMITTED
                                               to this crop (== n_frag), not
                                               the stored count; can be 0
        frag_store_idx: (n_frag,)     int64    row in the store's fragments/*
                                               (not collated; for scorers)
        endpoints : (2, tile_size)    float32  band-free NB targets
        mask      : (tile_size,)      bool     valid positions

    ``min_N`` vs ``frag_N``.  ``min_N`` gates which (sample, region) pairs
    are indexed, on the STORED count ``indptr[u+1] - indptr[u]``, at
    construction and before any crop.  ``frag_N`` counts what survives the
    crop (``admit_fragments``).  The stored rows span ``l_target``, wider
    than the crop, so an item that passes ``min_N=10`` can still emit
    ``frag_N < 10``, including 0.  A train-mode item's ``frag_N`` also varies
    from draw to draw with the jitter.  Consumers must not assume
    ``frag_N >= min_N``.

    Jitter.  ``train_mode=False`` always uses ``j = 0`` (the centred crop).
    ``train_mode=True`` draws ``j`` uniformly from ``[-jitter, jitter]``,
    one fresh draw per ``__getitem__`` call, from a stream keyed on
    ``(seed, worker)``: ``worker`` is the DataLoader worker id, or the main
    process when there are no workers.  The same ``seed``, the same item
    order AND the same ``num_workers`` therefore give the same jitters in
    every run and every process; a different ``num_workers`` gives a
    different sequence.  Two workers never share a stream.  The stream is
    sequential, not a function of the item index, so an item gets a fresh
    ``j`` each time it is drawn (the design relies on this to recover
    jitter diversity).  With non-persistent workers each epoch restarts
    every worker's stream; the shuffle still pairs the draws with different
    items.
    The stream used to be keyed on ``(seed, os.getpid())``, which no seed
    could reproduce.  The change moved every train-mode jitter, and so moved
    training numbers once.  For scale: two unseeded runs of identical code
    under the PID key (also no global seed, unseeded shuffle) differed by
    ~0.002 nats on best-epoch val (``docs/pending/training_analysis.md``
    §11, Caveats, item 2).
    """

    def __init__(
        self,
        store_path: str,
        model_input_size: int,
        model_L_min: int,
        model_L_max: int,
        split: str,
        min_N: int = 10,
        train_mode: bool = True,
        jitter: Optional[int] = None,
        seed: Optional[int] = None,
        preload: bool = True,
    ):
        if split not in SPLIT_CODES:
            raise ValueError(f"split must be one of {sorted(SPLIT_CODES)}")
        if seed is not None and seed < 0:
            # default_rng rejects a negative entropy word, and would only say
            # so on the first train-mode draw, inside a DataLoader worker.
            raise ValueError(f"seed must be None or >= 0, got {seed}")

        self.store_path = store_path
        self.model_input_size = int(model_input_size)
        self.L_min = int(model_L_min)
        self.L_max = int(model_L_max)
        self.min_N = int(min_N)
        self.train_mode = bool(train_mode)
        self._seed = seed

        root = zarr.open_group(store_path, mode="r")

        # ── config / guards ──────────────────────────────────────────────
        self.config = CutSiteStoreConfig.from_json(root.attrs["config_json"])
        self.config_hash = root.attrs["config_hash"]
        self.split_version = root.attrs["split_version"]
        validate_length_range(root, model_L_min, model_L_max)

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
        N_totals = root["totals/N"][:]           # (S, T, 2)
        self.n_tiles = N_totals.shape[1]
        self.n_samples = N_totals.shape[0]

        split_code = SPLIT_CODES[split]
        tile_ok = np.nonzero(split_arr == split_code)[0]

        # min_N filters on the STORED fragment count per (sample, region),
        # indptr[u+1] - indptr[u], before any crop.  It is not the admitted
        # count: an indexed item can still emit frag_N < min_N (see class
        # docstring).
        frag_indptr = root["fragments/indptr"][:]  # (S*T + 1,)
        index = []
        for s in range(self.n_samples):
            for t in tile_ok:
                u = s * self.n_tiles + int(t)
                n_frags = int(frag_indptr[u + 1] - frag_indptr[u])
                if n_frags >= self.min_N:
                    index.append((s, int(t)))
        self.index = index

        # ── worker-safe state ────────────────────────────────────────────
        self._root = None
        self._root_pid = None
        self._rng = None
        self._rng_owner = None

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

    def _jitter_rng(self) -> np.random.Generator:
        """The train-mode jitter stream for the current worker.

        Keyed on ``(seed, worker)`` only, so it is reproducible.  The PID is
        used solely to notice that this object now lives in a different
        process (a forked worker inherits the parent's ``_rng`` state); it
        never enters the key.
        """
        info = get_worker_info()
        # Main process -> 0, worker w -> w + 1, so no worker shares the
        # main process's stream.
        worker = 0 if info is None else info.id + 1
        owner = (os.getpid(), worker)
        if self._rng is None or self._rng_owner != owner:
            base = _DEFAULT_JITTER_SEED if self._seed is None else int(self._seed)
            self._rng = np.random.default_rng([base, worker])
            self._rng_owner = owner
        return self._rng

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
            j = int(self._jitter_rng().integers(-self.jitter, self.jitter + 1))
        else:
            j = 0

        # ── Crop: concentric crops with shared jitter offset ─────────────
        # model_input_size already includes the conv margin (+5) and
        # even-parity rounding — use it directly.
        x_tokens = jitter_matrix(seq_full, j, self.model_input_size)

        # Mask and endpoints → tile_size
        mask = jitter_matrix(mask_full, j, self.tile_size)

        # Endpoint tracks: densify over l_target, then crop to tile_size
        ep_dense = densify_endpoints(ep_pos, ep_track, ep_data, self.l_target)
        ep_cropped = jitter_matrix(ep_dense, j, self.tile_size)
        # Zero masked positions (same policy as 12-track dataset)
        ep_cropped = ep_cropped * mask[None, :]

        # ── Fragment coordinates: start-based admission ──────────────────
        # Crop origin in l_target coordinates
        crop_start = crop_origin(self.l_target, self.tile_size, j)

        # Admission: start in [0, tile_size), in-band length.
        frag_p_crop, keep = admit_fragments(
            frag_start, frag_length, crop_start,
            self.tile_size, self.L_min, self.L_max,
        )

        # Row of each admitted fragment in the store's flat fragments/*
        # arrays, so a per-fragment scorer can join back to the store without
        # re-deriving this admission rule.
        frag_store_idx = flo + np.nonzero(keep)[0]

        frag_m = frag_p_crop[keep]
        frag_L = frag_length[keep].astype(np.int64)
        frag_s = frag_strand[keep].astype(np.int64)
        frag_L_idx = frag_L - self.L_min  # 0-based index into n_L

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
            "frag_store_idx": torch.from_numpy(frag_store_idx.astype(np.int64)),
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
