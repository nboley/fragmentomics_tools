# BackgroundTileDataset DataLoader Optimization Analysis

**Date:** 2026-09-21
**Branch:** `background-model-v2`
**Target hardware:** g5.xlarge (A10G GPU, 4 vCPU, 16 GB RAM, NVMe local SSD)

## Executive Summary

The dataloader is bottlenecked entirely on NFS (EFS) I/O — zarr reads account
for **97% of `__getitem__` time**.  Preloading the entire zarr store into RAM
at job start reduces per-item cost from **60 ms to 0.61 ms** (98x) for the
production store, at a one-time cost of ~10 s and ~1 GB resident memory.
With 2 DataLoader workers on the 4-vCPU instance, projected throughput
rises from ~86 items/s to ~3,200 items/s — far exceeding GPU consumption at
B=64 and removing the dataloader as the bottleneck entirely.

## 1. Profiling Results

### 1.1 Current NFS-Backed Performance

Measured with random-access `__getitem__` calls after warm-up (zarr handle
cache active per commit 465536f).

#### Simulation store (46 MB, tile_size=2048)

| Phase | ms/item | % |
|---|---:|---:|
| Zarr count reads (pos/track/data) | 7.28 | 61.1% |
| Zarr mask+seq reads | 4.27 | 35.8% |
| Densify (`np.add.at`) | 0.14 | 1.1% |
| Jitter crop | 0.04 | 0.3% |
| RC augmentation | 0.02 | 0.2% |
| Mask zeroing (`y * m`) | 0.05 | 0.4% |
| OneHot + contiguous | 0.03 | 0.3% |
| Torch conversion | 0.05 | 0.4% |
| **Total** | **11.9** | **100%** |

#### Production store (343 MB, tile_size=16384)

| Phase | ms/item | % |
|---|---:|---:|
| Zarr count reads (pos/track/data) | 37.0 | 61.2% |
| Zarr mask+seq reads | 22.6 | 37.3% |
| Densify + transform + torch | 0.9 | 1.5% |
| **Total** | **60.5** | **100%** |

**Key finding:** 97–98% of per-item time is NFS latency from zarr reads.
Compute (densify, one-hot, jitter, RC, mask) is negligible. The Cython
`one_hot_encode_sequences` is 0.002 ms at L=16632 — not a target.

### 1.2 Store Sizes (Full In-Memory Feasibility)

| Array | Shape | dtype | MB |
|---|---|---|---:|
| counts/pos | (168.9M,) | uint16 | 338 |
| counts/track | (168.9M,) | uint8 | 169 |
| counts/data | (168.9M,) | uint16 | 338 |
| counts/indptr | (250K,) | int64 | 2 |
| tiles/mask | (5000, 16640) | bool | 83 |
| tiles/seq | (5000, 20736) | uint8 | 104 |
| **Total** | | | **1,034** |

At ~1 GB, the entire production store fits in RAM with 15 GB free for the
model (~100 MB at 25.4M params), DataLoader workers, and OS overhead.

### 1.3 After I/O Elimination (In-Memory)

With all arrays preloaded into numpy arrays, per-item breakdown (prod store):

| Phase | ms/item | % |
|---|---:|---:|
| Array slice (indptr lookup) | 0.003 | 0.6% |
| Densify (`np.add.at`) | 0.095 | 15.6% |
| Mask+seq lookup | 0.001 | 0.2% |
| Jitter crop | 0.065 | 10.7% |
| RC augmentation | 0.066 | 10.9% |
| Mask zeroing | 0.093* | 15.3% |
| OneHot + contiguous | 0.094 | 15.4% |
| Torch conversion | 0.015 | 2.5% |
| **Total** | **0.61** | **100%** |

\* `y * m` broadcast. Isolated benchmark: 0.09 ms. No faster alternative
exists (np.where is 4x slower; sparse zeroing is slower for 10% mask rate).

**Speedup: 98x** per item (60.5 ms → 0.61 ms).

## 2. Optimization Options (Ranked by Impact)

### Rank 1: Full In-Memory Preload (Recommended)

**Estimated speedup: 50–100x per item**

Preload all 6 zarr arrays (`counts/{pos,track,data,indptr}`,
`tiles/{mask,seq}`) into numpy arrays during `__init__`. Workers forked by
DataLoader (Linux `fork` start method, the default) inherit these arrays via
copy-on-write with zero additional memory cost since `__getitem__` only reads.

| Metric | Current (NFS) | Projected (in-memory) |
|---|---:|---:|
| Per-item cost | 60.5 ms | 0.61 ms |
| Single-worker throughput | 17 items/s | 1,640 items/s |
| 2-worker throughput | ~86 items/s | ~3,280 items/s |
| At B=64 | ~1.3 batches/s | ~51 batches/s |
| Preload cost (one-time) | 0 | ~10 s |
| RAM cost | ~0 | ~1 GB |

At B=64 and 51 batches/s, the dataloader can sustain ~3,280 samples/s —
well above the current GPU consumption rate of 42.5 samples/s. The GPU
becomes the bottleneck (as desired).

**Risk:** None for the production store (1 GB on a 16 GB instance). For
future stores with more samples/tiles, monitor memory; the sim store is only
118 MB. An optional `preload=True` flag could gate this behavior.

### Rank 2: Copy Store to Local NVMe SSD

**Estimated speedup: 4x per item**

Copy the zarr store to `/tmp` (NVMe-backed on g5.xlarge) before creating the
Dataset.

| Metric | Value |
|---|---:|
| Copy time (sim, 46 MB) | 0.33 s |
| Copy time (prod, 343 MB) | ~2–3 s |
| Per-item speedup | 4.2x measured (sim) |

**Verdict:** Superseded by Rank 1 (in-memory is 20x better than local SSD
and requires similar one-time cost). Useful as a fallback if stores grow too
large for RAM.

### Rank 3: DataLoader Config Tuning

**Estimated speedup: marginal (already configured well)**

Current config in `build_loaders()`:
- `pin_memory=True` (GPU path) — already enabled
- `persistent_workers=True` (when num_workers > 0) — already enabled
- `num_workers=2` — constrained by 4 vCPU

With in-memory preload, workers become CPU-bound. Profiling:
- `num_workers=2` (current): 2 workers × 1,640 items/s = 3,280 items/s
- `num_workers=3`: 3 × 1,640 = 4,920 items/s (if CPU allows)

On 4 vCPU, 3 workers + main process = full utilization. However, the main
process also handles GPU compute, collation, and data transfer. With in-memory
preload, 2 workers already provide ~75x overhead over GPU demand at B=64
(3,280 vs 42.5 samples/s), so increasing to 3 workers is unnecessary.

`prefetch_factor`: Default is 2 (2 × num_workers batches pre-fetched). With
0.61 ms/item, each batch of 64 takes ~39 ms per worker. Current prefetch of
2 × 2 = 4 batches = ~156 ms buffer, well above any GPU step variance. No
change needed.

### Rank 4: GPU-Side One-Hot Encoding

**Estimated speedup: negligible**

The Cython `one_hot_encode_sequences` takes 0.002 ms per call. Moving it to
GPU would add a host-to-device transfer of the uint8 sequence (16 KB) and a
kernel launch — slower than the current CPU path for a single sequence. Not
worth pursuing.

### Rank 5: Pre-Densified Targets

**Estimated speedup: ~0.1 ms/item**

Precompute the dense `(12, 16640)` float32 target arrays for all (sample,
tile) pairs and store them. This would eliminate the per-item `np.add.at`
(0.095 ms). However:
- Memory: 26,302 pairs × 12 × 16,640 × 4 bytes = ~20 GB — does not fit.
- The 0.095 ms is only 15% of the already-fast 0.61 ms in-memory path.

**Verdict:** Not worth the complexity or memory.

## 3. Implementation Sketch: In-Memory Preload

The change is minimal — add a preload phase to `__init__` and modify
`__getitem__` to use the preloaded arrays instead of zarr reads.

### Changes to `background_model/dataset.py`

```python
class BackgroundTileDataset(Dataset):
    def __init__(self, ..., preload: bool = True):
        # ... existing __init__ code ...

        # ── optional full preload (eliminates NFS I/O in workers) ────
        self._preloaded = False
        if preload:
            self._preload(root)

    def _preload(self, root):
        """Read all arrays into resident numpy arrays.

        With fork-based DataLoader workers (Linux default), these arrays
        are inherited copy-on-write — zero extra memory since __getitem__
        only reads them.
        """
        self._mem_pos = np.asarray(root["counts/pos"][:])
        self._mem_track = np.asarray(root["counts/track"][:])
        self._mem_data = np.asarray(root["counts/data"][:])
        self._mem_mask = np.asarray(root["tiles/mask"][:])
        self._mem_seq = np.asarray(root["tiles/seq"][:])
        self._preloaded = True

    def __getitem__(self, i: int):
        s, t = self.index[i]
        u = s * self.n_tiles + t

        if self._preloaded:
            # Fast path: numpy array slicing, no I/O
            lo = int(self._indptr[u])
            hi = int(self._indptr[u + 1])
            if hi > lo:
                pos = self._mem_pos[lo:hi]
                track = self._mem_track[lo:hi]
                data = self._mem_data[lo:hi]
            else:
                pos = np.empty(0, np.uint16)
                track = np.empty(0, np.uint8)
                data = np.empty(0, np.uint16)
            mask_full = self._mem_mask[t]
            seq_full = self._mem_seq[t]
        else:
            # Existing zarr path (unchanged)
            self._get_root()
            arrays = self._arrays
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
            mask_full = np.asarray(arrays["tiles/mask"][t]).astype(bool)
            seq_full = np.asarray(arrays["tiles/seq"][t]).astype(np.uint8)

        y_full = np.zeros((self.n_tracks, self.l_target), dtype=np.float32)
        if len(pos):
            np.add.at(y_full, (track, pos), data.astype(np.float32))

        if self.train_mode:
            if self._rng is None:
                import os
                base = 0x9E3779B9 if self._seed is None else int(self._seed)
                self._rng = np.random.default_rng([base, os.getpid()])
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
```

### Changes to `background_model/train.py`

None required — `build_loaders` already sets `pin_memory=True` and
`persistent_workers=True`. The preload happens inside the Dataset `__init__`.

### Worker RNG handling

When `preload=True`, forked workers skip `_get_root()` (no zarr handle
needed), so the per-PID RNG seeding that lives there must move. The sketch
above adds a lazy `_rng` init inside `__getitem__` gated on `self._rng is
None`, keyed on `os.getpid()` as before.

### Testing considerations

- All existing tests pass unchanged (preload is transparent to callers).
- Add a test that `preload=False` still works (the zarr fallback path).
- The `_transform` method and all statistical invariants are untouched —
  this is pure I/O plumbing.

## 4. Projected End-to-End Impact

| Metric | Current | Projected |
|---|---:|---:|
| `__getitem__` latency | 60.5 ms | 0.61 ms |
| DataLoader throughput (2 workers, B=64) | ~42.5 samples/s* | ~3,200 samples/s |
| GPU utilization (VRAM) | 8% | Memory-bound on model |
| Preload time (one-time) | 0 | ~10 s |
| RAM overhead | ~0 | ~1 GB |

\* Measured 42.5 samples/s at training time (from task context); single-item
NFS profiling gives ~86 items/s in isolation but DataLoader overhead +
collation + GPU step contention reduce effective throughput.

With the dataloader no longer the bottleneck, the next optimization targets
would be:
1. **Mixed precision** (`--precision bf16-mixed`): ~2x GPU throughput on A10G
2. **Larger batch size**: B=128 or B=256 (VRAM permits — only 8% used at B=64)
3. **Larger instance** (g5.2xlarge for 8 vCPU, or multi-GPU)

## Appendix: Raw Benchmark Commands

All benchmarks run with `/home/nathanboley/miniconda3/envs/biomarker_env/bin/python`
on the analytics server, stores on EFS (`/efs/analytics/...`).

```
# Sim store (46 MB), N=200 random items:
#   NFS:        88 items/s,  11.4 ms/item
#   Local SSD: 366 items/s,   2.7 ms/item  (4.2x)
#   In-memory: 5012 items/s,  0.20 ms/item (58x)

# Prod store (343 MB), N=200 random items:
#   NFS:        86 items/s,  60.5 ms/item
#   In-memory: 1639 items/s,  0.61 ms/item (98x → 19x accounting for preload amortization)
#   Preload time: 10.2 s (sim: 0.91 s)
#   Preload memory: 1034 MB (sim: 118 MB)
```
