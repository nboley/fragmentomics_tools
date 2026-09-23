# K-mer Embedding Network (KEN) — Implementation & Testing Plan

**Date:** 2026-09-22
**Branch:** `background-model-v2`
**Prerequisite:** `docs/pending/attention_architecture.md` (design rationale)
**Status:** Implementation plan — ready for execution

---

## 0. Executive Summary

The CNN background model captures only 42% of the known hexamer bias
(per-count multinomial NLL: 7.6037 vs oracle 7.5745, gap 0.0292 nats).
The bottleneck is not receptive field but combinatorial representation: a
Conv1d cannot efficiently encode the 4^6 = 4,096-entry hexamer lookup
table. An `nn.Embedding(4096, d)` IS that lookup table.

This plan specifies the implementation of the K-mer Embedding Network
(KEN), a model that replaces the convolutional k-mer recognition with an
explicit embedding table, followed by optional context layers. It
integrates with the existing training harness, loss functions, and
evaluation pipeline without modifying the frozen statistical core
(`background_model_core.py`).

**What the model predicts:** The KEN predicts joint endpoint profiles,
not individual cut-site biases. The simulation applies
`w6(left_cut) * w6(right_cut_RC)` — bias at BOTH fragment ends. The
observed endpoint profile at each position is the result of this
two-ended bias convolved with the fragment-length distribution. This is
the same quantity the CNN predicts. The embedding table learns the
effective per-hexamer contribution to this joint profile, not a raw w6
lookup.

**RC weight tying (Phase 1):** The 4,096 hexamers form 2,016 RC pairs
plus 64 palindromes = 2,080 canonical entries. RC weight tying in the
embedding table halves the effective parameter count without any
complexity cost, and is included from the start.

---

## 1. Architecture Specification

### 1.1 Class: `BackgroundModelKEN`

**File:** `background_model_core.py` — new class alongside `BackgroundModel`.
The class inherits from `L.LightningModule` directly (not from
`BackgroundModel`) to avoid coupling to the CNN's geometry logic, but
reuses the same loss classes, masking convention, and `predict_profile`
interface.

**Rationale for same file:** The loss functions, track naming, masking,
`jitter_matrix`, and `reverse_complement_track_permutation` all live in
`background_model_core.py`. Adding the KEN class here keeps it under the
same "frozen statistical core" umbrella and avoids circular imports. The
new class adds no new loss semantics — it only provides an alternative
trunk.

### 1.2 Forward Pass — Pseudocode

```
Input: one_hot  (B, 4, L_in)     where L_in = L_out + (k-1) + n_ctx*(k_ctx-1), rounded up to even

Stage 1 — K-mer Index Computation (fixed, no gradient):
  base_idx = one_hot.argmax(dim=1)              # (B, L_in), long
  patches  = base_idx.unfold(1, k, 1)           # (B, L_unfold, k)
  powers   = [4^(k-1), 4^(k-2), ..., 4^0]      # (k,), registered buffer
  kmer_idx = (patches * powers).sum(dim=-1)     # (B, L_unfold), long

  # L_unfold = L_out (k odd) or L_out + 1 (k even).
  # Each output position i gets the k-mer CENTERED on it:
  #   for k=6, position i -> seq[i-3:i+3] in output coordinates
  # This centering is achieved by the dataset's center-crop of L_in
  # from the stored sequence (k//2 extra bases on each side for even k).

Stage 2 — K-mer Embedding (THE lookup table, with RC weight tying):
  canonical_idx = to_canonical[kmer_idx]         # RC pair -> canonical index
  h = self.embed(canonical_idx)                  # (B, L_unfold, d_embed)
  h = h.transpose(1, 2)                         # (B, d_embed, L_unfold) — conv layout
  h = self.embed_dropout(h)                      # SpatialDropout

Stage 3 — Context CNN (optional, unpadded — trims (k_ctx-1) per layer):
  for conv_layer in self.context_layers:
    h = conv_layer(h)                            # Conv1d(d, d, k_ctx, padding=0)
    h = GELU(h)
  # residual + LayerNorm omitted in v1 for simplicity

Stage 4 — Trim (even-k only):
  if k is even:
    h = h[..., :-1]                              # drop the extra position from
                                                 # even-parity rounding

Stage 5 — Shape Head:
  shape_logits = self.shape_head(h)              # Conv1d(d, n_tracks, 1)
                                                 # (B, n_tracks, L_out)

Stage 6 — Dispersion Head (when loss != "multinomial"):
  dispersion_bp = self.dispersion_head(h)        # Conv1d(d, n_tracks, 1)
                                                 # (B, n_tracks, L_out)
  (pooled identically to BackgroundModel._pooled_log_dispersion)

Output: shape_logits (B, n_tracks, L_out), dispersion_bp or None
```

### 1.3 Key Design Decisions

**Centered hexamer alignment.** The simulation's ground truth applies
`w6` at cut positions with a 3-in/3-out convention: the hexamer for a
cut at position `c` is `seq[c-3:c+3]` — three bases inside and three
outside the fragment (see `scripts/sim_fragments.py:precompute_region`,
lines 310-335). Each output position IS a potential cut site, so the
embedding at output position `i` must encode the hexamer CENTERED on
`i`, not the one starting there.

Centering is achieved through the input geometry and the dataset's
center-crop:

```
L_in = L_out + (k-1) + n_ctx*(k_ctx-1), rounded up to the next even number

For k=6, 2 context layers k_ctx=15:
    raw = L_out + 5 + 2*14 = L_out + 33 (odd) → L_in = L_out + 34
    Center-crop: 17 extra bases on each side
    After unfold (trims k-1=5): L_out + 29 positions
    After 2 unpadded convolutions (trims 2*14=28): L_out + 1 positions
    After even-k trim (1): L_out — CORRECT
    The hexamer at output position i covers seq[i-3:i+3] — CORRECT

For k=6, 0 context layers:
    raw = L_out + 5 (odd) → L_in = L_out + 6
    Center-crop: 3 extra bases on each side
    After unfold: L_out + 1 positions
    After even-k trim: L_out — CORRECT
```

When k is even, the rounding adds one extra input position, and
`unfold` produces `L_out + 1` k-mers. The forward pass trims the last
position (which corresponds to a cut site beyond the output region).
When k is odd, `unfold` produces exactly `L_out` k-mers and no
trimming is needed.

**Without centering** (the naive `seq[i:i+6]` approach), the
embedding-only ablation (no context layers) would look up the WRONG
hexamer at every position, displaced by `k//2 - (k-1)//2 = 1` position.
Context layers could partially compensate, but the embedding would not
learn the hexamer table directly. With correct centering, the
embedding-only model IS a lookup table, and its weights should correlate
directly with the ground-truth w6.

**Position-level aggregation: single centered k-mer (v1).**
Each position gets exactly one k-mer embedding — the centered hexamer.
The context CNN layers (kernel 15, unpadded) then mix information
from neighboring positions, implicitly aggregating the effects of
overlapping k-mers and longer-range sequence context. A future Phase 2
ablation could explicitly aggregate all k overlapping k-mers per
position (requiring `L_in = L_out + 2*(k-1)`), but this is deferred.

**Joint endpoint profile, not individual cut-site weights.** The model
predicts the observed endpoint profile — the count distribution across
positions. At each position, the observed count is the sum of left-end
fragments (bias `w6(forward_hex)`) and right-end fragments (bias
`w6(RC_hex)`), integrated over all fragment lengths. The resulting
profile is a function of hexamers at ALL nearby positions (within one
fragment-length), not just the local hexamer. The embedding captures the
local hexamer effect; the context layers capture interactions between
positions. This is the same quantity the CNN predicts, and the same
quantity the loss functions expect.

**GC x fragment-length bias is out of the context CNN's reach.** The
simulation includes a 2-D GC x length bias surface in addition to
hexamer bias. The GC% of a fragment depends on the sequence across
its entire span (40-175 bp for the modeled FL bands). The KEN's
context CNN has an effective receptive field of:

| Layers | Kernel | Effective RF |
|--------|--------|-------------|
| 2 (default) | 15 | 15 + 14 = 29 bp |
| 4 | 15 | 15 + 3 x 14 = 57 bp |
| 2 | 31 | 31 + 30 = 61 bp |

Even 4 layers with k=15 only cover 57 bp — insufficient for
mononucleosomal fragments (120-175 bp) and barely marginal for short
fragments (40-65 bp). By comparison, the CNN has RF = 248 bp (kernel=32,
2 dilated residual layers).

**Consequence:** The KEN targets hexamer cut-site bias specifically. It
will capture the dominant hexamer effect but will underperform the CNN on
the GC x FL component of the simulation. This is expected and
informative: the gap between KEN and oracle, minus the gap attributable
to GC bias, isolates the hexamer representation quality. If GC x FL
capture becomes important, options include: (a) adding dilated context
layers to expand the RF, (b) a separate GC model combined with the
hexamer model, or (c) increasing to 6+ context layers with larger
kernels.

**Geometry:**
```
L_in = L_out + (k-1) + n_ctx*(k_ctx-1), rounded up to even

def calc_input_region_size(output_region_size, k):
    raw = output_region_size + k - 1
    return raw + (raw % 2)
```
For k=6 with 2 context layers k_ctx=15: `L_in = 2048 + 34 = 2082`. The store's `l_seq` is
`tile_size + 2*(jitter + rf_budget) = 2048 + 2*(128 + 2048) = 6400`,
vastly exceeding the 2054 needed. No store rebuild required.

### 1.4 Layer Dimensions and Parameter Counts

**KEN-v1 (default configuration, with RC weight tying):**

| Component | Shape | Parameters |
|-----------|-------|------------|
| `nn.Embedding(2080, 64)` | 2080 x 64 | 133,120 |
| `SpatialDropout(0.15)` | — | 0 |
| `Conv1d(64, 128, 15, padding=0)` | 64 x 128 x 15 + 128 | 123,008 |
| `Conv1d(128, 128, 15, padding=0)` | 128 x 128 x 15 + 128 | 245,888 |
| `Conv1d(128, 12, 1)` (shape head) | 128 x 12 + 12 | 1,548 |
| `Conv1d(128, 12, 1)` (dispersion head) | 128 x 12 + 12 | 1,548 |
| **Total** | | **505 K** |

For multinomial loss (no dispersion head): **504 K** parameters.
Compare: CNN 512k/3L = 25.4M (50x larger).

The 2,080 canonical entries come from: 4,096 hexamers = 2,016 RC pairs
+ 64 palindromes. Each RC pair shares one embedding row; palindromes
get their own row. The `_to_canonical` index buffer (4,096 int64) maps
any hexamer to its canonical row.

### 1.5 Configurable Hyperparameters

```python
class BackgroundModelKEN(L.LightningModule):
    def __init__(
        self,
        output_tracks: Optional[List[str]] = None,
        k: int = 6,                        # k-mer size
        d_embed: int = 64,                 # embedding dimension
        d_context: int = 128,              # context conv channels
        n_context_layers: int = 2,         # number of context conv layers
        context_kernel_size: int = 15,     # context conv kernel size
        dropout: float = 0.15,             # SpatialDropout on embeddings
        learning_rate: float = 1e-3,
        loss: str = "multinomial",
        dispersion_window_size: int = 256,
        log_dispersion_init: float = 7.0,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
        freeze_dispersion: bool = False,
        dispersion_lr_scale: float = 1.0,
        weight_decay: float = 0.0,        # L2 on embedding table
    ):
```

### 1.6 VRAM Estimate

| Component | Memory (bf16, B=64, L_out=2048) |
|-----------|---------------------------------|
| Parameters (505K x 2 bytes) | 1.0 MB |
| Optimizer states (Adam, fp32 master + moments) | 5.9 MB |
| Input one-hot (64 x 4 x 2054 x 4 bytes) | 2.0 MB |
| K-mer indices (64 x 2049 x 8 bytes) | 1.0 MB |
| Embeddings (64 x 2048 x 64 x 2 bytes) | 16.8 MB |
| Context CNN activations (2 layers x 64 x 128 x 2048 x 2) | 67.1 MB |
| Gradients (~ params + activations) | ~90 MB |
| **Total training** | **< 200 MB** |
| **A10G budget used** | **< 1%** |

B=2048 would use ~3.5 GB — still comfortable. The bottleneck will be
data loading, not GPU compute.

---

## 2. Integration with Existing Code

### 2.1 What is REUSED unchanged

| Component | File | Why no change |
|-----------|------|---------------|
| Loss functions | `background_model_core.py:372-544` | Same `(B, C, L)` shape convention |
| Masking (`_prepare_mask`) | `background_model_core.py:342-349` | Same mask shape `(B, L)` |
| `masked_mean_pool` | `background_model_core.py:353-369` | Used by `_pooled_log_dispersion` |
| Track naming / RC perm | `background_model_core.py:99-156` | Track convention is model-independent |
| `jitter_matrix` | `background_model_core.py:164-228` | Augmentation primitive, used by dataset |
| `SpatialDropout` | `background_model_core.py:236-249` | Reused directly |
| Dataset | `background_model/dataset.py` | No changes — `model_input_size` is passed in |
| DataLoaders | `background_model/train.py:240-253` | No changes |
| Inference/stitching | `background_model/inference.py` | Uses `calc_input_region_size` and `predict_profile` |
| Correction | `background_model/correction.py` | Operates on `predict_profile` output |
| Config | `background_model/config.py` | Store geometry unchanged |

### 2.2 What CHANGES

#### 2.2.1 `background_model_core.py` — add `BackgroundModelKEN` class

New code only; no modification to `BackgroundModel` or any loss.
The class implements:

- `__init__`: builds embedding table (with RC weight tying), context layers, heads
- `calc_input_region_size(L_out) -> int`: returns `L_out + (k-1) + n_ctx*(k_ctx-1)`, rounded up to even
- `forward(x) -> (shape_logits, dispersion_bp_or_None)`
- `_pooled_log_dispersion(dispersion_bp, mask)`: identical logic to
  `BackgroundModel._pooled_log_dispersion` — copy the method rather
  than factoring out a shared base class, to avoid touching the frozen
  core
- `_step(batch, log_name)`: identical logic to `BackgroundModel._step`
- `training_step`, `validation_step`: delegate to `_step`
- `configure_optimizers`: Adam with optional weight_decay on embeddings
- `predict_profile`: identical interface to `BackgroundModel.predict_profile`

Also adds two standalone functions:
- `rc_kmer_permutation(k)`: fixed permutation mapping k-mer indices to RC partners
- `one_hot_to_kmer_indices(one_hot, k, powers)`: converts one-hot DNA to k-mer indices

#### 2.2.2 `background_model/train.py` — model selection

Add `--model {cnn,ken}` flag to the CLI. Extend `build_model`:

```python
def build_model(cfg: TrainConfig) -> L.LightningModule:
    if cfg.model == "ken":
        return InstrumentedBackgroundModelKEN(
            k=cfg.k, d_embed=cfg.d_embed, d_context=cfg.d_context,
            n_context_layers=cfg.n_context_layers,
            context_kernel_size=cfg.context_kernel_size,
            loss=cfg.loss, learning_rate=cfg.lr, dropout=cfg.dropout,
            weight_decay=cfg.weight_decay,
            freeze_dispersion=cfg.freeze_dispersion,
            dispersion_lr_scale=cfg.dispersion_lr_scale,
        )
    else:
        return InstrumentedBackgroundModel(...)
```

`InstrumentedBackgroundModelKEN` subclasses `BackgroundModelKEN` the
same way `InstrumentedBackgroundModel` subclasses `BackgroundModel`:
adds per-track loss logging, grad norms, dispersion trajectory, and
the cross-family `_log_multinomial_nll` metric.

#### 2.2.3 `TrainConfig` dataclass — add KEN fields

```python
@dataclass
class TrainConfig:
    ...
    model: str = "cnn"           # "cnn" or "ken"
    k: int = 6                   # k-mer size (KEN only)
    d_embed: int = 64
    d_context: int = 128
    n_context_layers: int = 2
    context_kernel_size: int = 15
    weight_decay: float = 0.0
```

#### 2.2.4 `run_meta.json` — record KEN hyperparameters

Add `model`, `k`, `d_embed`, `d_context`, `n_context_layers`,
`context_kernel_size`, `weight_decay` to the metadata dict.

### 2.3 What is NOT changed

- `BackgroundModel` class — untouched
- Loss functions — untouched
- Store format — unchanged
- Store build scripts — unchanged
- Simulation scripts — unchanged
- Existing tests — no modifications

---

## 3. K-mer Index Computation — Detailed Implementation

### 3.1 `one_hot_to_kmer_indices` (standalone function)

```python
def one_hot_to_kmer_indices(one_hot: torch.Tensor, k: int,
                            powers: torch.Tensor) -> torch.Tensor:
    """Convert one-hot DNA (B, 4, L) to integer k-mer indices (B, L-k+1).

    Each k-mer is encoded as a base-4 integer:
        index = base[0]*4^(k-1) + base[1]*4^(k-2) + ... + base[k-1]*4^0

    Uses integer arithmetic; positions with ambiguous bases (all-zero columns
    in one-hot, i.e. N bases) are mapped to index 0.

    Args:
        one_hot: (B, 4, L) float tensor (one-hot encoded DNA)
        k: k-mer length
        powers: (k,) long tensor [4^(k-1), ..., 4^0], registered as a buffer
    Returns:
        (B, L-k+1) long tensor of k-mer indices in [0, 4^k)
    """
    base_idx = one_hot.argmax(dim=1)                # (B, L), long
    patches = base_idx.unfold(dimension=1, size=k, step=1)  # (B, L-k+1, k)
    kmer_idx = (patches * powers).sum(dim=-1)       # (B, L-k+1)
    return kmer_idx
```

**N-base handling:** `argmax` on an all-zero column returns 0, so N
bases map to `base_idx=0` (same as A). This is acceptable because:
(a) N positions are always masked in the blacklist mask, so they never
contribute to the loss or to N, and (b) the same behavior occurs in the
simulation's `hexamer_indices` function (which uses `_BASE_LUT` with a
255 sentinel, then marks those windows invalid).

**Powers buffer:** registered as a non-parameter buffer in `__init__`:
```python
self.register_buffer(
    "_powers",
    4 ** torch.arange(k - 1, -1, -1),  # [4^(k-1), ..., 1]
)
```

### 3.2 Alignment with Simulation Ground Truth

The simulation (`scripts/sim_fragments.py:145-155`) computes hexamer
indices with the SAME base-4 big-endian encoding:
```python
_POW = 4 ** np.arange(KMER - 1, -1, -1)
idx = (win * _POW).sum(axis=1)
```
where `win` is the sliding window of 2-bit base codes (A=0, C=1, G=2,
T=3). The KEN uses the identical encoding (via `argmax` on the same
ACGT one-hot channel order), so the embedding table entries are
index-compatible with the simulation's `w6` array.

**Centering alignment:** The simulation applies the hexamer bias at a
cut position `c` using the hexamer centered on `c`:
`seq[c-3:c+3]` (see `precompute_region`, lines 310-335, which pads
the region sequence by `HEX_HALF=3` on each side). The KEN's input
geometry ensures the same centering: for k=6, `L_in = L_out + 6`
gives 3 extra bases on each side after center-cropping, so `unfold`
position `i` covers output positions `[i-3, i+3)` — exactly the
centered hexamer. This enables direct comparison:
`model.embed.weight[:, 0]` vs `ground_truth["w6"]` (modulo the joint
endpoint convolution effect; see §1.3).

### 3.3 Correctness Invariant

For a sequence with no N bases:
```
one_hot_to_kmer_indices(one_hot, 6, powers)[b, i]
== sim_fragments.hexamer_indices(seq_bytes)[0][i]   (for valid positions)
```
This will be verified in a unit test.

---

## 4. Reverse-Complement Equivariance

### 4.1 Phase 1: RC Weight Tying in the Embedding + Data Augmentation

**RC weight tying** is included from Phase 1 because it is simple,
reduces the embedding table by ~50%, and eliminates the need for data
augmentation to independently learn that RC partner hexamers should have
the same bias. Without tying, the model must learn `embed[i] ≈
embed[rc(i)]` from augmentation alone — wasteful, since it halves the
effective sample size.

**RC mapping in the embedding table:**

Under reverse complement, k-mer with index `i` maps to k-mer with
index `rc(i)`. This is a fixed permutation of `{0, ..., 4^k - 1}`,
computable once at init.

For k=6 with the base encoding A=0, C=1, G=2, T=3:
- complement: A<->T (0<->3), C<->G (1<->2)
- reverse complement of k-mer `[b0, b1, ..., b5]` is
  `[3-b5, 3-b4, ..., 3-b0]`

```python
def rc_kmer_permutation(k: int) -> np.ndarray:
    """Fixed permutation mapping each k-mer index to its RC partner."""
    n = 4 ** k
    idx = np.arange(n)
    codes = np.zeros((n, k), dtype=np.int64)
    rem = idx.copy()
    for j in range(k):
        codes[:, k - 1 - j] = rem % 4
        rem //= 4
    rc_codes = (3 - codes)[:, ::-1]
    powers = 4 ** np.arange(k - 1, -1, -1)
    return (rc_codes * powers).sum(axis=1)
```

This is exactly `scripts/sim_fragments.py:128-139` (`RC_PERM`).

**Weight tying implementation:** Partition the 4^k k-mers into RC
pairs. For k=6: 2,016 non-palindromic pairs + 64 palindromes (a 6-mer
is palindromic iff `b[i] + b[5-i] = 3` for all `i`, giving 4^3 = 64).
Store one canonical embedding per pair (the one with the lower index).
At forward time, map each k-mer index to its canonical representative:

```python
# In __init__:
rc_perm = rc_kmer_permutation(k)
canonical = np.minimum(np.arange(vocab_size, dtype=np.int64), rc_perm)
_, to_canonical = np.unique(canonical, return_inverse=True)
self.register_buffer("_to_canonical",
                     torch.from_numpy(to_canonical.astype(np.int64)))
n_canonical = int(to_canonical.max()) + 1  # 2080 for k=6
self.embed = nn.Embedding(n_canonical, d_embed)

# In forward:
kmer_idx = one_hot_to_kmer_indices(x, k, self._powers)
canonical_idx = self._to_canonical[kmer_idx]
h = self.embed(canonical_idx)
```

**Context layers: data augmentation only.** The context CNN layers are
NOT weight-tied for RC in Phase 1. The existing RC augmentation in
`BackgroundTileDataset._transform` (50% probability RC flip with track
permutation) handles equivariance for the full model. This is the same
approach the CNN uses and provides a fair comparison. The embedding
tying eliminates the table-level redundancy that augmentation handles
poorly (sparse gradients, 2x effective sample size loss), while leaving
the dense context layers to augmentation where it works well.

### 4.2 Future: Full Architectural Equivariance (Phase 2)

For the context layers, full RC equivariance would require either:
- Group-equivariant convolutions (RC as a group action)
- Weight averaging: `W_tied = (W + P @ W @ P^T) / 2` per layer
- Explicit RC forward pass averaging at inference

This adds complexity and is deferred to Phase 2 ablations. The Phase 1
approach (embedding tying + data augmentation for context layers) is
sufficient to verify the hexamer embedding hypothesis.

### 4.3 Verification

Verify that the model produces the correct RC relationship:
```python
# For a trained model:
out_fwd = model.predict_profile(onehot_fwd)["probs"]   # (C, L)
out_rc  = model.predict_profile(onehot_rc)["probs"]     # (C, L)
# Under RC: probs should satisfy out_rc ≈ out_fwd[rc_perm, ::-1]
np.testing.assert_allclose(out_rc, out_fwd[rc_perm, ::-1], atol=0.01)
```

With RC weight tying in the embedding, the deviation should be smaller
than with data augmentation alone (the embedding is exactly equivariant;
remaining deviation comes from the untied context layers).

---

## 5. Multiple k Values

### 5.1 Approach: Separate Runs First, Multi-k Later

**Phase 1:** Train separate models with k=4, 5, 6, 7, 8. The `k`
parameter is a simple hyperparameter — no code changes beyond what's
already in the configurable `__init__`.

| k | Vocab size | Canonical entries | Embedding params (d=64) | Per-kmer training examples* |
|---|-----------|-------------------|------------------------|----------------------------|
| 4 | 256 | 136 | 8.7 K | ~102,000 |
| 5 | 1,024 | 524 | 33.5 K | ~25,600 |
| 6 | 4,096 | 2,080 | 133 K | ~6,400 |
| 7 | 16,384 | 8,224 | 526 K | ~1,600 |
| 8 | 65,536 | 32,896 | 2.1 M | ~400 |

\* 12,800 training pairs x 2,048 positions = 26.2M positions; each
k-mer appears `26.2M / 4^k` times on average.

**Phase 2 (if warranted):** Multi-k model that concatenates embeddings
from multiple k values. Implementation:

```python
class MultiKEmbedding(nn.Module):
    def __init__(self, k_values, d_per_k):
        self.embeddings = nn.ModuleDict({
            str(k): nn.Embedding(4**k, d_per_k[k])
            for k in k_values
        })

    def forward(self, one_hot):
        # Compute indices for each k, embed, pad to same length, concat
        parts = []
        for k, emb in self.embeddings.items():
            k = int(k)
            idx = one_hot_to_kmer_indices(one_hot, k, self._powers[k])
            h = emb(idx)  # (B, L-k+1, d_k)
            # Pad to L_out = min(L-k+1 for all k) -- use the largest k
            # to determine L_out, trim the others
            parts.append(h[:, :L_out, :])
        return torch.cat(parts, dim=-1)  # (B, L_out, sum(d_k))
```

**Geometry for multi-k:** `L_in = L_out + max(k) - 1`. All shorter-k
embeddings produce slightly longer sequences that are center-cropped
to match.

Defer multi-k to Phase 2 ablations because single-k=6 should capture
the dominant signal, and multi-k adds complexity.

---

## 6. Dispersion Head

### 6.1 Start with Multinomial Only

The training analysis conclusively showed that:
- Multinomial is the best-performing loss (41.7% bias captured)
- DM adds nothing on simulation data
- NB-offset is noisy and recovers less bias

**For Phase 1:** Use multinomial loss exclusively. No dispersion head.
This eliminates the dispersion complexity and focuses the comparison
on the representation bottleneck.

### 6.2 Dispersion Head Architecture (for future use)

When adding DM or NB-offset, the dispersion head is:
```python
self.dispersion_head = nn.Conv1d(d_context, n_tracks, 1)
```
Pooled via `_pooled_log_dispersion` (copied from `BackgroundModel`),
offset by `log_dispersion_init=7.0`. Identical to the CNN's approach.

---

## 7. Complete Model Code — Reference Implementation

This is the target implementation. The implementation agent should use
this as a reference, adapting as needed for integration.

```python
def rc_kmer_permutation(k: int) -> np.ndarray:
    """Fixed permutation mapping each k-mer index to its RC partner.

    Complement in 2-bit code is (3 - code) (A0<->T3, C1<->G2); RC also
    reverses base order.  Involution: perm[perm[i]] == i.
    """
    n = 4 ** k
    idx = np.arange(n, dtype=np.int64)
    codes = np.zeros((n, k), dtype=np.int64)
    rem = idx.copy()
    for j in range(k):
        codes[:, k - 1 - j] = rem % 4
        rem //= 4
    rc_codes = (3 - codes)[:, ::-1]
    powers = 4 ** np.arange(k - 1, -1, -1)
    return (rc_codes * powers).sum(axis=1)


def one_hot_to_kmer_indices(one_hot: torch.Tensor, k: int,
                            powers: torch.Tensor) -> torch.Tensor:
    """Convert one-hot DNA (B, 4, L) to integer k-mer indices (B, L-k+1).

    Each position i in the output gets the k-mer centered on it (after
    the dataset's center-crop ensures the correct input offset). The
    k-mer is encoded as a base-4 big-endian integer, matching the
    simulation's hexamer_indices encoding.
    """
    base_idx = one_hot.argmax(dim=1)
    patches = base_idx.unfold(dimension=1, size=k, step=1)
    return (patches * powers).sum(dim=-1)


class BackgroundModelKEN(L.LightningModule):
    """K-mer Embedding Network for cfDNA fragment-endpoint background modeling.

    Replaces the CNN trunk with an explicit k-mer lookup table
    (nn.Embedding) followed by optional context Conv1d layers.
    Same loss functions, masking, and predict_profile interface as
    BackgroundModel.

    The model predicts joint endpoint profiles: the observed count at each
    position reflects hexamer bias at BOTH fragment ends (left cut forward,
    right cut RC), convolved with the fragment-length distribution. The
    embedding table learns the effective per-hexamer contribution to this
    joint profile, not raw individual cut-site weights.

    RC weight tying: each k-mer and its reverse-complement share the same
    embedding row (2,080 canonical entries for k=6 instead of 4,096).
    """

    def __init__(
        self,
        output_tracks: Optional[List[str]] = None,
        k: int = 6,
        d_embed: int = 64,
        d_context: int = 128,
        n_context_layers: int = 2,
        context_kernel_size: int = 15,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        loss: str = "multinomial",
        dispersion_window_size: int = 256,
        log_dispersion_init: float = 7.0,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
        freeze_dispersion: bool = False,
        dispersion_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES} (got '{loss}')")
        if output_tracks is None:
            output_tracks = list(DEFAULT_OUTPUT_TRACKS)
        for t in output_tracks:
            track_name_to_index_key(t)
        self.save_hyperparameters()
        self.output_tracks = output_tracks

        n_tracks = len(output_tracks)
        vocab_size = 4 ** k

        # Stage 1: k-mer index computation (fixed)
        self.register_buffer(
            "_powers",
            4 ** torch.arange(k - 1, -1, -1, dtype=torch.long),
        )

        # RC weight tying: map each k-mer to its canonical representative
        rc_perm = rc_kmer_permutation(k)
        canonical = np.minimum(np.arange(vocab_size, dtype=np.int64), rc_perm)
        _, to_canonical = np.unique(canonical, return_inverse=True)
        self.register_buffer(
            "_to_canonical",
            torch.from_numpy(to_canonical.astype(np.int64)),
        )
        n_canonical = int(to_canonical.max()) + 1  # 2080 for k=6

        # Stage 2: embedding table (canonical entries only)
        self.embed = torch.nn.Embedding(n_canonical, d_embed)
        self.embed_dropout = SpatialDropout(dropout)

        # Stage 3: context CNN (unpadded, consistent with BackgroundModel)
        layers = []
        in_ch = d_embed
        for _ in range(n_context_layers):
            layers.append(
                torch.nn.Conv1d(in_ch, d_context, context_kernel_size,
                                padding=0)
            )
            layers.append(torch.nn.GELU())
            in_ch = d_context
        self.context = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        trunk_out_ch = d_context if n_context_layers > 0 else d_embed

        # Stage 4: shape head
        self.shape_head = torch.nn.Conv1d(trunk_out_ch, n_tracks, 1)

        # Stage 5: dispersion head (only for overdispersed losses)
        self.dispersion_head = (
            None if loss == "multinomial"
            else torch.nn.Conv1d(trunk_out_ch, n_tracks, 1)
        )

        # Loss
        if loss == "multinomial":
            self.loss_fn = MaskedMultinomialNLLLoss()
        elif loss == "dirichlet_multinomial":
            self.loss_fn = MaskedDirichletMultinomialNLLLoss()
        else:
            self.loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
                max_dispersion_ratio=max_dispersion_ratio,
                clamp_margin=clamp_margin,
            )

        if freeze_dispersion and self.dispersion_head is not None:
            for p in self.dispersion_head.parameters():
                p.requires_grad = False

        # Even-k trimming flag: when k is even, calc_input_region_size
        # rounds up to even, producing one extra unfold position.
        self._trim = (k - 1) % 2  # 1 when k is even, 0 when k is odd

    def calc_input_region_size(self, output_region_size: int) -> int:
        """Input length for the given output length.

        The k-mer unfold needs (k-1) extra bases. Each unpadded context
        layer trims (context_kernel_size - 1) positions. For even k the
        raw total may be odd, violating the dataset's even-parity
        requirement — round up. This rounding also ensures the center-crop
        places k//2 extra bases on each side, giving correctly centered
        hexamers (see sec 1.3).
        """
        k = self.hparams.k
        n_ctx = self.hparams.n_context_layers
        k_ctx = self.hparams.context_kernel_size
        raw = output_region_size + (k - 1) + n_ctx * (k_ctx - 1)
        return raw + (raw % 2)

    def forward(self, x):
        # x: (B, 4, L_in) one-hot
        kmer_idx = one_hot_to_kmer_indices(x, self.hparams.k, self._powers)
        canonical_idx = self._to_canonical[kmer_idx]
        h = self.embed(canonical_idx)             # (B, L_unfold, d_embed)
        h = h.transpose(1, 2)                     # (B, d_embed, L_unfold)
        h = self.embed_dropout(h)
        h = self.context(h)                       # (B, d_context, L_unfold)

        # Trim the extra position from even-parity rounding (even k only)
        if self._trim:
            h = h[..., :-1]                       # (B, d_context, L_out)

        shape_logits = self.shape_head(h)         # (B, n_tracks, L_out)
        if self.dispersion_head is None:
            return shape_logits, None
        return shape_logits, self.dispersion_head(h)

    def _pooled_log_dispersion(self, dispersion_bp, mask):
        L = dispersion_bp.shape[-1]
        if self.hparams.loss == "dirichlet_multinomial":
            out_size = 1
        else:
            w = self.hparams.dispersion_window_size
            assert L % w == 0
            out_size = L // w
        pooled = masked_mean_pool(dispersion_bp, mask, out_size)
        pooled = pooled + self.hparams.log_dispersion_init
        if self.hparams.loss == "dirichlet_multinomial":
            pooled = pooled.squeeze(-1)
        return pooled

    def _step(self, batch, log_name):
        x, y, mask = batch
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)
        if self.hparams.loss == "multinomial":
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)
        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        scale = self.hparams.dispersion_lr_scale

        # Apply weight decay only to the embedding table
        embed_params = list(self.embed.parameters())
        embed_ids = {id(p) for p in embed_params}

        if self.dispersion_head is not None and not self.hparams.freeze_dispersion and scale != 1.0:
            disp_ids = {id(p) for p in self.dispersion_head.parameters()}
            main_params = [p for p in self.parameters()
                          if id(p) not in embed_ids and id(p) not in disp_ids]
            return torch.optim.Adam([
                {"params": embed_params, "lr": lr, "weight_decay": wd},
                {"params": main_params, "lr": lr, "weight_decay": 0.0},
                {"params": list(self.dispersion_head.parameters()),
                 "lr": lr * scale, "weight_decay": 0.0},
            ])
        main_params = [p for p in self.parameters() if id(p) not in embed_ids]
        return torch.optim.Adam([
            {"params": embed_params, "lr": lr, "weight_decay": wd},
            {"params": main_params, "lr": lr, "weight_decay": 0.0},
        ])

    @torch.no_grad()
    def predict_profile(self, one_hot_seq: np.ndarray,
                        mask: Optional[np.ndarray] = None):
        self.eval()
        x = torch.as_tensor(one_hot_seq, dtype=torch.float32, device=self.device)
        shape_logits, dispersion_bp = self(x[None])
        mask3 = None
        if mask is not None:
            mask3 = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            mask3 = mask3[None, None, :]
            shape_logits = shape_logits.masked_fill(~mask3, float("-inf"))
        probs = torch.softmax(shape_logits, dim=-1)[0].cpu().numpy()
        log_dispersion = None
        if dispersion_bp is not None:
            log_dispersion = (
                self._pooled_log_dispersion(dispersion_bp, mask3)[0].cpu().numpy()
            )
        return {"probs": probs, "log_dispersion": log_dispersion}
```

---

## 8. Training Harness Changes

### 8.1 `InstrumentedBackgroundModelKEN`

Subclass `BackgroundModelKEN` with the same diagnostic logging as
`InstrumentedBackgroundModel`:

```python
class InstrumentedBackgroundModelKEN(BackgroundModelKEN):
    """BackgroundModelKEN + per-track loss, grad-norm, and multinomial NLL."""

    def _step(self, batch, log_name):
        # Identical to InstrumentedBackgroundModel._step but calls
        # BackgroundModelKEN.forward
        ...  # (copy the _log_per_track, _log_dispersion_trajectory,
             #  _log_multinomial_nll methods from InstrumentedBackgroundModel)
```

The four `_log_*` methods are identical — they operate on detached
`shape_logits` and `log_disp` tensors and don't depend on model
internals. Copy them rather than extracting a mixin, to avoid touching
the frozen `InstrumentedBackgroundModel`.

### 8.2 CLI Changes

Add to `build_arg_parser()`:

```python
p.add_argument("--model", choices=["cnn", "ken"], default="cnn")
p.add_argument("--k", type=int, default=6)
p.add_argument("--d-embed", type=int, default=64)
p.add_argument("--d-context", type=int, default=128)
p.add_argument("--n-context-layers", type=int, default=2)
p.add_argument("--context-kernel-size", type=int, default=15)
p.add_argument("--weight-decay", type=float, default=0.0)
```

### 8.3 Dataset Integration

The dataset receives `model_input_size` from `build_datasets`:
```python
model_input_size = model.calc_input_region_size(_tile_size)
```

For KEN with k=6 and tile_size=2048: `model_input_size = 2054` (even).
The dataset's geometry check verifies `l_seq >= model_input_size + 2*jitter`.
For the simulation store: `l_seq = 6400 >= 2054 + 256 = 2310`. Passes.

**Even-parity:** `calc_input_region_size` always returns an even number
(the `raw + (raw % 2)` formula). The forward pass trims one position
when k is even. No other code needs to know about the rounding.

---

## 9. Testing Plan

### 9.1 Unit Tests (file: `tests/test_background_model_ken.py`)

All tests are CPU-only, seeded, tiny configs. Follow the pattern of
`tests/test_background_model_core.py`.

#### T1: K-mer index computation correctness

```python
def test_kmer_indices_match_sim_fragments():
    """Verify KEN index encoding matches simulation ground truth."""
    # Construct a known sequence, compute indices both ways
    seq = "ACGTACGTNN"  # 10 bases, 5 hexamers
    one_hot = encode_to_one_hot(seq)  # (1, 4, 10)
    powers = 4 ** torch.arange(5, -1, -1, dtype=torch.long)
    idx = one_hot_to_kmer_indices(one_hot, 6, powers)
    # Compare against manual base-4 encoding
    # ACGTAC = 0*4^5 + 1*4^4 + 2*4^3 + 3*4^2 + 0*4 + 1 = 0+256+128+48+0+1 = 433
    assert idx[0, 0].item() == 0*1024 + 1*256 + 2*64 + 3*16 + 0*4 + 1  # = 433
```

#### T2: Geometry — forward matches `calc_input_region_size`

```python
@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("loss", LOSSES)
def test_geometry_forward_matches(k, loss):
    model = BackgroundModelKEN(k=k, d_embed=16, d_context=32,
                               n_context_layers=1, context_kernel_size=5,
                               loss=loss, dropout=0.0)
    L_out = 512  # must be even
    L_in = model.calc_input_region_size(L_out)
    assert L_in % 2 == 0, "model_input_size must be even"
    x = random_one_hot(rng, 2, L_in)
    shape_logits, disp = model(x)
    assert shape_logits.shape == (2, N_TRACKS, L_out)
    if loss != "multinomial":
        assert disp.shape == (2, N_TRACKS, L_out)
```

#### T3: Gradient finiteness with masking

```python
@pytest.mark.parametrize("loss", LOSSES)
def test_gradients_finite_with_mask(loss):
    model = BackgroundModelKEN(k=6, d_embed=16, d_context=32,
                               n_context_layers=1, loss=loss, dropout=0.0)
    L_out = 512
    L_in = model.calc_input_region_size(L_out)
    x = random_one_hot(rng, 2, L_in)
    mask = np.ones((2, L_out), dtype=bool)
    mask[:, 100:150] = False
    y = rng.integers(0, 6, size=(2, N_TRACKS, L_out)).astype(np.float32)
    y *= mask[:, None, :]
    batch = (x, torch.as_tensor(y), torch.as_tensor(mask))
    loss_val = model.training_step(batch, 0)
    assert torch.isfinite(loss_val)
    loss_val.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
```

#### T4: predict_profile interface compatibility

```python
@pytest.mark.parametrize("loss", LOSSES)
def test_predict_profile_shape_and_normalization(loss):
    model = BackgroundModelKEN(k=6, d_embed=16, d_context=32,
                               n_context_layers=1, loss=loss, dropout=0.0)
    L_out = 512
    L_in = model.calc_input_region_size(L_out)
    seq = random_one_hot(rng, 1, L_in)[0].numpy()
    out = model.predict_profile(seq)
    assert out["probs"].shape == (N_TRACKS, L_out)
    np.testing.assert_allclose(out["probs"].sum(axis=-1), 1.0, atol=1e-4)
    # With mask
    mask = np.ones(L_out, dtype=bool)
    mask[50:100] = False
    out_m = model.predict_profile(seq, mask=mask)
    assert (out_m["probs"][:, ~mask] == 0).all()
    np.testing.assert_allclose(out_m["probs"][:, mask].sum(axis=-1), 1.0, atol=1e-4)
```

#### T5: Embedding gradient sparsity

```python
def test_embedding_gradients_are_sparse():
    """Only k-mers that appear in the batch should receive gradients."""
    model = BackgroundModelKEN(k=6, d_embed=16, d_context=0,  # no context
                               n_context_layers=0, loss="multinomial", dropout=0.0)
    L_out = 64
    L_in = model.calc_input_region_size(L_out)
    # Constant sequence: all A's -> only one k-mer (AAAAAA = index 0)
    x = torch.zeros(1, 4, L_in)
    x[0, 0, :] = 1.0  # all A
    y = torch.ones(1, N_TRACKS, L_out)
    mask = torch.ones(1, L_out, dtype=torch.bool)
    loss = model.training_step((x, y, mask), 0)
    loss.backward()
    grad = model.embed.weight.grad
    assert grad is not None
    # Only the canonical entry for AAAAAA should have nonzero gradient
    assert grad[0].abs().sum() > 0
    assert grad[1:].abs().sum() == 0
```

#### T6: Checkpoint save/load roundtrip

```python
def test_checkpoint_roundtrip(tmp_path):
    model = BackgroundModelKEN(k=6, d_embed=16, d_context=32,
                               n_context_layers=1, loss="multinomial")
    ckpt = tmp_path / "test.ckpt"
    torch.save(model.state_dict(), ckpt)
    model2 = BackgroundModelKEN(k=6, d_embed=16, d_context=32,
                                n_context_layers=1, loss="multinomial")
    model2.load_state_dict(torch.load(ckpt))
    # Verify same output
    x = random_one_hot(np.random.default_rng(0), 1, model.calc_input_region_size(128))
    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = model2(x)
    torch.testing.assert_close(out1, out2)
```

#### T7: Parameter count verification

```python
def test_parameter_count():
    model = BackgroundModelKEN(k=6, d_embed=64, d_context=128,
                               n_context_layers=2, context_kernel_size=15,
                               loss="multinomial")
    total = sum(p.numel() for p in model.parameters())
    # 2080*64 + (64*128*15+128) + (128*128*15+128) + (128*12+12) = 503,564
    assert 480_000 < total < 530_000, f"unexpected param count: {total}"
```

#### T8: RC weight tying correctness

```python
def test_rc_weight_tying():
    """RC partner k-mers must map to the same embedding row."""
    model = BackgroundModelKEN(k=6, d_embed=16, d_context=0,
                               n_context_layers=0, loss="multinomial", dropout=0.0)
    rc_perm = rc_kmer_permutation(6)
    # For every k-mer, it and its RC partner should map to the same canonical index
    to_canonical = model._to_canonical.numpy()
    for i in range(4096):
        assert to_canonical[i] == to_canonical[rc_perm[i]], \
            f"k-mer {i} and its RC {rc_perm[i]} have different canonical indices"
    # Verify the embedding table has the expected number of rows
    assert model.embed.num_embeddings == 2080

def test_rc_kmer_permutation_is_involution():
    """RC permutation applied twice should give the identity."""
    perm = rc_kmer_permutation(6)
    np.testing.assert_array_equal(perm[perm], np.arange(4096))
```

### 9.2 Simulation Benchmark

#### Experiment: KEN-v1 on sim_store_B.zarr

**Store:** `/efs/analytics/nathanboley/background_model/simulation_v2/stores/sim_store_B.zarr`

**Command:**
```bash
cd /home/nathanboley/src/fragmentomics_tools
PYTHONPATH=. python -m background_model.train \
    --model ken --k 6 --d-embed 64 --d-context 128 \
    --n-context-layers 2 --context-kernel-size 15 \
    --loss multinomial --lr 1e-3 --batch-size 64 \
    --precision bf16-mixed --lr-patience 4 --max-lr-reductions 3 --max-epochs 200 \
    --store /efs/analytics/nathanboley/background_model/simulation_v2/stores/sim_store_B.zarr \
    --runs-root /efs/analytics/nathanboley/background_model/simulation_v2/runs \
    --run-name sim_v2_B_ken6_multinomial \
    --seed 1337
```

**Expected runtime:** 10-20 min to early-stop on A10G (~$0.20-0.35).
The model is 50x smaller than the CNN and uses padded convolutions
(simpler data path), so it should be significantly faster per epoch.

**Key metrics to extract:**
1. `val_loss` (= per-count multinomial NLL, directly comparable)
2. `val_multinomial_nll` (cross-family metric, same as val_loss for multinomial)

**Success criteria (from design doc sec 8):**

| Outcome | KEN NLL | Gap | Bias Captured | Interpretation |
|---------|---------|-----|---------------|----------------|
| Strong success | < 7.5900 | < 0.0155 | > 69% | Embedding captures hexamer table |
| Moderate success | < 7.5990 | < 0.0245 | > 51% | Better than CNN, confirms bottleneck |
| Marginal | < 7.6037 | < 0.0292 | > 42% | Matches CNN with 50x fewer params |
| Failure | >= 7.6037 | >= 0.0292 | <= 42% | No improvement over CNN |

Reference points:
- Oracle: 7.5745 (gap = 0.0000, 100%)
- CNN 512k/3L best: 7.6037 (gap = 0.0292, 41.7%)
- Uniform: 7.6246 (gap = 0.0501, 0%)

**Note on GC x FL bias:** The simulation includes GC x FL bias in
addition to hexamer bias. The KEN's context CNN (RF=29bp) cannot
capture GC% over fragment-length scales (40-175bp). This means the
KEN will not reach the oracle even with perfect hexamer recovery. To
isolate the hexamer representation quality, also compare against the
oracle *with GC bias removed* (uniform GC bias surface). The gap
attributable to GC bias is the CNN's advantage from its larger RF.

### 9.3 Interpretability Test

After training, extract the learned embedding table and compare against
the ground-truth w6:

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load ground truth
gt = np.load(".../simulation/B/ground_truth.npz", allow_pickle=True)
w6 = gt["w6"]  # (4096,) ground-truth hexamer weights

# Load trained model
model = InstrumentedBackgroundModelKEN.load_from_checkpoint(ckpt_path)
embed = model.embed.weight.detach().cpu().numpy()  # (2080, d_embed)
to_canonical = model._to_canonical.numpy()         # (4096,) -> [0, 2080)

# The shape head maps d_embed -> n_tracks. The effective per-hexamer
# contribution to logits for track c is:
#   logit_contribution(kmer) = shape_head.weight[c] @ embed[to_canonical[kmer]]
#                              + shape_head.bias[c]
W = model.shape_head.weight.detach().cpu().numpy()  # (n_tracks, d_embed, 1)
b = model.shape_head.bias.detach().cpu().numpy()     # (n_tracks,)
W = W.squeeze(-1)  # (n_tracks, d_embed)

# Effective logit for each of the 4096 hexamers, each track
full_embed = embed[to_canonical]  # (4096, d_embed), RC partners share rows
eff_logit = full_embed @ W.T + b  # (4096, n_tracks)

# For each track, correlate against log(w6)
for c in range(n_tracks):
    r, p = pearsonr(eff_logit[:, c], np.log(w6))
    print(f"Track {c}: Pearson r = {r:.4f}, p = {p:.2e}")
```

**Important caveat:** The embedding does not learn raw w6 values
directly. It learns the effective per-hexamer contribution to the
*joint* endpoint profile (which integrates left and right cut biases
over all fragment lengths). The correlation with log(w6) should still
be strong since hexamer bias is the dominant effect, but it won't be
perfect. Pearson r > 0.7 would be strong evidence that the embedding
is capturing the hexamer table.

### 9.4 Test Suite Baseline

After implementing, run the full test suite:
```bash
cd /home/nathanboley/src/fragmentomics_tools
python -m pytest tests/ -q
```

Current baseline: **217 passed, 0 skipped**. The new tests should bring
the total to ~207+ passed.

---

## 10. Experimental Plan

### Phase 1: Baseline Comparison (1-2 days)

1. **Add `rc_kmer_permutation` and `one_hot_to_kmer_indices`** to
   `background_model_core.py` (standalone functions)
2. **Add `BackgroundModelKEN`** with RC weight tying to
   `background_model_core.py` (after `BackgroundModel`)
3. **Add `InstrumentedBackgroundModelKEN`** to `background_model/train.py`
4. **Add CLI flags** for model selection and KEN hyperparameters
5. **Write unit tests** (T1-T8, including RC weight tying)
6. **Run unit tests** — all must pass
7. **Run simulation benchmark** — KEN-v1 on sim_store_B.zarr
8. **Extract interpretability metrics** — embedding vs w6 correlation

**Expected cost:** ~$0.25 per training run on A10G.

### Phase 2: Ablations (2-3 days, if Phase 1 succeeds)

Run each ablation as a separate training run:

| Experiment | What changes | Run name | Expected cost |
|------------|-------------|----------|---------------|
| **k sweep** | k=4, 5, 7, 8 (k=6 is Phase 1) | `sim_v2_B_ken{k}_multinomial` | $0.25 each = $1.00 |
| **No context** | `n_context_layers=0` | `sim_v2_B_ken6_nocontext` | $0.20 |
| **Deeper context** | `n_context_layers=4` | `sim_v2_B_ken6_deep` | $0.30 |
| **Larger embedding** | `d_embed=128` | `sim_v2_B_ken6_d128` | $0.25 |
| **Embedding-only** | `n_context_layers=0, d_embed=12` | `sim_v2_B_ken6_direct` | $0.15 |
| **Weight decay** | `weight_decay=1e-4` | `sim_v2_B_ken6_wd` | $0.25 |
| **LR sweep** | lr=1e-2, 3e-3, 3e-4 | `sim_v2_B_ken6_lr*` | $0.75 |

**Total Phase 2 cost:** ~$3.00.

**Key ablations to prioritize:**

1. **Embedding-only (no context layers):** Tests whether the k-mer
   embedding alone is sufficient. With correct centering, the
   embedding-only model IS a hexamer lookup table. If this matches the
   full KEN on hexamer bias recovery, context layers add nothing for
   the hexamer component (they may still help with GC effects).

2. **k sweep:** Tests whether k=6 is optimal (it should be, since the
   ground truth IS a hexamer table).

3. **d_embed=12 (one scalar per track):** The minimal embedding — each
   k-mer gets exactly one logit per output track. This is the most
   direct analog of the w6 lookup table. If this works as well as
   d_embed=64, the higher-dimensional embedding is unnecessary overhead.

### Phase 3: Real Data (if simulation succeeds)

**Store:** `/efs/analytics/nathanboley/background_model/stores/bg_store_b67d7c95.zarr`

```bash
PYTHONPATH=. python -m background_model.train \
    --model ken --k 6 --d-embed 64 --d-context 128 \
    --n-context-layers 2 --context-kernel-size 15 \
    --loss multinomial --lr 1e-3 --batch-size 64 \
    --precision bf16-mixed --lr-patience 4 --max-lr-reductions 3 --max-epochs 200 \
    --store /efs/analytics/nathanboley/background_model/stores/bg_store_b67d7c95.zarr \
    --runs-root /efs/analytics/nathanboley/background_model/runs \
    --run-name ken6_multinomial_real \
    --seed 1337
```

**Evaluation:** CTCF footprint recovery, QQ calibration on held-out
inactive regions. Compare against CNN baseline using the same
`sim_evaluate.py` framework adapted for the real store.

**Note on real-data geometry:** The real store uses `tile_size=16384`,
`jitter=128`, `rf_budget=2048`, so `l_seq=20736`. KEN needs
`model_input_size=16384+6=16390` (even). The store's
`l_seq=20736 >= 16390+256=16646`. Passes.

### Hyperparameter Ranges

| Hyperparameter | Phase 1 | Sweep range (Phase 2) |
|----------------|---------|----------------------|
| k | 6 | {4, 5, 6, 7, 8} |
| d_embed | 64 | {12, 32, 64, 128} |
| d_context | 128 | {0, 64, 128, 256} |
| n_context_layers | 2 | {0, 1, 2, 4} |
| context_kernel_size | 15 | {7, 15, 31} |
| dropout | 0.15 | {0.0, 0.10, 0.15, 0.25} |
| lr | 1e-3 | {1e-2, 3e-3, 1e-3, 3e-4} |
| weight_decay | 0.0 | {0, 1e-5, 1e-4, 1e-3} |
| batch_size | 64 | {64, 256, 512} |

**Do NOT sweep everything.** Start with Phase 1 defaults. Only sweep
parameters that the Phase 1 results suggest are important (e.g., if
Phase 1 shows overfitting, sweep dropout and weight_decay).

---

## 11. Risk Mitigation

| Risk | Likelihood | Detection | Mitigation |
|------|-----------|-----------|------------|
| Even-parity assertion fails | Eliminated | Unit test T2 | `calc_input_region_size` returns even; forward trims |
| KEN overfits (2080 free embedding entries) | Medium | val_loss diverges from train_loss | RC tying already halves table; also dropout, weight decay, reduce d_embed |
| N-bases produce wrong k-mer indices | Low | Unit test T1 | N positions are masked; argmax(all-zero)=0 is safe |
| LR too high for embedding (large sparse gradients) | Medium | NaN or exploding loss | Start at 1e-3 (same as 128k/1L CNN); reduce if needed |
| Context layers dominate, embedding doesn't learn | Low | Check embed.weight.grad norm | Embedding-only ablation (Phase 2) |
| Multi-k geometry alignment fails | Medium | Unit test at multi-k impl time | Center-crop all k's to min output length |
| GC x FL bias missed due to small RF | Expected | Compare KEN vs CNN gap to GC contribution | Documented limitation; test with no-GC simulation if needed |

---

## 12. File Inventory — What Gets Created/Modified

### New files:
- `tests/test_background_model_ken.py` — unit tests (T1-T8)

### Modified files:
- `background_model_core.py` — add `rc_kmer_permutation()`,
  `one_hot_to_kmer_indices()`, and `BackgroundModelKEN` class (~170 lines)
- `background_model/train.py` — add `InstrumentedBackgroundModelKEN`,
  `--model` CLI flag, KEN hyperparameter args, update `build_model()`
  and `TrainConfig` (~80 lines)

### NOT modified:
- Loss functions, masking, track naming, jitter, RC permutation
- Dataset, DataLoaders, config, inference, correction
- Simulation scripts, existing tests
- Any `.zarr` store

---

## 13. Implementation Order

The implementation agent should follow this exact sequence:

1. **Add `rc_kmer_permutation` and `one_hot_to_kmer_indices`** to
   `background_model_core.py` (standalone functions, above the
   `BackgroundModel` class)

2. **Add `BackgroundModelKEN`** to `background_model_core.py` (after
   `BackgroundModel`, before the calibration diagnostics section)

3. **Add `InstrumentedBackgroundModelKEN`** to `background_model/train.py`
   (after `InstrumentedBackgroundModel`)

4. **Extend `TrainConfig`** with KEN fields

5. **Extend `build_model`** and `build_arg_parser` for model selection

6. **Extend `_write_run_meta`** to record KEN hyperparameters

7. **Write `tests/test_background_model_ken.py`**

8. **Run `python -m pytest tests/ -q`** — all 196 existing + new tests pass

9. **Run simulation benchmark** on A10G

10. **Extract and report interpretability metrics**

Each step should be committed independently with a descriptive message.
Do not combine unrelated changes.

---

## 14. Review Notes — Changes from Critical Review (2026-09-22)

This section documents the five issues identified during critical review
and how each was resolved.

### Issue 1: Position alignment — hexamer centering (FIXED)

**Problem:** The plan described each output position `i` as getting the
k-mer `seq[i:i+6]` (left-aligned). But the simulation's ground truth
applies `w6` at the CUT POSITION with a 3-in/3-out convention:
`w6(seq[cut-3:cut+3])`. The left-aligned encoding is off by 1 position
from the centered hexamer, and the embedding-only ablation would look
up the wrong hexamer at every position.

**Root cause:** With `L_in = L_out + 5` (the raw `k-1` formula), the
dataset's center-crop gives `resize_start = 5//2 = 2`, putting 2 extra
bases on the left and 3 on the right. Unfold at position `i` covers
output `[i-2, i+4)` — not `[i-3, i+3)`.

**Fix:** The even-parity rounding (`L_in = L_out + 6` for k=6) already
produces the correct centering: `resize_start = 6//2 = 3`, giving 3
extra bases on each side, so unfold at position `i` covers output
`[i-3, i+3)`. The fix was to RECOGNIZE that the parity rounding is not
just a compatibility hack but is essential for correct centering, and to
document this throughout (sec 1.2, 1.3, 3.2, 8.3).

**Changed sections:** 1.2, 1.3, 3.2, 7, 8.3.

### Issue 2: RC weight tying — moved to Phase 1 (DONE)

**Problem:** The plan deferred RC weight tying to Phase 2, requiring the
model to learn `embed[i] ≈ embed[rc(i)]` from data augmentation alone.
This halves the effective sample size for embedding learning.

**Fix:** RC weight tying is simple to implement (a `_to_canonical` index
buffer mapping 4,096 k-mers to 2,080 canonical entries) and does not
complicate the model. Moved to Phase 1. Context layers still use data
augmentation only (tying dense conv weights for RC requires more
thought and is genuinely Phase 2 material).

**Changed sections:** 0, 1.4, 4.1, 4.2, 5 (table), 7, 9.1 (T7, T8),
10, 11, 12.

### Issue 3: GC x FL bias — receptive field documented (DONE)

**Problem:** The simulation has GC x FL bias (2-D surface). The KEN's
context CNN with kernel=15, 2 layers has effective RF=29bp — insufficient
to estimate GC% of fragments spanning 40-175bp.

**Fix:** Added explicit receptive field calculation and comparison table
to sec 1.3. Documented that the KEN targets hexamer bias specifically
and will underperform on the GC x FL component. Added a note to sec 9.2
suggesting a no-GC-bias control experiment to isolate hexamer
representation quality.

**Changed sections:** 1.3, 9.2, 11.

### Issue 4: Even-parity — simplified (DONE)

**Problem:** The plan spent ~150 lines deliberating on the even-parity
issue in sec 8.3, exploring multiple approaches with inline code
fragments, retracted ideas, and "let me reconsider" passages.

**Fix:** Replaced with the 4-line solution: `calc_input_region_size`
returns `raw + (raw % 2)`, forward trims one position when k is even.
The motivation is now clearly linked to centering (Issue 1), not just
compatibility. All the deliberation was removed.

**Changed sections:** 8.3 (reduced from ~150 to ~10 lines). The
solution is also cleanly stated in 1.3 and the reference implementation
in sec 7.

### Issue 5: Joint endpoint profile prediction — documented (DONE)

**Problem:** The plan did not explain what the model actually predicts.
The simulation applies `w6(left_cut) * w6(right_cut_RC)` — bias at
BOTH ends. The model predicts the resulting profile (the convolution of
left and right biases with the FL distribution), which is the same
quantity the CNN predicts, but this was not stated.

**Fix:** Added explicit documentation in sec 0 (executive summary),
sec 1.3 (design decisions), sec 7 (class docstring), and sec 9.3
(interpretability caveat about what the embedding learns vs raw w6).

**Changed sections:** 0, 1.3, 7, 9.3.
