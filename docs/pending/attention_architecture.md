# Attention-Based Architecture for the Background Model

**Date:** 2026-09-22
**Branch:** `background-model-v2`
**Status:** Design proposal (no code changes)

---

## 1. Problem Analysis: Why the CNN Plateaus at 42%

### 1.1 The ground truth is a lookup table

The simulation study (§5 of `training_analysis.md`) established that the
dominant sequence-driven bias signal is a **hexamer cut-site weight table**:
4,096 entries mapping each possible 6-mer to a cleavage propensity. The oracle
that knows this table perfectly achieves a per-count NLL of 7.5745; the uniform
(no model) baseline is 7.6246. The gap is 0.0501 nats.

The best CNN (512 kernels, 3 dilated residual layers, 25.4M parameters) closes
41.7% of this gap (NLL 7.6037). **58% of the hexamer signal remains
unrecovered despite a 497 bp receptive field — 83x larger than the 6 bp
hexamer context.**

The model can *see* the hexamer. It cannot efficiently *use* it.

### 1.2 Why convolutions are the wrong inductive bias here

Consider what the model must learn: an arbitrary function
`f: {A,C,G,T}^6 → R` over 4,096 inputs with no exploitable low-rank
structure (the ground-truth table is drawn log-normal).

**A Conv1d(4, C, k=6) is a linear map.** Applied to a one-hot input, each
output channel computes a dot product between the 6-mer's binary encoding and a
learned weight vector. This produces C linear features of the k-mer identity.
But 4,096 distinct inputs require at least 4,096 linearly independent features
to distinguish them all. With C=512 channels, the first conv layer can produce
at most 512 independent features — roughly 1/8 of what the lookup table
demands.

Multiple layers with nonlinearities CAN represent arbitrary functions
(universal approximation). But the representation is *indirect*: the model must
build up the 4,096-entry table through compositions of convolutions and
activations, each operating on partial features of the k-mer. This is a much
harder optimization problem than directly indexing into a table.

**Evidence from the sweep:** scaling from 1.2M to 25.4M parameters (21x)
improved recovery from 29% to 42% (+13 pp). Extrapolating this sublinear
scaling, the CNN would need >100M parameters to approach oracle performance —
absurd for a 49K-entry lookup table.

### 1.3 The hexamer ≠ the whole story

Two bias mechanisms operate in the simulation:

| Mechanism | Scale | Representation needed |
|-----------|-------|-----------------------|
| **Hexamer cut-site bias** | 6 bp (local) | Lookup table: 4,096 entries |
| **GC × fragment-length bias** | Whole fragment (tens to hundreds of bp) | Smooth 2-D surface |

The CNN may actually capture the GC×FL component reasonably well (it's smooth,
low-dimensional, and within the receptive field). The unrecovered 58% is likely
dominated by the hexamer component. An architecture that directly models the
lookup while retaining a pathway for longer-range context should outperform.


## 2. Candidate Architectures

### 2.1 Candidate A: K-mer Embedding Network (KEN)

**Core idea:** Replace the convolutional approach to k-mer recognition with an
explicit lookup table — `nn.Embedding(4^k, d)` — followed by optional
convolutional/MLP layers for longer-range context.

```
one-hot (4, L)
     │
     ▼
 argmax → base indices (L,)
     │
     ▼
 unfold(k) → k-mer indices (L-k+1,)   ← integer base-4 encoding
     │
     ▼
 nn.Embedding(4^k, d) → (L-k+1, d)    ← THE LOOKUP TABLE
     │
     ▼
 [optional: 1-2 Conv1d layers for GC context]
     │
     ▼
 Shape head: Conv1d(d, n_tracks, 1) → logits (n_tracks, L_out)
     │
     ▼
 masked softmax → profile probabilities
```

**Why it fits this problem:**

- The embedding IS a lookup table. With k=6 and d=12 (one scalar per output
  track), the embedding has 4,096 × 12 = 49,152 parameters — enough to store
  the exact ground-truth table with no approximation.
- Gradient descent directly updates each k-mer's entry based on the loss at
  positions where that k-mer appears. No compositional decomposition needed.
- The k-mer index computation is a fixed (non-learned) operation — zero
  gradient noise, no compositional indirection.

**Parameters:**

| Component | Config | Parameters |
|-----------|--------|------------|
| 6-mer embedding | 4,096 × 64 | 262 K |
| Optional 4-mer embedding | 256 × 32 | 8 K |
| Optional 8-mer embedding | 65,536 × 32 | 2.1 M |
| 2 Conv1d layers (d=128, k=15, padded) | 128 × 128 × 15 × 2 | 492 K |
| Shape head (Linear) | 128 × 12 | 1.5 K |
| **Total (6-mer only + 2 conv)** | | **~0.8 M** |
| **Total (multi-k + 2 conv)** | | **~2.9 M** |

**Pros:**
- Directly models the known signal structure (lookup table)
- 100-1000x more parameter-efficient than the CNN for this problem
- Trivial to interpret: inspect the learned embedding table vs ground truth
- Fast training: each k-mer's gradient is independent, no layer composition
- RC equivariance achievable by weight tying (§5)
- Simple geometry: only the k-mer extraction trims (k-1) positions; padded
  convolutions keep the rest length-preserving

**Cons:**
- Inductive bias is NARROW: assumes the dominant signal is k-mer-local.
  If important signals exist at scales not captured by any k in the multi-k
  set, they'll be missed (though the optional conv layers partially address this).
- 8-mer and above: 4^k grows exponentially. At k=10, the table has 1M entries
  and many will be seen rarely in training data. However, k=6 is the known
  ground truth for this simulation, and real-data hexamer bias is well-established
  in the literature.
- New code path: the current `calc_input_region_size` geometry assumes stacked
  unpadded convolutions. The k-mer approach simplifies this but requires
  adapting the existing training/inference harness.

**Risk: k-mer sparsity in training data.**
With 12,800 training pairs × 2,048 positions = 26.2M training positions, each
of the 4,096 6-mers appears ~6,400 times on average — plenty. For 8-mers
(65,536 entries), each appears ~400 times — adequate but thinner. For 10-mers
(1M entries), many would be seen <10 times, so regularization or
dimensionality reduction becomes necessary.


### 2.2 Candidate B: Enformer-lite (CNN Stem + Transformer)

**Core idea:** Replace the deeper dilated ResNet blocks with transformer
layers. A short CNN stem extracts local features (including implicit k-mer
representations), then self-attention layers perform the combinatorial mixing
that CNNs struggle with.

```
one-hot (4, L)
     │
     ▼
 CNN stem: Conv1d(4, d, k=15) → LayerNorm → GELU → Conv1d(d, d, k=7)
     │
     ▼
 positional encoding (sinusoidal or learned, 2048 positions)
     │
     ▼
 N × Transformer block:
   ├─ LayerNorm → Multi-head self-attention (H heads, flash)
   └─ LayerNorm → FFN (d → 4d → d, GELU)
     │
     ▼
 Shape head: Linear(d, n_tracks) → logits
     │
     ▼
 masked softmax → profile probabilities
```

**Why it might help:**

Self-attention computes pairwise interactions between all positions. For the
hexamer signal, each position can "attend to" its neighbors within the 6-mer
context and learn arbitrary functions of the joint identity. The attention
mechanism is a weighted sum over value vectors, with weights determined by
key-query dot products — structurally similar to a soft dictionary lookup.

More importantly, self-attention captures interactions at *any* distance, not
just within a fixed receptive field. This could help with longer-range effects
like GC context.

**Parameters (d=256, H=8, N=4 layers):**

| Component | Parameters |
|-----------|------------|
| CNN stem (2 conv layers) | 59 K |
| 4 transformer layers | 3.2 M |
| Shape head | 3 K |
| **Total** | **~3.3 M** |

**VRAM estimate (B=64, L=2048, bf16, flash attention):**

| Component | Memory |
|-----------|--------|
| Parameters + optimizer states | ~40 MB |
| Activations (CNN stem) | ~67 MB |
| Activations (4 transformer layers, flash) | ~270 MB |
| Peak (incl. gradient accumulation) | ~600 MB |
| **Total** | **< 1 GB** |

Comfortable within the 24 GB A10G budget. Could scale to B=512 or d=512
without concern.

**Pros:**
- General-purpose: captures both local (hexamer) and global (GC, composition)
  signals.
- Well-understood architecture with extensive tooling (flash attention in
  PyTorch 2.x, gradient checkpointing, etc.).
- Precedent in genomics: Enformer/Borzoi demonstrate transformer success on
  sequence-to-profile prediction.

**Cons:**
- **No inductive bias for the lookup structure.** The transformer CAN learn a
  lookup table, but it must discover this structure from data, just as the CNN
  must. Attention heads must jointly implement: (1) identify which 6-mer is
  at this position, (2) retrieve the associated weight. This is possible but
  not guaranteed — it depends on optimization, initialization, and capacity.
- L=2048 is short enough that self-attention is cheap (2048² = 4.2M entries
  per head per layer), but this also means there isn't much long-range signal
  TO capture. The hexamer is 6 bp — the transformer's global attention range is
  wasted.
- Positional encoding adds complexity (sinusoidal vs learned, how it interacts
  with RC augmentation).
- The CNN-stem-first approach means the transformer still processes CNN-derived
  features, not raw k-mer identities. If the CNN stem can't distinguish
  k-mers (the same problem as Candidate C below), the transformer inherits
  the limitation.


### 2.3 Candidate C: CNN + Cross-Attention Codebook

**Core idea:** Keep the CNN trunk for feature extraction, but add a
cross-attention layer where each position queries a learned "hexamer codebook"
— K learned key-value pairs (K ≈ 4,096) that act as a soft lookup table.

```
one-hot (4, L)
     │
     ▼
 CNN trunk (same as current, or shallower)
     │                                 ┌──────────────┐
     ▼                                 │  Codebook     │
 per-position features (d, L)          │  Keys: (K, d) │
     │                                 │  Values:(K,d) │
     ▼                                 └──────┬───────┘
 Cross-attention:                              │
   Q = Linear(features)                        │
   K, V = codebook                             │
   out = softmax(Q·K^T / √d) · V    ←─────────┘
     │
     ▼
 Shape head → logits → masked softmax
```

**Pros:**
- The codebook explicitly provides the "dictionary" structure that the lookup
  table needs. Each codebook entry can specialize to one k-mer (or group of
  similar k-mers).
- The CNN trunk provides the features that the cross-attention queries against.
- Codebook size K is a tunable parameter independent of the vocabulary size.

**Cons:**
- Still depends on the CNN trunk to produce good features (the same
  bottleneck, potentially).
- The soft attention over K entries adds K × L compute per position. With
  K=4096 and L=2048, the cross-attention matrix is 2048 × 4096 = 8.4M entries
  — larger than self-attention.
- More complex to implement and debug than either A or B.
- The codebook is essentially reinventing an embedding table with extra steps:
  if the codebook learns to map each position to one entry (hard attention),
  it IS an embedding table but with the overhead of soft attention.


## 3. Comparison

| Criterion | A (KEN) | B (Enformer-lite) | C (Codebook) |
|-----------|---------|-------------------|--------------|
| **Inductive bias match** | Exact (IS a lookup table) | Weak (must discover lookup structure) | Moderate (provides dictionary structure) |
| **Parameter efficiency** | 0.8-2.9 M | 3.3 M | ~5 M |
| **Optimization ease** | Easy (direct gradient to each k-mer) | Medium (attention must learn to route) | Medium (codebook must specialize) |
| **Interpretability** | High (inspect embedding table) | Low | Medium |
| **Longer-range context** | Via optional conv layers | Native (self-attention) | Via CNN trunk |
| **RC equivariance** | Simple weight tying | Complex (positional encoding issues) | Medium |
| **Implementation effort** | Low (new module + adapt harness) | Medium (new architecture) | Medium-high |
| **Risk of not improving** | Low (directly addresses the bottleneck) | Medium (may repeat CNN's indirect encoding) | Medium |


## 4. Recommendation: K-mer Embedding Network (Candidate A)

Candidate A directly addresses the identified bottleneck (indirect
combinatorial encoding) with the structurally correct inductive bias (a lookup
table). It is the simplest, most parameter-efficient, most interpretable, and
lowest-risk option.

**If Candidate A closes most of the 58% gap:** it confirms that the hexamer
lookup was indeed the bottleneck and that the problem is well-characterized.

**If Candidate A closes only part of the gap:** the remaining signal involves
longer-range context (interactions, GC effects), and Candidate B becomes the
natural next step — possibly as a hybrid (k-mer embedding + transformer
layers).

**Candidate B as a follow-up, not a starting point:** the Enformer-lite
architecture is a reasonable next step if the k-mer embedding alone is
insufficient, but starting with it would conflate two questions (architecture
vs. representation) and make it harder to diagnose results.


### 4.1 Recommended Architecture: KEN-v1

```
┌──────────────────────────────────────────────────────────────┐
│                        KEN-v1                                │
│                                                              │
│  Input: one-hot DNA (4, L_in)                                │
│                                                              │
│  Stage 1 — K-mer Embedding (local, exact)                    │
│  ├─ argmax → base indices (L_in,)                            │
│  ├─ unfold(k=6) → 6-mer indices (L_in - 5,)                 │
│  ├─ nn.Embedding(4096, d_local) → (L_in - 5, d_local)       │
│  └─ [optional: concatenate 4-mer and 8-mer embeddings]       │
│                                                              │
│  Stage 2 — Contextual CNN (longer-range, smooth)             │
│  ├─ Conv1d(d_local, d_ctx, k=15, padding="same") → GELU     │
│  ├─ Conv1d(d_ctx, d_ctx, k=15, padding="same") → GELU       │
│  └─ [optional: residual connection, LayerNorm, dropout]      │
│                                                              │
│  Stage 3 — Heads                                             │
│  ├─ Shape head: Conv1d(d_ctx, n_tracks, k=1) → logits       │
│  │   → masked softmax → profile probabilities                │
│  └─ [optional: dispersion head, same structure]              │
│                                                              │
│  Output: shape logits (n_tracks, L_out)                      │
│  where L_out = L_in - (k - 1) = L_in - 5                    │
└──────────────────────────────────────────────────────────────┘
```

**Concrete hyperparameters for the first experiment:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| k (k-mer size) | 6 | Matches ground truth; 4,096 entries is manageable |
| d_local (embedding dim) | 64 | Each 6-mer → 64-D vector, plenty for 12 tracks |
| d_ctx (context channels) | 128 | Modest — most work is done by the embedding |
| Context conv kernel size | 15 | Covers ~15 bp for GC/composition context |
| Context conv layers | 2 | Minimal — just for smoothing and interaction |
| Context padding | "same" | Length-preserving; simplifies geometry |
| Dropout | 0.15 | Match current model for fair comparison |
| Loss | multinomial | Best performer in the simulation study |
| Estimated parameters | ~0.8 M | 30x fewer than the 25.4M CNN |

**Optional extensions (test incrementally, not all at once):**
- Multi-k: concatenate 4-mer (256 × 32) and 8-mer (65,536 × 32) embeddings
  alongside the 6-mer. Let the model combine short- and long-range k-mer
  signals.
- Deeper context: add 1-2 more conv layers or a single transformer layer
  after the embeddings for cross-position interactions.
- Learnable positional encoding: a per-position bias term in the shape head,
  if tile-boundary effects matter.


### 4.2 K-mer Index Computation

Computing integer k-mer indices from one-hot encoding:

```python
def one_hot_to_kmer_indices(one_hot: torch.Tensor, k: int) -> torch.Tensor:
    """Convert one-hot DNA (B, 4, L) to integer k-mer indices (B, L-k+1).

    Each k-mer is encoded as a base-4 integer:
        index = base[0]*4^(k-1) + base[1]*4^(k-2) + ... + base[k-1]*4^0

    This uses integer arithmetic to avoid floating-point precision issues.
    Positions with ambiguous bases (all-zero one-hot) are mapped to index 0.
    """
    B, C, L = one_hot.shape
    assert C == 4

    # One-hot → base index {0,1,2,3}
    base_idx = one_hot.argmax(dim=1)  # (B, L), long

    # Compute k-mer index via dot product with powers of 4
    powers = 4 ** torch.arange(k - 1, -1, -1, device=one_hot.device)  # [4^(k-1), ..., 1]
    patches = base_idx.unfold(dimension=1, size=k, step=1)  # (B, L-k+1, k)
    kmer_idx = (patches * powers).sum(dim=-1)  # (B, L-k+1)

    return kmer_idx
```

Key property: this operation is **fixed** (no learnable parameters, no
gradients). The gradients flow through the embedding table alone, which
receives a direct gradient for every k-mer that appears in the batch.


### 4.3 Training Configuration

Match the simulation study setup for a fair comparison:

| Setting | Value | Notes |
|---------|-------|-------|
| Store | `sim_store_B.zarr` (regime B, jitter=128) | Same as best CNN run |
| Splits | 16 train, 4 heldout; 800/100/100 tiles | Same |
| Batch size | 64 | Same |
| Precision | bf16-mixed | Same |
| LR | 1e-3 (start here, reduce if needed) | Smaller model may tolerate higher LR |
| Patience | 15 | Same |
| max_epochs | 200 | Same |
| Evaluation | Per-count multinomial NLL on val tiles | Apples-to-apples with CNN results |

**Expected runtime:** with ~0.8M parameters and padded convolutions (no
complex geometry), training should be significantly faster than the CNN.
Estimated 15-20 min to early-stop on A10G ($0.25-0.35).


## 5. Reverse-Complement Equivariance

### 5.1 The current approach: data augmentation

The current model uses data augmentation: with 50% probability, the input
sequence is reverse-complemented, and the target tracks are permuted via
`reverse_complement_track_permutation` (+ ↔ -, first ↔ last). This works but
doubles the effective training time and introduces noise (the model sees RC
variants independently, not as constrained transformations).

### 5.2 Weight tying for k-mer embeddings

The k-mer embedding enables a clean architectural approach:

**Under RC, k-mer `s` maps to its reverse complement `rc(s)`.** The mapping
`kmer_index → rc_kmer_index` is a fixed permutation `P` of {0, ..., 4^k - 1},
computable once at init. For palindromic k-mers (e.g., AACGTT), the index maps
to itself.

**RC equivariance constraint:** the embedding of a k-mer and its RC partner
must produce outputs that are related by the track permutation:

```
embed[kmer] == perm(embed[rc(kmer)])
```

where `perm` is the track-channel permutation (swap + ↔ -, first ↔ last).

**Implementation:** store only half the embedding table (one k-mer per RC
pair) and derive the other half by permuting channels. This halves the
embedding parameters and guarantees RC equivariance by construction.

For the contextual CNN layers: use **RC averaging** (Borzoi's approach):

```
forward:
    h_fwd = embed(kmer_indices_fwd)        # forward strand
    h_rev = perm(embed(kmer_indices_rev))   # RC strand, track-permuted
    h = (h_fwd + h_rev) / 2                # average
    ... context layers ...
```

This is simpler than constraining all downstream layers and provides exact
RC equivariance without weight tying in the conv layers.

### 5.3 Recommendation

Start with data augmentation (same as current model) for the first experiment,
to isolate the effect of the k-mer embedding from the RC equivariance change.
Add weight tying in a second experiment to measure its independent contribution.


## 6. Integration with the Existing Codebase

### 6.1 Loss Functions

**No changes needed.** All three losses (`MaskedMultinomialNLLLoss`,
`MaskedDirichletMultinomialNLLLoss`, `MaskedNegativeBinomialOffsetNLLLoss`)
operate on the same `(B, n_tracks, L_out)` shape logits and target. The
KEN model produces outputs of the same shape. The masking, softmax, and N
conditioning logic is unchanged.

### 6.2 Masking

**No changes needed.** The masked softmax convention (mask → -inf before
softmax, exclude from N) applies identically. The mask is `(B, L_out)` and
aligns with the output positions.

### 6.3 Input/Output Geometry

**Simplified.** The current CNN uses unpadded convolutions and requires:

```
L_in = calc_input_region_size(L_out)
     = L_out + 2*(k-1) + sum((k-1)*2^i for i in 1..N_layers)
```

For the 512k/3L model: L_in = L_out + 496.

The KEN model with padded context convolutions requires:

```
L_in = L_out + (k_embed - 1)
```

For k=6: L_in = L_out + 5.

This is a **major simplification**. However, it changes the geometry that the
dataset/store was built for (RF_BUDGET=2048 in the store builder). Options:

1. **Keep the existing store:** the stored sequences have 2048 bp of margin per
   side (RF_BUDGET). The KEN model only needs 5 bp of margin. Simply ignore the
   extra margin — the dataset crops to the needed input size. Wasteful in
   storage but compatible.
2. **Override `calc_input_region_size`:** implement the new formula in the
   KEN model class. The training harness already calls this method to determine
   sequence crop size.

Option 1 (keep existing store) is recommended for the initial experiment to
avoid rebuilding stores.

### 6.4 Dispersion Head

For the DM and NB-offset losses, the dispersion head operates on the same
trunk features. In the KEN model, the trunk output is the post-context
representation `(B, d_ctx, L_out)`. The dispersion head is a
`Conv1d(d_ctx, n_tracks, 1)`, pooled via `masked_mean_pool` per the existing
logic. No changes to `_pooled_log_dispersion`.

### 6.5 Training Harness

The `BackgroundModel` class in `background_model_core.py` is the statistical
specification. The KEN model should be a **separate class** (e.g.,
`BackgroundModelKEN`) that implements the same Lightning interface
(`training_step`, `validation_step`, `predict_profile`, etc.) and shares the
same loss functions. The training harness (`background_model/train.py`) adds
a `--model` flag to select between CNN and KEN.

This avoids modifying the frozen statistical core.

### 6.6 Inference and Correction

`predict_profile()` returns `{"probs": ..., "log_dispersion": ...}` — same
interface. The correction module (`background_model/correction.py`) and
inference module (`background_model/inference.py`) operate on these outputs
and do not depend on model internals. No changes needed.


## 7. VRAM and Compute Estimates

### 7.1 KEN-v1 (recommended)

| Component | Memory (bf16) |
|-----------|---------------|
| Parameters (0.8M) | 1.6 MB |
| Optimizer states (Adam, fp32 master) | 10 MB |
| Input + embeddings (B=64, L=2053, d=64) | 17 MB |
| Context CNN activations (2 layers, d=128) | 34 MB |
| Gradients | ~20 MB |
| **Total training** | **< 100 MB** |
| **Ratio of A10G budget** | **< 0.5%** |

This leaves enormous headroom. B=512 would use ~600 MB. B=2048 would use
~2.5 GB. Throughput would likely be bound by data loading, not GPU compute.

### 7.2 Scaling options within VRAM budget

| Configuration | Params | VRAM (B=64, bf16) |
|---------------|--------|-------------------|
| KEN-v1 (6-mer, d=64, 2 conv) | 0.8 M | ~100 MB |
| KEN-v1 + 8-mer (multi-k) | 2.9 M | ~200 MB |
| KEN-v1 + 4 transformer layers (d=256) | 4.0 M | ~500 MB |
| Current CNN (512k/3L) | 25.4 M | ~2.9 GB |
| Enformer-lite (d=512, 8 layers) | 26 M | ~3 GB |

All options are comfortably within the 24 GB A10G budget.


## 8. Experimental Plan

### Phase 1: Baseline comparison on simulation data

1. **Implement `BackgroundModelKEN`** as a new class alongside the existing
   `BackgroundModel`. Share loss functions, masking logic, and the Lightning
   interface.
2. **Train on `sim_store_B.zarr`** with multinomial loss (the best-performing
   loss from the simulation study). Use identical splits, batch size, precision,
   and evaluation.
3. **Evaluate**: per-count multinomial NLL, hexamer recovery (correlation with
   ground-truth w6 table), corrected flatness. Compare directly against the
   42% baseline.
4. **Inspect the learned embedding table** vs ground truth w6. This is a
   uniquely powerful diagnostic: if the model learns an embedding whose first
   principal component correlates strongly with w6, it confirms the
   architecture is working as intended. No such interpretability is possible
   with the CNN.

### Phase 2: Ablations (if Phase 1 succeeds)

5. **Multi-k ablation:** add 4-mer and 8-mer embeddings. Does longer k-mer
   context improve beyond the 6-mer?
6. **Context depth:** add more conv layers or a single transformer layer.
   Does longer-range context help?
7. **RC equivariance:** replace data augmentation with weight tying. Does it
   improve sample efficiency?

### Phase 3: Real data (if simulation succeeds)

8. **Train on the real store** (`bg_store_b67d7c95.zarr`, 160K training pairs).
   Compare CTCF footprint recovery and QQ calibration against the CNN baseline.

### Success criteria

| Outcome | Interpretation | Next step |
|---------|----------------|-----------|
| KEN closes >80% of the gap | Hexamer lookup was the bottleneck | Adopt KEN for production |
| KEN closes 50-80% | Lookup helps but longer-range context needed | Add transformer layers (hybrid) |
| KEN closes <50% | Signal is not purely hexamer-local | Investigate Enformer-lite (Candidate B) |
| KEN ≤ CNN (no improvement) | The CNN's bottleneck is not representation but optimization/data | Revisit assumptions |


## 9. Comparison with Existing Genomics Models

This problem differs fundamentally from the Enformer/Borzoi setting:

| Dimension | Enformer/Borzoi | Background Model |
|-----------|-----------------|------------------|
| Input size | 196,608 bp | 2,048 bp |
| Signal scale | Long-range (enhancer-promoter, 100kb+) | Local (hexamer = 6 bp) |
| Output | Gene expression (128 bp bins) | Per-bp fragment endpoint profile |
| Resolution | ~128 bp | 1 bp |
| Training data | Thousands of CAGE/ATAC tracks | 12 tracks × ~50 samples |
| Attention purpose | Long-range regulatory interactions | Combinatorial k-mer identity |
| Model size | ~250M parameters | 0.8-25M parameters |

The key difference is that Enformer's success comes from attention's ability
to model *long-range interactions* (e.g., enhancer-promoter pairs 100kb
apart). Our problem has no long-range interactions — the signal is entirely
local (6 bp hexamer + ~100 bp GC context). What we need from attention (or
embedding) is not range but **combinatorial capacity**: the ability to learn
an arbitrary function of a 6-mer identity.

This is why the k-mer embedding (Candidate A) is preferred over the
Enformer-style architecture (Candidate B): it provides the combinatorial
capacity without the unnecessary long-range machinery.


## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| K-mer embedding is trivially better | Medium | Low (good outcome) | None needed |
| Hexamer is necessary but not sufficient; GC context matters | Medium | Medium | Multi-k + context conv layers |
| Training fails to converge (embedding overfits) | Low | Medium | Dropout on embeddings, weight decay, multi-k regularization |
| Real data has different signal structure than simulation | Medium | High | Phase 3 is the real test; simulation is the validation of methodology |
| K-mer index computation has edge effects (N bases, tile boundaries) | Low | Low | Mask boundary positions; N bases map to index 0, excluded via blacklist mask |

The overall risk profile is favorable: the experiment is cheap ($0.25-0.35 per
run), the architecture is simple to implement, and the diagnostic power (direct
embedding inspection) means failures are informative even if the model doesn't
outperform.
