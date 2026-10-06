# Cut-site fragment model — integration design

Status: DESIGN COMPLETE, implemented on worktree-cut-site-model.

This is the integration and plumbing spec for a model whose native output is the
fragment weight itself — one logit per `(m, L, strand)` — trained on the
multinomial likelihood of the observed fragments. It **joins the bake-off**
alongside the 12-track model class (`BackgroundModel` / `BackgroundModelKEN` /
`BackgroundModelHybrid` in `background_model_core.py`). Both classes coexist and
are trained and compared on their own stores (two stores, deliberately not
unified — see below).

The per-fragment NLL metric, the normalisation domain `D`, the cut-site index
convention, and the uniform/oracle anchors are defined in the sibling doc
`simulator_and_fragment_nll.md` (the NLL doc). This doc specifies the
model, its store, its dataset, its objective, and its evaluation.

## The problem

The 12-track model outputs `2 strands × 2 fl_bands × 3 coverage_types = 12`
per-position tracks (`background_model/tracks.py`). Two properties the new model
removes:

1. **`FL_BANDS` is baked into the output channel axis**, and all lengths in a
   band share one per-position propensity, so a 125 bp and a 174 bp fragment are
   indistinguishable to a track.
2. **~49% of fragments fall outside every band** and contribute to no track.

The new model emits a logit for every `(m, L, strand)` and trains on the
fragment multinomial. Bands become a reporting choice, not a model
parameterisation.

## Geometry

**Start admission.** A fragment is admitted iff `min(fragment_start,
fragment_end)` lies in `[0, tile_size)`. `m` is the fragment START position,
not the centre. This gives `|D_L| = P` for every in-band `L` (length-
independent — start in `[0, P)` gives `P` positions for every `L`).

`WINDOW_HALF = 181`.  Derivation: the model's reachable cut range in hex-idx
space is `[WINDOW_HALF, WINDOW_HALF + P + L_max - 2]`.  The widest margin
needed is `L_max - 1 = 179`, so `WINDOW_HALF >= 179`.  181 is chosen so
`rf_budget = WINDOW_HALF + 5 = 186` keeps its existing definition.  Symmetric
by design — the unused left margin is the price of reviewability after the
off-by-3 incident.

Window `W` = 2 × WINDOW_HALF = 362 bases.

```
left tap   o_l = WINDOW_HALF           (independent of L)
right tap  o_r = WINDOW_HALF + L       (L apart = cut site spacing)
```

The taps are exactly `L` apart, indexing **CUT SITES** (`m` and `m+L`).

The interior branch's offset axis spans `n_offsets = L_max` (180), covering
`[o_l_min, o_r_max)` where `o_l_min = WINDOW_HALF`.  Under start admission
the left tap doesn't depend on `L`, so `o_l_prime = 0` for all fragments and
the prefix-sum difference simplifies to just `U[L]`.
The prefix axis has `n_offsets + 1 = 181` slots.

## Model architecture

### L1 — cut-site propensity

`N1` = 128 width-6 convolutions over the one-hot sequence → a per-cut-site
propensity track.

**There MUST be a nonlinearity after the convolution.** A width-6 linear filter
is a PWM (24 weights), and any linear combination of PWMs is still a PWM. The
truth has 4096 independent hexamer values, so a linear path would fail
structurally. Embedding alternative deferred to Stage 3.

### L2a — endpoint branch

The **truth has FOUR hexamer tables**: `{start, end} × {forward, reverse}`,
**untied** — no RC tying. This data is short-read single-stranded, so the two
ends are not related by reverse complement.

**Four full banks — `a`, `B`, `W` for each `(tap, strand)`** in
`{start, end} × {forward, reverse}`, **untied**. Two distinct products
`start_f·end_f` and `start_r·end_r` require two independent banks per tap; a
shared bank plus a per-strand *additive* bias will not do — it exponentiates to a
single product, exactly the strandless failure this branch exists to avoid. Each
bank has bank dimension `K` = **128**: `a (K, N1)`, per-group multiplicative gain
`B (K, 10)`, projection `W (32, K)`. Per bank `16,384 + 1,280 + 4,096 = 21,760`;
per tap (both strands) `43,520`; all four banks `87,040`.

`K = 128` rationale: the factorisation is kept for a **statistical** reason —
`a` pools estimation across all lengths, `B` is a small per-group correction.

**A free per-`L` additive term `len_bias (n_L,)` is required**, separate from the
length groups. The groups carry the length × *sequence* interaction at 15 bp
resolution; `len_bias` carries the length distribution itself at 1 bp resolution.
The length component is worth **~0.4255 nats, about 5× the whole sequence-bias
gap of ~0.084**, so forcing it through a piecewise-constant 15 bp approximation
would be a large avoidable loss.

**Placement: on the logit, after the nonlinearity** — not into the
pre-activation. `logit[m,L,s] = w_out · act(pre) + len_bias[L]`. This is
`(n_L,)` = **155 parameters**, strand-shared (the FL/GC length term is
strand-independent).

**Length groups: 10, half-open, 15 wide from 25 with the LAST 20 wide:**

```
[25,40) [40,55) [55,70) [70,85) [85,100) [100,115) [115,130) [130,145) [145,160) [160,180)
```

Index `g = min((L-25)//15, 9)`; the clamp folds `[175,180)` into the last group.
Limitation: 15 bp resolution cannot isolate a sharp effect at ~167.

### L2b — interior branch

A **second, separate** bank `F (N2=32, 4, n_offsets=180)` with weights **learned
per offset**, spanning the live range `[o_l_min, o_r_max)` (see §Geometry).
Re-indexed as `o' = o - o_l_min`. Prefix sum along the re-indexed offset axis:

```
u[n,m,o'] = sum_c F[n,c,o'] * x[c, m + o_l_min + o']   # x in h-coordinates
U          = prefix_sum(u, axis=o')       # exclusive, n_offsets+1 = 180 slots
span       = U[n,m,o_r'] - U[n,m,o_l']   # o_l' = o_l - o_l_min, o_r' = o_r - o_l_min
```

Under start admission `o_l' = 0` for all L, so the subtraction is **one subtraction per pair** — not
a per-candidate 256-wide convolution, which would be ~256× the work. Feed
**both** `span` and `span/L` explicitly; a dense layer cannot form that ratio
from `[span, L]`.

### L3, strand, output

`concat(L2a, L2b)` → dense `H = 32` → nonlinearity → **one logit per
`(m, L, strand)`**.

**The model HAS a strand axis — two logits per `(m, L)`.** The marginal is
`0.5·(start_f·end_f) + 0.5·(start_r·end_r)`, a **sum of two products**, while an
additive strandless logit exponentiates to a **single** product — so a strandless
model structurally cannot represent it. Representing the two distinct products
needs the four full endpoint banks of §L2a (one per `(tap, strand)`); a shared
bank with a per-strand additive bias still exponentiates to one product and does
not suffice. Strand is observed, so conditioning on it is legitimate.

## The fold-down, and its condition

Both branches fold to per-position tensors:

- **L2a** folds because the endpoint term is a **sum** of two per-position
  projections: `M_l[g] = W_l diag(B_l[:,g]) a_l` collapses to ten `(32,128)`
  matrices applied directly to `h`, so the `K`-wide `e_l` is never materialised.
- **L2b** folds because the span is a **difference** of two values from one
  per-`(m,o)` tensor, and `W·(span/L) = (W·span)/L` folds too.

Unfolded, the per-pair concat would be `e_l(128) + e_r(128) + span(32) +
mean(32) = 320` wide. Folded, the pair grid holds only `H = 32`.

**CONDITION: the endpoint path must stay LINEAR until it is summed into the
pre-activation.** A nonlinearity on the per-pair features destroys the fold-down.

### Shapes across the layers

```
B    = 64     regions per step          N1  = 128   L1 cut-site bank
P    = 1024   scored start positions    K   = 128   L2a bank dim (x4 banks)
n_L  = 156    in-band L = 25..180       N2  = 32    L2b interior bank
W    = 362    window bases              H   = 32    hidden width
L_chunk = 32                            l_in  = 1392 (calc_input_region_size(1024))
```

```
                  ┌────────────────────────────────────────────┐
 one-hot seq      │  (B, 4, 2310)                    2.4 MB    │
                  └────────────────────────────────────────────┘
                          │  L1: Conv1d(4 -> 128, k=6) + nonlinearity
                          v
 h  cut-site       ┌───────────────────────────────────────────┐
    propensity     │  (B, 128, 2304)                  75 MB    │  PER-POSITION
                   └───────────────────────────────────────────┘
                     │                              │
        ┌────────────┘                              └──────────────┐
        v  L2a                                                     v  L2b
 ── ENDPOINT BRANCH ─────────────────────        ── INTERIOR BRANCH ──────────────
 a[t,s] : (128,128) x4     65,536 params        F : (32, 4, 179)   22,912 params
 B[t,s] : (128,10)  x4      5,120 params        W_span, W_mean : (32,32) each
 W[t,s] : (32,128)  x4     16,384 params
                                                 u never materialised whole;
 FOLD: M_l[g,s] = W_l[s] diag(B_l[s,g]) a_l[s]   prefix-sum over o then difference
       -> 20 matrices of (32,128)  (10 groups x 2 strands)
                                                 v[h,m,o] = sum_c G[h,c,o]·x[..]
 f_l[s] = M_l[g,s] @ h                           U = cumsum(v, axis=o)
        │                                               │
        v                                               v
 ┌──────────────────────────┐                 ┌──────────────────────────────┐
 │ (B,10,2,32,2304) 378 MB  │                 │ (B, 32, 2048, 180)  3.02 GB  │
 │ x2 taps = 756 MB         │                 │ chunk over m -> 377 MB       │
 └──────────────────────────┘                 └──────────────────────────────┘
        │  gather at o_l, o_r                         │  difference at o_r, o_l
        └───────────────┬─────────────────────────────┘
                        v
              pre[h, m, L, s]  =  f_l[s] + f_r[s]           <-- LINEAR until summed
                                + (V_span[o_r'] - V_span[o_l'])
                                + (V_mean[o_r'] - V_mean[o_l']) / L
                        │
                        v   <-- THE ONLY PER-PAIR TENSOR
              ┌──────────────────────────────────────────────┐
              │ hid = act(pre)  (B,32,2048,L_chunk=32,2)     │
              │        1074 MB fp32  /  537 MB bf16          │
              └──────────────────────────────────────────────┘
                        │  logit = w_out · hid + len_bias[L]
                        v
              ┌──────────────────────────────────────────────┐
              │ logit  (B, 2048, 155, 2)          strand axis │
              └──────────────────────────────────────────────┘
                        │  streaming logsumexp over L blocks
                        v
              log Z  (B,)  ->  fragment multinomial NLL
```

In the diagram `V_span = W_span @ U` and `V_mean = W_mean @ U` — the projection
folded *into* the prefix tensor, so the difference is taken after projecting to
`H`. The forward-pass listing below shows the same arithmetic unfolded
(`span = U[o_r] − U[o_l]`, then `W_span·span`); the two are the same quantity,
`W` being the matrix and `V` the folded prefix.

**Where the cost is:** the offset-axis prefix tensor `(B, 32, 2048, 180)`. The
prefix sum is cumulative over `o'`, so the axis cannot be tiled. All 179 entries
are live (dead offsets trimmed). **Chunk over `m`, not over `o`**.

### Forward pass

```
h = L1(seq)                              # (B, N1, model_input_size), post-nonlinearity

# L2a endpoint (per strand s): shared tap weight + per-group gain, folded
o_l, o_r = 128 - L//2, 128 - L//2 + L    # window offsets of the two CUT SITES
g        = min((L - 25)//15, 9)          # length group
f_l[s] = M_l[g,s] @ h ; f_r[s] = M_r[g,s] @ h   # M_x[g,s] = W_x[s] diag(B_x[s,g]) a_x[s]

# L2b interior: re-indexed offsets o' = o - o_l_min
u[n,m,o'] = sum_c F[n,c,o'] * x[c, m + o_l_min + o']   # (B, N2, P, n_offsets)
U          = prefix_sum(u, axis=o')                      # (B, N2, P, n_offsets+1), exclusive
span       = U[:,:,o_r'] - U[:,:,o_l']
mean       = span / L

pre[h,m,L,s] = f_l[s] + f_r[s] + W_span·span + W_mean·mean   # LINEAR
hid          = act(pre)                                         # (B, H, P, L_chunk, 2)
logit[m,L,s] = w_out · hid + len_bias[L]                       # len_bias on the LOGIT

ep_shape     = endpoint_shape_head(h_centre)  # (B, 2, P) trainable shape for NB loss
disp_bp      = dispersion_head(h_centre)      # (B, 2, P) FROZEN at log_dispersion_init
```

## Store schema

A **separate zarr store** from the 12-track store (two stores, deliberately). It
reuses the existing store's mechanical conventions (zarr v2 format, `zarr==2.18.3`
pin, CSR triples, `split`/`role` codes, bulk single-write build) but carries a
different payload: per sample × region the **fragment list** (the object the
multinomial trains on) and the **two per-position endpoint count tracks** (the
ancillary NB targets), both derived from the identical fragment list at build
time.

```
/attrs: config_json, config_hash, created_utc, split_version, max_len, sim_dir, sim_study
/tiles/{contig, start, stop, strand, region_id, split, seq, mask}
/samples/{library, role, total_fragments}
/fragments/ (CSR over flat index u = s*T + t)
    indptr   (S*T + 1,)  int64
    start    (nnz,)      uint16   # region-LOCAL left endpoint p (base)
    length   (nnz,)      uint16   # fragment length L (= stop - start)
    strand   (nnz,)      uint8    # 0 = +, 1 = -   (the model conditions on strand)
/endpoints/ (CSR over flat index u = s*T + t)
    indptr   (S*T + 1,)  int64
    pos      (nnz2,)     uint16   # per-base position
    track    (nnz2,)     uint8    # 0 = first (left endpoint), 1 = last (right)
    data     (nnz2,)     uint16   # count
/totals/N  (S, T, 2)    uint32    # per-endpoint-track totals over the centre tile
```

- **`start`/`length`, not `start`/`stop`.** The multinomial indexes `(p, L)`
  directly. `L ≤ max_len (256)` and `p ≤ l_target (≤ 2560)` both fit `uint16`.
- **Strand IS stored** because the model conditions on it. (The 12-track store
  omits strand; this one must not.)
- **Endpoint tracks are the 2 band-free NB targets**, strand-pooled: `first[p]
  += 1` and `last[p+L-1] += 1` per fragment. The right endpoint base is `p+L-1`.
  A build-time test asserts these tracks equal a recomputation from
  `/fragments/`, closing the ±1 gap at the source.
- **`/totals/N` has 2 columns**, not 12 — the count axis differs, which is why
  this store is separate.
- **Geometry** is recorded as the sim store does today (`tile_size`, `jitter`,
  `rf_budget`, derived `l_target = tile_size + 2·jitter`, `l_seq = tile_size +
  2·(jitter + rf_budget)`). The v4 stores run at `tile_size ∈ {2048, 1024}`,
  `jitter = 256` → `l_target ∈ {2560, 1536}`; invoke the builder with
  `--jitter 256`.

### Config hashing and drift guard

Do **not** add `max_len` to the shared `PlumbingConfig`: hashing a new field
would rehash *every* store, including the real-data production store. The new
store records its geometry in a separate small config (`tile_size`, `jitter`,
`rf_budget`, `max_len`, `seed`, `n_train_samples`), hashed with the same
`hashlib.sha256(canonical_json)` discipline, giving it its own drift guard while
`config.py` stays frozen for the old class. `split_version` carries over.

**Domain guard.** Because `max_len` defines `D`, the Dataset MUST assert
`model.max_len == store.attrs["max_len"]` at construction and fail loudly. A
silent `max_len` mismatch changes `|D|` and reinterprets every reported NLL.

## Dataset / dataloader

This model is **sequence-only**: the `(m, L, strand)` logit map depends only on
the tile's sequence. At **S = 1** (current), the dataset indexes by
`(sample, tile)`. The structure supports `S > 1` region-batching later without
restructuring.

Item = one region, yielding:

```
x         : (4, l_in)              float32   one-hot sequence
frag_m    : (n_frag,)              int64     start positions in crop coords
frag_L_idx: (n_frag,)              int64     L index (0-based, L - L_MIN)
frag_s    : (n_frag,)              int64     strand (0 or 1)
frag_N    : scalar                 int64     total fragment count
endpoints : (2, tile_size)         float32   the two NB target tracks
mask      : (tile_size,)           bool      valid positions
```

The multinomial shares one per-region `logp` map across the `S` samples:
`NLL_s = -Σ_{(m,L,s)∈frags_s} logp[m,L,s] / N_s`, averaged over samples.

**Crop.** The jittered crop uses `jitter_matrix` with concentric widths
(seq → `l_in`, targets → `tile_size`). A fragment survives iff
`min(fragment_start, fragment_end)` lies in `[0, tile_size)` (start-based
admission — see §Settled parameters). `frag_m` is the fragment start in
crop coordinates.  Train draws `j ∈ [-jitter, jitter]`; **val and
test use `j = 0` (centred crop) — jitter is training-only**. `jitter_matrix`'s
crop requires its input and output lengths to share **matched parity** (not even
parity per se — see `calculate_start_and_stop_from_jitter`): `l_in` =
`model.model_input_size` (already includes +5 for the conv margin and
even-parity rounding — no separate +5 step in the dataset).
The model trims the extra position after the width-6 conv
(same pattern as `BackgroundModelKEN`). The parity assertion covers `tile_size` /
`l_target` / `l_seq` / `l_in`.

**Shared-jitter coupling.** Because one sequence forward serves all `S` samples,
all samples of a region in a step share one jitter offset — the crop defines the
position frame the shared `logp` map lives in. This is accepted: it is the price
of the shared-forward speedup (the whole point of region-batching). Diversity is
recovered across steps/epochs (fresh `j` per region each draw) and across
regions within a batch. Document this in the dataset docstring so it is not
"fixed" later.

**RC augmentation is DROPPED: `rc_prob = 0`.** There is no RC-equivariance
requirement on this model and no RC-equivariance test.

**`min_N` = 10.** Filter on **total fragment count per `(sample, region)`**:
regions with fewer than 10 fragments for a sample are skipped.

**Worker safety** is inherited verbatim: lazy per-PID zarr handle, optional
in-memory preload with copy-on-write fork sharing, no CUDA in the dataset.

## The model module — where it lives

**New module `background_model/cut_site_model.py`.** It imports primitives from
the frozen `background_model_core.py` but does **not** modify it: `jitter_matrix`
(via the dataset); `one_hot_to_kmer_indices`, `rc_kmer_permutation` only if L1
uses the embedding arm; `MaskedNegativeBinomialOffsetNLLLoss` reused unchanged
for the endpoint NB term (a `(B, C=2, L)` call); the per-window dispersion
machinery reused unchanged — `_BackgroundModelMixin._pooled_log_dispersion`
(`background_model_core.py`), which mean-pools `disp_bp (B, 2, P)` to the NB
loss's window level `(B, 2, W)`, `W = P // dispersion_window_size`. Because the
class does not subclass the mixin (see below), copy this method verbatim rather
than inheriting it. The design changes no frozen-core statistical semantics.

The class is a `lightning.LightningModule`. It does **not** subclass
`_BackgroundModelMixin`, whose `_step` is hardwired to the per-track core losses
and whose `predict_profile` softmaxes `(n_tracks, L)` over `L` — neither fits a
`(m, L, strand)` output. It implements its own `training_step`/`validation_step`.
A `predict` entry point will be added when correction wiring is in scope.

## Objective

The **fragment multinomial over `(m, L, strand)` is the test; all other losses
are ancillary.** Per sample, over the region's shared
`logp = log_softmax(logit over D)`:

```
L_mult = mean_s [ -Σ_{(p,L,s)∈frags_s} logp[p,L,s] / N_s ]
```

Plus an **NB term on each of the two band-free endpoint tracks**, one
`MaskedNegativeBinomialOffsetNLLLoss` call on the `(B, C=2, L)` target
(`first` = 0, `last` = 1), with the **dispersion head FROZEN (not trained)**.
The endpoint tracks are band-free by design, so `N_first = N_last = N_fragments
= N_s` and both terms are per-fragment nats on the same scale; this equality
would break if the tracks were banded.

The endpoint term is an **annealed warmup**, not a fixed-weight co-objective:

```
loss = L_mult + w(t) · L_NB
w(t) = w0 · max(0, 1 - t / T_anneal)      # linear decay to 0
```

It gives the cut-site propensity a strong, well-conditioned gradient early, then
fades so the converged objective is the pure fragment likelihood. `w0` and
`T_anneal` are engineering hyperparameters with no principled value — **set them
in the first smoke run** (start `w0 = 1` and `T_anneal` at a few epochs, then
tune). `L_mult` and `L_NB` are both **per-fragment nats on the same scale** (the
band-free `N_first = N_last = N_s` equality above), so `w0` is a dimensionless
mixing ratio, not a unit-bridging factor. **Evaluation is the fragment
multinomial only.**

## Evaluation

The currency is the **per-fragment NLL**. The NLL doc owns the normalisation
domain `D` (in-band ∧ in-crop ∧ strand), the uniform (`log|D|`) and oracle
anchors, the `log 2` strand term and its cancellation, and the
`% bias captured = (uniform − model)/(uniform − oracle)` definition — including
that both anchors are recomputed per store. Only the model-specific scoring form
belongs here.

For this model, over the region's shared `logp`:

```
NLL_model = -logp[p, L, s]           gathered over the observed fragments in D
logp      = log_softmax(logit over D)
```

The `(m, L, strand)` logits are softmaxed over exactly `D` (the NLL doc's set).
**No external `len_p[L]` or GC factor** — the model's logit already carries both;
applying the track-model product would double-count length.

**Old 12-track models** are scored via `w_old[p, L] = first_b[p] · last_b[p+L-1]`,
**scored natively on strand**, with **no `len_p` supplied**. With `first_b`/
`last_b` identical for every `L` in a band and no `len_p`, an old model implicitly
predicts **uniform lengths within each band**, so its NLL is dominated by length
mis-specification rather than by sequence bias. Structurally fair, but it does
**not** isolate sequence-bias recovery.

### Why the model cannot reach the oracle exactly

`% captured` is **capped below 100 by construction**, and the cap is the
**receptive field**, not the joint form of the model.

The sequential truth carries a per-start normaliser `Z_s(p)` (NLL doc). Because
`−log Z_s(p)` is a function of `p` **alone**, a joint log-linear model with a free
per-position term absorbs it **exactly** — the joint-vs-sequential distinction is
*not* the obstruction. The obstruction is spatial reach: the model's per-position
term is `M_l[g,s] @ h[:,p]`, and L1 is a single width-6 convolution, so `f_l(p)`
sees only **6 bp**. But `Z_s(p) = Σ_q end_s[hex(q)] · FLGC` sums over ~155 lengths
and so depends on **~256 bp** of surrounding context. The model cannot see far
enough to represent it.

`Z_s(p)` is an average over ~155 positions, so it is **smooth**; but its variation
across positions could still be of the same order as the whole available bias gap
(~0.084 nats), which would make the cap material. **Measure `var(log Z_s(p))` over
the region set early**: it bounds the unreachable fraction, and without it a
sub-100% result is uninterpretable. Widening L1's receptive field (or adding an
explicit context branch) is the lever that shrinks the gap.

## Cost

**Not yet measured on the implemented model.** The prior benchmark
(`scripts/bench_fragment_logit_sweep.py`) was run on a strandless, unfolded
prototype that differs from this architecture on both the strand axis and the
interior branch. `L_chunk = 32` is mandatory (unchunked OOMs). Re-measure
before budgeting GPU time.

## Required tests

- **±1 bp alignment in CUT-SITE space.** Assert that the cut sites the model
  indexes are the same genomic cut sites the sampler drew from: the taps at
  `o_l = 128 - L//2` and `o_r = o_l + L` index the cut sites `p` and `p+L` that
  the shared `build_region_weights` reads (`fwd_cut[p]` / `rc_cut[p+L]`), noting
  the taps are `L` apart (`p` and `p+L`, **not** `p` and `p+L-1`). Pin the
  property, not any line number, in one test on a small region with known
  fragments: the store's per-base derivation (`first[p]==1`, `last[p+L-1]==1`),
  the model's cut-site tap indexing, and the centre↔endpoint round-trip
  `p = m - L//2` for **both parities of `L`** — a shift here is silent, and a
  test that only exercises even `L` would miss it.
- **No RC-equivariance test** — it is void now.
- **Store consistency:** `/endpoints/` recomputed from `/fragments/` at build
  time must match the stored tracks exactly.
- **Domain guard:** constructing the Dataset against a store whose
  `attrs["max_len"]` ≠ the model's `max_len` must raise loudly.
- **Oracle below uniform:** on sim data the oracle's own fragment NLL must come
  in strictly below `log|D|`.
- **Shared builder exercised by both paths:** the oracle scorer and the sampler
  must call one `build_region_weights`, tested together so a change to one
  cannot silently diverge.

Follow CLAUDE.md testing discipline: run the `background_model` suite via
`make test PYTEST_ARGS="tests/ -q"` before and after, and measure the baseline
yourself (it moves per commit).

## Settled parameters

- `min_N` = 10 (total fragment count per sample×region).
- **Start-based fragment admission**: D = {(m, L, s) : m ∈ [0, P), L in-band,
  s ∈ {0,1}}. |D| = P × n_L × 2 = 319,488. |D_L| = P for all L — no length bias.
- `WINDOW_HALF` = 181, `rf_budget` = 186 (WINDOW_HALF + 5).
- **F offset range**: `n_offsets = L_max` (180),
  `o_l_min = WINDOW_HALF`. Under start admission the left tap is fixed.
- **NB endpoint loss kept, dispersion frozen** at `log_dispersion_init`.
- **Conv L1** (not embedding). Embedding arm is a Stage 3 improvement.
- **len_bias (n_L,)**: scalar per-L on the logit (not per-H).
- **S = 1 currently.** Dataset indexes by `(sample, tile)`.
- **Separate val AND test** region sets; val/test use the centred crop.
- **Two stores**, one per model class.
- Simulator `max_len` = 256, with filtering to the bands before inference.

## Limitations (accepted, not solved here)

- The current sampler draws lengths from a **global** PMF, so a
  region-dependent length distribution is untestable on current sim data. Open
  TODO, not solved here.
- `bias_correction/` is v1 and superseded; this design does not build on it.
- Wiring the new model's `(m, L, strand)` output into `correction.py` (which
  assumes a per-position SHAPE head with the 12-track gather) is out of scope
  until a model is chosen.
