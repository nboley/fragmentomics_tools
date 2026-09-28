# Cut-site fragment model — integration design

Status: DESIGN, not implemented. Doc-only.

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
   indistinguishable to a track. This is the structural problem, and it does not
   go away by choosing better bands.
2. **Fragments outside every band contribute to no track at all** — ~24% of
   realised fragments under the current `((25,110),(110,180))`. Widening the bands
   shrank this (it was ~49% under `((40,65),(120,175))`) but cannot remove it,
   because any banding leaves a remainder.

The new model emits a logit for every `(m, L, strand)` and trains on the
fragment multinomial. Bands become a reporting choice, not a model
parameterisation.

## Geometry

Window `W` = 256 bases = **257 cut sites**. A fragment of length `L` sits
centred in the window:

```
left tap   o_l = 128 - L//2
right tap  o_r = o_l + L
```

Integer **floor** division; an odd `L` sits half a base off-centre,
deterministically and always in the same direction. The taps are exactly `L`
apart, so they index **CUT SITES** (`p` and `p+L`), matching the simulator's
`fwd_cut[p]` / `rc_cut[p+L]` — **not** per-base end positions.

The map is indexed by the window **centre**: `(m, L)`, with `p = m - L//2`.

**The offset axis is 219 slots.** Scoring is in-band (`L ≤ 179`), so the largest
offset reached is `o_r = 128 + ceil(179/2) = 218`. Sizing the axis to the store's
`max_len` instead (256 ⇒ 257 slots) would waste ~15% of the dominant
`(B, 32, 2048, ·)` prefix tensor on offsets never touched. Scoring out-of-band
lengths would require widening it; nothing else would.

The store's `max_len` is a separate quantity, checked by its own domain guard
(see Store schema) — it is not what sizes this axis.

## Model architecture

### L1 — cut-site propensity

`N1` = 128 width-6 convolutions over the one-hot sequence → a per-cut-site
propensity track.

**There MUST be a nonlinearity after the convolution.** A width-6 4-channel
filter has 24 weights and its hexamer response is `sum_i w[base_i, i]`, i.e. a
PWM score — and any *linear* combination of such filters is still a PWM score.
The simulator's `build_w6` draws **4096 independent** values, so a linear path
could only fit a 24-parameter PWM against a 4096-value target and would fail
structurally, presenting as a training failure. The hexamer-embedding
alternative (`one_hot_to_kmer_indices` + `nn.Embedding`, as in
`BackgroundModelKEN`) is a free 4096-entry lookup and can represent it exactly;
the conv form is expected to generalise better on real data, so **conv is the
default and the embedding is a comparison arm**.

### L2a — endpoint branch

The truth has **four hexamer tables**, `{start, end} × {forward, reverse}`, and
they are **untied**: this is short-read single-stranded data, so the two ends are
not related by reverse complement. The model mirrors that with **four full banks,
one per `(tap, strand)`** — each carrying its own `a`, `B` and `W`.

Each bank has bank dimension `K` = **128**: `a (K, N1)`, per-group multiplicative
gain `B (K, 10)`, projection `W (32, K)`. Per bank
`16,384 + 1,280 + 4,096 = 21,760`; per tap (both strands) `43,520`; all four
`87,040`.

**Why four full banks and not a shared bank with a per-strand bias:** the two
strands need two *distinct products*, `start_f·end_f` and `start_r·end_r`. An
additive per-strand bias exponentiates to a single product scaled by a constant,
which cannot represent their mixture. This is the one argument the strand axis
rests on; §L3 states the consequence, not the argument again.

`K = 128` rationale: the factorisation is kept for a **statistical** reason —
`a` pools estimation across all lengths, `B` is a small per-group correction.

**A free per-`L` additive term `len_emb (H, n_L)` is required**, separate from the
length groups. The groups carry the length × *sequence* interaction at 15 bp
resolution; `len_emb` carries the length distribution itself at 1 bp resolution.
These must not be conflated: the length component is worth **~0.4255 nats, about
5× the whole sequence-bias gap of ~0.084**, so it is the dominant term in the
likelihood, and forcing it through a piecewise-constant 15 bp approximation would
be a large avoidable loss. At `H = 32` and `n_L = 155` this costs 4,960
parameters. It is additive and per-`L` only, so it folds like everything else,
and it is **shared across strand** — the FL/GC (length) term `FLGC(L, gc)` is
strand-independent in the generative form, so there is one `len_emb`, not one per
strand.

**Length groups: 10, half-open, 15 wide from 25 with the LAST 20 wide:**

```
[25,40) [40,55) [55,70) [70,85) [85,100) [100,115) [115,130) [130,145) [145,160) [160,180)
```

Index `g = min((L-25)//15, 9)`; the clamp folds `[175,180)` into the last group.

Limitation, stated precisely because it is easy to overstate: the **length ×
sequence interaction** is resolved at 15 bp, so a cut-site preference specific to
~167 bp cannot be isolated from its neighbours. The **length distribution itself**
is unaffected — `len_emb` carries that at 1 bp.

### L2b — interior branch

A **second, separate** bank `F (N2=32, 4, 256)` with weights **learned per
offset**. Aggregate by a prefix sum along the **filter-offset** axis:

```
u[n,m,o] = sum_c F[n,c,o] * x[c, m-128+o]
U        = prefix_sum(u, axis=o)          # exclusive; 219 slots (max scored o_r + 1)
span     = U[n,m,o_r] - U[n,m,o_l]
```

The centred masks are **nested**, so this is **one subtraction per pair** — not
a per-candidate 256-wide convolution, which would be ~256× the work. Feed
**both** `span` and `span/L` explicitly; a dense layer cannot form that ratio
from `[span, L]`.

### L3, strand, output

`concat(L2a, L2b)` → dense `H = 32` → nonlinearity → **one logit per
`(m, L, strand)`**.

**The model HAS a strand axis — two logits per `(m, L)`.** The strand marginal is
`0.5·(start_f·end_f) + 0.5·(start_r·end_r)`, a sum of two products, which a single
strandless logit cannot represent (§L2a). Strand is observed, so conditioning on
it is legitimate rather than a modelling liberty.

## The fold-down, and its condition

Both branches fold to per-position tensors:

- **L2a** folds because the endpoint term is a **sum** of two per-position
  projections: `M_l[g,s] = W_l[s] diag(B_l[s,g]) a_l[s]` collapses to **20**
  `(32,128)` matrices (10 length groups × 2 strands) applied directly to `h`, so
  the `K`-wide `e_l` is never materialised.
- **L2b** folds because the span is a **difference** of two values from one
  per-`(m,o)` tensor, and `W·(span/L) = (W·span)/L` folds too.

Unfolded, the per-pair concat would be ~320 wide; folded, the pair grid holds only
`H = 32`. §Cost's 5.4 GB measurement was taken on a 320-wide *strandless* unfolded
implementation, so treat it as the order of magnitude the fold avoids rather than
as this design's number.

**Materialise `f_l`/`f_r` per chunk, not for all groups at once.** A given `L` uses
only its own group `g(L)`, so the full `(B, 10, 2, 32, 2304)` tensor in the diagram
below (378 MB per tap) is the naive reading: an `L_chunk` of 32 spans 2–3 groups,
so computing `M[g,s] @ h` for just those groups cuts this several-fold.

**CONDITION: the endpoint path must stay LINEAR until it is summed into the
pre-activation.** A nonlinearity on the per-pair features destroys the fold-down.

### Shapes across the layers

```
B    = 64     regions per step          N1  = 128   L1 cut-site bank
P    = 2048   scored centres m (crop)   K   = 128   L2a bank dim (x4 banks)
n_L  = 155    in-band L = 25..179       N2  = 32    L2b interior bank
W    = 256    window bases (257 cuts)   H   = 32    hidden width
O    = 219    offset axis (max o_r + 1) L_chunk = 32
l_in = P + 256 + 5 = 2309 (pre-conv)    model_input_size = P + W = 2304 (crop out)
```

```
                  ┌────────────────────────────────────────────┐
 one-hot seq      │  (B, 4, 2309)                    2.4 MB    │
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
 a[t,s] : (128,128) x4     65,536 params        F : (32, 4, 256)   32,768 params
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
 │ (B,10,2,32,2304) 378 MB  │                 │ (B, 32, 2048, 219)  3.67 GB  │
 │ x2 taps = 756 MB         │                 │ chunk over m -> 459 MB       │
 │ (naive; do 2-3 groups    │                 │                              │
 │  per L_chunk instead)    │                 │                              │
 └──────────────────────────┘                 └──────────────────────────────┘
        │  gather at o_l, o_r                         │  difference at o_r, o_l
        └───────────────┬─────────────────────────────┘
                        v
              pre[h, m, L, s]  =  f_l[s] + f_r[s]           <-- LINEAR until summed
                                + (V_span[o_r] - V_span[o_l])
                                + (V_mean[o_r] - V_mean[o_l]) / L
                                + len_emb[h, L]              <-- free per-L term
                        │
                        v   <-- THE ONLY PER-PAIR TENSOR
              ┌──────────────────────────────────────────────┐
              │ hid = act(pre)  (B,32,2048,L_chunk=32,2)     │
              │        1074 MB fp32  /  537 MB bf16          │
              └──────────────────────────────────────────────┘
                        │  w_out : (32,)
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

**Where the cost is:** the offset-axis prefix tensor `(B, 32, 2048, 219)`. The
prefix sum is cumulative over `o`, so the axis cannot be tiled — a chunk would lose
the running sum — and in-band `o_l`/`o_r` span `o ∈ [39, 218]`, essentially all of
it. So **chunk over `m`, not over `o`**.

> These sizes are arithmetic, not measured, and the §Cost benchmark measured a
> **strandless** unfolded implementation with a 320-wide per-pair concat at
> 5.4 GB. The strand axis and the learned-per-offset interior branch both change
> the intermediate; re-measure before budgeting on these numbers.

### Forward pass

```
h = L1(seq)                              # (B, N1, P) cut-site propensity, post-nonlinearity

# L2a endpoint (per strand s): shared tap weight + per-group gain, folded
o_l, o_r = 128 - L//2, 128 - L//2 + L    # window offsets of the two CUT SITES
p_l, p_r = m - L//2, m - L//2 + L        # genomic cut sites p and p+L
g        = min((L - 25)//15, 9)          # length group
f_l[s] = M_l[g,s] @ h ; f_r[s] = M_r[g,s] @ h   # per strand s; M_x[g,s] = W_x[s] diag(B_x[s,g]) a_x[s]

# L2b interior: weights learned per offset
u[n,m,o] = sum_c F[n,c,o] * x[c, m-128+o]   # (B, N2, P, W)
U        = prefix_sum(u, axis=o)            # (B, N2, P, W+1), exclusive
span     = U[:,:,o_r] - U[:,:,o_l]
mean     = span / L                         # both fed explicitly

pre[h,m,L,s] = f_l[s] + f_r[s] + W_span·span + W_mean·mean
             + len_emb[h, L]
hid          = act(pre)                                      # (B, H, P, L_chunk, 2)
logit[m,L,s] = w_out · hid                                   # (B, P, n_L, 2)
disp_bp      = dispersion_head(h)   # (B, 2, P) per-position, seq-indexed, FROZEN
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

**`fl_bands` must be recorded and guarded too.** This model is band-free in its
*parameterisation*, which makes it tempting to skip — but scoring is **in-band**,
so `|D|`, the uniform anchor and the oracle anchor all depend on the band
definition. A band change would silently move every percentage for this model
exactly as it would for the track models. Record `fl_bands` in the store's config
and assert it against the constant at construction, reusing
`config.check_fl_bands`.

## Dataset / dataloader

This model is **sequence-only**: the `(m, L, strand)` logit map depends only on
the tile's sequence, so **one forward serves all samples of a region**. The
dataset therefore batches **by region**, not by `(sample, tile)`.

Item = one region, yielding:

```
x         : (4, l_in)              float32   one-hot sequence (shared)
frag_p    : list over samples of (n_frag_s,) uint16 region-local left endpoints
frag_L    : list over samples of (n_frag_s,) uint16 lengths
frag_s    : list over samples of (n_frag_s,) uint8  strand
endpoints : (S, 2, tile_size)      float32   the two NB target tracks per sample
mask      : (tile_size,)           bool      valid positions
```

The multinomial shares one per-region `logp` map across the `S` samples:
`NLL_s = -Σ_{(p,L,s)∈frags_s} logp[p,L,s] / N_s`, averaged over samples.

**Crop.** The jittered centre crop is the frozen `jitter_matrix`, applied
identically to `seq`, `mask`, the endpoint tracks, and the fragment coordinates
(a fragment survives iff both `p` and `p+L-1` land inside the crop; its `p` is
re-based to the crop origin). Train draws `j ∈ [-jitter, jitter]`; **val and
test use `j = 0` (centred crop) — jitter is training-only**. `jitter_matrix`'s
crop requires its input and output lengths to share **matched parity** (not even
parity per se — see `calculate_start_and_stop_from_jitter`): the jittered crop
runs from the stored tile to the **even** `model_input_size = P + W = 2304`, and
the width-6 convolution's 5 bp margin is added **outside** that crop, so the
crop's input and output parities match. The odd `l_in = 2309` (§Shapes) is
therefore the pre-convolution fetch width — `model_input_size + 5`, the one-hot
actually handed to L1 — **not** a crop target; it does not enter the parity
assertion, which covers only the even `tile_size` / `l_target` / store `l_seq` /
`model_input_size`.

**Shared-jitter coupling.** Because one sequence forward serves all `S` samples,
all samples of a region in a step share one jitter offset — the crop defines the
position frame the shared `logp` map lives in. This is accepted: it is the price
of the shared-forward speedup (the whole point of region-batching). Diversity is
recovered across steps/epochs (fresh `j` per region each draw) and across
regions within a batch. Document this in the dataset docstring so it is not
"fixed" later.

**RC augmentation is DROPPED: `rc_prob = 0`.** There is no RC-equivariance
requirement on this model and no RC-equivariance test.

**`min_N` = 0 for these stores.** The old `N.min(axis=2) ≥ min_N` filter does
not transfer (its axis is now the 2 endpoint tracks, and training needs
fragments, not per-track depth). Filter on **total fragment count per
`(sample, region)`** with default **0**: keep every region a sample has any
fragments in; regions with zero fragments for a sample contribute a zero
multinomial term and are skipped.

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
`(m, L, strand)` output. It implements its own `training_step`/`validation_step`
and its own `predict` entry point.

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

## Cost — measured

A10G (g5.xlarge), torch 2.5.1, `bf16` via `torch.autocast`, `B=64`, `P=2048`,
`L` 25–256, **442,772 `(p,L)` pairs/region**, 54 fragments/example. Median of 20
timed steps after 5 warmup, `cuda.synchronize()` around timings, peak VRAM reset
per cell. `scripts/bench_fragment_logit_sweep.py`.

fwd+bwd ms / peak VRAM, **naive** batching (map recomputed per sample):

| hidden width \ L-chunk | 32 | 64 | all (232) |
|---|---|---|---|
| linear | 515.7 ms / 4.9 GB | 511.1 ms / 9.7 GB | **OOM** |
| 32     | 597.6 ms / 5.4 GB | 593.0 ms / 10.6 GB | **OOM** |
| 64     | 644.5 ms / 5.9 GB | 639.3 ms / 11.7 GB | **OOM** |

- **The exact `(p,L)` sweep is affordable; no sampled softmax / NCE is needed.**
- **`L_chunk = 32` is MANDATORY** — unchunked OOMs in all 12 configs. 32 beats
  64: same wall time, half the VRAM (5.4 vs 10.6 GB at H=32).
- **`H = 32` costs +16%** over a linear head (597.6 vs 515.7 ms) — take it. The
  linear head cannot express any interaction between cut-site identity and
  fragment length, and that interaction is strong and sign-flipping in the truth
  (high/low-GC bias ratio 0.65 at L=24 vs 4.08 at L=75).
- **`bf16` is for time, not memory.** Faster than fp32 at `H > 0` but peak VRAM
  is slightly *higher* (autocast keeps fp32 masters beside bf16 copies). Do not
  cite it as a memory saving.

Two caveats that must be stated:

1. The benchmark's 8–10× amortisation came from sharing one region forward
   across 16 samples, and **training is now single-sample**, so budget the naive
   **~9.3 ms/example** (597.6 ms ÷ 64 at H=32, `L_chunk=32`), **~1.8×** the
   330–360 ms/batch baseline per example.
2. The learned-per-offset interior weights need an offset-axis prefix tensor the
   benchmark did not build, and the strand axis roughly doubles the endpoint
   branch — neither was in the measured grid, so **re-measure** rather than
   assuming it fits.

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

- `min_N` = 0.
- **Separate val AND test** region sets; val/test use the centred crop, jitter
  is training-only.
- **Two stores**, one per model class, deliberately not unified — streamlining
  happens after a model is chosen.
- Simulator `max_len` = 256, with filtering to the bands before inference.

## Limitations (accepted, not solved here)

- The current sampler draws lengths from a **global** PMF, so a
  region-dependent length distribution is untestable on current sim data. Open
  TODO, not solved here.
- `bias_correction/` is v1 and superseded; this design does not build on it.
- Wiring the new model's `(m, L, strand)` output into `correction.py` (which
  assumes a per-position SHAPE head with the 12-track gather) is out of scope
  until a model is chosen.
