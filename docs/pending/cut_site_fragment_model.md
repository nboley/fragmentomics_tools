# Cut-site fragment model — integration design

Status: implemented at HEAD 7ea790e on worktree-cut-site-model. Start admission is in force. No store or hex table has been rebuilt under it. No result under start admission has been measured. Centre-admission anchors are retired; see §Store compatibility.

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

**Start admission.** A fragment is admitted iff its START lies in `[0, P)`
in crop coordinates (`P = tile_size`). `m` is the fragment START position,
not the centre. This gives `|D_L| = P` for every in-band `L` (length-
independent — start in `[0, P)` gives `P` positions for every `L`).

Admission reads `start` directly. No `min(start, end)` remains in the store
builder or the dataset. The store is verified, not assumed, to hold fragments
left-to-right. `scripts/cut_site_model/build_store.py` raises `ValueError` on any
fetched fragment with `stop < start`, before dedup (commit 7ea790e). The check
covers every fragment `fetch_array` returns. That set is an overlap superset,
so the check also sees fragments whose start lies outside the region.

- **Admission results are unchanged by that check.** For every build that
  completes, the keep set equals the one the former `min()` filter produced.
  A build that contains a reversed fragment now fails instead of dropping it
  silently. [Verified: 0 of 30,754,614 fragments reversed, chr1/2/20/X, per the
  coordination log, 2026-10-06 section "EM FINDING".]
- **`np.abs` was rejected.** It would admit reversed fragments that are dropped
  today. That changes which fragments enter the model, so it needs owner
  sign-off.

`WINDOW_HALF = 181` (`cut_site_model/model.py:79`). The constructor enforces
`WINDOW_HALF >= L_max - 1 = 179` (`cut_site_model/model.py:288`). Derivation: the
deepest hex gather is at `WINDOW_HALF + P + L_max - 2`, the 3' cut at `m = P-1`
with `L = L_max`. The k-mer for that cut ends 2 bases past it. §Margin
arithmetic gives the tight bound, which is one base lower. 181 is the owner's
choice (Q1, 2026-10-06). Symmetric by design — the unused left margin is the
price of reviewability after the off-by-3 incident.

`rf_budget = WINDOW_HALF + 5 = 186` (`CutSiteStoreConfig.rf_budget` in
`cut_site_model/store.py`). It is pinned by
`tests/test_hex_registration.py::test_rf_budget_equals_window_half_plus_5`,
not imported. Importing `WINDOW_HALF` would pull torch and lightning into
`cut_site_model/store.py`, a numpy/zarr module that the AWS Batch builder depends on.

Window `W` = 2 × WINDOW_HALF = 362 bases.

Index conventions. Hex index `j` is the k-mer that starts at one-hot position
`j`. The tile begins at one-hot position `pad = WINDOW_HALF + 3`, and
`CUT_OFFSET_IN_KMER = 3`. A cut at tile base `c` therefore reads hex index
`WINDOW_HALF + c`. The taps in use:

| Tap | Tile base | Hex index | Used by |
|---|---|---|---|
| 5' endpoint | `m` | `WINDOW_HALF + m` | hex prior `start_pos` (`cut_site_model/model.py:193`); L2a left tap (`:650`) |
| 3' **cut base** (junction-centred) | `m + L` | `WINDOW_HALF + m + L` | hex prior `stop_pos`; L2a right tap; the table counter — all via the one definition `fragment_cut_bases` |
| fragment's last base | `m + L - 1` | — | store `last[p+L-1]`. A fact about the fragment, NOT a registration choice |

**RESOLVED 2026-10-10 (owner ruling, commit `9aa8ef5`): the two branches now
agree at `m + L`.** The hex prior previously read `m+L-1` while the L2a branch
read `m+L`, so the analytic prior and the learned correction keyed the same cut
to different 6-mers. `CUT_OFFSET_IN_KMER = 3` registers a cut at base `s` with
the k-mer `s-3 .. s+2`, junction mid-k-mer; a fragment occupying `m .. m+L-1`
has its 3' junction between `m+L-1` and `m+L`, so the junction-centred cut base
is `m+L`. **L2a and the simulator were already right; the prior and the table
counter moved to them.**

Do not confuse the two rows above. `m+L-1` is still the fragment's last base —
that never changed. What moved is which base *indexes the 3' cut*.

Both the reader and the counter now derive the positions from a single
`fragment_cut_bases` definition, so they cannot drift apart again, and the npz
carries a registration key the loader refuses to mismatch. Measured effect on
real data: `+0.002730` nats on test, which is ~1.1x the paired per-region SE —
**the justification was consistency, not NLL, and it should not be quoted as a
measured win.** See §Open discrepancies, item 1.

**Registration is enforced, not emergent.** The construction-time assertion
`(l_in - tile_size)//2 - WINDOW_HALF == CUT_OFFSET_IN_KMER` runs in
`CutSiteFragmentModel.__init__` (`cut_site_model/model.py:299-306`). The reason is
not obvious. `raw = P + 2*WINDOW_HALF + 5` is odd only because `P` is even.
Rounding up to even gives `pad = WINDOW_HALF + 3`. An ODD `tile_size` gives an
even `raw`, so `pad` falls to `WINDOW_HALF + 2`. The k-mer then slides to
offset 2, and registration breaks with no error. [Verified: the comment at
`cut_site_model/model.py:295-298`. The arithmetic follows from `:437-438`.] The
assertion turns that case into a construction error. The test is
`tests/test_hex_registration.py:391`.

**Hot-path width check.** `compute_log_p0` checks `hex_idx.shape[1] >=
WINDOW_HALF + tile_size + L_max - 1` (`cut_site_model/model.py:199-207`). The check
uses Python ints, so it adds no device-to-host sync per forward. It replaces the
tensor `assert` on `start_pos.min()` and `stop_pos.max()`, which forced a sync.
It runs on every call, not at construction. [Verified: 7ea790e diff.]

### Margin arithmetic

Only even `P` is valid, which the constructor enforces. For even `P`,
`l_in = P + 2*WINDOW_HALF + 6`, because the `+5` conv margin is rounded up to
even (`cut_site_model/model.py:437-438`). So `pad = WINDOW_HALF + 3` and the right
margin past the tile is `WINDOW_HALF + 3`. [Inferred from the formula above.]

- The deepest 3' cut is at tile base `P + L_max - 2`. Its k-mer ends 2 bases
  later, at tile base `P + L_max`.
- That base must lie inside the input: `L_max <= WINDOW_HALF + 2`.

So the tight bound at even `P` is `WINDOW_HALF >= L_max - 2 = 178`. The
constructor enforces `L_max - 1 = 179`, one base stricter. The code is not
changed for this. [Inferred. Not probed. A probe at `WINDOW_HALF = 178` would
confirm it.]

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
The length component is worth **0.173 nats on real data and 0.272 on the full
simulation** (measured 2026-10-10 as `uniform - (uniform cut-site + marginal
p_L)`, test split), so forcing it through a piecewise-constant 15 bp
approximation would be a large avoidable loss.

An earlier version of this line claimed ~0.4255 nats, "about 5× the whole
sequence-bias gap of ~0.084". Both figures are superseded and the ordering is
inverted: measured on the same test splits, the sequence component is **larger**
than the length component (0.359 vs 0.173 on real, 0.331 vs 0.272 on the full
sim), not one fifth of it.

**Placement: on the logit, after the nonlinearity** — not into the
pre-activation. `logit[m,L,s] = w_out · act(pre) + len_bias[L]`. This is
`(n_L,)` = **156 parameters** (`n_L = L_max - L_min + 1`, `cut_site_model/model.py:369`), strand-shared (the FL/GC length term is
strand-independent).

**Length groups: 10, half-open, 15 wide from 25 with the LAST 20 wide:**

```
[25,40) [40,55) [55,70) [70,85) [85,100) [100,115) [115,130) [130,145) [145,160) [160,180]
```

Index `g = min((L-25)//15, 9)`; the clamp folds `[175,180]` into the last group,
so it is closed at the inclusive `L_max = 180`.
Limitation: 15 bp resolution cannot isolate a sharp effect at ~167.

### L2b — interior branch

A **second, separate** bank `F (N2=32, 4, n_offsets=180)` with weights **learned
per offset**, spanning the live range `[o_l_min, o_r_max)` (see §Geometry).
Re-indexed as `o' = o - o_l_min`. Prefix sum along the re-indexed offset axis:

```
u[n,m,o'] = sum_c F[n,c,o'] * x[c, m + o_l_min + o']   # x = one-hot sequence (cut_site_model/model.py:494)
U          = prefix_sum(u, axis=o')       # exclusive, n_offsets+1 = 181 slots
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
 one-hot seq      │  (B, 4, 1392)                    1.4 MB    │
                  └────────────────────────────────────────────┘
                          │  L1: Conv1d(4 -> 128, k=6) + nonlinearity
                          v
 h  cut-site       ┌───────────────────────────────────────────┐
    propensity     │  (B, 128, 1386)                  45 MB    │  PER-POSITION
                   └───────────────────────────────────────────┘
                     │                              │
        ┌────────────┘                              └──────────────┐
        v  L2a                                                     v  L2b
 ── ENDPOINT BRANCH ─────────────────────        ── INTERIOR BRANCH ──────────────
 a[t,s] : (128,128) x4     65,536 params        F : (32, 4, 180)   23,040 params
 B[t,s] : (128,10)  x4      5,120 params        W_span, W_mean : (32,32) each
 W[t,s] : (32,128)  x4     16,384 params
                                                 u never materialised whole;
 FOLD: M_l[g,s] = W_l[s] diag(B_l[s,g]) a_l[s]   prefix-sum over o then difference
       -> 20 matrices of (32,128)  (10 groups x 2 strands)
                                                 v[h,m,o] = sum_c F[h,c,o]·x[..]
 f_l[s] = M_l[g,s] @ h                           U = cumsum(v, axis=o)
        │                                               │
        v                                               v
 ┌──────────────────────────┐                 ┌──────────────────────────────┐
 │ (B,10,2,32,1024) 168 MB  │                 │ (B, 32, 1024, 181)  1.52 GB  │
 │ x2 taps = 336 MB         │                 │ chunk over m                 │
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
              │ hid = act(pre)  (B,32,1024,L_chunk=32,2)     │
              │        537 MB fp32  /  268 MB bf16          │
              └──────────────────────────────────────────────┘
                        │  logit = w_out · hid + len_bias[L]
                        v
              ┌──────────────────────────────────────────────┐
              │ logit  (B, 1024, 156, 2)          strand axis │
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

**Where the cost is:** the offset-axis prefix tensor `(B, 32, 1024, 181)`. The
prefix sum is cumulative over `o'`, so the axis cannot be tiled. All 180 offsets
are live; none are trimmed. **Chunk over `m`, not over `o`**. The diagram sizes
are shape arithmetic at `B = 64`, fp32, `P = 1024`. They are not measured.

### Forward pass

```
h = L1(seq)                              # (B, N1, model_input_size), post-nonlinearity

# L2a endpoint (per strand s): shared tap weight + per-group gain, folded
o_l, o_r = WINDOW_HALF, WINDOW_HALF + L  # L2a right tap is m+L; see §Index conventions
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
pin, CSR triples, tile `split` codes, bulk single-write build) but carries a
different payload: per sample × region the **fragment list** (the object the
multinomial trains on) and the **two per-position endpoint count tracks** (the
ancillary NB targets), both derived from the identical fragment list at build
time.

```
/attrs: config_json, config_hash, created_utc, split_version, ...
        (the length range is in config_json as L_min, L_max; there is no max_len attr)
/tiles/{contig, start, stop, strand, region_id, split, seq, mask}
/samples/{library, total_fragments}
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
/totals/N  (S, T, 2)    uint32    # per-endpoint-track totals over the tile
```

- **`start`/`length`, not `start`/`stop`.** The multinomial indexes `(p, L)`
  directly. `L ≤ L_max (180)` and `p ≤ l_target (≤ 2560)` both fit `uint16`.
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

Do **not** add the length range to the shared `PlumbingConfig`: hashing a new field
would rehash *every* store, including the real-data production store. The new
store records its geometry in a separate small config (`tile_size`, `jitter`,
`rf_budget`, `L_min`, `L_max`, `seed`, `n_train_samples`), hashed with the same
`hashlib.sha256(canonical_json)` discipline, giving it its own drift guard while
`config.py` stays frozen for the old class. `split_version` carries over.

**Domain guard.** Because `L_min` and `L_max` define `D`, the Dataset MUST
check them against the store at construction and fail loudly. It does so in
`validate_length_range` (`cut_site_model/store.py`), which raises `ValueError` on
any mismatch. The Dataset also asserts `l_seq >= model_input_size + 2 * jitter`
(`CutSiteFragmentDataset.__init__`, `cut_site_model/dataset.py`). A silent
length-range mismatch changes `|D|` and reinterprets every reported NLL.
[Verified: `validate_length_range`, and the call to it and the `l_seq` check
in `CutSiteFragmentDataset.__init__`.]

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
frag_N    : scalar                 int64     fragments ADMITTED to this crop (== n_frag)
endpoints : (2, tile_size)         float32   the two NB target tracks
mask      : (tile_size,)           bool      valid positions
```

`frag_N` is not the stored count. `min_N` gates on the STORED count, so an
indexed item can still emit `frag_N < min_N`, including 0 (see `min_N` below).

The multinomial shares one per-region `logp` map across the `S` samples:
`NLL_s = -Σ_{(m,L,s)∈frags_s} logp[m,L,s] / N_s`, averaged over samples.

**Crop.** The jittered crop uses `jitter_matrix` with concentric widths
(seq → `l_in`, targets → `tile_size`). A fragment survives iff its START
lies in `[0, tile_size)` in crop coordinates, and its length lies in
`[L_min, L_max]` (both inclusive). The rule has one implementation,
`admit_fragments` in `cut_site_model/dataset.py`, with the crop offset from
`crop_origin`; `stage1_hex_baseline.compute_hex_tables` calls both. `frag_m` is that start. The Dataset
does not recompute an end. The store guarantees start ≤ end.  Train draws `j ∈ [-jitter, jitter]`; **val and
test use `j = 0` (centred crop) — jitter is training-only**. "Centred" names the
crop position, not centre admission. `jitter_matrix`'s
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

**`min_N` = 10.** Filter on the **stored fragment count per `(sample,
region)`** (`indptr[u+1] - indptr[u]`, over all of `l_target`, before any
crop): regions with fewer than 10 stored fragments for a sample are skipped.
It does not bound `frag_N`, the count admitted to the crop.

**Jitter stream.** Train mode draws one fresh `j` per `__getitem__` from a
NumPy generator keyed on `(seed, worker)`: `worker` is the DataLoader worker
id + 1, or 0 in the main process. The PID is not in the key, so `--seed`
reproduces a run for the same item order and the same `num_workers`. The
stream is sequential, not a function of the item index. `train.py` uses
persistent workers whenever `--workers > 0`, so each worker's stream
continues across epochs and every epoch draws fresh jitters.

**Worker safety** is inherited verbatim: lazy per-PID zarr handle, optional
in-memory preload with copy-on-write fork sharing, no CUDA in the dataset.

## The model module — where it lives

**New module `background_model/cut_site_model/model.py`.** It imports primitives from
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

**`L = 180` in `scripts/eval_fragment_nll.py` (owner decision 185).** The
evaluator admits `L_MIN..L_MAX = 25..180` from `background_model.constants`
(inclusive), but `FL_BANDS` is half-open, so `L = 180` lies in no band. In that
script ONLY, an `L = 180` fragment is scored with the TOP band's tracks
`(110, 180)`, and `log Z` includes that cell. This is the single deliberate
exception to half-open bands (`_band_index`); `FL_BANDS` stays half-open
everywhere else. A fragment that still falls in no `(strand, band)` group raises.

### Why the model cannot reach the oracle exactly

`% captured` is **capped below 100 by construction**, and the cap is the
**receptive field**, not the joint form of the model.

The sequential truth carries a per-start normaliser `Z_s(p)` (NLL doc). Because
`−log Z_s(p)` is a function of `p` **alone**, a joint log-linear model with a free
per-position term absorbs it **exactly** — the joint-vs-sequential distinction is
*not* the obstruction. The obstruction is spatial reach: the model's per-position
term is `M_l[g,s] @ h[:,p]`, and L1 is a single width-6 convolution, so `f_l(p)`
sees only **6 bp**. But `Z_s(p) = Σ_q end_s[hex(q)] · FLGC` sums over ~156 lengths
and so depends on **~256 bp** of surrounding context. The model cannot see far
enough to represent it.

`Z_s(p)` is an average over ~156 positions, so it is **smooth**; but its variation
across positions could still be of the same order as the whole available bias gap
(~0.084 nats), which would make the cap material. **Measure `var(log Z_s(p))` over
the region set early**: it bounds the unreachable fraction, and without it a
sub-100% result is uninterpretable. Widening L1's receptive field (or adding an
explicit context branch) is the lever that shrinks the gap.

## Cost

**Not yet measured on the implemented model.** The prior benchmark
(`scripts/cut_site_model/bench_fragment_logit_sweep.py`) was run on a strandless, unfolded
prototype that differs from this architecture on both the strand axis and the
interior branch. `L_chunk = 32` is mandatory (unchunked OOMs). Re-measure
before budgeting GPU time.

## Required tests

- **Registration (hex prior).** Implemented in `tests/test_hex_registration.py`.
  The 5' and 3' lookups read the k-mer that starts `CUT_OFFSET_IN_KMER` before
  the cut (`:46`, `:83`). Builder and model agree (`:116`). The right-edge lookup
  covers every cut (`:210`). The minus-strand gathers are probed by value
  (`:302`, `:344`). An odd `tile_size` raises (`:391`).
- **3' endpoint convention.** Both sides are pinned at `m+L`, and the tests now
  pin them **to each other** rather than each to its own position. The L2a
  forward is pinned by `tests/test_cut_site_training.py::TestLogitInputSupport`
  through the model's own gradients: `test_support_is_start_to_right_tap`
  checks that one logit's input support ends at the right tap's window, and
  `test_each_tap_bank_on_its_own` checks each tap's 6-base window on its own.
  The hex prior is pinned by `tests/test_hex_registration.py`:
  - **T12a** measures the prior's receptive field by swapping individual bases
    and comparing the measured field against L2a's taps and the
    junction-centred k-mer derived from `CUT_OFFSET_IN_KMER`.
  - **T12b** ties the prior's reads to the simulator's `cut_site_hexamers`,
    joining by k-mer STRING so the comparison is independent of either stack's
    index convention.
  - **T4b** asserts the table counter counts at the same index
    `compute_log_p0` reads, over all 4096 cells x 4 tables.
  - **T13** refuses an npz or checkpoint carrying another registration.

  **Why this wording changed:** the previous pin,
  `test_stop_endpoint_uses_the_same_registration`, was a TAUTOLOGY. It was
  named for the right invariant and documented it — "the 3' cut must be
  registered identically to the 5' cut" — then asserted `m+L-1`, comparing
  `compute_log_p0`'s gather against a hand-written restatement of the same
  formula. It passed for any formula, which is why a suite that already
  guarded registration never caught the mismatch. Non-tautology is now proved
  by MUTATION, reproduced independently by implementer and reviewer: reverting
  both producer and reader to `m+L-1` fails 12 tests; reverting the READER
  only — the keyed-one-way-read-another state — fails 6, including T4b, T12a
  and T12b.
- **Fragment start.** `frag_m` equals the stored start minus the crop origin:
  `tests/test_cut_site_store_dataset.py::test_frag_m_is_stored_start_both_parities`.
- **Store consistency.** `tests/test_build_cut_site_store.py::TestBuilderIntegration::test_endpoint_consistency`.
- **Domain guard.** `tests/test_cut_site_store_dataset.py::TestDomainGuards::test_length_range_mismatch_raises`.
  It checks `L_min`/`L_max`, not `max_len`.
- **Not implemented in this worktree.** The shared `build_region_weights` test
  and the oracle-below-uniform test. They belong to the sampler and NLL doc. RC
  equivariance is dropped, as the design says.
- **No RC-equivariance test** — it is void now.

Follow CLAUDE.md testing discipline: run the `background_model` suite via
`make test PYTEST_ARGS="tests/ -q"` before and after, and measure the baseline
yourself (it moves per commit).

## Settled parameters

- `min_N` = 10 (stored fragment count per sample×region, before the crop).
- **Start-based fragment admission**: D = {(m, L, s) : m ∈ [0, P), L in-band,
  s ∈ {0,1}}. |D| = P × n_L × 2 = 319,488. |D_L| = P for all L — no length bias.
- `WINDOW_HALF` = 181, `rf_budget` = 186 (WINDOW_HALF + 5).
- **`tile_size` is required** (owner decision 176): `CutSiteStoreConfig` and
  `CutSiteFragmentModel` have no default. `train_cut_site.py` and
  `gpu_benchmark.py` read it from the store's `config_json`, as the dataset
  does; their `--tile-size` is required and must match the store. The
  hash of `CutSiteStoreConfig(tile_size=2048)` is unchanged, `e71256c72cbacd4e`.
- **F offset range**: `n_offsets = L_max` (180),
  `o_l_min = WINDOW_HALF`. Under start admission the left tap is fixed.
- **NB endpoint loss kept, dispersion frozen** at `log_dispersion_init`.
- **Conv L1** (not embedding). Embedding arm is a Stage 3 improvement.
- **len_bias (n_L,)**: scalar per-L on the logit (not per-H).
- **S = 1 currently.** Dataset indexes by `(sample, tile)`.
- **Separate val AND test** region sets; val/test use the centred crop.
- **Two stores**, one per model class.
- Simulator `max_len` = 256, with filtering to the bands before inference.

## Store compatibility (measured)

Measured at 7ea790e. The `model_input_size`, `l_seq`, and slack figures come
from the coordination log, 2026-10-06 PRE-FLIGHT. The hashes below were
recomputed in this session. [Verified: `CutSiteStoreConfig(tile_size=1024,
jitter=256, rf_budget=…)` with other fields at defaults. The results match the
log exactly.]

| Quantity | Value |
|---|---|
| `model_input_size` (= `l_in`) | 1392 |
| Required `l_seq` (`>= model_input_size + 2*jitter`) | 1904 |
| OLD store, `rf_budget = 133` | `l_seq = 1802`. REJECTED by the `l_seq` check in `CutSiteFragmentDataset.__init__` |
| NEW store, `rf_budget = 186` | `l_seq = 1908`. Slack 4 |
| Minimum viable `rf_budget` | 184 |
| `config_hash`, old → new | `e0699b0902fafafa` → `d5e89191901ea991` |

- [Verified] The new store goes to a NEW path. `quiet_tile1536_v2.zarr` must not
  be overwritten. It is the only artifact still tied to the retired
  centre-admission anchors.
- [Verified] The old store fails at dataset construction. It cannot load silently.
- [Inferred] The slack is 4 bases. With `rf_budget = 186`, a dataset `jitter`
  above 258 breaks the guard. No check at the store boundary covers that
  relation.
- [Inferred] Changing `L_max` rejects any existing store (`validate_length_range`
  in `cut_site_model/store.py`). The constructor also fails for `L_max >= 183` at
  `WINDOW_HALF = 181` (`cut_site_model/model.py:288`).

## Open discrepancies

Originally recorded against `7ea790e`. **Items 1 and 3 are now closed** — see
each. Item 2 remains open. Each remaining fix changes computed results, so it
needs an owner decision.

1. ~~**3' endpoint, one base.**~~ **RESOLVED 2026-10-10, owner ruling, commit
   `9aa8ef5`.** The hex prior and the table counter moved to `m+L`, matching
   the L2a forward and the simulator, both of which were already correct.
   `CUT_OFFSET_IN_KMER = 3` means the junction-centred 3' cut base of a
   fragment occupying `m .. m+L-1` is `m+L`. Reader and counter now share one
   `fragment_cut_bases` definition, the npz carries a registration key the
   loader refuses to mismatch, and four tests pin it (§Required tests). The
   store's `last[p+L-1]` is unchanged and was never wrong — that is the
   fragment's last base, a different quantity from the cut base.
   Measured on real data: `+0.002730` nats on test, ~1.1x the paired
   per-region SE, so **consistency was the justification, not NLL.**
2. [Verified] **Interior window.** L2b reads one-hot positions
   `WINDOW_HALF + m + o'` for `o'` in `[0, L)` (`cut_site_model/model.py:494`). With
   `pad = WINDOW_HALF + 3`, these are tile bases `m-3` to `m+L-4`. The window
   sits 3 bases left of the fragment body `m … m+L-1`. The filter is learned,
   so this is not a hex registration error. The cut-site code and this doc do
   not state whether the shift is intended.
3. ~~**Tight margin.**~~ **SUPERSEDED 2026-10-10, and the old guard turned out
   to be BROKEN, not merely loose.** This item said the tight bound was
   `WINDOW_HALF >= 178` and the constructor enforced 179. The binding limit is
   actually L2a's `h`, which spans `WINDOW_HALF` each side of `P`, so the
   farthest right tap is `WINDOW_HALF + (P-1) + L_max` and the requirement is
   **`WINDOW_HALF >= L_max`** — not `hex_idx`, which carries one extra
   rounding base. The old guard `WINDOW_HALF < L_max - 1` would have admitted
   `L_max = 182`, whose farthest tap sits at the boundary of `h`'s width, so it
   was a guard that did not guard. The constructor now enforces
   `WINDOW_HALF >= L_max` and is tested: `CutSiteFragmentModel(L_max=WINDOW_HALF+1)`
   must raise. Verified independently by the Phase-0b reviewer, which is how
   the off-by-one was found — the implementer and the EM had both derived it
   as "drop the `-1`".

## Limitations (accepted, not solved here)

- The current sampler draws lengths from a **global** PMF, so a
  region-dependent length distribution is untestable on current sim data. Open
  TODO, not solved here.
- `bias_correction/` is v1 and superseded; this design does not build on it.
- Wiring the new model's `(m, L, strand)` output into `correction.py` (which
  assumes a per-position SHAPE head with the 12-track gather) is out of scope
  until a model is chosen.
