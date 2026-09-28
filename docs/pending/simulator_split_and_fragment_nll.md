# Simulator design and per-fragment NLL

Status: DESIGN, not implemented. Specifies the simulator's generative form and
the per-fragment NLL metric that scores both model classes. The band widening
this doc once described is **done and committed** — see "Bands" below.

**This simulator is written from scratch.** Existing code may be reused where it
is genuinely useful — `load_duphist` / `build_cell_map`
(`scripts/ztnb_from_duphist.py`), the `hexamer_indices` + `cum_gc` precompute,
`empirical_length_pmf`, region/BED loading, the `sample_<i>.npz` output layout
(`region_idx, start, stop, strand`), the multiprocessing scaffolding, and
`sim_build_store.py` — but this is not a refactor of `sim_fragments.py`.

**The existing simulator stays live and must keep working.** KEN and Hybrid
retain their RC-tied k-mer embeddings so they can be validated against the
current RC-symmetric simulator, which requires `sim_fragments.py` and its stores
to remain intact. This work is **additive**, not a replacement.

## Why

1. **The oracle must not be a second implementation of the generative model.**
   An oracle that re-derives the propensity independently has to agree with the
   sampler exactly or every "% bias captured" figure is wrong — and it has
   already disagreed once, by a 128-position crop offset. Two implementations of
   one rule drift silently. Writing from scratch means designing the single
   shared implementation in from the start rather than retrofitting it.
2. **Losses in different families are not comparable.** Multinomial and
   `nb_offset` score on different scales, so the architecture ranking can
   reverse between them with no way to tell why.
3. **A comparable, real-data-computable metric is needed** — the per-fragment
   NLL below.

## Goals

- **One implementation of the fragment weight**, consumed by both the sampler
  and the scorer, so they cannot disagree.
- A metric comparable across parameterisations and losses, computable on real
  data.

## Scope note

Fragments longer than `max_len` = 256 are **excluded from the normalisation
domain `D`** rather than treated as an error — applied identically to oracle and
model, so it does not affect comparisons. Because a store built under different
`max_len` or different bands is scored over a different domain, **state the
domain alongside any number.**

## Sizing facts this design rests on

| fact | value |
|---|---|
| valid `(p,L)` pairs per 2560 region, `max_len` 256 | 622,720 |
| materialised `all_w_arr`, 10k / 50k regions | 46 GB / 129 GB |
| factorised artifact per region | **31 KB** |
| FL mass lost to `max_len` 256 | 0.05% |

The factorised artifact reconstructs the weight exactly (its only 2-D term is an
integer-indexed GC LUT), is ~150× smaller than the materialised array, and is
what enables the shared-builder design below.

**PMF mass vs realised-fragment share are different quantities.** The realised
length distribution is not the length PMF, because the sampler reweights `(p,L)`
by the cut-site and GC terms, and GC bias varies strongly with length
(high/low-GC ratio 0.65 at L=24 vs 4.08 at L=75). A PMF-mass figure and a
realised-share figure over the same bands will differ by several points and can
look exactly like an arithmetic error. **Any in-band figure must say which of
the two it is.** Neither is load-bearing for the metric: the per-fragment NLL
denominator is a measured PMF sum applied identically to oracle and model, so
nothing computed depends on in-band mass — the only risk is a wrong constant
being hardcoded.

## Stage 1 — propensity builder

Input: region set BED, reference, `w6` params, GC/FL capture model, FL PMF.

Output per region — all exact, not approximations of the materialised array:

| array | dtype | size |
|---|---|---|
| `fwd_cut` | int32 `(region_len+1,)` | 10 KB |
| `rc_cut` | int32 `(region_len+1,)` | 10 KB |
| `cum_gc` | int32 `(region_len+1,)` | 10 KB |

Plus, once per run: `w6`, the integer-indexed GC lookup, `len_p`, `max_len`, and
the region table.

**The single shared function** `build_region_weights(region_arrays, globals,
L_range)` returns `(p, q, w)`. Stage 2 calls it to sample; the scorer calls it
for the truth term. That shared call is where the correctness benefit lives —
not in the artifact format.

## Stage 2 — generative form

The truth has **four hexamer tables**: `{start, end} × {forward, reverse}`,
**untied** (this is short-read single-stranded data, so the two ends are not
related by reverse complement). A fragment is drawn sequentially:

```
1. s ~ Bernoulli(1/2)
2. start p ~ start_s[hex(p)] / sum_p' start_s[hex(p')]
3. end   q ~ end_s[hex(q)] * FLGC(L, gc(p,q)) / sum_q' (same), over q' giving L in range
```

**The end normaliser is per-start, `Z_s(p)` — this is a SEQUENTIAL scheme, not a
symmetric joint over `(p, L)`.** On the minus strand the 5' end is at the
**higher** coordinate, so for `s = -` the `start` table applies at `q` and the
`end` table at `p`, with hexamers read reverse-complemented.

The length/GC factor:

```
FLGC(L, gc)        = observed_len_p(L) * capture(L, gc) / capture_marginal(L)
capture(L, gc_pct) = 1 / GCFlDistModel.predict(L, gc_pct)
```

`GCFlDistModel.predict` is **inverse** capture: `predict(25) = 2.753` means a
25 bp fragment is seen only ~36% of the time, so using `predict` directly as a
positive weight would invert the assay — hence the reciprocal. Its length
profile spans only **2.58×** while the observed length PMF spans **7.3×**, so
capture alone cannot carry length structure; the observed PMF carries it. The
trailing ratio `capture / capture_marginal` averages to 1 per length, adding GC
dependence without re-applying the length effect the PMF already holds.

Output layout: `sample_<i>.npz` with `region_idx`, `start`, `stop`, `strand`.
Overdispersion is a count-drawing option on top of the shared propensity, which
is what the "can excess variance be learned" question needs.

The split simulator has **no** regime B, `--regime`, per-sample `w6` jitter,
`log_jitter`, `acceptance_normaliser`, or propose-and-reject sampling. Anything
referencing those after the change is stale — flag it, do not maintain it.

## Bands (done and committed)

`FL_BANDS = ((25,110), (110,180))`. Track count stays 12, so no shape check
catches a mismatch: `dataset.py` verifies only `C == len(DEFAULT_OUTPUT_TRACKS)`
and never compares the store's recorded `fl_bands` to the constant, so every
existing store would be silently reinterpreted (including the real-data
production store `bg_store_b67d7c95`). The guard — in `dataset.py` **and**
`sim_build_store.py` — **asserts the store's recorded `fl_bands` equals
`FL_BANDS` at construction and fails loudly.**

## Per-fragment NLL

For fragment `(p, L)` in region `r`, under model `m`:

```
NLL       = -log( w_m[p, L] / sum_{(p',L') in D} w_m[p', L'] )
w_m[p, L] = first_m[p] * last_m[p+L-1] * gc_correction(L, GC%) * len_p[L]
```

where `first_m`, `last_m` are the model's per-base track predictions for the
fragment's band and strand.

**Index spaces — state the mapping, then test it.** The simulator weight uses
`w6[rc_cut[p+L]]` and the model weight uses `last_m[p+L-1]`; these are different
subscripts because they are different spaces. `fwd_cut`/`rc_cut` have length
`region_len+1` and index **between-base cut sites**; `first_m`/`last_m` are
**per-base tracks** of length `region_len`. Convention: a fragment occupying
bases `[p, p+L)` has left endpoint at base `p`, right endpoint at base `p+L-1`,
and its cut sites are `p` and `p+L`.

**Required test:** assert the endpoint bases the scorer indexes are the same
genomic positions as the cut sites the sampler drew from, over a small region
with known fragments. A silent 1 bp offset corrupts every NLL, and this project
has already paid for exactly this class of error once (a 128-position crop
misalignment), which is why a shift-correlation proof is owed on every geometry
change.

- `midpoint` is **not** a factor. All three coverage types are deterministic
  functions of the same fragment; multiplying them would triple-count one piece
  of evidence. It remains available as a consistency check.
- `len_p[L]` is not a constant offset — it varies with `L`, so omitting it
  changes the distribution rather than shifting it. Taken from the empirical
  PMF, identical across models.

### Reporting conventions

| convention | value |
|---|---|
| strand `log 2` term | **INCLUDED** |
| which fragments are scored | **IN-BAND ONLY** (`L` in at least one FL band) |

**Strand `log 2` is INCLUDED** because strand is modelled and observed.
Simulator strand is exactly 50/50, so oracle, uniform and model all gain the
same `log 2`, and it **cancels in `(uniform − model)/(uniform − oracle)`** — %
captured is invariant and only the absolute nats move.

**In-band only** because out-of-band fragments have no `first_m`/`last_m` track
prediction — the weight is undefined, not merely inconvenient. Consequence to
state with every number: the denominator changes when the bands change, so a
per-fragment NLL under one banding is **not** comparable to one under another,
even on identical data. Report the bands with the number, always.

**Normalisation domain `D`** = pairs *entirely* inside the centred evaluation
crop; only fragments in `D` are scored. A fragment with one endpoint in the crop
and its partner outside would put numerator and denominator on different sets —
the same class of error as the crop misalignment.

### The two anchors

`% bias captured = (uniform − model) / (uniform − oracle)`. The per-fragment NLL
is an absolute number in nats, so it needs both anchors in its own units.

| anchor | definition |
|---|---|
| **uniform** | `log|D|` — every pair in the normalisation domain equally likely |
| **oracle** | the same NLL evaluated with the TRUE simulator weights |

The oracle anchor is cheap because the true weights are exactly what the shared
`build_region_weights` returns — the same call the sampler makes, so the anchor
cannot drift from the generative model.

**Both anchors are recomputed per store and never carried across.** Under the
per-track metric, uniform's deficit below `log(tile_size)` tracks the empty-mass
fraction and has ranged over a 33× spread across stores; a carried baseline
silently misstates every percentage.

**GC source.** For simulation the oracle uses the **true simulator surface**
(available now, unblocks sim scoring). The real-data GC source is a deferred
owner decision (KEN's 34 bp receptive field cannot compute GC over a 175 bp
span, so asking models to predict `gc_correction` would rank receptive fields
more than anything else — the recommendation is to supply GC externally); it
blocks only real-data scoring.

**Cost.** `log Z` needs the sum over ~622,720 pairs per region — factorised, a
few vector ops per length. Affordable for validation, plausibly affordable as a
loss.

## Validation

- **No parity requirement of any kind.** Written from scratch, this owes the old
  sampler neither byte equality nor distributional equivalence. New seed lineage;
  spend no effort comparing against the old output.
- **Self-consistency instead:** the drawn fragments' empirical distribution must
  match the weights they were drawn from, on a small region. This is the property
  the shared-builder design exists to guarantee, and it is checkable without
  reference to any previous implementation.
- Realised-parameter recovery: the generative tables must be recoverable from the
  emitted fragments (realised hexamer-table recovery, GC slope by length).
- The shared builder must be exercised by **both** paths in one test, so a
  change to one consumer cannot silently diverge from the other.
- The oracle's own fragment NLL must come in strictly below `uniform` on sim
  data.
