# Simulator and per-fragment NLL

Status: DESIGN, not implemented. Specifies the simulator's generative model, the
data that informs it, and the per-fragment NLL that scores both model classes.

Written **from scratch**, in `background_model/simulator/`. Existing code may be
reused where useful — `load_duphist` / `build_cell_map`
(`scripts/ztnb_from_duphist.py`), the `hexamer_indices` + `cum_gc` precompute,
`empirical_length_pmf`, region/BED loading, the `sample_<i>.npz` layout, the
multiprocessing scaffolding, `sim_build_store.py` — but this is not a refactor of
`sim_fragments.py`, and **there is no parity requirement of any kind**.

**The existing simulator stays live and must keep working.** KEN and Hybrid retain
RC-tied k-mer embeddings so they can be validated against it, which requires
`sim_fragments.py` and its stores intact. This work is additive.

---

## 1. The generative model

### The shared weight

One function, `build_region_weights`, returns the fragment probability. **Both the
sampler and the oracle call it** — that is the only thing preventing them from
drifting apart, and a drift here is silent.

```
w(c5, c3, s) = 1/2 · start_s[hex(c5)]/S_s · end_s[hex(c3)] · FLGC(L, gc) / Z_s(c5)

S_s     = sum_c5' start_s[hex(c5')]                     per-strand start mass
Z_s(c5) = sum_c3' end_s[hex(c3')] · FLGC(L', gc')       per-5'-site end mass
```

**`w` is a fully normalised probability, not an unnormalised weight.** Each
normaliser has a job:

- `1/Z_s(c5)` makes the end step a proper conditional. Omit it and globally
  normalising `start·end·FLGC` gives the **symmetric** joint — a different model.
- `1/S_s` makes the start step a proper conditional **per strand**. Omit it and the
  total becomes `S_+ + S_-`, so the strand marginal comes out `∝ S_s` rather than ½.
  Harmless only if `S_+ = S_-`, which **untied tables do not guarantee**.
- the `1/2` is the strand prior the sampler draws from.

With all three, `sum_Ω w = 1` exactly over the generative domain `Ω` (the sum
telescopes: `sum_c3` of the end term is `Z_s(c5)/Z_s(c5) = 1`, leaving
`sum_c5 start_s/S_s = 1`), and the strand marginal is exactly ½.

### Sampling

```
1. s  ~ Bernoulli(1/2)
2. c5 ~ start_s[hex(c5)] / S_s
3. c3 ~ end_s[hex(c3)] · FLGC(L, gc) / Z_s(c5)     over c3 giving L in range
```

Sequential, with a **per-start normaliser** — not a symmetric joint over `(c5,c3)`.

**`c5` is the 5′ cut site, `c3` the 3′.** On the plus strand `c5 = p`, `c3 = p+L`;
**on the minus strand they swap**, because the 5′ end sits at the *higher*
coordinate — `c5 = p+L`, `c3 = p`, hexamers read reverse-complemented. The formula
is written in `(c5, c3)` rather than `(p, q)` deliberately: an implementer coding
from a `p`/`q` form gets the minus strand backwards, which corrupts strand
asymmetry silently.

### Four hexamer tables

`{start, end} × {forward, reverse}`, **untied** — this is short-read
single-stranded data, so the two ends are not related by reverse complement.
Strand is drawn first and selects which pair applies.

### The FL/GC term

```
FLGC(L, gc)        = observed_len_p(L) · capture(L, gc) / capture_marginal(L)
capture(L, gc_pct) = 1 / GCFlDistModel.predict(L, gc_pct)
gc_pct             = 100 · (cum_gc[p + L] − cum_gc[p]) / L
```

**GC uses the GENOMIC span `[p, p+L)`, not `c5`/`c3`.** GC content is a property of
the span and is strand-independent; `c5`/`c3` are 5′/3′ *labels*, and on the minus
strand `c5 > c3`, so `cum_gc[c3] − cum_gc[c5]` would be negative. The hexamer terms
use `c5`/`c3` because they genuinely are strand-dependent; GC does not.

Two measured facts fix this form; neither is optional.

**Direction: `predict` is *inverse* capture.** `predict(25) = 2.753` on RD-56153
means a 25 bp fragment is seen only ~36% of the time, so using `predict` as a
positive weight would **invert the assay** — hence the reciprocal.

**Magnitude: capture cannot carry length structure.** Its length profile spans
**2.58×** while that sample's observed length PMF spans **7.3×**, and `end_s` is
hexamer-indexed so it cannot manufacture length structure on average. The PMF
carries length; capture adds GC dependence *conditional on* length.

`capture_marginal(L)` is **the average capture over the length-`L` fragments the
hexamer terms would favour, pooled across both strands.** One value per `L`.

Concretely: enumerate every candidate fragment of length `L` in the region set — for
each strand `s`, every `(c5, c3)` placement with that length, using the strand's own
5′/3′ convention from "Sampling" above. For each such candidate define

```
v = start_s[hex(c5)] · end_s[hex(c3)]     its hexamer weight, with NO FLGC factor
g = 100 · (cum_gc[p+L] − cum_gc[p]) / L    ITS OWN GC, over the genomic span
```

then

```
                       sum over all candidates of  v · capture(L, g)
capture_marginal(L) = ───────────────────────────────────────────────
                       sum over all candidates of  v
```

`g` varies candidate to candidate — that is the whole point. If `capture` were
constant across the sum the ratio would be identically 1 and the term would do
nothing.

Computed once per `(region set, L)` and cached beside `gcfl_model.json`.

**It does not cancel out of `w`.** `Z_s(c5)` sums `end_s · FLGC` over `c3`, hence
over *different* lengths `L'`, so `capture_marginal(L')` varies inside that sum and
cannot be simplified away. Anyone "tidying" it will change the distribution.

**It is a one-pass approximation, deliberately.** The measure the sampler actually
realises also includes `FLGC` itself and `1/Z_s(c5)`, and `FLGC` contains
`capture/capture_marginal` — so the exact quantity is a fixed point needing
iteration. Weighting by `start·end` alone is one pass and non-circular; the
residual is second-order (the covariance between `capture` and `FLGC`'s own GC tilt
at fixed `L`, plus a per-strand term from pooling). Recorded so nobody "fixes" it
by iterating without knowing what that buys. One iteration would close most of the
remaining gap if it ever matters.

### Per-region counts

**Constant in Layer 1**: 54 fragments per 2560 bp region, 37 per 1536 bp.

Matching each region's depth to a real sample's realised count — which introduces
per-region depth variation correlated with region features — is a **named future
stage**, not Layer 1. The machinery exists (`--real-count-dir`), and is held out so
recovery is first tested against a clean constant-count generative model.

`max_len` = 256 for simulation, then filter to the bands before inference. The
out-of-band fragments consume counts, so the **retained** per-region count varies
even though the sampled count is constant. The store-build filter lands at
`config.max_frag_len = max(hi for fl_bands) = 180`.

### Bands

`FL_BANDS = ((25,110), (110,180))`, and the track count stays 12 — so a store built
under other bands is caught not by any shape check but by `config.check_fl_bands`,
asserted at construction in **both** `dataset.py` and `sim_build_store.py`, failing
loudly and naming both tuples.

---

## 2. Data that informs it

| what | where | supplies |
|---|---|---|
| region sets | `quiet_v2_pad1200_repeats_removed_tile2560` (11,505 tiles), `..._tile1536` (66,649) | the regions; `region_len = tile_size + 2·jitter` |
| reference | `/efs/analytics/nathanboley/data_resources/genome/hg38.fa` | sequence for hexamers and `cum_gc` |
| hexamer tables | `build_w6(seed, dynamic_range)` — synthetic, **4096 independent** log-normal draws per table | the four `{start,end}×{fwd,rev}` tables |
| length PMF | `empirical_length_pmf(h5)` | `observed_len_p(L)` |
| capture surface | `duphist_merged/<sid>__duphist_wg.tsv.gz` → `load_duphist` → `build_cell_map` → `GCFlDistModel().fit(...)` → `save()` | `capture(L, gc)` |
| the class | `flgc.model.GCFlDistModel`, needs `PYTHONPATH=/home/nathanboley/src/biomarker` | **runtime dependency, including in the Batch container** |

**Fit bins: FL in 1 bp bins, GC in 5% bins covering 0–100.** Do not take the `flgc`
defaults — their top length bin is `(101,200)`, which would collapse the whole high
band `[110,180)` into one bin and leave `capture` constant in `L` across it, and
their GC range stops short of 0–100 so extreme-GC fragments would fall out of bin
onto `max_weight` (→ `capture = 1/3`). Fit covers lengths 25–180 **inclusive** (156
bins, one wider than the 155-length scoring band) × 20 GC bins = 3,120 cells, of
which 2,057 fit at `MIN_CELL_SIZE = 200`; the starved remainder holds **0.0333% of
molecule mass**, so the unfitted floor is immaterial. Other `fit` options take the
defaults (`MIN_P=1e-6`, `MAX_WEIGHT=3.0`, `K_FIT=25`).

**Bin boundaries must be contiguous.** `_bin_index` tests `lo ≤ v ≤ hi` and
`gc_pct` is **continuous**, so integer bins like `(0,4),(5,9),…` would drop every
non-integer value — 4.5 matches no bin and silently lands on `max_weight`, the
exact failure the 0–100 range was chosen to avoid. Use `[0,5), [5,10), …, [95,100]`
with the last inclusive. Length bins `(25,25)…(180,180)` are unambiguous.

**GC is percent (0–100) throughout the `flgc` path** — `build_cell_map` keys on the
duphist's percent column, `gc_bins` are percent, `predict()` takes percent.

Output layout: `sample_<i>.npz` with `region_idx`, `start`, `stop`, `strand`.

---

## 3. The metric

For fragment `(p, L, s)` under model `m`:

```
NLL = -log( w_m[p,L,s] / sum_{(p',L',s') in D} w_m[p',L',s'] )
```

`w_m` is **model-class specific**:

- **Track models** (12-track / KEN / Hybrid):
  `w_m = first_m[p] · last_m[p+L-1] · gc_correction(L, GC%)`, with `first_m`/`last_m`
  from the fragment's band and strand. **No `len_p` is supplied.**
- **Cut-site model**: its own `log_softmax` logits over `D`. It needs no external
  `len_p` or GC factor — the logit carries both, and applying the track-model
  product would double-count length.

`midpoint` is **not** a factor for either. All three coverage types are
deterministic functions of the same fragment, so multiplying them triple-counts one
piece of evidence. It stays available as a consistency check.

### `Ω` and `D` are different sets

`Ω` is everything the simulator can emit, and `sum_Ω w = 1`. **`D` is the scoring
domain: `{(p, L, strand)}` with the fragment entirely inside the centred evaluation
crop AND `L` in band** — a strict subset. Strand is a dimension, so
`|D| = pairs_inband_in_crop × 2` and `log|D|` already contains the `log 2`.

`D` **is** the set that is scored; "in-band only" is this definition, not a filter
applied after it. Three ways to get it wrong, all silent:

- including out-of-band pairs makes `log|D|` count cells no model is ever charged for;
- admitting a fragment with one endpoint outside the crop puts numerator and
  denominator on different sets;
- hardcoding `|D|` — it is **computed**, and in particular is **not** the 622,720
  figure, which is the *sampling* count over all `L ∈ [1,256]` across the full
  2560 bp region.

### The anchors

`% bias captured = (uniform − model) / (uniform − oracle)`, all three in
per-fragment nats.

| anchor | definition |
|---|---|
| **uniform** | `log\|D\|` |
| **oracle** | `−log w(x) + log W_D`, the true weights **renormalised over `D`** |

`W_D = sum_{x in D} w(x) < 1`. Because scoring is conditional on being in `D`, the
true conditional is `w/W_D`. **`−log w` alone is NOT the oracle**: `log W_D < 0`, so
it overstates the oracle — makes it too weak — by exactly `|log W_D|`, and a model
that correctly learns the conditional then **prints above 100% captured**, which
reads as an implementation bug rather than a mis-specified anchor.

**`W_D` is PER REGION; `|D|` is not.** `w` is normalised within a region, so `W_D`
differs region to region and the oracle averages it over scored fragments. `|D|` is
purely geometric — identical for every region at a given geometry — so `log|D|` is
one number. Treating `W_D` as a single scalar reintroduces the same
mis-normalisation, averaged.

**Both anchors are recomputed per store and never carried across.** Under the
per-track metric, uniform's deficit below `log(tile_size)` tracked the empty-mass
fraction over a ~33× spread across stores; a carried baseline silently misstates
every percentage.

**Strand `log 2` is INCLUDED.** Simulator strand is exactly 50/50, so oracle,
uniform and model all gain the same `log 2` and it **cancels** in the ratio — %
captured is invariant, only absolute nats move.

**Report the bands with every number.** The denominator changes when the bands
change, so an NLL under one banding is not comparable to one under another even on
identical data.

**GC source for scoring.** Simulation uses the true simulator surface. The
real-data GC source is a deferred owner decision — KEN's 34 bp receptive field
cannot compute GC over a 175 bp span, so asking models to predict `gc_correction`
would rank receptive fields more than bias models; the recommendation is to supply
GC externally. Blocks real-data scoring only.

### Index spaces — the ±1 trap

The simulator weight uses `hex(c3)` at a **cut site**; a track model's weight uses
`last_m[p+L-1]`, a **per-base** track. These are different subscripts because they
are different spaces: `fwd_cut`/`rc_cut` have length `region_len+1` and index
between-base cut sites, while `first_m`/`last_m` are per-base of length
`region_len`.

Convention: a fragment occupying bases `[p, p+L)` has endpoints at bases `p` and
`p+L-1`, and cut sites at `p` and `p+L`.

**Required test:** assert the endpoint bases the scorer indexes are the same
genomic positions as the cut sites the sampler drew from, on a small region with
known fragments. A silent 1 bp offset corrupts every NLL, so a shift-correlation
proof is owed on **every** geometry change.

---

## 4. Validation

- **No parity with the old sampler** — neither byte equality nor distributional
  equivalence. New seed lineage; spend no effort comparing.
- **Self-consistency:** the drawn fragments' empirical distribution must match the
  weights they were drawn from, on a small region. This is the property the shared
  builder exists to guarantee, and it needs no reference implementation.
- **Realised-parameter recovery:** the generative tables must be recoverable from
  the emitted fragments — realised hexamer-table recovery, GC slope by length.
- **The shared builder exercised by both paths in one test**, so a change to one
  consumer cannot silently diverge from the other.
- **The oracle's own fragment NLL must come in strictly below uniform** on sim data.
- **`var(log Z_s(p))` measured over the region set.** A joint log-linear model
  absorbs `−log Z_s(p)` only through a per-position term, and a width-6 L1 sees 6 bp
  while `Z_s(p)` depends on ~256 bp — so `% captured` is capped below 100 before
  training starts. This variance bounds the unreachable fraction; without it a
  sub-100% result cannot be distinguished from a defect.
