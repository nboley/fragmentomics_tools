# Plan — layered simulator

Single sample throughout. `FL_BANDS = ((25,110),(110,180))`.

## 1. Simulation layers

### Layer 1 — hexamer bias + FL/GC bias
Fixed FL distribution across regions. Fragment counts constant per region.

**Four hexamer tables**, `{start, end} × {forward, reverse}` (owner, 2026-09-27).
No RC tying between them: most of this data is single-stranded, so the two ends
are not related by reverse complement. Strand is drawn **first**, and it selects
which pair of tables applies.

Sampling is sequential — **the end normaliser is per-start**, `Z_s(p)`, so this is
NOT a symmetric joint over `(p, L)`:

```
1.  s        ~  Bernoulli(1/2)
2.  start p  ~  start_s[hex(p)] / Σ_p' start_s[hex(p')]
3.  end  q   ~  end_s[hex(q)] · FLGC(L, gc(p,q))  /  Σ_q' (same)
                over all q' giving a length in range,  L = |q − p|
```

On the minus strand the 5′ end is at the **higher** coordinate, so for `s = −`
the `start` table applies at `q` and the `end` table at `p`, hexamers read
reverse-complemented.

**The `FLGC` term — measured, do not simplify it to `capture` alone.** Length
structure comes from the observed PMF; capture supplies GC dependence *conditional
on* length:

```
FLGC(L, gc) = observed_len_p(L) · capture(L, gc) / capture_marginal(L)
```

The trailing ratio averages to 1 at each length, so it adds GC dependence without
re-applying the length effect the PMF already carries. Equivalently, keep three
factors with `latent_len_p = observed_len_p × predict` — the same thing as a
deconvolution.

Two measurements on `RD-56153` are why this form rather than `end_s · capture`:

| | |
|---|---|
| `predict` length-profile dynamic range | **2.58×** |
| observed length-PMF dynamic range (25–180) | **7.3×** |

`end_s` is indexed by hexamer and cannot manufacture length structure on average,
so capture alone yields a length distribution ~3× flatter than reality. The two
shapes do agree — `predict`'s trough at L=80–100 matches the PMF's trough at
L=80–120 — which is the confirmation that `observed ≈ latent / predict`.

**The length distribution is short-dominated, and that is correct.** This is
short-read **single-stranded** data (owner-confirmed 2026-09-27), which recovers
short and degraded fragments that dsDNA prep misses. Measured over three samples,
full range:

| | RD-56153 | RD-56154 | RD-56156 |
|---|---|---|---|
| **mode** | **L=52** | **L=51** | **L=52** |
| mass 25–100 | 47.1% | 67.2% | 58.9% |
| mass 141–180 | 30.0% | 17.3% | 20.8% |
| mass 181–1000 | 2.4% | 2.6% | 2.2% |

The main FL peak is at ~50 with a **secondary** mononucleosome bump at 141–180.

> **`scripts/analysis/ztnb_from_duphist.py`'s docstring is WRONG on this point.** It cites
> "the mononucleosome mode at 166bp" as evidence the length axis is fragment
> length. The axis *is* fragment length, but 166 is the secondary bump, not the
> mode. Do not use that line to validate anything.

Three consequences, none of them defects:

- **It supports the new bands on data grounds.** The old band 1 `(40,65)` was
  narrow and sat on the flank of the real mode; `[25,110)` captures the whole short
  peak, and `[110,180)` captures the mono shoulder.
- **`[25,110)` will carry the majority of fragments** and `[110,180)` a minority,
  so any per-band figure will look lopsided by design.
- **Nucleosome phasing is carried by 17–30% of fragments**, which calibrates what
  to expect from the CTCF flagship rather than indicating a problem.
- The 181–256 range used for count variance is ~2.4% of mass, consistent with the
  ~3% CV estimated for that mechanism.

`capture` comes from the **biomarker model class** `flgc.model.GCFlDistModel`,
via its own `predict()`. **The capping is part of the model**, so it is used as
the model defines it — not worked around:

```
capture(L, gc_pct) = 1 / model.predict(L, gc_pct)
gc_pct             = 100 · (cum_gc[p+L] − cum_gc[p]) / L        # percent, per predict()'s contract
```

`predict()` returns the correction weight `min(1/P(seen), max_weight)`, so its
reciprocal is the capture probability the simulator injects, floored at
`1/max_weight` (= 1/3 at the default 3.0). Taking the weight itself as the
generative factor would enrich poorly-captured fragments, i.e. the wrong
direction; the reciprocal is the capture side of the same quantity.

**The surface is fitted by the `flgc` ZTNB code. Precomputed artifacts are not
used.** The pipeline, all four steps through the class:

```
load_duphist(sid)                      # duphist TSV, length 1..1000
build_cell_map(df)                     # (length, gc) -> (k_vals, obs, kmax, seen_unique)
GCFlDistModel().fit(cell_map, length_bins=..., gc_bins=..., min_cell_size=...)
model.save(<sim>/gcfl_model.json)      # cached with the simulation
```

`load_duphist` and `build_cell_map` already exist in
`scripts/analysis/ztnb_from_duphist.py` and are reused rather than reimplemented; only the
bins differ.

| | |
|---|---|
| import | `from flgc.model import GCFlDistModel`, requires `PYTHONPATH=/home/nathanboley/src/biomarker` — a **runtime dependency**, including in the AWS Batch container |
| fit | `fit(cell_map, length_bins, gc_bins, *, min_p=1e-6, max_weight=3.0, single_end=False)` plus `min_cell_size`, `k_fit` |
| predict | `predict(length, gc)`, vectorised over arrays |
| persistence | `save` / `load` — JSON, round-trips the surface, bins, method and `max_weight` |

**GC is percent (0–100) throughout the `flgc` path** — `build_cell_map` keys on
the duphist's percent `gc` column, `gc_bins` are percent bins, and `predict()`
takes percent. There is no GC-count interface here.

Because the model assigns **unfitted and out-of-range cells the maximum weight**,
capture is defined everywhere — no NaN rule and no out-of-range rule are needed.
Unfitted cells enter as `1/max_weight`.

**Caching.** Fit once per sample, `save()` the model JSON into the simulation's own
artifact directory, and materialise a `predict()` lookup table over the whole
`(L, gc_pct)` grid for the sampling loop. Both are caches of our own fit, so the
simulation stays self-contained and reproducible. Per-call numpy overhead in the
previous GC implementation was 77% of simulator runtime, which the table avoids.

### Layer 2 — region-varying FL distributions + sample-specific counts
Two additions:

- **Region-varying FL.** Global FL distribution as prior; Bayesian posterior update
  per region. Replaces the single `len_p` with `len_p_r`.
- **Sample-specific count distributions.** Per-region counts stop being constant and
  are drawn from that sample's own count distribution, so depth heterogeneity
  (mappability, GC, copy number) enters. `sim_fragments.py` already has the
  machinery: `--real-count-dir` joins counts to their own regions per sample,
  `--real-count-tsv` / `--real-store` pool them.

### Layer 3 — blacklist / repeat regions
Reintroduce excluded regions. Requires model-side masking changes.

## 2. Datasets

| input | source | detail |
|---|---|---|
| Reference | `/efs/analytics/nathanboley/data_resources/genome/hg38.fa` | hg38 |
| Region set (2048 tiles) | `quiet_v2_pad1200_repeats_removed_tile2560` | 11,505 tiles; `region_len` 2560, `jitter` 256, `tile_size` 2048 |
| Region set (1024 tiles) | `quiet_v2_pad1200_repeats_removed_tile1536` | 66,649 tiles; `region_len` 1536, `jitter` 256, `tile_size` 1024 |
| Hexamer bias `s`, `e` | `build_w6(seed, dynamic_range)` in `sim_fragments.py` | synthetic, seeded; p95/p5 ≈ 4.0. Fitting from real data is deferred — §7 |
| FL distribution `len_p` | `empirical_length_pmf(h5)` from one real library | 500 lengths |
| FL/GC bias | `flgc.model.GCFlDistModel` in `/home/nathanboley/src/biomarker` | **fitted by us** per sample; cached as `gcfl_model.json` + a `predict` lookup table |
| ZTNB fit input | `/efs/analytics/nathanboley/ctcf_fa_cache/duphist_merged/<sid>__duphist_wg.tsv.gz` | `length, gc(percent), multiplicity, molecule_keys`; unbinned, so any bin resolution. 97 samples |
| Repeats (L3) | `/efs/analytics/nathanboley/data_resources/genome/hg38_rmsk.txt.gz` | UCSC rmsk |
| Blacklist (L3) | `/efs/analytics/nathanboley/data_resources/genome/hg38-blacklist.v2.bed.gz` | ENCODE hg38 v2 |
| Store | built by `scripts/sim_build_store.py` | zarr, 12 tracks, `min_N` 0 |

Our own FL/GC fit is cached into each simulation's artifact directory as
`gcfl_model.json` plus the derived `predict` lookup table, so the simulation is
self-contained.

## 3. Testing plan

**The test is the NLL over the simulated fragments.** All other losses are
ancillary.

Per layer, on held-out regions of the same sample:

| quantity | definition |
|---|---|
| model | −log P(fragment) under the model, summed over observed fragments in `D`, per fragment |
| oracle | the same NLL under that layer's own fragment selection probability |
| uniform | `log |D|` |
| result | `% captured = (uniform − model) / (uniform − oracle)` |

- The oracle is redefined at each layer to match that layer's simulator exactly,
  and both anchors are recomputed per store.
- `D` = `(p, L)` pairs entirely inside the centred evaluation crop.
- Models: **KEN, then CNN**, run at every layer.
- `val` / `test` = held-out regions, same single sample, centred crop; jitter is
  training-only.

## 4. Settled parameters

| item | value |
|---|---|
| Per-region fragment count | **54** per 2560 bp, **37** per 1536 bp — constant per region |
| Geometries | **both, in parallel** |
| Samples | **random selection, starting with one** |
| `GCFlDistModel.fit` options | **the defaults** — `MIN_P=1e-6`, `MAX_WEIGHT=3.0`, `K_FIT=25`, `MIN_CELL_SIZE=200` |
| Fit binning | **FL in 1 bp bins** (every length, 25–180 = 156 bins) × **GC in 5% bins**. Verified on `RD-56153`: 2,057 of 3,120 cells fitted, and only **0.0333% of molecule mass** falls in starved cells, so the `1/max_weight` artifact is negligible. GC bins should span **0–100** — out-of-range GC also receives `max_weight` |
| `max_len` | **256 for simulation**, then **filtered to the bands before inference** |
| Hexamer bias | **`build_w6`, synthetic** — fitting from real data deferred to §7 |
| v4 stores | **not rebuilt** — abandoned |
| Simulator rework | **folded in**, pending discussion |
| Cut-site fragment model | **built in parallel** with the layers |
| Scoring | **in-band only** |
| `len_p` for old models | **not supplied** |
| Strand for old models | **scored natively** |
| Endpoint NB loss | **kept, with the dispersion head frozen** |
| Splits | **separate val and test** region sets |

Unfitted cells and out-of-range lengths need no rule: the model assigns both the
maximum weight, and that convention is part of the model.

## 5. Still open

| # | item |
|---|---|
| 1 | Layer 2 posterior form and prior strength |

### Why simulate to 256 and then filter
Fragments outside the bands are sampled, consume counts, and are then dropped before
the store is built — so the **retained** count per region varies even though the
sampled count is constant. Measured mass outside 25–180 on `RD-56153`: 2.40% at
181–256 plus 2.10% below 25, ~4.5% total.

Magnitude to expect: at 54 sampled fragments with ~95.5% retention, the retained
count is `Binomial(54, 0.955)` — mean ≈ 51.6, **sd ≈ 1.5, CV ≈ 3%**. Real per-region
depth is far more dispersed than that (the 1536 sim measured mean 37.4 against median
33, right-skewed). So this mechanism supplies *some* count variance, not depth
heterogeneity; that is what Layer 2's sample-specific counts add. Retention is
sequence-dependent, so the variance is mildly correlated with region features.

## 6. Consequences to expect, not bugs

- **Old models have no length term at all.** With `len_p` not supplied and
  `first_b`/`last_b` identical for every `L` in a band, an old model implicitly
  predicts fragment lengths as uniform within each band. Its fragment NLL will be
  dominated by length mis-specification rather than by sequence bias. The
  comparison is still structurally fair — the new architecture can represent
  length and the old cannot — but the headline gap will not isolate sequence-bias
  recovery. A supplied-`len_p` variant is the diagnostic that separates them.
- **The learned start propensity will not equal the hexamer table.**
  Sequential sampling makes `P(p,L) ∝ s(p)·len_p(L)·e(p+L) / Z(p)`; the `−log Z(p)`
  term is a function of `p` alone and gets absorbed into any per-position start
  term. Judge recovery on the fragment likelihood, not on correlating learned
  weights against the table.

## 7. Deferred to a future phase

Not in scope for the layers above. Listed so they are resurfaced rather than lost.

| item | note |
|---|---|
| **Hexamer bias fitted from real data**, saved per sample | Replaces synthetic `build_w6`. Needs cut-site counts by hexamer from a real sample, normalised against genomic hexamer frequency, and a decision on whether the two ends share one table read on opposite strands or need two. Makes the oracle an estimate rather than a known table, so "recovery" would then mean recovering our own fit. **Open when it happens: which fragment population the bias should describe** — a read filter changes the answer, since low-mapq reads concentrate in repeats (skewed hexamer composition) and PCR duplicates weight a site by amplification efficiency rather than cut propensity. For reference, the real-data preprocess path uses `min_mapq=10` (`config.py:81` -> `preprocess.py:184`) and optional dedup (`preprocess.py:188`); nothing in the current simulator plan goes through that path |
| Between-sample variation | Parked by the single-sample decision, which removes the axis entirely rather than deferring it |

Per-region depth heterogeneity was here; it has moved **into Layer 2** as
sample-specific count distributions.
