# Simulator and per-fragment NLL

The simulator produces, per sample, a **fragment h5 plus a manifest sufficient
to reconstruct the sampling weights** (decision 79). It does not build the zarr
store, choose the scoring domain `D`, or compute the anchors — those are the
model agent's, along with `cut_site_fragment_model.md`.

Steps 1–5 are the generative model, Step 6 the manifest. Steps 7–8 are the
model agent's, recorded here only as the contract the output must support.

Implementation: `background_model/simulator/` (`weights.py`, `precompute.py`,
`sampler.py`, `emit.py`, `capture.py`).

---

## Domain and conventions

A fragment is `(c5, c3, s)`: strand `s`, 5′ cut site `c5`, 3′ cut site `c3`.
These are **cut sites, not bases** — for bases the fragment occupies `[p, p+L)`,
endpoints `p` and `p+L-1`, cut sites `p` and `p+L`.

- **Plus strand:** `c5 = p`, `c3 = p + L`.
- **Minus strand:** the ends **swap** — `c5 = p + L`, `c3 = p` — because the
  5′ end sits at the higher coordinate; hexamers read reverse-complemented.
- **Length** `L = |c3 - c5|`, restricted to **`L = 25..180`, 156 values** —
  the full support of the capture surface.
- **GC** over the genomic span (strand-independent):
  `gc_pct = 100 · (cum_gc[max(c5,c3)] - cum_gc[min(c5,c3)]) / L`.
  Using `cum_gc[c3] - cum_gc[c5]` goes negative on the minus strand.
- **Hexamer tables:** four, `{start, end} × {fwd, rev}`, **untied**
  (short-read single-stranded data — the two ends are not related by reverse
  complement). Plus strand uses `hex_fwd` for both `c5` and `c3`; minus strand
  uses `hex_rc` for both. Strand selects which `(start_s, end_s)` pair applies.

Derivations: Appendix D.

---

## Procedure

### Step 1 — Fit the capture surface

- **In:** duphist `duphist_merged/<sid>__duphist_wg.tsv.gz`, columns
  `(length, gc, multiplicity, molecule_keys)`.
- **Do:** `load_duphist` → `build_cell_map` →
  `GCFlDistModel().fit(cell_map, length_bins, gc_bins, min_cell_size=200)`.
  Length bins: 1 bp over 25–180 (156 bins). GC bins: 5% inclusive-integer bins
  `(0,4),(5,9),…,(95,100)` — contiguous on the integer grid, last inclusive.
  Other `fit` options default (`min_p=1e-6`, `max_weight=3.0`, `k_fit=25`).
  Build a `predict` LUT over `(L, gc_bin)` by evaluating `predict(L, gc_mid)` at
  each bin's midpoint (2.5, 7.5, ..., 97.5). The midpoint evaluation is exact
  only if `predict` is piecewise-constant on those bins — true for the ZTNB
  method (which bins GC internally via `_bin_index`), not for `spike_grid` (which
  interpolates). `predict_lut_from_model` asserts both conditions: model is ZTNB
  and `model.gc_bins == SIM_GC_BINS`. The LUT is built once and reused across
  regions and across both the sampler (Step 5) and the oracle (Step 7).
- **Out:** fitted `GCFlDistModel`, persisted via `save()` to
  `gcfl_model.json`. `predict(L, gc_pct)` returns
  `min(1/P(seen), max_weight)`, **GC in percent**. Dividing by `predict` is
  multiplying by capture.
- Runtime dependency: `flgc.model` needs
  `PYTHONPATH=/home/nathanboley/src/biomarker`, **including in the Batch
  container**.
- Bin semantics: Appendix F.

### Step 2 — Build `marginal_fl(L)`

- **In:** the same duphist (its de-duplicated `molecule_keys` column).
- **Do:** for each `L`, sum `molecule_keys` across all GC values; restrict to
  `L = 25..180`; normalise to sum 1. Nothing is deconvolved.
- **Out:** `marginal_fl(L)`, the empirical unweighted length marginal.

### Step 3 — Per-region precompute

- **In:** reference `hg38.fa`, one region set.
- **Do:** per region, compute hexamer indices for the forward and
  reverse-complement cut-site tracks (`hex_fwd`/`hex_rc`, length
  `region_len+1`), cumulative GC `cum_gc`, and validity mask `valid` (False
  where the hexamer window contains a non-ACGT base).
- **Out:** per-region `RegionPrecompute(hex_fwd, hex_rc, cum_gc, valid)`.
- Index conventions: Appendix D.

### Step 4 — Build `w` (`build_region_weights`)

- **In:** the four hexamer tables as `HexamerTables(start_fwd, end_fwd,
  start_rev, end_rev)`, `marginal_fl`, `predict_lut`, per-region
  `hex_fwd`/`hex_rc`/`cum_gc`/`valid`, `region_len`.
- **Do:** per region, per strand `s`, with `c3(L) = c5 + σ·L`
  (`σ = +1` plus, `−1` minus):

  ```
  S_s        = Σ_{c5 : Z_s(c5) > 0} start_s[hex(c5)]
  E_s(c5,L)  = end_s[hex(c3(L))] · marginal_fl(L) / predict(L, gc(L))     L = 25..180
  Z_s(c5)    = Σ_{L=25..180} E_s(c5,L)
  w(c5,c3,s) = ½ · start_s[hex(c5)]/S_s · E_s(c5,L)/Z_s(c5)
  ```

- **Hexamer track:** plus strand uses `hex = hex_fwd` for both `c5` and `c3`;
  minus strand uses `hex = hex_rc` for both.
- **`S_s` range:** sums only over `c5` with `Z_s(c5) > 0` — positions where at
  least one fragment is achievable. Including `Z_s(c5) = 0` positions breaks
  the Appendix A cancellation: the strand marginal falls below ½ by the
  dead-mass fraction.
- **Edge rule:** `Z_s(c5)` sums only those `L` whose `c3(L)` stays in
  `[0, region_len]`; off the region `hex(c3)` is undefined. This per-`c5`
  truncation is exactly what makes `|Ω| = 2 · Σ_L (region_len − L + 1)` correct.
- **`predict` is a cached `(L, gc_bin)` LUT**, gathered from, not called per
  element. Built once and reused across regions and across both the sampler
  (Step 5) and the oracle (Step 7).
- **Sequential form.** `1/Z_s(c5)` makes the end step a proper conditional,
  leaving the start marginal a clean `start_s/S_s`. `1/S_s` makes the start
  step a proper conditional per strand; without it the strand marginal is
  `∝ S_s` rather than ½, which untied tables do not guarantee. The `½` is the
  strand prior. (Globally normalising with one denominator — the symmetric
  joint — gives a different model where the start marginal is tilted by
  `Z_s(c5)`.)
- **Out:** `w`, a fully normalised probability over the generative domain `Ω`:
  `Σ_Ω w = 1` **exactly**, strand marginal exactly ½ (Appendix A).
- **One implementation.** Both the sampler (Step 5) and the oracle (Step 7)
  call `build_region_weights`. This shared call is the only thing preventing
  sampler/scorer drift — such a drift is silent. `Σ_Ω w = 1` and the exact-½
  strand marginal are cheap post-conditions that catch most breakages.

### Step 5 — Draw fragments

- **In:** the factors of `w`; the per-region target count (**54** at
  region_len 2560, **37** at 1536).
- **Do:** per region, per sample, repeat until the target count is reached:

  ```
  1. s  ~ Bernoulli(½)
  2. c5 ~ start_s[hex(c5)] / S_s
  3. c3 ~ E_s(c5,L) / Z_s(c5)   over L = 25..180,  c3 = c5 + σ·L
  ```

- **Out:** fragments `(start, stop, strand)` per sample, in BED coordinates.
- Related bound: Appendix E (`var(log Z_s)`).

### Step 6 — Emit fragments and the manifest

The simulator's deliverable is the **fragment h5 plus the manifest**. The model
agent owns store construction and the scoring domain.

**BED format:** one sorted, bgzipped, tabix-indexed BED per sample — 8 columns:

```
contig  start  stop  name  score  strand  mapq1  mapq2
```

with **all MAPQs = 60**. **GC is not emitted** — `build-fragments-h5` computes
it from the FASTA via `get_g_or_c_cumsum`, so store GC comes from the real
reference through production code. Then:

```
build-fragments-h5 sample_<i>.bed.gz sample_<i>.h5 --fasta hg38.fa
```

The simulator stops here. `background_model preprocess` and the zarr store are
the model agent's (decision 79).

- **One BED and one h5 per sample, not one overall.** The fragment h5 is a
  per-sample artifact; building it per-sample from the start is what makes
  `S > 1` work without restructuring.

- **MAPQ trap (silent-failure).** Unknown MAPQs store as `-1`;
  `config.min_mapq = 10`; `-1 >= 10` is False — so without real MAPQs **every
  fragment is filtered and the store comes out empty, with no error**. Each repo
  can pass its own suite and still fail to compose. Assert on a **non-empty**
  store.

- **Out:** the per-sample fragment **h5** and the **manifest**. Not the store.

### Step 7 — Compute the anchors

> **Owner: the model agent** (decision 79). Steps 7–8 are specified here
> because they define what the simulator's output must support.
>
> **Binding contract:** the oracle MUST obtain `w` by calling
> `build_region_weights`, reconstructed from the manifest's stored factors. It
> must not reimplement the weight. `Σ_Ω w = 1` and the exact-½ strand marginal
> are cheap post-conditions.

- **In:** the store, `w` via `build_region_weights`, the scoring domain `D`.
- **Do:**
  - `D = {(m, L, s) : m ∈ [0, P), L in band, s ∈ {0,1}}`, centre-based — the
    fragment centre `m = p + L//2` lands in the (centred, val/test) crop of
    width `P = tile_size` (2048 for region_len 2560, 1024 for 1536).
    `|D_L| = P` for every `L`, so `|D| = 155 · P · 2` (155 in-band lengths,
    2 strands).
  - `uniform = log|D|`.
  - `oracle = mean over scored fragments of ( -log w(x) + log W_D )`, where
    `W_D = Σ_{x ∈ D} w(x)`, computed **per region**.
- **Out:** `uniform`, `oracle` in per-fragment nats, **recomputed per store and
  never carried across**.
- Derivation: Appendix C.
- **Scope (decision 112).** Sharing `build_region_weights` between generator
  and scorer means a bug inside it is invisible to the oracle — and that is
  **intended**. The oracle measures NLL under the model that generated the
  data (the anchor for how much variance is capturable), nothing more.
  Simulator correctness is realised-parameter recovery's job (decision 88).
- **Accumulation precision (decisions 115/117).** The oracle averages ~2.47M
  log-probs of magnitude 13–18 nats. A naive float32 accumulator costs
  **~1.9 nats** — catastrophic on a ~13-nat quantity. All summary statistics
  and reported results use **float64 accumulation** (standing rule). Per-tile
  loss reductions inside the training step stay float32 by explicit scoping.
  **Lightning disclosure:** Lightning accumulates logged metrics in float32,
  contributing ~0.008 nats over ~5000 batches (**~1.05 pp of %-captured**).
  Published figures quoted to 2 dp do not disclose this.

### Step 8 — Score a model

> **Owner: the model agent** (decision 79b). Specified here only as the
> contract the simulator's output must support.

- **In:** the store, `D`, the model's weight `w_m`.
- **Do:** for fragment `(p, L, s)`:

  ```
  NLL = -log( w_m[p,L,s] / Σ_{(p',L',s') ∈ D} w_m[p',L',s'] )
  ```

  - **Track models** (12-track / KEN / Hybrid):
    `w_m = first_m[p] · last_m[p+L-1] · gc_correction(L, GC%)`, with
    `first_m`/`last_m` from the fragment's band and strand. **No `len_p`.**
  - **Cut-site model:** its own `log_softmax` logits over `D`; no external
    `len_p` or GC factor (the logit carries both; the track-model product
    would double-count length).
  - `midpoint` is **not** a factor for either — all three coverage types are
    deterministic functions of the same fragment, so including it triple-counts
    one piece of evidence. Kept only as a consistency check.
- **Out:** per-fragment NLL, and
  `% bias captured = (uniform - model) / (uniform - oracle)`, all in
  per-fragment nats.
- **Report the bands with every number** — the denominator changes with the
  bands, so an NLL under one banding is not comparable to one under another.
- Strand `log 2` is **included** everywhere and cancels in the ratio (simulator
  strand is exactly 50/50); only absolute nats move.

### Measured anchors — cut-site store val split

Measured from `scripts/cut_site_oracle.py` (commit `62bee76`) on the 1536-geometry
sim stores, val split. The script calls `build_region_weights` from
`background_model/simulator/weights.py` — the same function the sampler uses —
and scores each stored fragment's exact `w(c5, L, s)`, accumulating in float64.

Both stores were built from the same h5 (`RD-56670.fragments.h5`) and the same
manifest, so the hexamer tables, `marginal_fl`, and `predict_lut` are identical.
They differ in `L_max`: v1 (09-30) caps at 179, excluding L=180; v2 (10-01) caps
at 180, covering the full generative domain.

#### v1 — `sim_tile1536.zarr` (2026-09-30, `L_max`=179)

| Quantity | Value |
|---|---|
| Oracle NLL (val, 246,134 fragments) | **12.240380** nats |
| `−mean(log w)` | 12.242086 |
| `log W_D` | −0.001707 (pooled retention 2,461,808 / 2,466,013 = 0.998295) |
| Uniform NLL (per-region) | 13.005489 |
| `log\|D\|` geometric | 13.005492 (deficit 0.000004 — simulated data has essentially no N-masking) |
| **Gap (uniform − oracle)** | **0.765109** |
| `\|Ω\|` / `\|D\|` at region_len 1536 | 447,564 / 444,850 |
| Missing fragments | 4,205 of 2,466,013 (0.170%): 3,468 from L=180 exclusion + 737 dedup collisions |
| Check 1 — fragments on a `w == 0` cell | **0 of 246,134 — PASS** |
| Check 2 — per-region entropy 12.244610 vs sample mean 12.242092 | diff 0.002519, SE 0.002565, ratio 0.98 — **PASS** |

#### v2 — `sim_tile1536_v2.zarr` (2026-10-01, `L_max`=180)

| Quantity | Value |
|---|---|
| Oracle NLL (val, 246,496 fragments) | **12.244812** nats |
| `−mean(log w)` | 12.245111 |
| `log W_D` | −0.000299 (pooled retention 2,465,276 / 2,466,013 = 0.999701) |
| Uniform NLL (per-region) | 13.011571 |
| `log\|D\|` geometric | 13.011575 (deficit 0.000003) |
| **Gap (uniform − oracle)** | **0.766759** |
| `\|Ω\|` / `\|D\|` at region_len 1536 | 447,564 / 447,564 (`D = Ω` — full generative domain scored) |
| Missing fragments | 737 of 2,466,013 (0.030%): all dedup collisions (732 tiles×1 + 5 tiles×2) |
| Check 1 — fragments on a `w == 0` cell | **0 of 246,496 — PASS** |
| Check 2 — per-region entropy 12.244610 vs sample mean 12.245092 | diff 0.000482, SE 0.002568, ratio 0.19 — **PASS** |

**v2 notes.** The oracle rises by 0.004432 nats (v1→v2) because L=180
fragments sit at the low end of `marginal_fl` and carry lower `w`. The gap
widens from 0.765109 to 0.766759 (+0.001650), meaning the newly scored domain
adds slightly more uniform mass than oracle mass. Per-region entropy is
identical (12.244610) because it is computed over the full `Ω` regardless of
store `L_max`. The v2 store records provenance attrs (`source_h5`,
`source_regions`, `source_reference`, `l_target`, `l_seq`, `git_sha`) but
`git_sha` is the literal string `"unknown"` — it does not resolve to a commit.
This is the honest failure mode (claims nothing false), but means the store
cannot be tied to the code that built it.

**Open reconciliation: the 11.932 figure.** An earlier empirical MLE baseline
of 11.932 nats has circulated. Scored against this store's anchors it would read
`(13.005489 − 11.932) / 0.765109 = 140.3%` captured — above 100%, which is
exactly the symptom Appendix C predicts for a mis-specified anchor. The two
numbers are **not on the same footing**: the oracle (12.240380) is computed on
the val split from the simulator's own `build_region_weights` with both
correctness checks passing decisively; 11.932 is either in-sample, on a
different domain, or under a different normalisation. Flagged for reconciliation,
not as a reason to doubt the oracle. Do not use 11.932 as a scoring target until
its provenance is established.

**Precision disclosure.** Lightning accumulates logged metrics in float32 at
~0.008 nats, which is **1.05 pp of %-captured** at this gap. Published figures
quoted to 2 dp (e.g. 65.57%, 71.09%) do not disclose this. The oracle itself
uses float64 and is not affected.

---

## Manifest

`w` factorises. `build_region_weights` takes the per-region precompute (derived
entirely from the reference and region coordinates) plus three small
free-parameter arrays. The manifest stores those arrays:

| stored | size |
|---|---|
| 4 hexamer tables (4 × 4096) | 128 KB |
| predict LUT (156 × 20) | 24 KB |
| `marginal_fl` (156) | 1.2 KB |
| **total** | **~154 KB** |

Against **~73.5 GB** for a dense `w` (region_len 2560 × 11,505 regions) — which
is why `w` is reconstructed rather than stored.

**Store the realised tables, not the recipe.** Not "seed 42 + `build_w6`", not
"re-fit from sample X" — the actual arrays. Recomputation is then deterministic
and exact, with no dependence on a library version, a fit being reproducible,
or NumPy's RNG stream being stable across releases.

**Hexamer tables are DataFrames keyed by the hexamer string** (decision 81),
not bare 4096-element arrays ordered by an implicit integer code. A bare array
makes the k-mer ordering a contract between producer and consumer; if a table
is produced under a different convention, **every weight is wrong and
`Σ_Ω w = 1` still holds**, because normalisation cannot see a relabelling.
Keyed by the string, a mismatched table fails to join rather than silently
misaligning, and the integer code becomes a private detail on each side. The
stored string is written 5′→3′ **along the strand of the fragment that produced
it**, so the `*_rev` tables already hold reverse-complemented 6-mers — do not
RC them again on load.

The manifest also carries: **reference identity + hash** (a different hg38
patch silently changes every weight); **region-set identity + hash** and
`region_len`; the `L` range; `FL_BANDS`; per-region counts; the RNG seed
(provenance only — no longer load-bearing once the realised tables are stored);
and the `build_region_weights` **commit sha**, since the one residual
dependency is that the function still behaves the same.

`jitter` and `tile_size` are **deliberately absent**: `Ω`, and therefore `w`,
depends on `region_len` alone. `jitter` is derived
(`region_len = tile_size + 2·jitter`) and is a training-time crop budget;
`tile_size` parameterises `D`. Both belong to the model agent.

### Provenance verification

Verification is not opt-in, and the guarantee is **two-tier**:

- **Mandatory — `reference` and `region_set`.** If the manifest records the
  hash and the caller does not supply the path, `load_manifest` raises
  `ManifestVerificationIncomplete`. For these a skipped check is
  unrepresentable: the caller always has access to the files, so omitting one
  is a caller error. A hash that disagrees raises `ManifestMismatch`.
- **Best-effort but explicitly marked — `simulator_script`.** If `git` cannot
  resolve the recorded path to a blob sha, the check does not run and
  `"simulator_script:unresolvable"` is recorded in `verified`. That entry means
  **the check did not run** — it is not a pass. Tolerated because `git` is
  genuinely absent from the Batch containers, where hard-failing would make
  manifests unloadable (decision 86).

`verify=False` bypasses everything and is explicit, so skipping provenance is a
visible decision at the call site.

**Paths are stored repo-relative and resolved against the repo root**, never
against the working directory. Resolving against the CWD makes verification
succeed only when the process happens to be standing in the repo, which defeats
the reason for storing a relative path — and fails on the `cd /tmp` +
`PYTHONPATH` pattern used for Batch. An absolute path keeps its literal
meaning. Any test for this must `chdir` outside the repo, or it cannot observe
the failure.

---

## Output

**The deliverable, per sample:**

| artifact | contents |
|---|---|
| `sample_<i>.h5` | the fragments, via `build-fragments-h5` from an 8-column sorted, tabix-indexed `sample_<i>.bed.gz` |
| **manifest** | the 4 hexamer tables (DataFrames, string-keyed), the `(L, gc_bin)` predict LUT, `marginal_fl`; plus reference identity+hash, region-set identity+hash, `region_len`, `L` range, `FL_BANDS`, per-region counts, RNG seed, and the `build_region_weights` commit sha |
| **weight sidecar** | per-fragment `w` as **float32** (decision 114). One value per emitted fragment, self-keyed `(contig, start, stop, strand, w)`. Not a `fragments_h5` schema change — the field has no meaning for real samples. Size: ~9.9 MB at 37 × 66,649 fragments |

The intermediate `sample_<i>.bed.gz` is a means, not a deliverable. **No
store.**

**Storage dtype (decision 116).** Store `w`, not `log w`. float32's error is
**relative**, so a relative error on `w` becomes a flat **6.0e-8 absolute
error** in log space, independent of magnitude. Storing `log w` instead gives
`|log w| · eps ≈ 1.1e-6` at `|log w| = 18`, growing with magnitude — 13–24×
worse. No underflow risk: `w_min ≈ e^{-18.1} ≈ 1.4e-8` vs float32 min normal
`1.2e-38`. Normalisation is **per region** (uniform `log w = −13.01`; it would
be `−24.12` if global).

The hexamer tables in the manifest are Layer 1's synthetic `build_w6`. Real
counted tables from `scripts/count_cut_site_hexamers.py` are a separate stream;
whichever is used, the realised arrays go in the manifest, so the h5 stays
self-describing either way. **Constraint (decision 91):** the end tables use
`observed/background`, which is a **biased approximation** — `Z_s(c5)` couples
every end weight, so this ratio cannot factor `end_s[h]` out. Nothing in this
document or downstream may claim these tables are the data's end-hexamer
preference; they are accepted as inputs to `w`, not as measurements of biology.

**Hexamer prior (decisions 107/110/111).** Per-sample posteriors are derived via
Dirichlet-multinomial shrinkage toward a count-weighted pooled proportion
`p_pool`, with concentration `α_0` estimated by method of moments (Ronning
1989). The split-half reliability gets the Spearman-Brown step-up (decision
110), and `p_pool` uses count-weighted proportions (decision 111, M1). The
shrinkage form is approved (decision 107); the per-sample posteriors in the
92-sample artifact `hexamer_prior_92samples.json` stand.

**GC source for scoring.** Simulation uses the true simulator surface; the
real-data GC source is out of scope for Layer 1.

**Per-region counts are constant in Layer 1** (54 / 37); matching each region's
depth to a real sample's realised count is out of scope.

---

## Validation

- **No parity with the old sampler** — neither byte equality nor distributional
  equivalence; new seed lineage; spend no effort comparing.
- **Self-consistency:** the drawn fragments' empirical distribution matches the
  weights they were drawn from.
- **Realised-parameter recovery** — the check that **catches errors every
  normalisation invariant passes**, because `Σ_Ω w = 1` survives any
  per-element reweighting. The two halves are not symmetric:
  - **Start tables are exactly identifiable.** The Appendix-A cancellation
    leaves a clean 5′ marginal `P(c5 | s) = start_s[hex(c5)] / S_s`, so
    `count_5_s[h] / B_s[h] ∝ start_s[h]` exactly, where `B_s[h]` counts the
    `c5` positions carrying `h` with `Z_s(c5) > 0`. Recovered up to one
    multiplicative constant.
  - **End tables are NOT identifiable from marginal 3′ counts.**
    `Z_s(c5) = Σ_L end_s[hex(c3(L))] · marginal_fl(L) / predict(...)` couples
    every end weight, so an observed-over-background ratio cannot factor
    `end_s[h]` out. Compare the observed 3′ hexamer marginal against the
    **model-predicted** one computed from `w`, and test with `χ²/dof`.
  - **Use `χ²/dof`, not a correlation threshold.** Its null value is 1 with a
    known interval, so it is calibrated in advance; a Pearson-r cutoff is not,
    and will fail a correct sampler at finite depth purely from the Poisson
    floor. Always state the noise floor alongside the statistic, or "close to
    truth" is unfalsifiable.
- **The manifest round-trips.** Reconstruct `w` from the manifest alone and
  confirm `Σ_Ω w = 1` and the exact-½ strand marginal. **This is the
  load-bearing property of the whole output** — if the manifest is insufficient
  or mis-keyed, the h5 is unusable for scoring and nothing else detects it.
- **Non-empty store** when the model agent builds from the h5 (the only check
  that catches the MAPQ `-1 >= 10` trap).
- **Oracle NLL strictly below uniform** on sim data *(model agent)*.
- **`var(log Z_s(c5))` measured over the region set** (Appendix E).
- **Name what must FAIL, not only the invariant that must hold.** `Σ_Ω w = 1`
  and the strand marginal are insensitive to which hexamer table is used where.
  Test wiring with fully asymmetric tables and mutation-test against plausible
  misroutings.
- **A test that was never observed failing is only an assertion about itself.**
  Demonstrate: mutate → fail → restore → pass.
- **`W_D` per-region:** two regions with different valid-position counts must
  yield different `W_D`. This must be a test, not prose.
- **Per-sample disattenuation gate (decision 108).** Before running the
  simulator with respect to a specific sample, measure the disattenuated `r`
  between that sample's counts and the pooled counts (Spearman-Brown step-up
  applied, decision 110). This is the operative reading of decision 92's
  "tested every time" — per-sample, not per-fit. **As built the gate measures
  but cannot fail:** no threshold, no consequence, and 26 of 92 samples sit
  below 0.95. A threshold has not been set; do not invent one.

---

## Data

| what | where | supplies |
|---|---|---|
| region sets | `quiet_v2_pad1200_repeats_removed_tile2560` (11,505 tiles), `..._tile1536` (66,649) — **both include chrX** (decision 94) | the regions; `region_len = tile_size + 2·jitter` |
| scored crop | `P = tile_size` (2048 for region_len 2560, 1024 for 1536) | the centre axis of `D`; `|D| = 155 · P · 2` |
| reference | `/efs/analytics/nathanboley/data_resources/genome/hg38.fa` | sequence for hexamers and `cum_gc` |
| hexamer tables | Layer 1: `build_w6(seed, dynamic_range)` — synthetic. Layer 2+: counted `observed/background` over `tile2560`, shrunk via Dirichlet-multinomial prior (92 samples; decisions 91/107) — a **biased approximation**, not a measurement of end-hexamer preference | the four `{start,end}×{fwd,rev}` tables |
| length marginal | `duphist_merged/<sid>__duphist_wg.tsv.gz`, deduped `molecule_keys` | `marginal_fl(L)` |
| capture surface | same duphist → `load_duphist` → `build_cell_map` → `GCFlDistModel().fit(...)` | `predict(L, gc)` = inverse capture |

---

## Appendices

### Appendix A — `Σ_Ω w = 1` and strand marginal ½

With `E_s(c5,L)` and `Z_s(c5) = Σ_{L} E_s(c5,L)` as in Step 4,

```
Σ_Ω w = Σ_s ½ · Σ_{c5} start_s[hex(c5)]/S_s · ( Σ_L E_s(c5,L) ) / Z_s(c5)
      = Σ_s ½ · Σ_{c5} start_s[hex(c5)]/S_s · Z_s(c5)/Z_s(c5)
      = Σ_s ½ · Σ_{c5} start_s[hex(c5)]/S_s
      = Σ_s ½ · S_s/S_s
      = Σ_s ½  =  1.
```

The strand marginal is `Σ_{c5} Σ_L w = ½` for each `s`, exactly.

The cancellation `Σ_L E_s(c5,L) = Z_s(c5)` holds **for any end-step factor**,
because `Z_s(c5)` is *defined* as that sum.

The step `Σ_{c5} start_s[hex(c5)]/S_s = S_s/S_s` requires that `S_s` sum over
**exactly** the `c5` that contribute to the numerator — i.e. those with
`Z_s(c5) > 0`. A `c5` with `Z_s(c5) = 0` contributes nothing above (its `E_s`
is identically zero), so including it in `S_s` would leave
`Σ_{Z>0} start_s / Σ_{all} start_s < 1` and the strand marginal short of ½.

### Appendix C — `Ω` vs `D`, `W_D`, and why `-log w` is not the oracle

`Ω` is everything the simulator can emit, and `Σ_Ω w = 1`. `D` is the scoring
domain: `{(m, L, strand) : m ∈ [0, P), L in band}`, centre-based — a strict
subset. Strand is a dimension, so `|D| = 155 · P · 2` and `log|D|` already
contains the `log 2`. "In-band" is part of this definition, not a filter
applied after it. Three silent ways to get it wrong: including out-of-band pairs
makes `log|D|` count cells no model is charged for; admitting a fragment with
one endpoint outside the crop puts numerator and denominator on different sets;
and hardcoding `|D|` — it is **computed**.

`|Ω| = 2 · Σ_{L=25}^{180}(region_len + 1 − L)`. At region_len 2560:
`|Ω| = 767,052`; at 1536: `|Ω| = 447,564`. `|D|` is strictly smaller for two
independent reasons: the centre axis spans only `P < region_len` positions per
length (when a centred crop is applied), and **`L = 180` is drawn but is not in
band** — the bands are `[25,110) ∪ [110,180)`, so 155 of the 156 sampled
lengths are scored. That second point is why the in-band clause is not vacuous
even though the sampler is confined to the capture surface's support.

Because scoring is conditional on being in `D`, the true conditional is `w/W_D`
with `W_D = Σ_{x ∈ D} w(x) ≤ 1`, so the oracle per-fragment NLL is
`-log(w/W_D) = -log w + log W_D`. **`-log w` alone is not the oracle:**
`log W_D < 0`, so it overstates the oracle (makes it too weak) by exactly
`|log W_D|`, and a model that correctly learns the conditional prints **above
100% captured**, reading as an implementation bug rather than a mis-specified
anchor.

**Computing `W_D`.** `W_D` is itself a probability under `w` — the probability
that a draw from `w` lands in `D`. The **retention fraction** `n_scored /
n_emitted` therefore estimates it directly: no domain rebuild is needed. A
simulator that draws from `w` and a store that drops out-of-`D` fragments
computes `W_D` as a side effect.

**Per-region vs pooled scalar.** `W_D` is in principle **per region**: `w` is
normalised within a region, so `W_D` differs region to region. Treating it as a
single pooled scalar reintroduces the same mis-normalisation, averaged.
**However:** when the only domain restriction is dropping `L = 180` (0.17% of
mass) and no centred crop is applied — as in the 1536-geometry store —
per-region variation is dominated by binomial noise. Measured: pooled
`log W_D = −0.001707` vs fragment-weighted per-region `−0.001683`, difference
**0.000023 nats**; per-region sd = 0.006805, matching the binomial expectation
at n = 37 (0.006780). **At this domain the pooled scalar is correct and
per-region estimation adds only noise. If a centred crop is ever applied,
per-region `W_D` returns** — the crop excludes different mass fractions in
different regions, and the variation is no longer binomial at fixed `n`.

Both anchors are recomputed per store and never carried across: uniform's
deficit below `log(tile_size)` tracks the empty-mass fraction, which varies
across stores by a large factor, so a carried baseline silently misstates every
percentage.

**Verification.** Two checks replace the correlation-grid approach (which
failed) and constitute the sanctioned oracle validation:

1. **No observed fragment may land on a `w == 0` cell.** A hit proves the
   coordinate mapping is wrong — hex indices, strand convention, or the
   `c5`/`c3` assignment.
2. **Per-region entropy must match the sample mean within MC error.** The exact
   per-region entropy `H = −Σ_{w>0} w log w` is computable from the weights
   with no emitted fragments. The sample mean of `−log w(fragment)` over a
   region's draws must agree with `H` within `SE ≈ sd / √R`. Disagreement
   proves the coordinate mapping is wrong even when check 1 passes — a
   consistent off-by-one can avoid zero cells while shifting every weight.

### Appendix D — Orientation and index conventions

A fragment occupying bases `[p, p+L)` has endpoint **bases** `p` and `p+L-1`,
and **cut sites** at `p` and `p+L`. The simulator weight uses `hex(c3)` at a
cut site; a track model's weight uses `last_m[p+L-1]`, a per-base track. These
are different subscripts because they are different spaces: `hex_fwd`/`hex_rc`
have length `region_len+1` and index between-base cut sites, while
`first_m`/`last_m` are per-base of length `region_len`.

The formula is written in `(c5, c3)` rather than `(p, q)` deliberately: on the
minus strand the 5′ end is at the higher coordinate, so `c5 = p+L`, `c3 = p`,
and hexamers read reverse-complemented. An implementer coding from a `p`/`q`
form gets the minus strand backwards, which corrupts strand asymmetry silently.
GC is the exception — it is a property of the genomic span and
strand-independent, so it uses `cum_gc[max(c5,c3)] - cum_gc[min(c5,c3)]`.

**Required test:** assert the endpoint bases the scorer indexes are the same
genomic positions as the cut sites the sampler drew from, on a small region
with known fragments. A silent 1 bp offset corrupts every NLL, so a
shift-correlation proof is owed on **every** geometry change.

### Appendix E — The `Z_s(c5)` receptive-field bound on % captured

A joint log-linear model absorbs `-log Z_s(c5)` only through a per-position
term, and a width-6 L1 sees 6 bp while `Z_s(c5)` depends on the full
`L = 25..180` span (~180 bp) around the start. So `% captured` is capped below
100 before training starts. `var(log Z_s(c5))` over the region set bounds that
unreachable fraction; measure it early, because without it a sub-100% result
cannot be distinguished from a defect.

### Appendix F — Bin boundary semantics for the capture fit

`_bin_index` tests `lo ≤ v ≤ hi` (inclusive) and returns the first match;
`gc_pct` is **continuous**. `SIM_GC_BINS` are `(0,4),(5,9),…,(95,100)` —
inclusive-integer ranges with **unit-width gaps** for non-integer input:
`_bin_index(4.5, SIM_GC_BINS)` returns `None` because `4.5 > 4` and `4.5 < 5`.
This is correct for the fit path, where duphist GC values are integers, so
non-integer values never arise. The per-fragment weight path uses continuous GC
and therefore uses `gc_bin_index` (floor-based, `floor(gc/5)` clamped to
`[0, 19]`), which has no gaps — it covers `[0, 100]` contiguously.

Length bins `(25,25)…(180,180)` are unambiguous integers. Do not use the flgc
defaults: their top length bin `(101,200)` collapses the high band `[110,180)`
into one bin, leaving `capture` constant in `L` across it, and their GC range
stops short of 0–100 so extreme-GC fragments fall out of bin onto `max_weight`.
GC is percent (0–100) throughout the flgc path — `build_cell_map` keys on the
duphist's percent column, `gc_bins` are percent, `predict()` takes percent.
