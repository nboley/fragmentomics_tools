# Simulator and per-fragment NLL

Status: DESIGN, not implemented. Specifies the simulator's generative model, the
data that informs it, and the per-fragment NLL that scores both model classes.
The sibling doc `cut_site_fragment_model.md` owns the models and their stores;
this doc owns the weight, the sampler, the scoring domain, and the anchors.

Written **from scratch** in `background_model/simulator/`. Reuse where useful —
`load_duphist` / `build_cell_map` (`scripts/ztnb_from_duphist.py`), the
`hexamer_indices` + `cum_gc` precompute, region/BED loading, the per-sample output
layout, the multiprocessing scaffolding. Not a refactor of
`sim_fragments.py`; **no parity requirement of any kind**. The existing simulator
stays live and must keep working (KEN and Hybrid validate against it), so
`sim_fragments.py` and its stores stay intact. This work is additive.

---

## Part 1 — Procedure

### Domain and conventions

A fragment is `(c5, c3, s)`: strand `s`, 5′ cut site `c5`, 3′ cut site `c3`.
These are **cut sites, not bases** — for bases the fragment occupies `[p, p+L)`,
endpoints `p` and `p+L-1`, cut sites `p` and `p+L`.

- **Plus strand:** `c5 = p`, `c3 = p + L`.
- **Minus strand:** the ends **swap** — `c5 = p + L`, `c3 = p` — because the 5′
  end sits at the higher coordinate; hexamers read reverse-complemented.
- **Length** `L = |c3 - c5|`, restricted to **`L = 25..180`, 156 values** — this
  is exactly the support of the capture surface. Nothing outside it is ever drawn.
- **GC** over the genomic span (strand-independent):
  `gc_pct = 100 · (cum_gc[max(c5,c3)] - cum_gc[min(c5,c3)]) / L`.
- **Hexamer tables:** four, `{start, end} × {fwd, rev}`, **untied** (short-read
  single-stranded data — the two ends are not related by reverse complement);
  strand selects which `(start_s, end_s)` pair applies.

Derivations: Appendix D.

### Step 1 — Fit the capture surface

- **In:** duphist `duphist_merged/<sid>__duphist_wg.tsv.gz`, columns
  `(length, gc, multiplicity, molecule_keys)`.
- **Do:** `load_duphist` → `build_cell_map` →
  `GCFlDistModel().fit(cell_map, length_bins, gc_bins, min_cell_size=200)`.
  Length bins: 1 bp bins over 25–180 (156 bins, the surface's full support).
  GC bins: 5% inclusive-integer bins `(0,4),(5,9),…,(95,100)` — contiguous
  on the integer grid (which is all the duphist fit sees), last inclusive. Other `fit` options default (`min_p=1e-6`, `max_weight=3.0`,
  `k_fit=25`). `save()` to `gcfl_model.json`; build a `predict` LUT over
  `(L, gc_bin)` by evaluating `predict(L, gc_mid)` at each bin's **midpoint**
  (2.5, 7.5, ..., 97.5). This midpoint evaluation is exact **only if**
  `predict` is piecewise-constant on those bins — true for the ZTNB method
  (which bins GC internally via `_bin_index` before lookup), but NOT for
  `spike_grid` (which interpolates). `predict_lut_from_model` asserts both
  conditions: model is ZTNB and `model.gc_bins == SIM_GC_BINS`.
  **Cache that LUT and hand it to Step 4 as an array to gather
  from** — built once, reused across regions and across both the Step 5 sampler
  and the Step 7 oracle. Step 4 records the measured cost of the per-element
  alternative.
- **Out:** fitted `GCFlDistModel`. `predict(L, gc_pct)` returns
  `min(1/P(seen), max_weight)`, **GC in percent**. Dividing by `predict` is
  multiplying by capture.
- Runtime dependency `flgc.model` needs `PYTHONPATH=/home/nathanboley/src/biomarker`,
  **including in the Batch container**.
- Derivation: Appendix F (bin boundary semantics).

### Step 2 — Build `marginal_fl(L)`

- **In:** the same duphist (its de-duplicated `molecule_keys` column).
- **Do:** for each `L`, sum `molecule_keys` across all GC values; restrict to
  `L = 25..180`; normalise to sum 1. Nothing is deconvolved out of it.
- **Out:** `marginal_fl(L)`, the empirical unweighted length marginal.

### Step 3 — Per-region precompute

- **In:** reference `hg38.fa`, one region set.
- **Do:** per region, compute hexamer indices for the forward and
  reverse-complement cut-site tracks (`fwd_cut`/`rc_cut`, length `region_len+1`)
  and cumulative GC `cum_gc`.
- **Out:** per-region `hex(c5)` / `hex(c3)` lookups and `cum_gc`.
- Derivation: Appendix D (index spaces).

### Step 4 — Build `w` (`build_region_weights`)

- **In:** the four hexamer tables as a single `HexamerTables` (fields, in order,
  `start_fwd`, `end_fwd`, `start_rev`, `end_rev`), `marginal_fl`, `predict`,
  per-region `hex_fwd`/`hex_rc`/`cum_gc`, strand.
- **Do:** per region, per strand `s`, with `c3(L) = c5 + σ·L`
  (`σ = +1` plus, `-1` minus):

  ```
  S_s        = Σ_{c5 : Z_s(c5) > 0} start_s[hex(c5)]
  E_s(c5,L)  = end_s[hex(c3(L))] · marginal_fl(L) / predict(L, gc(L))     L = 25..180
  Z_s(c5)    = Σ_{L=25..180} E_s(c5,L)
  w(c5,c3,s) = ½ · start_s[hex(c5)]/S_s · E_s(c5,L)/Z_s(c5)
  ```

- **Hexamer track:** plus strand uses `hex = hex_fwd` for **both** `c5` and `c3`;
  minus strand uses `hex = hex_rc` for **both**. (Stated here so Step 4 is
  followable without Appendix D.)
- **`S_s` range:** the sum runs only over `c5` with `Z_s(c5) > 0` — positions
  where at least one fragment is achievable. Including `Z_s(c5) = 0` positions
  (edge-truncated, or fully N-masked) breaks the Appendix A cancellation: the
  strand marginal falls below ½ by exactly the dead-mass fraction. It also
  conditions the start draw on "at least one fragment is possible here", which is
  the correct conditional for a rejection-free sampler.
- **Edge rule:** `Z_s(c5)` sums only those `L` whose `c3(L) = c5 + σ·L` stays in
  `[0, region_len]`; off the region `hex(c3)` is undefined. This per-`c5`
  truncation is exactly what makes `|Ω| = 2 · Σ_L (region_len − L + 1)` correct.
- **`predict` is supplied as a cached `(L, gc_bin)` lookup table** and gathered
  from, not called per element (owner decision, 2026-09-29). Built once and reused
  across regions and across **both** the Step 5 sampler and the Step 7 oracle.
  Measured motivation: the scalar `predict(L, gc)` form costs **767,052 calls and
  0.665 s per region** at `region_len` 2560 with a *no-op* `predict` — exactly one
  Python call per element of `Ω`. *Estimate:* ~1–2 s/region with the real
  `GCFlDistModel`, ~4–10 h over 11,505 regions once the oracle's second pass is
  counted.
- **Out:** `w`, a fully normalised probability over the generative domain `Ω`:
  `Σ_Ω w = 1` **exactly**, strand marginal exactly ½. `build_region_weights`
  returns this normalised `w`.
- **Constraint — one implementation.** Both the sampler (Step 5) and the oracle
  (Step 7) call `build_region_weights`. This shared call is the only thing
  preventing sampler and scorer from drifting apart, and such a drift is silent.
- Derivations: Appendix A (`Σ_Ω w = 1`, strand marginal ½), Appendix B
  (sequential form vs symmetric joint; the roles of `1/S_s` and `1/Z_s(c5)`).

### Step 5 — Draw fragments

- **In:** the factors of `w`; the per-region target count (**54** at region_len
  2560, **37** at 1536).
- **Do:** per region, per sample, repeat until the target count is reached:

  ```
  1. s  ~ Bernoulli(½)
  2. c5 ~ start_s[hex(c5)] / S_s
  3. c3 ~ E_s(c5,L) / Z_s(c5)   over L = 25..180,  c3 = c5 + σ·L
  ```

- **Out:** fragments `(region_idx, start, stop, strand)` per sample, serialised
  directly to the per-sample BED of Step 6 — no intermediate `.npz` format.
- Related bound: Appendix E (`var(log Z_s)`).

### Step 6 — Emit fragments; production builds the store

The simulator's job is to generate fragments; store layout is a model-class
concern. Emitting a BED and letting production build the store removes a second
implementation of the store rule — the same rule that forced `sim_build_store.py`
to be patched separately from `dataset.py` for the `FL_BANDS` guard, the
silent-divergence failure `CLAUDE.md` names.

- **In:** the drawn fragments, reference `hg38.fa`.
- **Do:** write **one sorted BED per sample** — `sample_<i>.bed.gz`, bgzipped and
  tabix-indexed — in the **8-column** format

  ```
  0 contig  1 start  2 stop  3 name  4 score  5 strand  6 mapq1  7 mapq2
  ```

  with **all MAPQs = 60**. **GC is not emitted** — `build-fragments-h5` computes
  it from the FASTA via `get_g_or_c_cumsum`, so store GC comes from the real
  reference through production code. Then hand off to production:

  ```
  build-fragments-h5 sample_<i>.bed.gz sample_<i>.h5 --fasta hg38.fa   # per sample
  background_model preprocess + store                                  # all samples -> one store
  ```

- **One BED and one h5 per sample, not one of each overall.** The fragment h5 is a
  per-sample artifact: `preprocess.py` calls `from_fragments_h5` once per sample and
  the sample sheet maps samples to h5 paths. The store then holds all samples on its
  sample axis. At `S = 1` this is a single file, but building it per-sample from the
  start is what makes `S > 1` work without restructuring.

- **Hard dependency:** columns 6–7 need a pending `fragments_h5` patch —
  `tsv_to_fragments` currently hardcodes `mapq1=None, mapq2=None`; the two-column
  MAPQ read is being added. This step is blocked until it lands.
- **Why MAPQ matters (silent-failure trap):** unknown MAPQs store as `-1`;
  `background_model/config.py` sets `min_mapq = 10` and `preprocess.py` passes it
  to `from_fragments_h5`, and `-1 >= 10` is False — so without real MAPQs every
  fragment is filtered and the store comes out **empty, with no error**.
- **Out:** the fragment h5 and the zarr store, both built by production code.

### Step 7 — Compute the anchors

- **In:** the store, `w` via `build_region_weights`, the scoring domain `D`.
- **Do:**
  - `D = {(m, L, s) : m ∈ [0, P), L in band, s ∈ {0,1}}`, centre-based — the
    fragment centre `m = p + L//2` lands in the (centred, val/test) crop of width
    `P = tile_size` (2048 for region_len 2560, 1024 for 1536). `|D_L| = P` for
    every `L`, so `|D| = 155 · P · 2` (155 in-band lengths, 2 strands).
  - `uniform = log|D|`.
  - `oracle = mean over scored fragments of ( -log w(x) + log W_D )`, where
    `W_D = Σ_{x ∈ D} w(x)`, computed **per region**.
- **Out:** `uniform`, `oracle` in per-fragment nats, **recomputed per store and
  never carried across**.
- Derivation: Appendix C (`Ω` vs `D`; `W_D`; why `-log w` alone is not the
  oracle; `W_D` per region while `|D|` is geometric).

### Step 8 — Score a model

- **In:** the store, `D`, the model's weight `w_m`.
- **Do:** for fragment `(p, L, s)`:

  ```
  NLL = -log( w_m[p,L,s] / Σ_{(p',L',s') ∈ D} w_m[p',L',s'] )
  ```

  - **Track models** (12-track / KEN / Hybrid):
    `w_m = first_m[p] · last_m[p+L-1] · gc_correction(L, GC%)`, with
    `first_m`/`last_m` from the fragment's band and strand. **No `len_p`.**
  - **Cut-site model:** its own `log_softmax` logits over `D`; no external
    `len_p` or GC factor (the logit carries both; the track-model product would
    double-count length).
  - `midpoint` is **not** a factor for either — all three coverage types are
    deterministic functions of the same fragment, so including it triple-counts
    one piece of evidence. Kept only as a consistency check.
- **Out:** per-fragment NLL, and
  `% bias captured = (uniform - model) / (uniform - oracle)`, all in per-fragment
  nats.
- **Report the bands with every number** — the denominator changes with the
  bands, so an NLL under one banding is not comparable to one under another.
- Strand `log 2` is **included** everywhere and cancels in the ratio (simulator
  strand is exactly 50/50); only absolute nats move.
- Derivations: Appendix D (the ±1 endpoint/cut-site trap), Appendix C (anchors).

### Validation

- **No parity with the old sampler** — neither byte equality nor distributional
  equivalence; new seed lineage; spend no effort comparing.
- **Self-consistency:** the drawn fragments' empirical distribution matches the
  weights they were drawn from, on a small region. This is what the shared
  builder guarantees; it needs no reference implementation.
- **Realised-parameter recovery:** the generative tables are recoverable from the
  emitted fragments (hexamer-table recovery, GC slope by length).
- **Oracle NLL strictly below uniform** on sim data.
- **`var(log Z_s(c5))` measured over the region set** (Appendix E).

---

## Part 2 — Appendices

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
`Z_s(c5) > 0`, per Step 4. A `c5` with `Z_s(c5) = 0` contributes nothing above
(its `E_s` is identically zero), so including it in `S_s` would leave
`Σ_{Z>0} start_s / Σ_{all} start_s < 1` and the strand marginal short of ½.

### Appendix B — Sequential form, not the symmetric joint

Globally normalising `start_s · end_s · marginal_fl / predict` over all `(c5,c3)`
with one denominator gives the **symmetric** joint — a different model. There the
start marginal is tilted by the total end-mass at each start, `∝ start_s[hex(c5)] ·
Z_s(c5)`. The per-start normaliser `1/Z_s(c5)` is exactly what removes that tilt,
making the end step a proper conditional and leaving the start marginal a clean
`start_s/S_s`. The per-strand normaliser `1/S_s` makes the start step a proper
conditional **per strand**: omit it and the total is `S_+ + S_-`, so the strand
marginal comes out `∝ S_s` rather than ½ — harmless only if `S_+ = S_-`, which
untied tables do not guarantee. The `½` is the strand prior the sampler draws
from.

### Appendix C — `Ω` vs `D`, `W_D`, and why `-log w` is not the oracle

`Ω` is everything the simulator can emit, and `Σ_Ω w = 1`. `D` is the scoring
domain: `{(m, L, strand) : m ∈ [0, P), L in band}`, centre-based — the fragment
centre `m = p + L//2` lands in the crop of width `P = tile_size` — a strict
subset. Strand is a dimension, so `|D| = 155 · P · 2` and `log|D|` already
contains the `log 2`. "In-band" is part of this definition, not a filter applied
after it. Two silent ways to get it wrong: including out-of-band pairs makes
`log|D|` count cells no model is charged for; and hardcoding `|D|` — it is
**computed**.

`|Ω|` and `|D|` differ on both axes, so neither substitutes for the other. For the
2560 geometry, `|Ω| = 2 · Σ_{L=25}^{180}(2561-L) = 2 · 383,526 = 767,052`, while
`|D| = 155 · P · 2 = 155 · 2048 · 2 = 634,880`. `|D|` is strictly smaller for two
independent reasons: the centre axis spans only `P = 2048 < region_len` positions
per length (against `2561 − L` in `Ω`), and **`L = 180` is drawn but is not in
band** — the bands are `[25,110) ∪ [110,180)`, so 155 of the 156 sampled lengths
are scored. That second point is why the in-band clause is not vacuous even though
the sampler is confined to the capture surface's support.

Because scoring is conditional on being in `D`, the true conditional is `w/W_D`
with `W_D = Σ_{x ∈ D} w(x) < 1`, so the oracle per-fragment NLL is
`-log(w/W_D) = -log w + log W_D`. **`-log w` alone is not the oracle:** `log W_D <
0`, so it overstates the oracle (makes it too weak) by exactly `|log W_D|`, and a
model that correctly learns the conditional then prints **above 100% captured**,
reading as an implementation bug rather than a mis-specified anchor.

`W_D` is **per region** — `w` is normalised within a region, so `W_D` differs
region to region and the oracle averages it over scored fragments. `|D|` is purely
geometric — identical for every region at a given geometry — so `log|D|` is one
number. Treating `W_D` as a single scalar reintroduces the same mis-normalisation,
averaged.

Both anchors are recomputed per store and never carried across: uniform's deficit
below `log(tile_size)` tracks the empty-mass fraction, which varies across stores
by a large factor, so a carried baseline silently misstates every percentage.

### Appendix D — Orientation and index conventions

A fragment occupying bases `[p, p+L)` has endpoint **bases** `p` and `p+L-1`, and
**cut sites** at `p` and `p+L`. The simulator weight uses `hex(c3)` at a cut site;
a track model's weight uses `last_m[p+L-1]`, a per-base track. These are different
subscripts because they are different spaces: `fwd_cut`/`rc_cut` have length
`region_len+1` and index between-base cut sites, while `first_m`/`last_m` are
per-base of length `region_len`.

The formula is written in `(c5, c3)` rather than `(p, q)` deliberately: on the
minus strand the 5′ end is at the higher coordinate, so `c5 = p+L`, `c3 = p`, and
hexamers read reverse-complemented. An implementer coding from a `p`/`q` form gets
the minus strand backwards, which corrupts strand asymmetry silently. GC is the
exception — it is a property of the genomic span and strand-independent, so it
uses `cum_gc[max(c5,c3)] - cum_gc[min(c5,c3)]` (i.e. always over `[p, p+L)`);
`cum_gc[c3] - cum_gc[c5]` would go negative on the minus strand.

**Required test:** assert the endpoint bases the scorer indexes are the same
genomic positions as the cut sites the sampler drew from, on a small region with
known fragments. A silent 1 bp offset corrupts every NLL, so a shift-correlation
proof is owed on **every** geometry change.

### Appendix E — The `Z_s(c5)` receptive-field bound on % captured

A joint log-linear model absorbs `-log Z_s(c5)` only through a per-position term,
and a width-6 L1 sees 6 bp while `Z_s(c5)` depends on the full `L = 25..180` span
(~180 bp) around the start. So `% captured` is capped below 100 before training
starts. `var(log Z_s(c5))` over the region set bounds that unreachable fraction;
measure it early, because without it a sub-100% result cannot be distinguished
from a defect.

### Appendix F — Bin boundary semantics for the capture fit

`_bin_index` tests `lo ≤ v ≤ hi` (inclusive) and returns the first match; `gc_pct`
is **continuous**. `SIM_GC_BINS` are `(0,4),(5,9),…,(95,100)` — inclusive-integer
ranges with **unit-width gaps** for non-integer input: `_bin_index(4.5,
SIM_GC_BINS)` returns `None` because `4.5 > 4` and `4.5 < 5`. This is correct
for the fit path, where duphist GC values are integers (whole-number percent), so
non-integer values never arise. The per-fragment weight path uses continuous GC
and therefore uses `gc_bin_index` (floor-based, `floor(gc/5)` clamped to
`[0, 19]`), which has no gaps — it covers the full `[0, 100]` range contiguously.

The design doc's "contiguous" claim applies to `gc_bin_index`, not to
`SIM_GC_BINS` fed to `_bin_index`. `SIM_GC_BINS` covers the integer grid
exactly — which is all the fit sees — while `gc_bin_index` covers the continuous
range the weight builder needs.

Length bins `(25,25)…(180,180)` are unambiguous integers. Do not take the `flgc`
defaults: their top length bin `(101,200)` would collapse the whole high band
`[110,180)` into one bin, leaving `capture` constant in `L` across it, and their
GC range stops short of 0–100 so extreme-GC fragments fall out of bin onto
`max_weight`. GC is percent (0–100) throughout the `flgc` path —
`build_cell_map` keys on the duphist's percent column, `gc_bins` are percent,
`predict()` takes percent.

---

## Data

| what | where | supplies |
|---|---|---|
| region sets | `quiet_v2_pad1200_repeats_removed_tile2560` (11,505 tiles), `..._tile1536` (66,649) | the regions; `region_len = tile_size + 2·jitter` |
| scored crop | `P = tile_size` (2048 for region_len 2560, 1024 for 1536) | the centre axis of `D`; `|D| = 155 · P · 2` |
| reference | `/efs/analytics/nathanboley/data_resources/genome/hg38.fa` | sequence for hexamers and `cum_gc` |
| hexamer tables | `build_w6(seed, dynamic_range)` — synthetic, **4096 independent** log-normal draws per table | the four `{start,end}×{fwd,rev}` tables |
| length marginal | `duphist_merged/<sid>__duphist_wg.tsv.gz`, deduped `molecule_keys` | `marginal_fl(L)` |
| capture surface | same duphist → `load_duphist` → `build_cell_map` → `GCFlDistModel().fit(...)` → `save()` | `predict(L, gc)` = inverse capture |

Output: one `sample_<i>.bed.gz` per sample (8 columns, sorted, tabix-indexed).

**GC source for scoring.** Simulation uses the true simulator surface; the
real-data GC source is out of scope for Layer 1.

**Per-region counts are constant in Layer 1** (54 / 37); matching each region's
depth to a real sample's realised count is out of scope for Layer 1.

---

## Implementation plan

Five phases, each independently testable and committed separately. Every phase
runs the same gate sequence: **implement → implementation-review → fix ALL
findings → EM critical review → architecture reflection → design reconciliation.**
A- is the minimum grade to advance. Nothing merges or pushes until all five are
done.

| Phase | Steps | Scope | Status |
|---|---|---|---|
| **1** | 4 | `build_region_weights` + the orientation/geometry helpers | **COMPLETE** — `76e020f`, `409c683`, `5197778`, `09a9b1b` |
| **2** | 1–3 | cached `(L, gc_bin)` `predict` LUT **first**, then capture-surface fit, `marginal_fl`, per-region precompute | IN PROGRESS |
| **3** | 5–6 | sampler; 8-column BED per sample; `build-fragments-h5` → store, proven **non-empty** | NOT STARTED |
| **4** | 7–8 | anchors (`uniform`, `oracle`, per-region `W_D`); model scoring | NOT STARTED |
| **5** | — | smoke run: a few hundred regions, one sample, all invariants asserted | NOT STARTED |

### Why this order

**Step 4 was built first**, ahead of the Steps 1–3 that feed it. Both the Step 5
sampler and the Step 7 oracle call `build_region_weights`, and that single shared
call is the only thing preventing sampler/scorer drift — a drift that is silent
and would invalidate every `% captured` number downstream. Its inputs were
injected as parameters until Phase 2 supplied them.

**Phase 2 lands the `predict` LUT before anything else it contains**, because the
scalar interface was a known breaking change and the builder still had
essentially one consumer. Deferring it to Phase 3 or later would have meant
rewriting every call site plus its tests.

### Standing requirements, learned from Phase 1

- **Name what must FAIL, not only the invariant that must hold.** Phase 1's worst
  finding was that `Σ_Ω w = 1`, the strand marginal, and `|Ω|` are all
  *insensitive* to which of the four hexamer tables is used where — so 26 passing
  tests coexisted with provably unforgeable-looking but untested wiring. The fix
  was one test with fully asymmetric tables, mutation-tested against **9**
  plausible misroutings (both strands × start/end × fwd/rc track), **9/9 caught**.
- **A test claimed to catch a bug, but never observed failing, is only an
  assertion about itself.** Demonstrate the failure: mutate → fail → restore →
  pass.
- **Measure both suites before and after, every phase.** Baselines move with
  almost every commit; a quoted one is useless as the regression check it exists
  to be.
- **Phase 4 specifically:** `W_D` is **per region**, and `|D|` is geometric and
  identical across regions. Computing one `W_D` store-wide reintroduces the error
  in averaged form. This must be enforced by a **test**, not by prose — two
  regions with different valid-position counts must yield different `W_D`.
- **Phase 3 specifically:** unknown MAPQ reads back as `-1`, `config.min_mapq` is
  `10`, and `-1 >= 10` is False — so if MAPQ is not carried through the 8-column
  BED, **every fragment is filtered and the store comes out empty with no error**.
  Assert on a non-empty store; each repo can pass its own suite and still not
  compose.
