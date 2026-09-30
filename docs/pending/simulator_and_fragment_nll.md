# Simulator and per-fragment NLL

Status: Phases 1–2 IMPLEMENTED; Phase 3 next. See the implementation plan at the end.

**What the simulator produces (owner decision 79):**

> **the per-sample fragment `h5`, plus the metadata needed to reconstruct the
> fragment weights — and nothing else.**

It does **not** build the zarr store, choose the scoring domain `D`, or compute the
anchors. Those belong to the **model agent**, which consumes the h5 and the manifest.

Everything here serves that output. The generative model (Steps 1–5) determines the
fragments; the manifest (Step 6) is what makes the weights behind them recomputable.
Steps 7–8 and Appendix C specify the **scoring contract the output must support** —
they are recorded so the manifest can be shown to be sufficient, not because the
simulator performs them.

The weights are **not stored dense.** `w` factorises, so the manifest carries three
small arrays (~154 KB) from which `build_region_weights` reconstructs `w` exactly —
against 73.5 GB for a dense `w`. That is the central design choice of Step 6.

The sibling doc `cut_site_fragment_model.md` owns the models and their stores.

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

### Step 6 — Emit fragments and the manifest; the model agent builds the store

**Scope (owner decision 79).** The simulator's deliverable is the **fragment h5 plus
the metadata needed to reconstruct the simulated data** — and it stops there. The
**model agent** owns store construction, the choice of the scoring domain `D`, and the
oracle. Emitting a BED and letting production build the h5 removes a second
implementation of the store rule — the same rule that forced `sim_build_store.py`
to be patched separately from `dataset.py` for the `FL_BANDS` guard, the
silent-divergence failure `CLAUDE.md` names.

#### The manifest — `w` is reconstructed from factors, not stored dense

> **RECONCILIATION PENDING (2026-09-30) — the provenance-verification behaviour
> described in this subsection is known-stale and is being changed right now.**
> `8eb9fe9` made the manifest's recorded hashes *enforced* rather than merely stored.
> The EM critical review then measured that the `simulator_script` check **silently
> evaporates outside the worktree that wrote the manifest**: the manifest records an
> absolute path, `git_blob_sha` relpaths it against a repo root derived from `emit.py`'s
> own location, and from any other checkout that becomes `../../../scripts/...`, the
> `git rev-parse` fails, the function returns `None`, and the loader then neither raises
> *nor* records the check as skipped. Measured: resolves in-worktree, returns `None`
> via the main checkout. So Phase 5's "3/3 checks ran" is an artifact of running in
> place — the check is absent exactly where provenance matters most (a Batch container,
> another checkout, or after this worktree is deleted).
>
> The accompanying architecture change: verification is currently **opt-in by
> omission** — the loader checks only what the caller remembered to pass, and the
> caller must inspect `verified` afterwards to discover what was actually checked.
> That assertion is moving into the library, so that **a skipped check becomes
> unrepresentable rather than merely visible.**
>
> The choice between *record the path relative to the repo root* and *fail loudly on an
> unresolvable path* was deliberately delegated to the implementer rather than fixed
> here. **Do not write the replacement spec until that choice is reported**, or this
> section will document a decision nobody made.

`w` **factorises**. `build_region_weights` takes
`(hex_fwd, hex_rc, cum_gc, valid, hex_tables, marginal_fl, predict_lut, region_len)`,
and the first four come *entirely* from `precompute_region(contig, gstart, gstop, fasta)`.
Only three small arrays are free parameters; `S_s` and `Z_s` are derived per region.

| stored | size |
|---|---|
| 4 hexamer tables (4 × 4096) | 128 KB |
| predict LUT (156 × 20) | 24 KB |
| `marginal_fl` (156) | 1.2 KB |
| **total** | **~154 KB** |

against **73.5 GB** for a dense `w` at region_len 2560 × 11,505 regions. Recompute
costs the measured 0.0675 s/region → ~13 min for the full set — affordable only
because the Step-1 LUT landed; at the pre-LUT 0.665 s it would have been 2.1 h.

**Store the *realised* tables, not the recipe.** Not "seed 42 + `build_w6`", not
"re-fit from sample X" — the actual arrays. Recomputation is then deterministic and
exact, with no dependence on a `flgc` version, on a fit being reproducible, or on
NumPy's RNG stream being stable across releases.

**Hexamer tables are dataframes keyed by the hexamer STRING** (owner decision 81),
not bare 4096-element arrays ordered by an implicit integer code. A bare array makes
the k-mer ordering and RC convention a *contract* between producer and consumer; if a
table is ever produced under a different convention, **every weight is wrong and
`Σ_Ω w = 1` still holds**, because normalisation cannot see a relabelling. Keyed by
the string, a mismatched table fails to *join* rather than silently misaligning, and
the integer code becomes a private detail on each side. The stored string is written
5′→3′ **along the strand of the fragment that produced it**, so the `*_rev` tables
already hold reverse-complemented 6-mers — do not RC them again on load.

The manifest also carries, because `w` is irreproducible without them:
**reference identity + hash** (`hex_fwd`/`hex_rc`/`cum_gc`/`valid` all derive from the
FASTA — a different hg38 patch silently changes every weight); **region-set identity +
hash** and `region_len`; the `L` range; the `FL_BANDS` in force; per-region counts; the
RNG seed (provenance only — no longer load-bearing once the realised tables are stored);
and the `build_region_weights` **commit sha**, since the one residual dependency is that
the function still *behaves* the same.

`jitter` and `tile_size` are **deliberately absent**: `Ω`, and therefore `w`, depends on
`region_len` alone. `jitter` is derived (`region_len = tile_size + 2·jitter`) and is a
training-time crop budget; `tile_size` parameterises `D`. Both belong to the model agent.

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
  ```

  **The simulator stops here.** `background_model preprocess` and the zarr store are
  the **model agent's** (decision 79).

- **One BED and one h5 per sample, not one of each overall.** The fragment h5 is a
  per-sample artifact: `preprocess.py` calls `from_fragments_h5` once per sample and
  the sample sheet maps samples to h5 paths. The store then holds all samples on its
  sample axis. At `S = 1` this is a single file, but building it per-sample from the
  start is what makes `S > 1` work without restructuring.

- **The two-column MAPQ dependency is SATISFIED — verified in code, not from a
  version string.** `fragments_h5` `tsv_to_fragments` reads `int(parts[6])` /
  `int(parts[7])`, range-checks 0–255, and rejects 7-column input explicitly.
  **Caution:** the env's dist-info reports **2.11.0** while the editable checkout it
  points at is **v2.14.0** — the version *understates* the code, so a check gated on
  `pip show` would wrongly conclude the support is missing. Resolve the `.pth` and read
  the source.
- **Why MAPQ matters (silent-failure trap):** unknown MAPQs store as `-1`;
  `background_model/config.py` sets `min_mapq = 10` and `preprocess.py` passes it
  to `from_fragments_h5`, and `-1 >= 10` is False — so without real MAPQs every
  fragment is filtered and the store comes out **empty, with no error**. Each repo can
  pass its own suite and still fail to compose: assert on a **non-empty** store.
- **Out:** the per-sample fragment **h5** and the **manifest**. Not the store.

### Step 7 — Compute the anchors

> **Owner: the MODEL AGENT, not the simulator** (decision 79). `D` is centre-based on
> the crop `P = tile_size`, which is store geometry, so the store's owner owns `D` and
> the anchors built on it. Steps 7–8 are specified here because they define what the
> simulator's output must *support*, not because the simulator performs them.
>
> **The one binding contract across that boundary:** the oracle MUST obtain `w` by
> calling **`build_region_weights`** — the same function the Step-5 sampler used —
> reconstructed from the manifest's stored factors. It must not reimplement the weight.
> Step 4's "one implementation" constraint previously held because sampler and oracle
> sat in one phase; it now spans an ownership line, where the failure is silent and
> nothing enforces it. `Σ_Ω w = 1` and the exact-½ strand marginal are cheap
> post-conditions that catch most breakages.

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

> **Owner: the MODEL AGENT** (decision 79b). Specified here only as the contract the
> simulator's output must support — `% bias captured` is meaningless without an oracle
> built from the same `w` the sampler drew from, so the manifest must be sufficient to
> reconstruct it. Nothing in Step 8 is the simulator's to implement.

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

Everything here is checkable from the simulator's own output — the h5 and the
manifest — without a store.

- **No parity with the old sampler** — neither byte equality nor distributional
  equivalence; new seed lineage; spend no effort comparing.
- **Self-consistency:** the drawn fragments' empirical distribution matches the
  weights they were drawn from, on a small region. This is what the shared
  builder guarantees; it needs no reference implementation.
- **Realised-parameter recovery:** the generative tables are recoverable from the
  emitted fragments (hexamer-table recovery, GC slope by length). **Stronger than it
  sounds, and it is the end-to-end check on the draw**: simulate from surface `S`,
  re-fit from the *simulated* fragments, recover `S′`; `S′ ≈ S` exercises the whole
  sampler and **would catch a broken draw that every normalisation invariant passes**,
  because `Σ_Ω w = 1` is preserved by any per-element weighting.
- **The manifest round-trips.** Reconstruct `w` from the manifest alone and confirm
  `Σ_Ω w = 1` and the exact-½ strand marginal. **This is the load-bearing property of
  the whole output** — if the manifest is insufficient or mis-keyed, the h5 is
  unusable for scoring and nothing else detects it.
- **The store is non-empty** when the model agent builds it from the h5. Cheap, and it
  is the only thing that catches the `-1 >= 10` MAPQ trap, where each repo passes its
  own suite and they still fail to compose.
- **Oracle NLL strictly below uniform** on sim data *(model agent)*.
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

> **`Ω` is the simulator's; `D` and `W_D` are the model agent's** (decision 79b).
> Kept here because `Ω` and `w` are defined by this doc and `D` is carved out of them —
> and because **the manifest must be sufficient to compute `W_D`**, which is the
> requirement this appendix imposes on the simulator's output.

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

### Output

**The deliverable, per sample:**

| artifact | contents |
|---|---|
| `sample_<i>.h5` | the fragments, via `build-fragments-h5` from an 8-column sorted, tabix-indexed `sample_<i>.bed.gz` |
| **manifest** | everything needed to reconstruct `w` — the 4 hexamer tables (**dataframes keyed by the hexamer string**), the `(L, gc_bin)` predict LUT, `marginal_fl`; plus reference identity+hash, region-set identity+hash, `region_len`, `L` range, `FL_BANDS`, per-region counts, RNG seed, and the `build_region_weights` commit sha |

The intermediate `sample_<i>.bed.gz` is a means, not a deliverable. **No store.**

**Why the manifest and not the weights.** `w` is ~154 KB of factors against 73.5 GB
dense, and storing the *realised* arrays rather than a recipe ("seed 42", "re-fit from
sample X") makes reconstruction exact and independent of a `flgc` version, a
reproducible fit, or NumPy's RNG stream. **The manifest is the product, as much as the
h5 is** — an h5 without it cannot be scored, because `% bias captured` needs an oracle
built from the same `w` the sampler drew from.

**The `hexamer tables` row above is Layer 1's synthetic `build_w6`.** Real counted
tables from `scripts/count_cut_site_hexamers.py` are a separate stream; whichever is
used, the realised arrays go in the manifest, so the h5 stays self-describing either way.

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
| **2** | 1–3 | cached `(L, gc_bin)` `predict` LUT **first**, then capture-surface fit, `marginal_fl`, per-region precompute | **COMPLETE** — `439d135`, `f7b10a3`, `277dc39` |
| **3** | 5–6 | sampler; 8-column BED per sample; `build-fragments-h5` → **h5 + manifest**. **No store** (decision 79) | **COMPLETE** — `3ea7947`, `3bd1775` (all 5 review findings), `f328862` (two-region weight test) |
| **4** | 7–8 | anchors + model scoring — **MOVED to the model agent** (decision 79b). Specified above only as the contract the simulator's output must support | OUT OF SCOPE HERE |
| **5** | — | smoke run: a few hundred regions, one sample, all invariants asserted | **COMPLETE** — `8eb9fe9` (provenance enforcement), `b3955b0` (driver). Passed; commits verified against `git log` after the agent returned `status=timeout` |

**Phase 5 result (2026-09-30).** Six assertions measured over 300 regions; worst
`|w_manifest - w|` was 4.441e-16, i.e. the manifest's factored reconstruction of `w`
agrees with a direct build to the last bit. Separately confirmed that `8eb9fe9`
**changed no computed result** — the same 300-region run is byte-identical before and
after it, so that commit added *enforcement* only. This is the load-bearing check on
decision 82's intent.

**A fix round follows Phase 5 and is IN FLIGHT as of 2026-09-30** (owner decision 83,
"fix these"): three EM-critical-review issues plus one architecture change. The
provenance subsection of Step 6 below is therefore **known-stale and deliberately not
yet reconciled** — see the note there.

**Decision 79 rescoped this plan after Phase 2.** The simulator's deliverable is the
fragment h5 plus the reconstruction manifest; store construction, the choice of `D`,
and the oracle belong to the model agent. Phase 3 shrank accordingly and Phase 4
largely left this stream, so Layer 1 here is Phases 1, 2, 3, 5.

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
  **This requirement has now been half-applied three times, which makes it the
  project's most-repeated defect — treat it as the first thing to check, not a note.**
  (i) Decision 81 keyed the hexamer tables by the actual hexamer string, which defends
  against k-mer *ordering* drift but leaves a swapped table *slot* just as undetectable:
  swapping `start_fwd` ↔ `end_fwd` through the round-trip is accepted silently, and
  `hex_table_to_dataframe` takes a `table_name` it never uses. The 9/9 mutation test
  covers orientation wiring, not this mutation. (ii) The `region_weights=` fast path
  added to `draw_fragments_for_region` documents in its own docstring that weights built
  from a *different* region "would sample from one region's weights while reporting
  another's coordinates, and no normalisation invariant would notice" — the hazard is
  named in prose and left unenforced. Both are in the decision-83 fix round.
  The pattern to recognise: an invariant that is a property of *one* object (the
  weights) cannot police how that object is *paired* with another (the region).
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
