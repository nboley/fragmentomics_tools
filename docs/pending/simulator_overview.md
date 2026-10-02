# cfDNA fragment simulator — high-level overview

Orientation document for `background_model/simulator/`. Describes what the
simulator does, where its parameters come from, and which parts are real
versus placeholder. Every claim here was read out of the code rather than
recalled; line numbers rot, so re-check those, but treat the described
behaviour as current as of this document's commit.

The statistical specification lives in `docs/pending/simulator_and_fragment_nll.md`.
This document does not restate it — it orients a reader who needs to know how
the pieces fit before reading that.


## What it produces

Given a region set and a reference, the simulator emits synthetic cfDNA
fragments whose positions and lengths follow a known generative law. Because
the law is known exactly, the simulated data provides an **oracle** — a floor
on achievable fragment-level NLL — against which a model's share of the
capturable signal can be measured.

Output today is a fragments h5 plus a manifest recording the **realised**
parameters (not the recipe that produced them), so a run is interpretable
without re-executing it.


## The generative law

A fragment is `(c5, c3, strand)`, expressed in **cut sites, not bases**: a
fragment occupying `[p, p+L)` has cut sites at `p` and `p+L`, not at `p` and
`p+L-1`.

| strand | 5′ cut site | 3′ cut site | hexamers read from |
|---|---|---|---|
| `+` | `c5 = p` | `c3 = p + L` | `hex_fwd` (both ends) |
| `-` | `c5 = p + L` | `c3 = p` | `hex_rc` (both ends) |

On the minus strand the 5′ end sits at the **higher** coordinate. Getting this
backwards is the single most common way to misread the simulator's output, and
it has cost real debugging time: an attempt to map stored `(start, length)`
pairs onto `(c5, c3)` in store order produced hexamer tables that correlated
with their empirical counterparts on the plus strand and not at all on the
minus.

The weight of a fragment within one region:

```
E_s(c5, L) = end_s[hex_s(c3)] · marginal_fl[L] / predict_lut[L, gc_bin(c5,c3)]
Z_s(c5)    = Σ_L E_s(c5, L)                        # per strand, per 5′ site
S_s        = Σ_{c5 : Z_s(c5) > 0} start_s[hex_s(c5)]   # per strand, per region
w(c5,L,s)  = ½ · start_s[hex_s(c5)]/S_s · E_s(c5,L)/Z_s(c5)
```

**Normalisation is per region**, not global: `Σ w = 1` within a region and each
strand marginal is exactly `½` by construction. There is no single global
normaliser, and `Z_s(c5)` varies by position, so it does not factor out.

Four hexamer tables — `start_fwd`, `end_fwd`, `start_rev`, `end_rev` — are
**untied by design**; strand selects which `(start_s, end_s)` pair applies.
`HexamerTables` binds them by name specifically to prevent misrouting.


## Pipeline

| Stage | Module | What happens |
|---|---|---|
| Precompute | `precompute.py` | From the region's sequence: `hex_fwd` / `hex_rc` (cut-site hexamer index at every position), `cum_gc`, and a `valid` mask marking windows containing non-ACGT |
| Weights | `weights.py` | `build_region_weights` → two `(region_len+1, 156)` arrays, one per strand, fully normalised |
| Sample | `sampler.py` | Hierarchical draw: strand (coin flip) → 5′ cut site (`∝ start/S_s`) → length given that site (`∝ E/Z_s(c5)`) |
| Emit | `emit.py` | BED → sort/bgzip/tabix → `build-fragments-h5` → h5, plus the manifest |

The sampler derives its conditionals **from `w` itself** rather than from a
parallel expression. There is therefore only one implementation of the law,
and no second formula that can drift away from it. Any consumer that needs a
fragment's probability should import `build_region_weights` rather than
re-deriving it.


## Where the parameters come from

This is the most important thing to know about the simulator's current state.

| Parameter | Source | Status |
|---|---|---|
| `predict_lut` (156 lengths × 20 GC bins) | `capture.py::fit_and_build`, fitted from a real sample | **real** |
| `marginal_fl` (156 lengths) | same fit | **real** |
| four hexamer tables (4096 each) | `run_simulator.py::build_hexamer_tables` — four independent log-normal draws from one RNG, each normalised by its max | **synthetic placeholder** |

So the capture surface and the fragment-length marginal come from data, while
the **sequence bias does not**. Simulated fragments carry a plausible-looking
but invented hexamer preference.

Two consequences worth stating plainly:

- Any oracle computed against a current simulated store is the oracle of a
  *synthetic-bias* simulation. That is a valid test of plumbing and of a
  model's ability to learn a known law. It is not a statement about real
  fragmentation.
- Replacing `build_hexamer_tables` with real, shrunk per-sample tables is the
  substantive outstanding change (decisions 100/104, and 78 for using real
  tables at all).


## Per-region fragment counts

Layer 1 uses a **fixed count per region**, from a hardcoded lookup in
`sampler.py`:

```
_REGION_COUNTS = {2560: 54, 1536: 37}
```

Any other `region_len` raises `KeyError` — deliberately, so an unknown
geometry fails loudly instead of silently taking a default.

The two entries do not imply the same density:

| region_len | count | fragments/bp |
|---|---|---|
| 2560 | 54 | 0.0211 |
| 1536 | 37 | 0.0241 |

Both bracket the 0.0230 fragments/bp median measured across the 92-sample
hexamer cohort, so they are the right order of magnitude, but they differ from
each other by about 14%. Worth knowing before comparing runs at the two
geometries.


## Fragment length: three different banding concepts

These are easy to conflate — they share numbers without sharing meaning, and
conflating them has already produced one incorrect bug report.

1. **The simulator does not band length at all.** It works at full resolution
   over `L ∈ [25, 180]` inclusive — 156 individual lengths, indexed
   `li = L - 25`. `L_MIN`, `L_MAX` and `N_LENGTHS` in `weights.py` are
   authoritative for the cut-site path.

2. **`FL_BANDS = ((25,110), (110,180))`** — two half-open bands, defined in
   `background_model/tracks.py`. Used by the **12-track aggregation** path
   (`tracks.py`, `dataset.py`, `train.py`, `correction.py`, `preprocess.py`,
   `config.py`). The simulator *records* `fl_bands` as a manifest field and
   **never uses it in generation** — `weights.py` and `sampler.py` contain
   zero references to it.

3. **Sixteen 10 bp bands** — `(25,35)`, `(35,45)` … `(165,175)`, `(175,181)` —
   present in the cut-site hexamer counting parquets. A property of that
   counting path, unrelated to either of the above.

Because the half-open upper band of (2) ends at 180 while (1) includes 180,
the shared number invites the conclusion that one of them is off by one. It
does not follow: they describe different things, and inclusive `L_MAX = 180`
is correct for the generative domain.


## Feeding in an external hexamer table

The deserialisation machinery already exists and is exercised on every run:

- `hex_tables_to_dict` / `dict_to_hex_tables` (`emit.py`) convert between
  `HexamerTables` and a dict of DataFrames keyed by the hexamer string.
- `load_manifest` reconstructs the four tables from a written manifest, and
  the driver uses that path for its round-trip assertion.

What is missing is only the **input wiring** — there is no CLI argument to
supply tables from an external artifact, so `run_simulator.py` synthesises
them instead. Swapping the synthetic call for a loader is therefore a small
change that reuses existing, tested serialisation rather than adding a second
representation.


## Validation, and its limits

Each run asserts: `Σ_Ω w = 1`; strand marginal exactly `½`; emitted count
equals expected; fragments read back from the h5 are contained in their
regions; and the manifest round-trips to the same tables.

**All of these are internal-consistency checks.** They establish that the
sampler is faithful to `w` and that the plumbing is correct. None of them can
distinguish a right `w` from a wrong one — they pass identically under
different parameter semantics. Anything that needs to validate the *parameters*
requires a different kind of check: realised-versus-real comparison on actual
data, not a self-consistency assertion.

Two checks that *do* bite, for a consumer scoring fragments against `w`:

1. No observed fragment may land on a `w == 0` cell. Zero means edge-truncated
   or N-masked, so the simulator could not have emitted it — a single hit
   proves the coordinate mapping is wrong, with no statistics required.
2. The exact per-region entropy `−Σ w log w` must match the mean of
   `−log w` over emitted fragments, within Monte Carlo error at the per-region
   count.


## Interface

Implemented today:

```
--region-set-bed  --region-set-name  --fasta  --sample  --n-regions
--out-dir  --seed  --w6-seed  --w6-dynamic-range  --n-roundtrip  --workers
```

Writes four files — `{sample}.fragments.h5`, `.manifest.json`,
`.gcfl_model.json`, `.smoke_summary.json` — plus `bed.gz` / `.tbi`, which
currently persist.

Decided and not yet implemented: two region-bed inputs so the hexamer and
simulation region sets can differ (101); a single metadata json absorbing the
manifest, the capture model and the assertion summary (102); `bed.gz`/`.tbi`
demoted to internal temporaries (103); real inputs and in-tool table building
(100); the pooled prior as a separate input artifact (104); and a per-fragment
weight sidecar (114/116), which makes the output contract three files rather
than the two that 103 specifies.


## Map

```
background_model/simulator/
  precompute.py   hexamer indices, cumulative GC, validity mask
  weights.py      the generative law; HexamerTables; domain-size helpers
  sampler.py      hierarchical draw; per-region target counts
  capture.py      GC/FL capture-surface fit
  emit.py         BED writing, h5 construction, manifest read/write
scripts/run_simulator.py   driver
```

Tests: `tests/test_simulator_weights.py` (31), `tests/test_simulator_phase2.py`
(29), `tests/test_simulator_phase3.py` (62).

Run suites with `make test` and the conda env's `bin/` on `PATH`, never bare
`pytest` — bare `pytest` manufactures two failures that do not exist, from a
missing `flgc` import and absent `bgzip`/`tabix`.
