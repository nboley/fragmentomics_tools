# Simulator spec

## 1. Inputs

| # | Input | Source |
|---|-------|--------|
| 1 | Fragment h5 | manifest key `ibd/frag_h5s/<SEQRUN>/<SAMPLE>-Lib1.hg38.fragments.h5` |
| 2 | Regions BED | `quiet_v2_pad1200_repeats_removed_tile1536.bed` |
| 3 | Reference FASTA | nominally `s3://karius-biomarker-data-assets/pipeline-assets/hg38/sequence.fa.bgz` — **but see below; no run has used it** |

Manifest: `~/src/biomarker-projects/tf_binding_site_classification/projects/ibd/manifests/ibd.data_manifest.tsv`
— 763 h5, mirror `s3://karius-biomarker-data-assets/projects`.

**The FASTA actually in use is GRCh38.p12**, not the asset named above
(owner, 2026-10-07, "accept p12 for now"):
`/efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz`. The
canonical `sequence.fa.bgz` is not present at any local path checked, and the
only local file of that name belongs to another user and has no `.fai`.
Coverage is fine — all 23 BED contigs are present — and GRCh38 patch releases do
not alter primary assembly sequence, though **that last point was not verified
here**.

**The consequence is provenance, not sequence.** The store's `config_hash`
hashes the reference `.fai`, so a store built against p12 will not match one
built against `sequence.fa.bgz` **even where the sequence is byte-identical**.
p12-derived artifacts are therefore not interchangeable with the rest of the
pipeline's. Record the reference path and its `.fai` md5 in any output, so a
later switch is detectable rather than silent, and re-raise this before
comparing a p12 artifact against a pipeline one.

### Regions BED options

In the main checkout's gitignored `data/region_sets/`.

| BED | Regions | Tile | Repeats |
|---|---:|---|---|
| `quiet_v2_pad1200_repeats_removed_tile1536.bed` | **66,649** | 1536 | removed |
| `quiet_v2_pad1200_repeats_removed_tile2560.bed` | 11,505 | 2560 | removed |
| `quiet_v2_pad1200_repeats_kept_tile1536.bed` | 904,975 | 1536 | kept |
| `quiet_v2_pad1200_repeats_kept_tile2560.bed` | 523,604 | 2560 | kept |

## 2. Build

Four stages. **Stages 1, 2 and 4 share one pass over the h5** — all three come
off the same attached frame, which is why `count_sample` returns that frame
rather than discarding it. Stage 3 consumes stage 2.

1. **Observed hexamer counts** `C(h)` — cut-site 6-mers of real fragments.
   Per sample.
2. **Fragment-length distribution** `f(L)` — `FragmentLengthDist`, built from
   the SRDF after §3 admission, so the same population `C(h)` counts.
   Per sample.
3. **Uniform hexamer counts** `N(h)` — the same tabulation as stage 1 with
   every start site equally likely. `N_start` is sequence-only, so per region
   set; `N_end` is `f(L)`-weighted.
4. **Region counts** — fragments per region.

## 3. Admission

A fragment is admitted to a region when, in the order `filter_fragments`
applies them:

1. `min(mapq_read1, mapq_read2) >= 10` — at fetch, not in the callback
2. it survives dedup on `(start, stop)`
3. its **length** lies in `[L_MIN, L_MAX]` = `[25, 180]`
4. its **start** lies in `[gstart, gstop)`

**Only 1-before-2 is load-bearing.** Two fragments sharing `(start, stop)` with
mapq 5 and 30 survive differently depending which runs first, because the dedup
keeps the *first occurrence*, not the best-mapped one. 3 and 4 commute with each
other and with 2 — they are independent per-fragment predicates, and any two
fragments sharing `(start, stop)` necessarily share a length and a start, so
dedup cannot reorder them. The list is in code order anyway, so the two cannot
drift.

### Sequence frame

    left_pad  = HEX_HALF = 3
    right_pad = max_fl + HEX_HALF = 183

A hexamer index then equals its region-local coordinate. `max_fl` is `L_MAX` for
the fragment pass and `fl.max_fl` for the uniform pass. After §3 admission
`fl.max_fl <= L_MAX`, with **equality only if a fragment of exactly `L_MAX` was
observed** — `FragmentLengthDist.from_srdf` takes the maximum *observed* length.
At real depth the two coincide; they are not equal by construction.

## 4. Sample

`n` comes from the region counts. Per fragment:

**0. Strand** — Bernoulli. It selects which tables the next two steps use,
because a minus-strand fragment's genomic start is its 3′ end:

| | genomic start `i` | genomic stop `i + L` |
|---|---|---|
| plus | `r_start_fwd(hex_fwd(i))` | `r_end_fwd(hex_fwd(i+L))` |
| minus | `r_end_rev(hex_rc(i))` | `r_start_rev(hex_rc(i+L))` |

**1. Start** — genomic start `i` over `[gstart, gstop)`, weighted by the left
column and normalised **within the region**:

    p[i] = r_start(hex(i))        for i in [gstart, gstop), 0 if hex(i) invalid
    P(i) = p[i] / p.sum()

**2. Stop** — given `i`, build the weight vector over lengths and normalise it
over the **valid** entries only:

    w[l] = f(l) · r_end(hex(i + l))     for every l in f's support,
                                        0 unless the fragment is VALID
    P(l | i) = w[l] / w.sum()

The support is `f`'s own, `[fl.min_fl, fl.max_fl]`, and it is bounded by
`[L_MIN, L_MAX]` **because §3 admission produced it** — not because the sampler
re-checks the constants. It does not know them. See Settled, "the length bound
is enforced once".

### When a length weight is zero

`w[l]` is a product of three factors, and any one of them zeroes a length. This
is the canonical statement; everything else in this document refers back here.

| factor | zero when | property of |
|---|---|---|
| `valid` | the end hexamer contains a non-ACGT base | the **reference** |
| `r_end` | `C(h) = 0` — the hexamer was never observed as a cut site in this sample | the **sample**, and its depth |
| `f(l)` | the length is absent from the distribution and densified to zero | the **sample** |

Two consequences:

- The denominator is **per start**. Two starts in the same region normalise over
  different length sets, so `w.sum()` is not a region constant. And because two
  of the three factors are sample properties, *the same region normalises over
  different length sets for different samples.*
- The code **cannot distinguish the three causes** — it tests `w.sum() > 0`. So
  "a length is valid when both cut-site hexamers are ACGT-only" is necessary but
  not sufficient, and a zero cannot be attributed to a cause after the fact.

## 5. Propensity

A hexamer's weight is observed over expected-under-uniform:

    r(h) = C(h) / N(h)

§4 normalises it per region for starts and per start for stops.

`C` and `N` are one tabulation counted two ways, so both cover the same
candidate positions. **They do NOT use the same validity gate**, and the
earlier claim here that they did was wrong: `C` requires BOTH of a fragment's
cut-site windows to be ACGT-only, while `N_start` gates only its own window.
Unmeasurable on the current region set — 0 invalid cut windows in 1,373,600 over
800 regions, and 4 of 66,649 regions hold any non-ACGT base — but it is a real
asymmetry and it bites a region set with gaps near cut sites. Not yet an owner
decision either way.

`N_start` weights every position 1; `N_end` weights by `f(L)` per route, which
per position is

    w(i) = F(min(i, max_fl)) − F(max(min_fl − 1, i − region_len))

for the `FragmentLengthDist` CDF `F` over its own support. `f` reaches only the
edge ramp: interior positions carry weight 1 under any normalised `f`.

### Which expectation each table divides by

**This is the pairing, stated explicitly, because leaving it implicit is what
allowed a live defect to sit here undetected.** The previous wording said only
that "the `_rev` tables divide by the permuted expectation,
`N_rc == N_fwd[rc_permutation()]`" — true but silent on *which* of
`N_start`/`N_end`, which is precisely where the bug was.

| table | tallies | divides by |
|---|---|---|
| `start_fwd` | genomic starts | `N_start` |
| `end_fwd` | genomic stops | `N_end` |
| `start_rev` | genomic **STOPS** | `N_end[perm]` |
| `end_rev` | genomic **STARTS** | `N_start[perm]` |

The `_rev` rows are **crossed relative to their names, on purpose.** Table names
are MOLECULE-relative — `start` means the molecule's 5′ cut site — but a
minus-strand fragment's 5′ cut site sits at its genomic STOP. Denominators must
be POSITION-relative, so the table applied at genomic starts needs the start
expectation whatever it is called.

This is not a relabelling: `N_start` weights every position 1 while `N_end`
carries the edge ramp above, so the two differ for every position within
`max_fl` of a region edge — 11.7% of a 1536 bp tile. Getting it backwards cost
a measured median 1.2% and max 16% error per `r` cell on real regions, silently,
with every total intact. Fixed 2026-10-07; `[perm]` applies because the `_rev`
tables index reverse-complement hexamers, which is a relabelling of the same
position.

## 6. Code

`background_model/simulator/count_hexamers_rdf.py`. Self-contained: stdlib,
numpy, pandas, `fragmentomics_tools.dataframe`.

| Function | Does |
|---|---|
| `filter_fragments` | dedup, length filter, admission. All filtering |
| `cut_site_hexamers` | per region → `start_hex`, `stop_hex`, `strand` |
| `counts_from_hexamers` | genomic start/stop + strand → the four tables |
| `count_srdf` | an attached frame → `C(h)`, per-row admitted counts, stats. Asserts only that BOTH strand tables are non-empty — there is no balance check, see Settled |
| `count_sample` | stages 1, 2 and 4 end to end → `(C(h), region_counts, stats, srdf)`. **Returns the frame**, which `f(L)` and the sampler both need |
| `simulate_fragments_to_bed` | draws for every region → the 8-column BED that `build_fragments_h5` consumes. Sorts its output; `n_drawn < n_requested` is reported, not raised |
| `FragmentLengthDist` | `counts`, `densities`, `min_fl`, `max_fl`, cached CDF |
| `uniform_hexamer_counts` | → `N(h)` |
| `fl_end_weight` | `w(i)` above |
| `propensities` | `C / N` |
| `sample_region` | draw `n` fragments → `starts_0`, `lengths`, `is_plus` |
| `hexamer_indices` | sliding 6-mer encode; `str`/`bytes`/`uint8`, case-folded |
| `rc_permutation` | RC as a 4096 permutation, derived from the encoder |

`C(h)`: `attach_fragment_arrays(callback=filter_fragments)` →
`attach_sequence` → one `parallel_apply` of `cut_site_hexamers` → four
`bincount`s.

`N(h)`: serial sequence walk.

`FragmentLengthDist` is built by `from_dataframe` (columns
`fragment_length`, `count`; absent lengths densify to zero) or `from_srdf`,
which delegates to it. Its three arrays are read-only, since an in-place edit
would leave the cached CDF stale.

`sample_region` is vectorised over fragments: all `n` in a region share its
arrays, so the length draw is one `(n, n_lengths)` block.

## Open

Needs action. Nothing here has been decided.

- **4 mutations from the test design have no test**: M12 (dedup before MAPQ),
  M15 (midpoint instead of start-in-region admission), M24 (which of the three
  length-weight factors zeroed), M25 (end-hexamer offset). Scoped out of the
  implementation on purpose, so this is a known gap rather than an oversight —
  but admission order and the length draw are both places a silent error would
  live, so it is the most valuable remaining test work.

- **`tests/conftest.py` cannot exist.** Its module name collides with
  `test/fragment_array/conftest.py` under pytest's `prepend` import mode, so the
  implementing agent deleted it and registered markers inline. That is a
  workaround: the collision returns the moment anyone re-adds the file, which is
  an ordinary thing to want. The fix is `importmode = "importlib"` in
  `pyproject.toml`, which changes pytest behaviour repo-wide and so needs its
  own validation run.

- **`C` and `N` disagree on the validity gate** (§5). `C` needs both cut sites
  valid, `N_start` only its own. Unmeasurable on this region set; an owner
  decision either way is still open.

- **Second copy of the encoder** in `simulator/precompute.py`, kept until the
  old simulator is deleted (deferred: ~8 files still import it, some belonging
  to another stream). The drift is now actually guarded —
  `test_encoder_matches_oracle_all_4096` checks all 4096 against an INDEPENDENT
  oracle, which is stronger than checking the two copies against each other
  since those could drift in step. Measured 2026-10-07: they agree exactly.
  **What remains is a naming lie**: `count_hexamers_rdf.py` still cites
  `test_encoder_matches_precompute`, which does not exist. Delete that citation.

### Closed since this section was last accurate

Kept briefly because the Open list claimed all three for longer than they were
true, and a reader who saw it mid-day would have acted on stale information.

- ~~**No tests.**~~ 47 now reference `count_hexamers_rdf` (`844f227`, `45b32ec`,
  `d9c6e90`) — 43 plus an independent oracle, plus 4 propensity tests. Mutation
  tested: 18 mutations applied programmatically, all now caught.
- ~~**`attach_sequence` near a contig end is UNVERIFIED.**~~
  `test_contig_ends_raise` and `test_frame_through_attach_sequence` pin it: it
  raises, and the padded length is asserted exactly, which pins both pads.

---

## Settled

Decided. Recorded so none of it is re-opened as an oversight — if something here
looks like a defect, read the reason before changing it.

- **Bulk GC is not modelled.** The length marginal matches by construction; the
  `(length, GC)` joint does not. Accepted consequence of dropping the capture
  model, not a defect.
- **The length bound is enforced ONCE, at fragment-array construction** — not
  downstream (owner, 2026-10-07). `filter_fragments` runs as the
  `fragment_array_callback`, so `subset_fragment_lengths(l_min, l_max + 1)` has
  already applied `[L_MIN, L_MAX]` before anything else sees the data. `f(L)`
  is then built from that frame, so its support is bounded by construction.

  Consequence, intended: `sample_region` draws over `[fl.min_fl, fl.max_fl]`,
  **`fl`'s own support, and does not know `L_MIN`/`L_MAX` at all.** §4's
  `[L_MIN, L_MAX]` describes where the bound comes from, not a second check the
  sampler performs. Do not add one — a downstream re-check is the two-
  implementations-of-one-rule failure this repo keeps hitting.

  The contract that makes this safe: **`fl` must come from a filtered frame.**
  An `fl` built from an unfiltered one is out of contract. The frame has one
  index of slack — the hexamer track is `region_len + 181` long and the furthest
  end index is `region_len - 1 + fl.max_fl` — so `fl.max_fl = 181` reads one
  past the intended frame silently, and `>= 182` raises `IndexError`.

- **`p_plus = 0.5`, by construction rather than by fitting** (owner,
  2026-10-07). A fragment is double-stranded and has no intrinsic orientation;
  the strand label records which of its two ends became read 1, and adapter
  ligation is symmetric, so the label is a fair coin independent of sequence.
  There is nothing to measure and no producer is needed.
  `count_srdf` **asserts only that BOTH strand tables are non-empty**, and
  reports `n_plus` / `n_minus` / `plus_frac` without asserting them. That one
  check needs no threshold and catches the silent failure that matters:
  `fragment_strands` is `<U1`, so comparing it against `b'+'` yields an
  all-False mask and an *entirely empty* strand table, after which
  `sample_region` skips that strand without error and the simulator emits
  strand-pure data.

  **There is no balance check. Removed by owner decision 2026-10-07; do not
  reintroduce one without naming a failure mode the non-empty check misses.**
  Three measured reasons: a wholesale plus/minus swap maps `plus_frac` to
  `1 - plus_frac` and preserves every total, so it was blind to the error it
  appeared to guard; the real-data fraction is **overdispersed** relative to
  independent fragment draws (0.3758 / 0.5296 / 0.5041 at 10 / 200 / 2,000
  regions, the first two deviating in *opposite* directions, mechanism never
  measured), so it false-positived on a legitimate 10-region run; and rescaling
  it to `n_regions` to fix that made it vacuous below 10 regions, since
  `|plus_frac - 0.5| <= 0.5` by construction.

  Also note `start_fwd.sum() == end_fwd.sum()` and the `rev` pair are
  **tautologies**, not routing protection — measured: feeding
  `counts_from_hexamers` a start/stop swap leaves both true, because each sum is
  just that strand's row count however the hexamers are routed. They stay only
  to catch a malformed `counts` dict from outside this module.

  Scope of that guard, so it is not mistaken for more than it is.
  `sample_region` skips a whole strand block on `if tot <= 0: continue`, with no
  error. That cannot be *sequence*-driven: `valid` is shared across strands —
  one array, with only `track` switching between `fwd` and `rc` — so an N-driven
  zero hits both strands alike and is just the deferred shortfall below. A
  strand-*specific* zero needs the table itself dead, all 4096 entries, meaning
  no fragments of that strand were counted, which is precisely what the
  assertion raises on. The only remaining path is an `r` that did not come from
  `propensities` on counts measured in the same run — hand-built, or loaded from
  a stored artifact. Not reachable from this pipeline; worth knowing if the model
  agent ever loads stored tables and calls `sample_region` directly.

- **Starts are drawn WITH replacement, and the resulting dedup is correct**
  (owner, 2026-10-07). The real assay has no UMIs, so a genuine duplicate
  coordinate pair is indistinguishable from a resampled one there too. The
  simulator reproducing that is faithful, not a defect. Measured cost of the
  collisions is ~0.0075% at the 1536 tile; see the dedup note in the
  coordination file.

- **Draw order is not part of the contract.** `sample_region` returns the plus
  block then the minus block, unsorted within each. Output goes straight into
  fragment-h5 construction, which sorts before `bgzip`/`tabix` regardless.

- **A start with no valid length is dropped**, so that region yields fewer than
  `n` fragments. `P(start)` does not condition on a valid fragment existing.

  **OBSERVED, on the first real run — this said "not observed" until
  2026-10-07.** A 10-region run requested 322 and drew 320, one region short.
  The earlier 38,637-for-38,637 over ~1000 regions also stands; both are true,
  and the difference is the useful part:

  **The shortfall rate tracks how sparsely `r(h)` is estimated, not the
  sequence.** At 10 regions only 108-151 of 4096 cells were nonzero, so most end
  hexamers carried `r = 0` and whole length sets vanished — the `r_end` factor
  in §4, not the `valid` factor. It therefore shrinks as the region set grows,
  and **a shortfall on a small run is expected rather than a defect.** Do not
  chase one without first checking how many `r` cells are populated.

  **DEFERRED by owner, 2026-10-06: documented only. No counter, no test.**
  Recorded so the next reader does not re-open it as an oversight.

  The region frame is **not** the cause, which is the first thing everyone
  asks. `right_pad = L_MAX + HEX_HALF = 183` is exactly sufficient: the
  furthest end hexamer a drawn start needs sits at region-local
  `region_len - 1 + L_MAX` and reads through `region_len + 182`, which the
  flank provides. There is no truncation and no out-of-range read. The start
  side is independently safe — starts are gated on `valid`, so a start whose
  own hexamer is invalid has weight 0 and is never drawn.

  A drop therefore needs **every length in `f`'s support zeroed at once** — see
  §4, "When a length weight is zero", for the three factors. On repeats-removed
  quiet tiles that needs a ≥183 bp N-run immediately downstream, or that many
  consecutive hexamers all unobserved in the sample. Hence the clean 38,637.

  **The partial case is the larger exposure, and it is not a defect.** A start
  that loses *most* of its lengths does not drop; it reshapes `P(l | i)`, which
  §4 licenses. Nothing reports it. A shallow sample with many `C(h) = 0` cells
  therefore samples lengths from a quietly narrowed support, and the only
  symptom is that the length marginal drifts from `f(L)`. If a future
  length-marginal check fails without an obvious cause, measure the reachable
  fraction of `f(L)` per start before looking anywhere else.
