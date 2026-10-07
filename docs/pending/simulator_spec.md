# Simulator spec

## 1. Inputs

| # | Input | Source |
|---|-------|--------|
| 1 | Fragment h5 | manifest key `ibd/frag_h5s/<SEQRUN>/<SAMPLE>-Lib1.hg38.fragments.h5` |
| 2 | Regions BED | `quiet_v2_pad1200_repeats_removed_tile1536.bed` |
| 3 | Reference FASTA | `s3://karius-biomarker-data-assets/pipeline-assets/hg38/sequence.fa.bgz` |

Manifest: `~/src/biomarker-projects/tf_binding_site_classification/projects/ibd/manifests/ibd.data_manifest.tsv`
— 763 h5, mirror `s3://karius-biomarker-data-assets/projects`.

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
candidate positions and the same validity gate. `N_start` weights every
position 1; `N_end` weights by `f(L)` per route, which per position is

    w(i) = F(min(i, max_fl)) − F(max(min_fl − 1, i − region_len))

for the `FragmentLengthDist` CDF `F` over its own support. `f` reaches only the
edge ramp: interior positions carry weight 1 under any normalised `f`. The
`_rev` tables divide by the permuted expectation,
`N_rc == N_fwd[rc_permutation()]`.

## 6. Code

`background_model/simulator/count_hexamers_rdf.py`. Self-contained: stdlib,
numpy, pandas, `fragmentomics_tools.dataframe`.

| Function | Does |
|---|---|
| `filter_fragments` | dedup, length filter, admission. All filtering |
| `cut_site_hexamers` | per region → `start_hex`, `stop_hex`, `strand` |
| `counts_from_hexamers` | genomic start/stop + strand → the four tables |
| `count_srdf` | an attached frame → `C(h)`, per-row admitted counts, stats. Asserts strand balance and the start/end identities |
| `count_sample` | the three passes end to end → `(C(h), region_counts, stats, srdf)`. **Returns the frame**, which `f(L)` and the sampler both need |
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

- **No tests.** Not "thin" — *none*. No test anywhere references
  `count_hexamers_rdf`. `sample_region` is the worst place for that, because its
  failure modes are silent: swapped 5′/3′ strand pairing inverts the asymmetry
  the four untied tables exist to capture, and an off-by-one in the padded frame
  reads a neighbouring hexamer while leaving every total plausible.

- **`attach_sequence` near a contig end is UNVERIFIED.** The frame in §3 assumes
  183 bp of right flank is always obtainable. A region within 183 bp of a contig
  end cannot supply it, and nothing has established whether `attach_sequence`
  pads, truncates, or raises there. `uniform_hexamer_counts` is documented as
  raising on a truncated fetch, but the sampler's path has not been checked.
  Verify with a region placed within 183 bp of a contig end on the test FASTA.
  A truncated sequence would shift every hexamer index silently, since the whole
  frame rests on `left_pad = HEX_HALF` making an index equal its coordinate.

- **Second copy of the encoder** in `simulator/precompute.py`, kept until the
  old simulator is deleted (deferred: ~8 files still import it, some belonging
  to another stream). **The guard named for this drift does not exist** —
  `count_hexamers_rdf.py` cites `test_encoder_matches_precompute` as what keeps
  the two from "diverging silently", and no such test is in the repo. Either
  write it or delete the claim; a comment asserting a guard that isn't there is
  worse than silence.

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
  `count_srdf` nevertheless **asserts** the observed fraction is within
  `strand_tol` (default 0.1) of 0.5, because the failure it catches is silent:
  `fragment_strands` is `<U1`, and comparing it against `b'+'` empties a strand
  table, after which `sample_region` skips that strand without error and the
  simulator emits strand-pure data. It also checks `start_fwd.sum() ==
  end_fwd.sum()` and the `rev` pair, which are exact identities and so catch
  broken strand routing that no aggregate total can reveal.

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
  Not observed on the 1536 tiles — 38,637 drawn for 38,637 requested over ~1000
  regions, reported by a prior session and not re-measured since — but it is a
  silent shortfall where it does occur.

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
