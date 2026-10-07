# Test design: `background_model/simulator/count_hexamers_rdf.py`

**Status (2026-10-07): DRAFT v4 for owner approval. Nothing here is implemented.**
Rebased on HEAD `0820409` (parents `844f227`, `a409186`) [Verified: `.git` reflog]. The premise
"the module has zero tests" is **wrong** now: `844f227` added 4 tests
(`tests/test_simulator_propensity_denominators.py`) and fixed F1. Those pin the pairing only.
The rest of this design is still needed. v3 adds the F1 status, the propensity test file
(§1.3), review round 1's contract fixes, and a site column in the mutation matrix. **v4
addresses round 2's review. HEAD moved underneath it, `0820409` → `b50ab4f`** [Verified:
`git log -3`], fixing F2/F3/F13/F4/F18/F19 and shifting the module by +26 lines; every cite
this revision wrote was re-grepped there, but untouched cites may still carry old numbers.
This design has two parts. First, it proposes tests. Second, it reports findings that need
an owner decision. It does not change module code. Authority: `docs/pending/simulator_spec.md`.
`attic/` is quarantined and not current. Evidence tags:
- `[Verified: …]`: measured or read in the research for this design.
- `[Verified: review]`: review round 1 measured it. This revision did not re-run it.
- `[Draft measurement]`: measured in the session that wrote v1. The later research did not
  re-verify it.
- `[Inferred]`: derived from that evidence.
- `[Unverified: …]`: names the check that settles the claim.

## TL;DR

- **Fixture split.** Each test uses the cheapest data that gives it a checkable answer (§2.4).
  - *Synthetic* for frame, routing, admission, null identity, sampler and validity. Every
    such h5 is built at test time from a text BED, against a ~6,000-bp synthetic contig
    `chrT`. Its core is a **de Bruijn sequence B(4,6)**, so every hexamer occurs once.
  - *Real data* (tier R, marker `real_data`) where real formats are the thing under test:
    R1 `count_sample`, R2 `uniform_hexamer_counts`, R4 a real-region h5 round trip on `chr21`
    only (replaces R3, §2.4/M1-M2) — the **one** test that builds an h5 against the real FASTA
    (~11 s, `chr21`-restricted); every other h5 here is the toy or the committed golden fixture.
  - *Committed*: the existing chr6 FASTA and golden h5 serve two always-run smoke tests.
    One new ~1 KB file holds ~20 real regions as a BED (10 on `chr21`, the rest elsewhere).
    No binary is committed.
- **Skips.** Only `real_data` tests may skip, and only on missing EFS data. `-rs` prints the
  reason. `REQUIRE_REAL_DATA=1` turns such a skip into a failure (§2.5). The owner's
  acceptance run uses it and must show 0 skipped. A missing pysam or fragments_h5 fails
  loudly.
- **Tiers.** T0 encoder/frame, T1 strand routing, T2 admission, T3 `count_sample` against a
  brute-force oracle, T4 expectation and propensity, T5 sampler, T6 BED/h5 round trip,
  T7 hygiene, R real data. Every oracle imports nothing from the module. Every test names
  at least one mutation that it must turn red (§4.2).
- **Finding F1 (fixed in `844f227`, owner-approved).** `propensities()` divided the minus
  tables by swapped denominators. The fix pairs `start_rev` with `N_end[perm]` and `end_rev`
  with `N_start[perm]` [Verified: `count_hexamers_rdf.py:838-841`]. `t4_null_identity` is the
  exact regression guard. Mutation M41 restores the old pairing and must turn two rows red.
- **Finding F2 (MEDIUM).** The docstrings say invalid windows carry index 0. They do not.
  `hexamer_indices` reads N as A inside the window [Verified: Research A]. A consumer that
  skips the validity gate miscounts into a *neighbouring* hexamer, not into cell 0.
- **Finding F17 (library, owner-gated).** A minus-strand region flip leaves the strand label
  unswapped. No test pins it until the owner rules (Q8).
- **Strand statistics.** For simulator output the unit is the fragment, conditional on the
  realised per-region strand counts. For real data the unit is the region, and no test
  asserts a real-data plus fraction. The real-data excess is overdispersion. Its mechanism is
  unmeasured (§4.4).
- **Runs where?** Only under `make test PYTEST_ARGS="tests/ -q"`, with `biomarker_env`
  first on PATH. Default `make test` does not collect `tests/` [Verified: Makefile:27-29].
  See Q3 and F16.

## 1. Ground truth

### 1.1 Baselines (worktree `background-model-work`, `biomarker_env`)

Measured at `a409186`. Re-measure at the implementation HEAD (P0). The four `844f227` tests
came later. If all pass, `tests/` reads 677 [Inferred].

| Command | Result | Source |
|---|---|---|
| `make test` (default `test/ fragmentomics_tools/`, `--doctest-modules`) | 2 failed, 399 passed, 3 skipped, 92.39 s, exit 2 | [Verified: coordinator log, at `a409186`] |
| `make test PYTEST_ARGS="tests/ -q"` | 673 passed, no skips reported, 218.63 s, exit 0 | [Verified: coordinator log, at `a409186`] |
| `make test` under the wrong python (claude-mcp, py3.13) | exit 2, 11 collection errors | [Verified: coordinator log] (F16) |

The 2 failures are the known missing-data cases (`test_slice_encode_big_wig`,
`test_get_one_hot_encoded_sequence`). The counts match the v1 draft. The worktree
CLAUDE.md states 4 failed / 386 passed / 3 skipped (2026-09-27). That line is stale against
these measurements [Verified: research file, decision 7].

The Makefile runs bare `python -m pytest` [Verified: Makefile:203-207]. So
`/home/nathanboley/miniconda3/envs/biomarker_env/bin` must be first on PATH. The Makefile
also prepends the stale `FLGC_PYTHONPATH` (F15). Default `make test` collects `test/` and
`fragmentomics_tools/` only, so it never runs `tests/` and never runs the doctests in
`background_model/`. Run directly, those doctests pass: `2 passed in 5.94 s`
[Verified: Research A].

Environment [Draft measurement]: Python 3.10.19, pysam 0.23.3, h5py 3.15.1. `fragments_h5`
imports from the editable checkout `/home/nathanboley/src/fragments_h5`. `bgzip`, `tabix`,
`bedtools`, `samtools` and `build-fragments-h5` are on PATH.

### 1.2 Measured facts this design relies on

- **Toolchain round trip** [Draft measurement, except where tagged]. Write an 8-column BED
  (`contig,start,stop,"",0,strand,mapq,mapq`, no header). Then run `pysam.tabix_index(bed,
  preset="bed", force=True)`, then `build_fragments_h5(gz, h5, fasta_filename=FASTA)`, then
  `count_sample`. `tabix_index` *consumes* the plain BED; only `X.bed.gz` and
  `X.bed.gz.tbi` remain. The h5 is 548,712 B for both 2 and 10 fragments. The build logs a
  harmless WARNING `fragment_end_clipped is unavailable in TSV/BED input`.
- **Build cost scales with contig length, not linearly** [Verified: Research C,
  `num_processes=1`, synthetic `chrT`]:

  | Contig | Fragments | Build |
  |---|---|---|
  | 6,000 bp | 40 | 0.15 s (process wall 0.706 s) |
  | 600 kb | 320 | 0.16 s |
  | 3 Mbp | — | 0.30 s |
  | 30 Mbp | — | 9.98 s |
  | real GRCh38 FASTA, `allowed_contigs=["chr1"]` | 10 | 76.26 s (sys 1m10.8 s, I/O-bound; cause not confirmed) |
  | real GRCh38 FASTA, `allowed_contigs=["chr21"]` | 10 | 10.99 s [Verified: review, `realbuild/rb.py`] — cost tracks contig length, not fragment count |
  | committed chr6 (170,805,979 bp) | 10 | 26.2 s [Draft measurement] |

- `RegionDataFrame(df, ref="hg38")` accepts contig `chrT`, and `count_sample(...,
  n_workers=1, verbose=False)` takes ~0.04 s on a 600-fragment toy [Draft measurement].
  Import of the simulator module costs 7.24 s wall and needs no flgc [Verified: Research C].
- **Oracle agreement** [Draft measurement]. Committed chr6 FASTA plus a hand-written
  10-fragment BED (an exact duplicate, MAPQ 5, lengths 24/25/180/181, a tile-straddler,
  both strands). `count_sample` returned region_counts `[5, 1]`, and all four tables equal
  a plain-python brute force. Read-back `fragment_strands` on the RegionFragmentArray is
  `<U1`.
- **Two strand encodings.** The raw reader `FragmentsH5.fetch_array(..., return_strand=True)`
  returns strand `|S1` **bytes** (`b"+"`/`b"-"`), and mapq `(n,2)` int32 [Verified: Research A;
  fragments_h5.py:680, inside `fetch_array` (def at `:473`, which does not itself show the
  dtype — L4)]. The RegionFragmentArray after attach holds `<U1` str [Draft
  measurement]. An oracle that reads raw records must decode.
- **Raw reader filters** [Verified: `fragments_h5.py:563-564, :638-640`, read in this revision].
  `fetch_array` defaults `max_frag_len` to the h5 maximum. It keeps `starts < region_stop`,
  `stops > region_start` and `lengths <= max_frag_len`, and nothing else. MAPQ, dedup and the
  length filter live in the library and the module. The oracle's 180-bp window on each side
  keeps every fragment that starts in the region.
- **Equal-key order** [Verified: review, one case]. `build_fragments_h5` kept BED order for equal
  `(start,stop)`: `(700,800,'+',5)` then `(700,800,'-',30)`. Dedup keeps the first occurrence
  (`fragmentomics_tools/fragment_array/fragment_array.py:1050-1052`, `np.unique(...,
  return_index=True)`) [Verified: read]. One case is not a law. The M12 fixture relies on it.
- **Frame** [Draft measurement]. With `left_pad=3`, the hexamer at region-local cut `c` is
  padded `seq[c:c+6]`, which is genomic `[g0+c-3, g0+c+3)`. Padded length = `region_len + 186`.
- **Contig ends** [Verified: review `proto.py`, 6,000-bp toy]. Region 5920-5990 raises
  `ValueError` "fetched 83 bases, expected 256 … the fetch was truncated". Regions 1-60 and
  0-60 raise "left_pad=3 runs off the start". The raise is in the library
  (`fragmentomics_tools/region.py:854-865`), not the module. Spec Open item 2 is answered:
  the library raises loudly.
- **Closed loop** [Draft measurement]. 6,000-bp toy, 600 synthetic fragments, 5 tiles of
  1,000 bp. `count_sample` gave `[136,135,135,108,86]`. Simulation gave `n_drawn` 600 and
  `n_rows_written` 600. The recount gave `[134,135,134,108,86]` = 597. The 3 missing rows are
  duplicate `(start,stop)` draws: starts are drawn with replacement, and the dedup key omits
  strand (both Settled). **The exact invariant is: recounted region_counts == number of
  distinct `(start,stop)` per region among the written rows.** It is not `n_drawn`.
- **chr6 FASTA** [Verified: Research C]. `.fai` = `chr6 170805979 6 170805979 170805980`.
  The window `chr6:99,110,000-99,130,000` holds 20,000 bases (A 6177, C 4143, G 3881,
  T 5799), no N, no lowercase. Outside the window is N (sparse 1 Mb-step check).
- **Real data** [Verified: Research C]. FASTA
  `/efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz` (938,986,662 B,
  `.fai` 593 lines, `.gzi` present). Real h5 groups chr1-22, X, Y, M; names match the FASTA
  and the BED. Region BED: 66,649 rows, 3 columns (no strand), chr1-22 and chrX. On 20
  regions: h5 open 0.12 s, FASTA open 0.01 s, fetch 0.004 s (30,720 bases),
  `count_sample` 1.83 s (689 counted), `uniform_hexamer_counts` 0.03 s, 0 invalid starts.
  The real h5 path is not in the research record [Unverified: pin it in P6].

### 1.3 Existing tests: old generation, do not copy

| File | Status |
|---|---|
| `tests/test_simulator_weights.py`, `_phase2.py`, `_phase3.py` | Old generation (capture/GC/flgc/ZTNB). They import `simulator.weights/precompute/capture/sampler/emit`. Phase3 `_require_fasta()` (`tests/test_simulator_phase3.py:893-913`) calls **`pytest.fail`** on a missing `/efs` hg38 path. Its docstring says this is deliberate. Phase2 `:451` reads the same path with **no guard** [Verified: Research C]. |
| `tests/test_sim_fragments.py` | Does not exist. Only a stale `.pyc` remains [Draft measurement]. |
| `tests/test_simulator_propensity_denominators.py` (`844f227`, 4 tests) | Pins the pairing. `test_the_rev_tables_are_not_swapped` names the swapped answer. All four use the module's own `rc_permutation()` (`:56`, `:87`, `:115`, `:128`). The kept one asserts a negation (the pairing is NOT swapped), so it still detects M41 [Inferred]; the other three re-assert the module. `test_permutation_is_an_involution` duplicates `t0_rc_permutation_matches_oracle`. **Decision:** keep `test_the_rev_tables_are_not_swapped`. `t0_rc_permutation_matches_oracle` and `t4_null_identity` supersede the rest. Deleting the rest is an owner call (Q1). The new suite does not depend on that call. |
| `tests/test_count_cut_site_hexamers.py` | Tests the *older script* `scripts/count_cut_site_hexamers.py` (containment rule, imports `simulator.precompute`). Two of its tests contradict the live start-in-region rule: `region_edge_rule_is_containment_not_overlap` and `a_fragment_straddling_a_tile_boundary_is_counted_in_neither_tile`. They are stale *relative to this module*, but correct for their own script [Draft measurement]. **Reuse its pattern, not its code**: plain-python `hidx`/`rc`/`hexamer_at` helpers, and paired assertions that the expected bin is hot AND the plausible-bug bin is cold. |

**Do not reuse that file's genome generator.** The generator is `'ACGT'[(i*7+(i*i)//3)%4]`.
It is periodic with period dividing 12. `floor((i+12)²/3) = floor(i²/3) + 8i + 48`, and
`7·12`, `8i` and `48` are all ≡ 0 mod 4. So it has at most 12 distinct hexamers out of
4096 [Inferred: arithmetic]. [Verified: review `exp2.py`, 20,000 bases: 12 distinct 6-mers.]
A frame test on such a genome cannot tell some shifts apart.

### 1.4 Stale items (report, do not test)

- `count_hexamers_rdf.py:32` and `:88-94`: say `simulator.precompute` stays the encoder, and
  cite a test `test_encoder_matches_precompute` (`:93`) that does not exist (spec Open 3,
  `:193-199`) [Verified: grep]. T0 resolves this. The comment should then cite T0.
- `count_hexamers_rdf.py:578`: the docstring says "strand, then start, then length". The
  code draws one binomial per region (`:596`) [Verified: Research A].
- **Superseded by `b50ab4f`** (F2, F18, F19 done): `hexamer_indices`'s docstring (`:135-142`)
  is fixed, and the balance/MAPQ-all messages (`:1035-1046`, `:1047-1058`) no longer say
  "clustered"/"MAPQ removed ALL". Residual: `_hexamers_at`'s own docstring (`:250-258`) still
  asserts the old wrong claim, so the two now disagree [Verified: read].
- `:40` ("midpoint") and `:309` ("capture") are benign wording [Verified: Research D].
- Removed-feature references outside the module (F15): `scripts/run_simulator.py:66`
  imports `simulator.capture` at module level (also `:7,16,33,161-209,300-306,339,430,513,622`);
  `background_model/simulator/__init__.py:7,12,16,21,25,28-30` (`build_predict_lut`,
  `midpoint_index_arrays`, capture import); `Makefile:194,199-203,206` (flgc comment,
  `FLGC_PYTHONPATH`) [Verified: Research D].
- Docs: `simulator_spec.md:181,196,209` (196 is Open 3, 209 historical);
  `simulator_overview.md:48,85,89,224,234`; `simulator_and_fragment_nll.md` and
  `simulator_basic_inputs.md` (heavy ZTNB/dup-histogram/`predict(L,gc)`) [Verified: Research D].
- `docs/pending/simulator_output_design.md` (unreviewed draft) [Verified: Research D]:
  - Stale line numbers (`sample_region` is now `:566-643`, `count_sample` `:1061-1148`,
    `propensities` `:790-850`, `filter_fragments` `:1036-1058`) [Verified: grep, 0820409]. Also
    `:24-27,120-122,135-149,470-481` reference removed features.
  - It says `count_sample` returns `(counts, stats)`. It returns a 4-tuple `(counts,
    region_counts, stats, srdf)` (`:1148`).
  - "Nothing writes draws to disk" is wrong: `simulate_fragments_to_bed` writes a BED (`:779`).
  - Its `p_plus` question is closed (fixed 0.5, owner 2026-10-07).
  - Its shortfall counter conflicts with Settled "no counter, no test" (see F14).
- Hazard outside the module: `RegionDataFrame.get_fasta_path()` hardcodes
  `/scratch/karius/annotation/GRCh38.p12.genome.fa.gz` for `ref="hg38"` and does not check
  it exists (`fragmentomics_tools/dataframe.py:485-487`) [Verified: grep]. `count_sample`
  takes the FASTA explicitly, so tests must always pass it.
- **Spec `docs/pending/simulator_spec.md`** [Verified: grep]: `:137-138` says only
  `N_rc == N_fwd[rc_permutation()]`. It does not state the crossed pairing that `844f227`
  adopted. Open "No tests" (`:178-182`) is partly stale: 4 tests exist since `844f227`.

## 2. Fixture design (central decision)

### 2.1 Synthetic and committed components

| Component | Decision | Why |
|---|---|---|
| Toy genome | Synthetic `chrT`, ~6,000 bp, built in `tmp_path_factory` (session scope), indexed with `pysam.faidx`. Deterministic, no RNG. | 6,000 bp builds in 0.15 s [Verified: Research C]. |
| Genome core | Linear de Bruijn B(4,6) (cyclic 4096 + first 5 bases = 4,101 bp), from the standard Lyndon-word algorithm in the oracle module. | Every hexamer occurs exactly once. So any frame shift, base-order error or rc error maps to a *different* cell. The fixture asserts this property at build time. |
| Planted features | (i) one tandem block of a non-palindromic hexamer, e.g. `AACGTC`×50 (rc `GACGTT`); (ii) a single `N`; (iii) a 10-bp N-run; (iv) a lowercase copy of a de Bruijn segment; (v) N-free filler to the contig end. Positions are named constants. | (i) makes the expectation `N` vary strongly. The fixture asserts `max(N)/min(N>0)` ≥ 10 for the regions used. On the de Bruijn core alone `N≈1`, so `C/N == C` and a counts-for-propensity mutation would pass vacuously. (ii)-(iii) give validity cases. (iv) gives soft-mask folding, which the uppercase-only chr6 cannot. |
| Fragments | In-test lists of `(contig,start,stop,strand,mapq)` written to a text BED. Then `tabix_index`, then `build_fragments_h5` against the toy FASTA, one h5 per scenario, session scope. | The text list is the reviewable source of truth. |
| Regions | `RegionDataFrame(..., ref="hg38")` on `chrT`. Contiguous tiles. First tile start ≥ 3. Last tile end ≤ contig_len − 183. | The contig-end limits are [Verified: review] (§1.2), not a draft measurement (L9: the two sections now agree). |
| Committed chr6 FASTA | `tests/data/GRCh38.p12.genome.chr6_99110000_99130000.fa.gz`. Used by `t4_uniform_hexamer_counts_chr6` and `t3_golden_h5_matches_bruteforce` (fetch, no build). | Real composition, the `chr6` name and natural N at the window edge, through the real `attach_sequence`. **Never build an h5 against it.** |
| Committed golden h5 | `tests/data/golden.small.chr6.frag.h5`, used by `t3_golden_h5_matches_bruteforce` only. [Verified: review `golden.py`] It holds 7 admitted fragments (4+, 3−) in the window, so it is a schema check, not statistics. Its sibling `golden.test_duplicates.frag.h5` raises the all-empty-admission error on the window region (worded "EVERY fragment was removed before counting" as of `b50ab4f`; "MAPQ removed ALL" pre-`b50ab4f`), so it is not used. | Pins the *production-built* h5 schema (keys `data`, `fragment_length_counts`, `index`; group `chr6`) [Verified: Research C]. |
| New: `tests/data/simulator_real_regions_20.bed` | ~20 rows, 3 columns: 10 on `chr21` (its 1,049-row subset [Verified: review, `grep -c`]), the rest drawn from other chromosomes (e.g. every 3000th row, as Research C did). ~1 KB. | The source BED is gitignored and absent from worktrees [Verified: research file, decision 5c]. A committed 1 KB file makes R1, R2 and R4 depend on EFS only, and gives R4 its `chr21`-only region set (M2). |

### 2.2 Rejected alternatives

- **Commit binary h5 fixtures.** Rejected. Each is ≥548 KB of fixed overhead [Draft
  measurement], opaque and not reviewable. It drifts silently when fragments_h5 changes its
  schema. The text BED plus a 0.15 s build is better.
- **Build an h5 against the real FASTA, unrestricted.** Rejected for any contig but a small
  one: 76.26 s for 10 fragments on chr1 [Verified: Research C]. **Revised (M2):** `chr21` alone
  costs ~11 s [Verified: review, `realbuild/rb.py`] — cost tracks contig length, not fragment
  count — so R4 (§2.4) pins "h5 holds exactly the fragments written" on real `chr21` data, in
  addition to the toy (T6). R3, which checked only the real-sequence half on BED text with no
  h5 and no killing mutation, is deleted: its checks are strictly implied by R4's recount (M1).
- **Duck-typed `rfa` stubs everywhere.** Used only in T0/T1, where the target is index
  arithmetic. Admission (T2/T3) goes through the real `attach_fragment_arrays` +
  `filter_fragments` + reader, because the silent failures there are cross-library.
- **Arithmetic or RNG genome.** Rejected. The arithmetic one has period 12 (§1.3). De Bruijn has
  a checkable coverage property.
- **The CLI path (`bgzip`/`tabix`/`build-fragments-h5`, as used by `emit.py`).** Out of
  scope. That is the old generation. The pysam path needs no binaries.

### 2.3 Core dependencies never skip

`pysam`, `fragments_h5` and `numpy`/`pandas` are hard dependencies of the module and of
`fragmentomics_tools.dataframe` [Draft measurement]. Tests use plain imports, with no
`importorskip` and no `skipif`. A skip on a core dependency turns "environment broken" into
"suite green".

Local precedents [Verified: Research C]:
- Phase3 `_require_fasta()` hard-fails on missing EFS data, deliberately. v1 of this
  design called it "a guard that read as protection and was not". That was wrong.
- Phase2 `:451` reads the same EFS path with no guard at all.
- `tests/test_bg_golden_counts.py:54-59,191` skips on missing in-repo fixtures. Its reason
  names the rebuild command. `tests/test_bg_inference.py:41-42` and
  `tests/test_bg_correction.py:41-42` skip on in-repo fixture paths.

### 2.4 Real-data tier and the fixture split

The question per test: what does the test need for a checkable answer?

| Group | Data | Tests | Justification |
|---|---|---|---|
| (a) Synthetic only | toy `chrT`, built at test time | T0-T6 except the two smoke tests | Only a constructed input has a derivable answer for frame, routing, admission boundaries, null identity, sampler recovery, the validity gate and N windows. Real data cannot plant a mapq-10 fragment or a single-N stop window. |
| (b) Real data | real h5, real FASTA (EFS), committed ~20-region BED | R1, R2, R4 | Real formats and real magnitudes are the thing under test: bytes strands at the raw reader, contig naming, half-open library bounds, unclipped overhangs, real MAPQ semantics on a pipeline-written h5, real fl. R4 additionally builds a real h5 (`chr21` only, M2). |
| (c) Committed | chr6 FASTA, golden h5 (existing); the new 1 KB BED | `t3_golden_h5_matches_bruteforce`, `t4_uniform_hexamer_counts_chr6`; R inputs | Cheap, always runs, pins the production h5 schema. |
| (d) Skipped | — | `real_data` tests only, when EFS is absent | See §2.5. |

**Real-tier tests** live in a separate file, `tests/test_count_hexamers_rdf_real.py`, so its
budget and skips stay separate from the synthetic file.

- **R1** `r1_count_sample_matches_raw_reader_oracle`. `count_sample` on the 20 regions with
  the real h5 and real FASTA. Oracle: `FragmentsH5.fetch_array(contig, g0 − 180, g0 + R + 180,
  return_mapqs=True, return_strand=True)` per region, strands decoded from bytes, then
  `bruteforce_count` with hexamers from `pysam.FastaFile.fetch`. The wide window makes the
  oracle independent of the reader's overlap semantics; the oracle applies the start and length
  rules itself. `fetch_array`'s own filters are overlap and `lengths <= max_frag_len` (§1.2),
  and neither removes a needed fragment. Assert exact equality of the four tables,
  `region_counts` and the `stats` keys `n_counted`, `n_plus`, `n_minus`.
- **R2** `r2_uniform_hexamer_counts_matches_pysam_enumeration`. Same regions, `f` from
  `FragmentLengthDist.from_srdf` on R1's frame. Oracle: `enumerate_expectation` on sequence
  fetched through pysam, including the flank. Exact (rtol 1e-12).
- **R4** `r4_real_region_round_trip_chr21`, **replaces R3** (M1-M2; §2.2 says why R3 has no
  distinct purpose left). Ten of the committed BED's `chr21` rows (§2.1) go through the full
  pipeline against real data: `count_sample` (real h5) → `FragmentLengthDist.from_srdf` →
  `uniform_hexamer_counts` → `propensities` → `simulate_fragments_to_bed` →
  `bgzip`/`tabix_index` → `build_fragments_h5` against the real FASTA with
  `allowed_contigs=["chr21"]` [kwarg verified: review, `realbuild/rb.py`] → read back through
  the real reader — the only test that builds an h5 against a real FASTA (§1.2, §2.2: ~11 s on
  `chr21` alone). Assert the T6 distinct-pair invariant (recounted `region_counts` equals the
  distinct `(start,stop)` per region among the written rows, not `n_drawn`) and exact
  coordinates/strands on read-back. Independent oracle: `bruteforce_count` on the written BED
  rows, read via pysam, not through the module. Mutation coverage: M28 and M29 (§4.2) each add
  `r4_…` as a second red case, so R4 is not oracle-only. Marked `real_data`. Budget ~11 s build
  [reviewer-measured] plus the rest of the pipeline [Inferred]; the real tier total must stay
  ≤ 30 s [Inferred],
  else that is a finding.

R1's BED is strandless (3 columns) [Verified: Research C]. So R1 never exercises the
flip of minus-strand regions. See Least sure of.

### 2.5 Skip policy and skip visibility

- Only tests marked `real_data` may skip. The skip reason names the missing path.
- `tests/conftest.py` (new) registers the `real_data` marker. It also adds a
  `@pytest.hookimpl(wrapper=True)` on `pytest_runtest_makereport`: when `REQUIRE_REAL_DATA=1`
  and the item carries `real_data`, a skip becomes a failure.
- The run commands add `-rs`, because the Makefile default has none.
- Scratch evidence [Verified: Research C, via `make test PYTEST_ARGS=...` under `/tmp`]:
  without `-rs`, `1 passed, 1 skipped` and no reason; with `-rs`,
  `SKIPPED [1] ...: real data missing: /efs/nope/x.h5`; with the hook and the env var,
  `1 passed, 1 error`, make exit 2; without the env var, exit 0.
- Gate on the marker, not on all skips: torch/flgc `importorskip` calls elsewhere in `tests/`
  also skip.
- Do not reuse the `slow` marker. The repo-root `conftest.py:29-54` already owns `slow`,
  `extra_slow`, `--runslow` and `--run-extra-slow`. A new `tests/conftest.py` limits the
  blast radius to `tests/`.
- **Acceptance:** the owner's run uses `REQUIRE_REAL_DATA=1` and must show **0 skipped**.
  A reader tells skip from pass by the `-rs` reason line, and by the REQUIRE run failing.
- Residual risk: a run without the env var still skips quietly when EFS is absent.

## 3. Independent oracle (`tests/cut_site_oracle.py`)

The oracle is plain Python with no numpy in the encoder path. **It imports nothing from
`background_model` or `fragmentomics_tools`** (T7, by AST). A sibling `import cut_site_oracle`
should work under pytest's prepend mode [Inferred; P1 confirms]. If not, inline it and keep the
AST check. It provides:

- `IDX(h)`: base-4 big-endian, `A,C,G,T = 0..3`, after `upper()`.
- `RC(h)`: string reverse-complement.
- `DECODE(i)`: index to hexamer.
- `hex_at(genome, c) = genome[c-3:c+3]`.
- `valid(h)`: all 6 characters in `ACGTacgt`.
- `de_bruijn(4, 6)`.
- `decode_strand(x)`: bytes or str to `'+'`/`'-'`; anything else raises.
- `bruteforce_count(fragments, regions, genome, min_mapq=10, l_min=25, l_max=180)`: applies
  MAPQ `min(mapq1,mapq2) ≥ 10`, then dedup on `(start,stop)` keeping the first in **h5
  order**, then length `[25,180]` inclusive, then start in `[g0, g0+R)`. Returns
  region_counts and four tables. A fragment needs **both** cut sites valid to be in the
  tables, but is in region_counts regardless (the accepted divergence of 2026-10-06). Routing:
  - plus → `start_fwd[IDX(hex(start))]` and `end_fwd[IDX(hex(stop))]`
  - minus → `start_rev[IDX(RC(hex(stop)))]` and `end_rev[IDX(RC(hex(start)))]`
- `enumerate_expectation(genome, regions, f)`, by literal double loop:
  - `N_start[h] = #{s ∈ [0,R): valid hex(g0+s) = h}`
  - `w(i) = Σ_L f(L)·[0 ≤ i−L < R]`
  - `N_end[h] = Σ_i w(i)·[valid hex(g0+i) = h]`
- `enumerate_null_counts(...)`: every `(s, L)` with weight `p_plus·f(L)` per strand, routed
  as above, as float tables.
- `sampler_start_p(...)` and `sampler_length_p(...)`: the spec §4 weights, as strings.

The routing above matches the module (`:314`, `:320-321`) and spec `:79-80`
[Verified: Research A/B].

## 4. Silent-failure classes, oracles and the mutation matrix

### 4.1 Classes and oracle constructions

1. **Encoder** (base order, endianness, case). Oracle: `IDX` over all 4096 strings, and over
   the de Bruijn core in one call. `hexamer_vocabulary()` (`:156`) and `rc_permutation()`
   (`:189`) derive from the encoder [Verified: Research A]. An encoder defect therefore
   propagates into both, and no internal consistency check sees it.
2. **Frame** (window not centred on the cut, wrong pad). Oracle: `hex_at` through the real
   `attach_sequence`, on de Bruijn, where every shift changes the cell.
3. **Validity gate.** N reads as A inside the window: `CGTTTN` → 1788 with `valid=False`,
   `NCGTTT` → 447; `AAAAAA` is index 0 [Verified: Research A, mechanism `:150`]. The test
   plants a fragment whose stop window has one N, such that N-as-A equals the hexamer of a
   second, valid, planted fragment. It asserts that the shared cell holds the valid
   fragment only. It **never pins the index value** of an invalid window.
4. **Strand routing** (minus tables swapped, perm omitted, whole plus/minus swap, bytes vs
   str labels). The runtime checks cannot see a whole swap (F3), and the pair identities are
   tautologies (F4). Oracle: a hand frame with four pairwise-distinct non-palindromic
   hexamers whose rcs are also distinct. The test also computes, in-test, the tables that
   each plausible wrong mapping gives, and asserts each differs from the expected tables.
5. **Admission boundaries**: MAPQ `≥` vs `>`; dedup before MAPQ; `l_max` vs `l_max+1` in the
   half-open library call (`:1056`); midpoint instead of start; inclusive upper start bound;
   right pad too short for a max-overhang stop; stops clipped to the region. Oracle:
   `bruteforce_count` on planted boundary fragments (T2) and on raw real records (R1).
6. **Expectation** (`fl_end_weight` ramp off by one, flank omitted from `N_end`). Oracle:
   `enumerate_expectation`, plus the identity `Σ_i w(i) = R` per region, because `Σ_L f = 1`
   per start. For regions 700+500+700 the sum is 1900 [Draft measurement; also the
   arithmetic sum].
7. **Propensity** (C instead of C/N, wrong denominators, wrong threshold). Oracle: C/N on the
   skewed fixture for the forward tables. The null identity (§4.3) covers all four.
8. **Sampler** (wrong table per strand, wrong track, validity mask dropped, `f` omitted,
   end offset `i+l±1`, `p_plus` ignored). Oracle: `sampler_start_p`/`sampler_length_p`
   through a recording rng (T5), plus statistical recovery (T5).
9. **Writer/reader contract** (column order, 0/1-based, strand flip, mapq, sort). Oracle:
   read back through the real reader, and recount with `bruteforce_count`.

### 4.2 Mutation matrix

Apply each mutation as a scratch edit at its **site** (module, oracle, or library under
`fragmentomics_tools/`). Never commit. Run the named tests, confirm each is **red** (a failure or
an unexpected exception), then revert. The PR table lists mutation, site, test and red/green. A
test that stays green under its mutation is not done. Line numbers are at HEAD `0820409`.

| # | Class | Site | Mutation | Cheapest distinguishing input | Tests that must go red |
|---|---|---|---|---|---|
| M1 | encoder | module `:111-113` | swap `_BASE_LUT` G↔T | any hexamer with G or T | `t0_encoder_matches_oracle_all_4096`, `t0_vocabulary_matches_oracle` |
| M2 | encoder | module `:117` | reverse `_POW` (little-endian) | `AAAAAC` (1 vs 1024) | `t0_encoder_matches_oracle_all_4096`. The doctests do not expose it [Verified: review `mut/chr_m2.py`] |
| M3 | case | module `:114` | drop lowercase from `_BASE_LUT` | lowercase stretch | `t0_lowercase_folds`, `t7_module_doctests_execute` [Inferred: the doctest asserts `valid[0]` on `"acgtAC"`] |
| M4 | rc | module `:157` (L4: the edit belongs in `hexamer_indices`'s `(3 - safe)[:, ::-1]`, not `rc_permutation` `:195-220`) | complement without reverse | any non-palindrome | `t0_rc_permutation_matches_oracle` |
| M5 | frame | module `:259-262` | window `[c-2,c+4)` | de Bruijn interior cut | `t0_frame_through_attach_sequence` |
| M6 | frame | module `:1140` | `left_pad=0` in `count_sample` | fragment at region start | `t3_count_sample_matches_bruteforce`, `r1_count_sample_matches_raw_reader_oracle` |
| M7 | gate | module `:294` | drop `valid` mask in `cut_site_hexamers` | single-N stop window colliding (as A) with a planted hexamer | `t3_n_window_fragment_dropped_from_tables` |
| M8 | routing | module `:320-321` | swap `start_rev`/`end_rev` sources | one minus fragment, start≠stop hex | `t1_counts_from_hexamers_routing` |
| M9 | routing | module `:320-321` | omit `perm` on minus | same | `t1_counts_from_hexamers_routing` |
| M10 | routing | module `:314` | `plus = strand != "+"` (whole swap) | one plus, one minus fragment | `t1_counts_from_hexamers_routing`, `t3_count_sample_matches_bruteforce`, `t6_recount_equals_distinct_pairs`, `r1_…` |
| M11 | admission | library `fragment_array.py:1751` | MAPQ `>` instead of `≥` | mapq 10 fragment | `t2_mapq_boundary`, `r1_…` |
| M12 | admission | library `fragment_array.py:1742-1751` | dedup `:1050` moved before MAPQ | A`(s,e,'+')` mapq 5 then B`(s,e,'-')` mapq 30 (L3 split; L7 strand discriminates since hexamers are equal) | `t2_mapq_filter_precedes_dedup` |
| M13 | admission | module `:1082` | `subset_fragment_lengths(l_min, l_max)` (half-open drops 180) | L=180 | `t2_length_bounds`, `r1_…` |
| M14 | admission | module `:1083` | `starts_0 <= length` | start at g0+R | **Equivalent mutant (H1): no test can turn this red.** The real reader keeps only `starts < region_stop` (`fragments_h5.py:638-640`), so `starts_0 == fa.length` is unreachable through `from_fragments_h5`; a hand-built RFA with `starts_0 == length` raises `FragmentDoesNotIntersect` inside `filter_fragments` instead (even with `validate_data=False`, since `drop_duplicate_fragments` re-validates). `t2_start_admission_half_open` is kept regardless: it still pins the tile assignment of g0−1, g0, g0+R−1 and g0+R through the real reader. |
| M15 | admission | module `:1083` | midpoint admission | tile straddler | `t2_straddler_counted_in_start_tile` |
| M16 | pad | module `:1140` | `right_pad=l_max` | start g0+R−1, L=180 | `t2_max_overhang_fragment_counted` |
| M17 | expectation | module `:459-461` (docstring formula at `:453`, L4) | `max(min_fl, i-R)` in `fl_end_weight` | R < max_fl, min_fl > 1 | `t4_fl_end_weight_matches_enumeration`, `r2_…` |
| M18 | expectation | module `:561` (L4: `:516` only trips the `:543-548` guard for the wrong reason; the meaningful site masks `w` to the region) | `N_end` over the region only (no flank) | any region | `t4_end_weight_total_equals_region_length_sum`, `t4_uniform_hexamer_counts_matches_enumeration_toy`, `t4_uniform_hexamer_counts_chr6`, `r2_…` |
| M19 | propensity | module `:840-848` | return `C` (no division) | skewed fixture, tandem block | `t4_propensities_forward_exact`, `t4_null_identity[start_fwd\|end_fwd\|start_rev\|end_rev]` |
| M20 | propensity | module `:846` | `d > min_expected` becomes `>=` | N == min_expected cell | `t4_propensities_forward_exact` |
| M21 | sampler | module `:608` | minus `s_tab=r['start_rev']` | asymmetric r | `t5_start_probabilities[minus]` |
| M22 | sampler | module `:607` (L4: the track choice itself, not `:608-609`'s table choice) | minus on `fwd` track | non-palindromic region | `t5_start_probabilities[minus]` |
| M23 | sampler | module `:608` | drop `valid` on starts | N in region | `t5_start_probabilities[plus]` |
| M24 | sampler | module `:616` | drop one factor of the length weight: `fl.densities`, the end validity mask, or `r_end` | the matching single-cause construction | `t5_zero_weight_cause[f_zero]`, `[non_acgt]`, `[r_zero]` respectively |
| M25 | sampler | module `:614-616` | end hex at `i+l-1` | point-mass r_end | `t5_point_mass_exact_output`, `t5_length_probabilities` |
| M26 | sampler | module `:596` | ignore `p_plus` (always 0.5) | p_plus = 1 | `t5_p_plus_extremes` |
| M27 | sampler | module `:603-604` | plus uses minus tables (end to end) | planted 20× start_fwd | `t5_planted_propensity_recovered` |
| M28 | writer | module `:773` | strand column from the wrong field / flipped | read-back | `t6_round_trip_through_real_reader`, `r4_…` |
| M29 | writer | module `:769` | 1-based start | read-back | `t6_round_trip_through_real_reader`, `r4_…` |
| M30 | contract | library `fragment_array.py:299` | remove the `U1` coercion. Bytes then reach the guard at `:331-334`, which raises `ValueError` [Verified: review] | any read-back | `t1_reader_strand_labels_are_str` |
| M31 | contract | module `:314` | `plus = strand == b"+"` (bytes comparison) | any str-labelled frame | `t1_counts_from_hexamers_routing`, `t3_golden_h5_matches_bruteforce`, `r1_…` |
| M32 | admission | module `:298` | clip `stops_0` to the region end before the hexamer lookup | overhanging fragment | `t2_max_overhang_fragment_counted`, `r1_…` |
| M33 | gate | module `:154` | `hexamer_indices` returns `valid` all True | single N | `t0_single_n_invalid_at_every_offset`, `t3_n_window_fragment_dropped_from_tables` |
| M34 | frame | library `region.py:854-865` | pad a truncated contig-end fetch with N instead of raising | region within 183 bp of the end | `t0_contig_ends_raise` |
| M35 | admission | library `fragment_array.py:1050-1052` | dedup key includes strand | same `(s,e)` on + and − | `t2_dedup_key_omits_strand` |
| M36 | admission | module `:1082` | skip the length filter | planted L=181 | `t2_length_bounds`, `t5_fl_from_filtered_frame_within_bounds` |
| M37 | writer | module `:775` | emit rows unsorted, or add a header line | ≥ 2 rows | `t6_bed_text_shape` |
| M38 | fl | module `:391-394` | densify without normalisation, or one bin off | `{25:1, 27:3}` | `t6_fragment_length_dist_guards[densify]` |
| M39 | guards | module `:349-355`, `:385`, `:390-395`, `:408-415`, `:509-514`, `:707-711`, `:714-717`, `:718-720`, `:738-742`, `:744-751`, `:913-919`, `:995-999`, `:1035-1046`, `:1047-1058` (re-grepped at `b50ab4f`; M4 drops the two tautology sites — see below — and adds `:385`, the `from_dataframe` missing-column raise, used in the catalogue) | delete the `raise` that a parametrised case names | per case | `t3_one_empty_strand_raises`, `t3_strand_balance_guard_fires`, `t3_count_guards`, `t6_writer_guards`, `t6_fragment_length_dist_guards`, `t6_input_guards`. The named case must fail with "DID NOT RAISE" |
| M40 | hygiene | oracle file; module import lines | oracle imports `background_model`; module imports `simulator.precompute` | AST | `t7_oracle_is_independent`, `t7_no_removed_feature_imports` |
| M41 | propensity | module `:840-841` | restore the pre-fix pairing: `start_rev` ÷ `N_start[perm]`, `end_rev` ÷ `N_end[perm]` | the tandem block and the null regions | `t4_null_identity[start_rev]`, `t4_null_identity[end_rev]` |
| M42 | expectation | module `:553`, `:561` | drop `valid` in `uniform_hexamer_counts` | planted N in a start window | `t4_uniform_hexamer_counts_matches_enumeration_toy`, `t4_uniform_hexamer_counts_chr6` |

### 4.2a Implementation status — read this before trusting the matrix above

**The matrix names a test for nearly every row. That is the DESIGN, not the state
of the repo.** 18 of 42 mutations have been applied programmatically and verified
red (commit `45b32ec` + the sweep); the rest are unverified, and four have no
test at all. Treat an unverified row as unknown, not as covered.

**Verified red** (sweep at `45b32ec`, re-verified at `d9c6e90`): M1, M2, M3, M4,
M7, M8, M9, M10, M19, M20, M26, M31, M33, M37, M38, M41, M42. M14 is an
equivalent mutant by construction and is excluded.

**M7 and M33 were NOT red on the first sweep, and the reason corrects this
table.** M7 came back with **zero** red tests. The trigger column above says
"single-N stop window colliding (as A) with a planted hexamer" — that diagnosis
is wrong. **The fixture planted no N-window fragment at all**: it placed the
fragment at `n_pos - 3`, whose cut-site window is `[n_pos-6, n_pos)`, which
EXCLUDES the N. Off by one. Compounding it, the assertion was
`n_admitted >= n_counted`, true by construction however the code behaves. Two
defects stacked: a fixture that planted nothing and an assertion that could not
have noticed. Fixed in `d9c6e90` — the fixture now plants one invalid-START and
one invalid-END fragment, and the expected drop is derived from the genome string
and asserted EXACTLY. M7 now 2 red, M33 1 → 3 red.
**Lesson for the rows below: a test named in this matrix is worth nothing until a
mutation has been shown to turn it red.**

### 4.2b The four mutations with no test

Scoped out of `45b32ec` deliberately, so this is a known gap. Both pairs sit on
genuinely silent paths, which is why they are the most valuable remaining work.

| Mutation | Test to write | What it must construct | Why it is worth it |
|---|---|---|---|
| **M12** — dedup moved before MAPQ | `t2_mapq_filter_precedes_dedup` | Two fragments sharing `(start, stop)`: A mapq 5, B mapq 30, on *opposite strands* so the surviving one is identifiable (their hexamers are equal, so only strand discriminates). Correct order keeps B; mutated order dedups to A and then MAPQ drops it, losing the pair entirely. | This is the ONE admission ordering that is load-bearing (spec §3). Everything else commutes. Getting it wrong silently loses real fragments. |
| **M15** — midpoint instead of start-in-region admission | `t2_straddler_counted_in_start_tile` | A fragment straddling a tile boundary whose START is in tile *k* and whose MIDPOINT is in tile *k+1*. Assert it is counted in *k* and absent from *k+1*. | Midpoint admission is the rule the rewrite REVERSED, and two agents have already read stale docs asserting it. A regression here reintroduces the whole pre-rewrite geometry. |
| **M24** — drop one factor of the length weight | `t5_zero_weight_cause[f_zero\|non_acgt\|r_zero]` | Three single-cause fixtures, one per factor of `w[l] = f(l)·r_end·valid`: a length with `f(l)=0`, an end window with a non-ACGT base, and an end hexamer with `r_end=0`. Each must change the drawn length set when its factor is removed. | §4 says the code "cannot distinguish the three causes" — it only tests `w.sum() > 0`. These are the only tests that pin each factor independently. |
| **M25** — end hexamer read at `i+l-1` | `t5_point_mass_exact_output`, `t5_length_probabilities` | A point-mass `r_end` on one hexamer, so the drawn length is deterministic and an off-by-one in the end index shifts it by exactly 1. | A one-position shift in the end lookup leaves every total plausible and every marginal nearly right — the definition of a silent failure. |

**M12 carries a caveat that must be settled first.** It depends on the h5
returning A before B for equal `(start, stop)`, which the Least-sure-of section
records as verified on exactly **one** case — "one case is not a law". If the
ordering turns out to be unspecified, M12 cannot be tested through
`from_fragments_h5` and needs a direct `filter_fragments` test on a hand-built
fragment array instead. Settle the ordering before writing the test, not after.

`r1_…` and `r2_…` abbreviate the full R ids of §2.4.

**Cross-check.** Every §5 test id appears in at least one row, and vice versa. No coverage
exemptions remain: R3's uncovered id is deleted (M1); R4 is covered by M28/M29. M14 is a
separate, **red/green** exemption (H1), excluded from P7's "every row red" only, not from this
id-coverage check. M2 no longer lists `t7`; M3 kills it; M41/M42 are new. P7 re-runs this by
script — re-run manually here: 42 rows, 47 ids, fully covered both ways [Verified: `xcheck.py`,
adapted for R4/no-R3].

### 4.3 The null identity (regression guard for F1)

Fixture: N-free regions of the toy, `f` positive over a band, `p_plus = 0.5`. The regions must
include the tandem block, and the test asserts `max(N_start) >= 10` in-test. Set
`C = enumerate_null_counts(...)`, the exact expectation under the spec's null of uniform
genomic starts and f-weighted genomic stops. Then `r = propensities(C,
uniform_hexamer_counts(...))`. **Assert `r == p_plus` (rtol 1e-12) in every cell with
`N > 0`, per table**, parametrised over the four tables. At `844f227` and later all four are
green. [Verified: review `f1.py`: 0.500000 min and max in all four tables on 3 toy regions.]
The test is exact and deterministic. No simulation is needed.

Mutation M41 restores the pre-fix pairing. Then `[start_rev]` and `[end_rev]` must go red
[Verified: review `f1.py`: `start_rev` 0.0–0.5, `end_rev` 0.0–54.0]. The `a409186` numbers
(`end_rev` up to 546 on 3×1000-bp regions) are history only [Verified: Research B].

N-free is required because of F6. With an N planted, C and N use different validity rules,
and the identity fails for a reason other than F1.

### 4.4 Strand statistics: the unit of replication

**Simulator output.** `sample_region` draws one `rng.binomial(n, p_plus)` per region
(`:596-599`) [Verified: Research A]. Conditional on the per-region `n_r`, the pooled plus
count is exactly Binomial(N, 0.5) [Inferred: code]. Tests call `sample_region` per region
and read its `is_plus`. So per-region expectations are exact, and a fragment-level binomial
bound, conditional on the realised per-region `n_plus`, is valid.

**Real data.** The real-data plus fractions 0.3758 (322 fragments, 10 regions), 0.5296 (200)
and 0.5041 (2000) on RD-56804 give **overdispersion**: 4.5, 5.5 and 2.2 σ at fragment level, in
opposite directions [Verified: `count_hexamers_rdf.py:995-999`, measured by the owner]. The
mechanism is **not measured**. Candidates: within-region clustering, coarser structure, and the
strand-blind dedup that keeps the first h5 row (`fragment_array.py:1050-1052`). Per-region strand
counts are not stored (`count_srdf`, `:966-977`), so they need RD-56804 [Unverified]. The simulator
draws strand as an independent binomial per region (`:596`), so the excess is not reproduced
downstream. The sampler's zero-length drops (`:618-622`) act on simulator output only.

**Measured false-positive rates** [Verified: Research D scratch simulation, gamma `n_r`
mean ~37; nominal two-sided 3σ = 0.0027]:

| Rule | R=10 | R=200 | R=2000 |
|---|---|---|---|
| fragment-level 3σ, simulator draw | 0.0025 | 0.0025 | 0.0040 |
| region-level SE 3σ, simulator draw | 0.0060 | 0.0000 | 0.0000 |
| the code's guard, simulator draw | 0.0000 | 0.0000 | 0.0000 |
| the code's guard, strand-pure regions, unequal `n_r` | 0.0038 | 0.0141 | 0.0000 |

The guard bound `1.5/√R` is `3 × 0.5/√R`. It assumes the maximum per-region sd, 0.5, so it
is conservative. For R > 225 it equals the 0.1 floor, about 54 fragment-σ at 73,545 fragments.
**`b50ab4f` (F3) caps it: `min(0.45, max(strand_tol, 1.5/√R))`, capped at 0.45 for R ≲ 11 where
the uncapped term would reach/exceed 0.5 and make the check vacuous.** At R=10 the bound is now
0.45, not the uncapped 0.474 this design previously quoted [Verified: read, `b50ab4f`]. The
false-positive table above predates the cap and wants re-measurement at small R [Unverified].

**Rules for this suite.**
- Prefer deterministic tests: `p_plus` 0 and 1, and exact table equality.
- Statistical tests on simulator output use a fixed **6σ** fragment-level bound, conditional
  on the realised per-region `n_plus`. The normal figure (2e-9) understates the tail at small
  counts. The exact upper Poisson tail beyond 6σ is 1.5e-7 at expected count 25 and 2.0e-8 at
  100 [Verified: review, scipy; upper tail only]. Every checked cell therefore needs an expected
  count ≥ 100 [Inferred]. T5 has about four such checks, so the file-level rate is near 1e-7
  [Inferred].
- No test asserts a real-data plus fraction. For real data the region is the unit, and its
  dispersion is unmeasured.
- Guard tests keep inputs far from the bound, so they do not freeze the 1.5 constant (an
  owner-level choice, F13). At R=200 the bound is max(0.1, 0.106) = 0.106. A 0.80 plus
  fraction raises and 0.50 passes. A mildly lopsided 0.55 also passes, by design (F3).

## 5. Test catalogue

Synthetic file `tests/test_count_hexamers_rdf.py`. Real file
`tests/test_count_hexamers_rdf_real.py` (§2.4). Oracle `tests/cut_site_oracle.py` (not
named `test_*`). Marker and hook in `tests/conftest.py` (§2.5). Session fixtures:
`toy_fasta`, `toy_regions`, `skewed_regions`, `admission_h5`, `bruteforce_h5`,
`roundtrip_out`, `real_inputs` (skips with `real_data` when EFS is absent). Every count path
uses `n_workers=1`, because CLAUDE.md records a fork deadlock in `parallel_apply`. The fixture
builds are safe without that rule. `build_fragments_h5` defaults to `num_processes=None`, which
is serial with no fork [Verified: `fragments_h5.py:1081`, `:1281`, read].

**T0 encoder and frame (pure, no h5)**
- `t0_encoder_matches_oracle_all_4096`: `hexamer_indices` on each of the 4096 strings, and
  on the de Bruijn core in one call. Compare fwd and rc with `IDX`/`IDX∘RC`.
- `t0_rc_permutation_matches_oracle`: `perm[IDX(h)] == IDX(RC(h))` for all h. Also checks
  the involution, and that the fixed points equal the oracle's palindrome set (size 64 = 4³).
- `t0_vocabulary_matches_oracle`: `hexamer_vocabulary()[i] == DECODE(i)`.
- `t0_lowercase_folds`: on the soft-masked stretch, through `hexamer_indices` and through
  `uniform_hexamer_counts`. Lowercase must equal uppercase.
- `t0_single_n_invalid_at_every_offset`: offsets 0..5, `N` and `n`. Asserts `valid=False` only.
- `t0_frame_through_attach_sequence`: regions at contig start 3 (the minimum) and in the
  interior. Asserts padded length `R+186`. Then `cut_site_hexamers` with a duck-typed rfa
  (`n_frags, starts_0, stops_0, fragment_strands`) at every cut. Hexamers must equal `hex_at`.
- `t0_contig_ends_raise`. Each region holds a planted fragment, so the "every fragment was
  removed" guard (`:1047-1058`, reworded per F19/`b50ab4f`) cannot fire. Under M34 the test
  then sees no raise. Through `count_sample`: start 0 and 1 give `ValueError`,
  `match="runs off the start"` (`region.py:856`); a right end within 183 bp of the contig end
  gives `ValueError`, `match="the fetch was truncated"` (`:863`). Module guards, as separate
  cases: `uniform_hexamer_counts` near the end gives `ValueError`,
  `match="hexamer windows, need at least"` (`:543-548`); `simulate_fragments_to_bed` on a frame
  whose `sequence` is one base short gives `AssertionError`, `match="flank was truncated"`
  (`:744-751`).

**T1 strand routing**
- `t1_counts_from_hexamers_routing`: the hand frame from §4.1(4), with the in-test
  wrong-mapping self-check.
- `t1_reader_strand_labels_are_str`: the real read-back `fragment_strands` is a str dtype
  with labels ⊆ `{'+','-'}`. This guards the contract that `== "+"` relies on (F5). It does
  **not** assert what the module does with `'.'` or bytes, because that would freeze F5.

**T2 admission (one h5 of planted boundary fragments)**
- `t2_mapq_boundary`: (60,9), (9,60) and (10,10). Expects drop, drop, keep.
- `t2_mapq_filter_precedes_dedup` (L7): A and B share `(s,e)` but differ in strand and MAPQ —
  A `(s,e,'+')` mapq 5, B `(s,e,'-')` mapq 30: hexamers from `(s,e)` alone are equal for A and
  B, so strand is the discriminator the oracle can see. A self-check confirms A precedes B in
  the unfiltered read-back (`min_mapq=0`); else "fixture cannot discriminate". Correct order
  (MAPQ before dedup) drops A on MAPQ, leaving only B, whose hexamer lands in `start_rev`/
  `end_rev` (`start_fwd`/`end_fwd` stay empty). Under M12 (dedup before MAPQ), dedup keeps A
  first, then MAPQ removes it — zero fragments survive, not one — visibly different either way
  [Verified: `t3.py` M12, red on count and strand table].
- `t2_dedup_key_omits_strand`: the same `(s,e)` on + and −. Expects region total 1, and
  exactly one fragment across the four tables. It does not assert *which* strand survives.
  Settled behaviour, kept because the T6 invariant depends on it.
- `t2_length_bounds`: 24/25/180/181. Expects drop, keep, keep, drop.
- `t2_start_admission_half_open`: starts at g0−1, g0, g0+R−1, g0+R over contiguous tiles.
  Each lands in exactly the tile that contains its start.
- `t2_straddler_counted_in_start_tile`: also asserts `Σ region_counts` over tiles == the
  oracle's admitted total.
- `t2_max_overhang_fragment_counted`: start g0+R−1, L=180, in the last tile. Its stop
  hexamer comes from the right pad and matches `hex_at`. The stop is not clipped to the region.

**T3 `count_sample` vs brute force (exact equality)**
- `t3_count_sample_matches_bruteforce`: a few hundred deterministic fragments (no RNG) over
  contiguous de Bruijn tiles, plus all T2 cases. Exact equality of the four tables,
  `region_counts` and `stats` keys `n_regions`, `n_after_filters`, `n_counted`, `n_plus`,
  `n_minus`, `plus_frac` (`:966-977`) [Verified: grep, `b50ab4f`].
- `t3_n_window_fragment_dropped_from_tables`: the M7 construction. Also asserts
  `region_counts.sum() − n_counted ==` the number of oracle fragments with an N cut site.
  This is the accepted divergence, asserted as specified.
- `t3_one_empty_strand_raises`, over **≤ 9 regions**. All-plus fragments raise `AssertionError`,
  `match="one strand table is EMPTY"` (check (1), `:994-1002`). Check (1) raises unconditionally
  whenever either table is empty, so check (2) (`:1033-1046`) is never reached here; deleting
  check (1) (M39) turns this test red. **Converse (M3):** a **balanced** 20 plus / 20 minus
  split over the same ≤ 9 regions (deviation 0) passes under any reading of the balance bound —
  spec's flat `strand_tol` or the code's `min(0.45, max(strand_tol, 1.5/√R))` (`:1034`) — so it
  shows check (1) stays quiet with both strands present, without freezing Q6. Do **not** use a
  39/1 split: `b50ab4f` capped the tolerance at 0.45 (previously uncapped and vacuous at R ≤ 9),
  so that 0.475 deviation now also raises via check (2) at R = 9 [Verified: this revision,
  `guards.py` rerun] — round 2's "does not raise" claim for it no longer holds.
- `t3_strand_balance_guard_fires` (parametrised, R = 200): plus fraction 0.80 raises with
  `match="strand fraction .* from 0.5"` (check (2), `:1033-1046`). 0.50 and 0.55 pass (§4.4); the
  0.45 cap does not bind at R = 200 (`1.5/√200 ≈ 0.106`), so `b50ab4f` leaves this case unchanged.
  Use a 600-kb toy contig if 200 tiles do not fit the 6,000-bp one; 0.16 s [Verified: review].
- `t3_count_guards` (parametrised, L2): missing `sequence` column → `ValueError`,
  `match="has no 'sequence' column"`; missing `fragment_array` column → its own
  `match="has no 'fragment_array' column"` (both `:913-919` — the two no longer share one
  match). All-empty admission (any cause) → `ValueError` (`:1047-1058`, reworded by `b50ab4f`/
  F19 from "MAPQ removed ALL" to "EVERY fragment was removed before counting"); match the short,
  stable `"EVERY fragment was removed"` rather than the full text (L5/L8: a further rewording
  means "update the match", not a defect).
- `t3_golden_h5_matches_bruteforce`: `golden.small.chr6.frag.h5` and its chr6 window FASTA,
  against the oracle over raw records from `FragmentsH5.fetch_array` (strands decoded). Always runs.

**T4 expectation and propensity**
- `t4_fl_end_weight_matches_enumeration`: R < max_fl and R > max_fl, `min_fl > 1`.
- `t4_end_weight_total_equals_region_length_sum`: the derived identity, per region set.
- `t4_uniform_hexamer_counts_matches_enumeration_toy`: N-free regions, exact (rtol 1e-12), where
  "N-free" covers the **padded** span `[g0−3, g0+R+max_fl+3)`, not just the region (M6). On the
  region holding the planted single N (`4400-4560`), assert **full equality** of `N_start`/
  `N_end` against the plain enumeration (which gates each window on its own validity) — not
  merely "N-containing windows get no mass at their own site" (too weak to kill M42)
  [Verified: review, `null.py`: full equality GREEN at HEAD, RED under M42, on this region].
- `t4_uniform_hexamer_counts_chr6`: N-free regions inside the populated window equal the
  enumeration. A region that starts at 99,110,000 gets natural N in its first start windows
  through the real fetch, and those windows get no mass. Always runs.
- `t4_propensities_forward_exact`: on the skewed fixture, with the in-test check
  `max(N)/min(N>0) ≥ 10` and an assertion that `C/N` differs from `C` in at least one cell.
  `r == C/N` for both forward tables, and 0 where `N ≤ min_expected`. The minus tables are
  covered by `t4_null_identity` only.
- `t4_null_identity[start_fwd|end_fwd|start_rev|end_rev]`: §4.3. All four green at HEAD.

**T5 sampler**

These tests use a *recording rng*. Its interface follows the module's exact call sequence
[Verified: Research A]:
- `binomial(n, p_plus)`, once per region, unconditional (`:596`). Returns a forced `n_plus`.
- `choice(pos, size=k, replace=True, p=w_s/tot)`, only if `k > 0` and `tot > 0` (`:612`).
  Records `p` and returns supplied values from `pos`. A strand block with `tot ≤ 0` makes
  no call (`:610-611`).
- `random((len(starts), 1))`, only if `live.any()` (`:632`). Returns supplied `u` of shape
  `(n, 1)`. `u` is drawn per **kept** row, after rows with zero length weight drop
  (`:618-622`).

Any other attribute access raises, so an interface change fails loudly. The stub also asserts
the call sequence. No real draw is pinned. **(L8) That call-sequence assertion is a white-box
coupling to today's implementation, not a spec contract (output order itself is non-contract,
`simulator_spec.md:261-263`); a harmless refactor should update the stub, not be read as a defect.**
- `t5_start_probabilities[plus|minus]`: recorded `p` == `sampler_start_p`. Plus uses `start_fwd`
  on the fwd track, and its region set must include an N region or M23 leaves it green (L6).
  Minus uses `end_rev` on the rc track. The recorded `p` is the raw start weight, before the
  `:618-622` drop. The emitted-start distribution, used by `t5_planted_propensity_recovered`,
  zeroes zero-length-row starts and renormalises.
- `t5_length_probabilities[plus|minus]`: for one start, `choice` returns that start M times.
  `random` returns the midpoint `u = CDF(l−1) + p_l/2` of every oracle step. The drawn
  lengths must equal the oracle support, in order. All chosen starts have positive row
  totals, so row drops cannot misalign `u`.
- `t5_zero_weight_cause[non_acgt|r_zero|f_zero]`: the length weight is a product of three
  factors. Each construction zeroes exactly one, so a zero has one known cause:
  - non_acgt: an N exactly at `i+l`.
  - r_zero: `r_end[hex(i+l)] = 0`, N-free.
  - f_zero: `f(l) = 0`.

  Cell level: length `l` gets no mass (midpoint sweep). Row level, for `non_acgt` and
  `r_zero` only: `f` is a point mass at `l`, and start `i` never appears in the output.
  `f_zero` cannot zero a row alone (`Σf > 0`). The test asserts **support only**, never the
  shortfall count. See Q4.
- `t5_point_mass_exact_output`: one-hot `r` on a de Bruijn start and end hexamer, and `f`
  a point mass. The output is exactly n copies of one fragment under any rng.
- `t5_p_plus_extremes`: `p_plus` 0 and 1 give all minus and all plus under a real rng
  (degenerate binomial, no draw pinned).
- `t5_planted_propensity_recovered`: statistical (§4.4 rules). All `r` = 1, except 20× on
  one non-palindromic hexamer h in `start_fwd` only. `default_rng(seed)`, `sample_region`
  per region, recount with `hex_at` (not `counts_from_hexamers`). Within **6σ**, conditional
  on the realised per-region `n_plus`/`n_minus`:
  - (a) plus starts at h match `Σ_r n_plus,r·p_h,r`, with `p_h` from `sampler_start_p`;
  - (b) minus fragments whose genomic-start hex is `RC(h)` stay at the unplanted rate.

  Sizing [Inferred]: h appears once in a ~3,900-bp region, so its start weight is 20/3,919 ≈
  0.5%. Review measured a 21.5σ gap at n = 10⁵ total draws [Verified: review `t5.py`]. The gap
  scales with √n. At **n = 4 × 10⁴** it is ~13.6σ (M27 ≥ 12σ), and h has an expected count of
  ~100 (§4.4). Expected cost ~2 s and ~300 MB, scaled from the review's 5.26 s and 767 MB.
- `t5_fl_from_filtered_frame_within_bounds`: `FragmentLengthDist.from_srdf` on a
  `count_sample` frame gives `max_fl ≤ 180` and `min_fl ≥ 25`. v1's closed loop gave 25/180
  [Draft measurement]. This contract protects the sampler's unchecked boundary (spec Settled:
  the bound is enforced once, at fragment-array construction).

**T6 writer and round trip** (one closed loop on the toy: count → fl → expectation →
propensities → simulate → index → build → recount)
- `t6_bed_text_shape`: 8 tab-separated columns, no header, sorted by `(contig,start,stop)`,
  `n_rows_written == n_drawn`.
- `t6_round_trip_through_real_reader`: bgzip and tabix outputs exist; coordinates and
  strands read back exactly; a written `mapq=9` gives 0 admitted at `min_mapq=10`; no cell
  barcode on read-back [Unverified: the reader attribute for barcodes].
- `t6_recount_equals_distinct_pairs`: recount `region_counts` == distinct `(start,stop)` per
  region, and the four tables == `bruteforce_count` on the written rows. The h5 thus holds
  exactly the fragments written, up to the Settled dedup.
- `t6_writer_guards` (parametrised). Each case has its own exception type and `match`
  [Verified: read, `:701-746`]:
  | case | exception | `match` | line |
  |---|---|---|---|
  | `.gz` out_path | `ValueError` | `"write a PLAIN bed"` | `:702` |
  | `region_counts` shape ≠ rows | `ValueError` | `"region_counts has shape"` | `:709` |
  | missing column (e.g. `stop`) | `ValueError` | `"has no 'stop' column"` | `:715` |
  | `fa.length` ≠ stop−start | `AssertionError` | `"fragment_array.length"` | `:733` |
  | sequence length ≠ R+186 | `AssertionError` | `r"sequence is \d+ b, expected"` | `:739` |

  The `AssertionError` cases are explicit `raise` statements, so `python -O` keeps them.
- `t6_fragment_length_dist_guards` (parametrised). Each case has `match=` copied from its raise.
  Cases and lines: non-1D or negative or zero-sum counts (`:345-350`, `match` on "counts");
  empty frame (`:384`); duplicate lengths (`:386`); missing column (`:380`);
  `from_srdf` with no `fragment_array` column (`:404`) or no fragments (`:410`). Plus `densify`:
  from `{25:1, 27:3}` the densities are 0.25 at 25, 0 at 26 and 0.75 at 27.
- `t6_input_guards`: `load_sample_dataframe` raises `ValueError`, `match="no samples given"`
  (`:226`). `uniform_hexamer_counts` raises `ValueError`, `match="WITHOUT fragment arrays"` (`:504`).

**T7 hygiene**
- `t7_module_doctests_execute`: `doctest.testmod(module)` asserts `attempted > 0` and
  `failed == 0`. The 2 doctests pass when run directly [Verified: Research A], but default
  `make test` never collects them.
- `t7_oracle_is_independent`: AST check that the oracle imports nothing from
  `background_model` or `fragmentomics_tools`.
- `t7_no_removed_feature_imports`: AST check that the module file does not import `flgc` or
  `simulator.capture/precompute/weights/sampler/emit`. This is an owner contract
  (self-contained module). The stale comment at `:30-32`/`:92-94` invites a re-import. It
  uses AST, not `sys.modules`, because other suites import the old modules. It does not
  cover `simulator/__init__.py`, which still imports `capture` (F15).

**R real data** (`real_data` marker; §2.4)
- `r1_count_sample_matches_raw_reader_oracle`
- `r2_uniform_hexamer_counts_matches_pysam_enumeration`
- `r4_real_region_round_trip_chr21` (§2.4; replaces `r3_bed_format_and_support_real`, deleted — M1/M2)

**Not included:** a test that `tabix_index` consumes the plain BED (pysam behaviour; the
fixture builder documents it). No test of `n_short_regions` (F14).

**Requester items → tests.** "Substituted" means the test checks a proxy, not the item itself.

| Requester item | Pinned by | Status |
|---|---|---|
| (1) strand routing and rc permutation | `t1_counts_from_hexamers_routing`, `t3_count_sample_matches_bruteforce`, `t5_start_probabilities[minus]`, `t4_null_identity[*]` | Covered |
| (2) asymmetric frame, no pad term | `t0_frame_through_attach_sequence`, `t2_max_overhang_fragment_counted` | Covered |
| (3) invalid window = real index | `t3_n_window_fragment_dropped_from_tables`, `t0_single_n_invalid_at_every_offset`, `t5_start_probabilities[plus]` | Covered. M42 adds the `uniform_hexamer_counts` gate |
| (4) length weight = 3-factor product | `t5_zero_weight_cause[non_acgt\|r_zero\|f_zero]` | Covered (Q4 open) |
| (5) propensity = C/N | `t4_propensities_forward_exact`, `t4_null_identity[*]` | Covered |
| (6) strand statistics; region is the unit | §4.4. Simulator output is i.i.d. per fragment given `n_r`. No real-data fraction is asserted | Covered by design |
| (7) U1 vs bytes, half-open bounds, unclipped overhangs | `t1_reader_strand_labels_are_str`, `t2_length_bounds`, `t2_max_overhang_fragment_counted` | Covered |
| BED round trip | `t6_round_trip_through_real_reader`, `t6_recount_equals_distinct_pairs` | Covered |
| Input guards | `t6_writer_guards`, `t6_input_guards`, `t3_count_guards` | Covered. Exception types per case |
| Emptied vs lopsided strand | `t3_one_empty_strand_raises` (≤ 9 regions), `t3_strand_balance_guard_fires` (R = 200) | Covered |
| 10 real regions → h5 with exactly the fragments written | T6 on the toy (`t6_recount_equals_distinct_pairs`), **and** `r4_real_region_round_trip_chr21` on real `chr21` data | **Pinned** (M2): a `chr21`-restricted build costs ~11 s, not 76 s (§2.2) |

**Run** (biomarker_env first on PATH):
- Synthetic: `make test PYTEST_ARGS="tests/test_count_hexamers_rdf.py -q -rs"`.
- Real, acceptance: `REQUIRE_REAL_DATA=1 make test PYTEST_ARGS="tests/test_count_hexamers_rdf_real.py -q -rs"`.
- Full suites before and after: `make test` and `make test PYTEST_ARGS="tests/ -q -rs"`.

**Runtime budget**

| Item | Cost | Source |
|---|---|---|
| Module import (once per session) | 7.24 s wall | [Verified: Research C] |
| Toy h5 build, ~4 scenarios + 1 for R=200 | 0.15-0.16 s each | [Verified: Research C] |
| Toy `count_sample` (~20 calls), chr6 fetches (T4), doctests | 0.04 s, ~5 ms, 5.94 s each | [Draft measurement; doctests Verified: Research A] |
| T5 recovery at n = 4 × 10⁴ total draws | ~2 s, ~300 MB | [Inferred: scaled from review's 5.26 s, 767 MB at n = 10⁵] |
| R1: h5 open + FASTA open + `count_sample` (20 regions) | 0.12 + 0.01 + 1.83 s | [Verified: Research C] |
| R1 oracle: raw fetch + Python brute force | unknown | [Unverified: time it in P6] |
| R2: `uniform_hexamer_counts` + Python double loop over ~30,720 bp × 156 lengths | 0.03 s + a few s | [Verified / Inferred] |
| R4: count/sample/simulate pipeline + build (`chr21` only) + recount | ~11 s build [Verified: review, `realbuild/rb.py`] + rest unknown | [Unverified: time the non-build steps in P6] |
| **Never, except R4:** an h5 build against the unrestricted real FASTA or chr6 | 76.26 s / 26.2 s | [Verified: Research C] / [Draft measurement]. R4's `chr21`-restricted build is the one exception, at ~11 s (M2). |

Budget: **≤ 30 s per file**, measured by the implementer. Above that is a finding.

## 6. Findings for the owner (report, do not fix)

| ID | Sev. | Finding | Evidence | Owner approval? |
|---|---|---|---|---|
| F1 | FIXED | `propensities()` divided the minus tables by swapped denominators at `a409186`. `counts_from_hexamers` puts `perm[hex(STOP)]` into `start_rev` (`:320-321`), so `E[start_rev] ∝ N_end[perm]`. The fix at `844f227` pairs them that way (`:840-841`, with a comment saying the crossing is on purpose). At `a409186` the real-data effect (800 of 66,649 regions, history) was median 1.2% and max 16% per cell. Found independently by Research B and by the closed loop in the `844f227` message. `t4_null_identity` guards it. M41 restores the old pairing. Spec `:137-138` still does not state the crossed pairing. | [Verified: Research B at `a409186`; `git` reflog; `count_hexamers_rdf.py:838-841`] | Done (owner-approved). Spec wording open |
| F2 | MED | Docstrings (`:135-137`, `:246-248`) say invalid windows carry index 0. They carry the N-as-A index (`safe = np.where(win==255, 0, win)`, `:150`). Fix the docstring only. | [Verified: Research A] | No (doc only) |
| F3 | DONE | **Fixed in `b50ab4f`.** The balance check was `tol = max(strand_tol, 1.5/√n_regions)`, vacuous for n_regions ≤ 9 (reaches/exceeds 0.5, and `plus_frac` cannot deviate from 0.5 by more than 0.5). Now `tol = min(0.45, max(strand_tol, 1.5/√n_regions))` (`:1034`), `\|plus_frac−0.5\| > tol` raises (`:1035`): the check can always fire. It is still flat ±0.1 above 225 regions. A whole table swap gives `plus_frac' = 1 − plus_frac` and keeps every total, so no runtime check sees it — that half of the finding stands, by design. | [Verified: read, `b50ab4f`] | Done |
| F4 | LOW, DONE | `:934-938` (even total) and `:954-960` (pair identities) are tautologies given `counts_from_hexamers`; they cannot fire. **`b50ab4f` corrects the claim in the code's own comment** (it no longer says these "catch broken strand routing", and spells out why, matching this finding) and keeps the checks only as a malformed-`counts` guard (M4). Do not count them as routing protection; M39 drops both sites from its list (M4). | [Verified: read, `b50ab4f`] | Done |
| F5 | MED | Any strand label other than `'+'` counts as minus (`:314`): `b'+'`, `'.'`, `''`. Check (1) (`:994-1002`) fires if **either** table is empty. So an all-bytes frame is caught loudly, but only if `n_counted > 0`: when both are zero the `if n_counted :=` guard skips the check. A mixture with some `'.'` or bytes silently inflates minus. The raw reader yields bytes (§1.2), so the str contract rests on the RFA layer. | [Verified: Research A] | Yes, if it becomes a raise |
| F6 | LOW/MED | Validity asymmetry. C needs both cut sites valid (`:294`). `N_start` gates only its own site (`:547-548`, `:556`). On this region set the effect is unmeasurable: 0 invalid cut windows in 1,373,600 (800 regions); 4 of 66,649 regions hold a non-ACGT base (11 of 114,769,578 bases, full scan 22.1 s). It can matter for region sets with gaps near cut sites. | [Verified: Research A/B] | **Yes** (Q2) |
| F7 | LOW | `propensities()` sets cells with `C>0, N=0` to 0 and raises no error (`:846-848`). | [Verified: grep, 0820409] | Owner, with F1 |
| F8 | LOW | `_hexamers_at` promises `IndexError` (`:242`, `:279`) but wraps a negative `pos` silently. On the research probe's sequence, `_hexamers_at(seq, np.array([-2]))` returned index 283 with `valid` True. That sequence is not recorded, so 283 is not a constant. On another sequence the wrap gave index 0 with `valid` True [Verified: review]. Only the `starts_0 ≥ 0` gate in `filter_fragments` prevents it, so tests cover the gate (T2), not the helper. | [Verified: Research A probe] | No |
| F9 | INFO | `rc_permutation()` returns a shared writable cached array (`:189`). A caller that writes to it corrupts later calls. Proposal: `setflags(write=False)` (engineering) plus a test that a write raises. No test lands unless that fix lands, because today it can only be red or re-assert the hazard. | [Verified: Research A] | No (engineering), but confirm |
| F10 | INFO | Stale comments and docs (§1.4). The module doctests pass but default `make test` never runs them. | [Verified: Research A] | No |
| F11 | LOW | `FragmentLengthDist` casts counts through int64 (`:343`, `:381-382`), so float counts truncate silently. Not tested: either assertion presumes a decision. | [Verified: Research A] | Yes, if changed |
| F12 | INFO | **v1 was wrong here.** The writer guards (`:732-746`) are explicit `raise AssertionError` (`:733`, `:739`), so `python -O` keeps them. The only bare `assert`s are `:175-176` in `hexamer_vocabulary`, which `-O` strips. They run once at build of a derived table, so the exposure is small. | [Verified: Research A] | No |
| F13 | MED | Spec/code conflict, **still open after `b50ab4f`**. Spec `:233-236` says the guard asserts within `strand_tol`. The code now uses `min(0.45, max(strand_tol, 1.5/√n_regions))` (`:1034`) — the F3 cap fixed the vacuous-below-10-regions defect, but did not reconcile the formula with the spec's flat `strand_tol`. Either the spec or the code still needs the owner's wording. Tests avoid the bound (§4.4), via a converse that passes under either reading (M3). | [Verified: read, `b50ab4f`] | **Yes** (Q6) |
| F14 | LOW | Spec Settled says dropped zero-length starts get "no counter, no test" (owner 2026-10-06). But `simulate_fragments_to_bed` returns `n_short_regions` (`:726`, `:785`): a counter exists. Flagged, not tested. | [Verified: Research D] | Confirm (Q7) |
| F15 | LOW | Removed-feature code still live: `scripts/run_simulator.py:66` imports `simulator.capture` at module level; `simulator/__init__.py` imports `capture`, `build_predict_lut`, `midpoint_index_arrays`; `Makefile` keeps `FLGC_PYTHONPATH`. Flag as stale; do not fix here. | [Verified: Research D] | Deletion is the owner's call |
| F16 | LOW | `make test` depends on PATH: it runs bare `python -m pytest`. The wrong python gave exit 2 and 11 collection errors. A fix pins the interpreter in the Makefile (engineering). | [Verified: coordinator log] | No (engineering) |
| F17 | MED, LIBRARY | A minus-strand region flip has **two** defects (M5; v3 under-described this as one). `_switch_plus_with_minus_and_minus_with_plus` (`fragmentomics_tools/fragment_array/fragment_array.py:123-127`, compare at `:126`) runs on the raw `\|S1` array from `from_fragments_h5` (call at `:1793`): the bytes compare with `"+"` gives all-False, with no error, so labels are **never swapped**. Separately, `starts_0`/`stops_0` (and methyl/gc) are reversed with `[::-1]` to flip coordinate order, but `fragment_strands` is **not reversed**, so after the swap-that-didn't-happen the labels are also **misaligned against the reordered coordinates** — a fix that only decodes bytes (swaps `+`/`-`) without also reversing the label array still mislabels some fragments. Measured on `chrT:500-1000,-` with `(600,700,+)`,`(650,760,+)`,`(800,900,-)`: got `[(100,200,'+'),(240,350,'+'),(300,400,'-')]`, expected `[(100,200,'+'),(240,350,'-'),(300,400,'-')]` — a decode-only fix gives `['-','-','+']`, wrong at 2 of 3. `generate_weights_callback` weights are also not reversed [Inferred]. The `U1` coercion runs later, at `:299`. Reached only by minus-strand regions; the current region BED is strandless, so not reachable now [Inferred]. | [Verified: review `minus.py`; numpy 2.2.6] | **Yes** (Q8). No test until ruled |
| F18 | LOW, DONE | **Fixed in `b50ab4f`.** The `count_srdf` balance-check message (`:1036-1046`) no longer says "Strand is clustered within regions"; it now states overdispersion, mechanism unmeasured, matching §4.4. | [Verified: read, `b50ab4f`] | Done |
| F19 | LOW, DONE | **Fixed in `b50ab4f`.** The all-empty-admission error (`:1047-1058`) no longer says "MAPQ removed ALL"; it now names the whole admission chain (MAPQ, dedup, length filter, start-in-region) and keeps MAPQ only as the first thing to check. `t3_count_guards`'s match string must track this wording (L2/L5). | [Verified: read, `b50ab4f`] | Done |

**F1 is closed.** The fix is in `844f227`, and the owner approved it. No xfail and no owner
gate apply. `t4_null_identity` (all four tables, plain assertions) is the regression guard.
Mutation M41 checks that the guard still catches the old pairing.

## 7. Deliberately not tested

- **Removed by owner decision** (spec): capture/GC modelling, `predict(L,gc)`, ZTNB fit,
  flgc, dup-histogram sidecar, store construction, model crop width, oracle. Any remnant is
  STALE (§1.4, F15).
- **Settled** (spec): bulk GC not modelled; no `L_MIN`/`L_MAX` re-check in the sampler (T5 tests
  the from-filtered-frame contract only); `p_plus = 0.5`; starts drawn with replacement, so the
  round trip uses distinct pairs; draw order is not contract; dropped zero-length starts get no
  counter or shortfall test (F14); the dedup key omits strand (asserted only for the T6 invariant).
- **The accepted divergence** `region_counts.sum() ≥ n_counted`. Not asserted to be equal.
- **Minus-table values from real data.** The exact check is `t4_null_identity` only.
- **Minus-strand region input** (F17). No test until the owner rules (Q8).
- **Any real-data plus fraction** (§4.4). The `1.5` constant itself (F13).
- **The index value of an invalid window** (F2). The old-generation tests and the CLI h5
  path. F5/F7/F11 behaviour, which a test would freeze.
- **The even-total and pair-identity checks** (`:934-938`, `:954-960`) are tautologies, not
  routing protection (F4, M4) — no test can turn either red, and `b50ab4f`'s own code comment
  now says so. M39 excludes both sites from its list.
- **Pinned `default_rng` draws.** None anywhere.

## 8. Implementation plan

| Phase | Work | Depends on | Exit criterion |
|---|---|---|---|
| P0 | Re-measure both baselines at the implementation HEAD, biomarker_env first on PATH. Re-run the review measurements that P2 and P6 cite | — | Numbers recorded. Last measured at `a409186`: 2F/399P/3S and 673P/0S. Expect `tests/` near 677. |
| P1 | `tests/conftest.py` (marker, REQUIRE hook). Oracle module and its self-tests (de Bruijn property, palindromes = 64). Confirm the sibling import. T0, T7. | P0 | Green. The hook turns a forced `real_data` skip into a failure under `REQUIRE_REAL_DATA=1`. |
| P2 | Fixture builders: toy FASTA (asserted properties), BED → tabix → h5 helper, regions. T1, T2, T3. | P1 | T3 exact equality green. The M12 self-check discriminates. |
| P3 | T4 (§4.3), with M41 run in the same phase | P2 | All four null-identity tables green at HEAD. M41 red on `[start_rev]` and `[end_rev]`. |
| P4 | Recording rng. T5. Time the recovery test. | P1 | Recovery gap ≥ 12σ under M27, by the T5 arithmetic. |
| P5 | Closed loop on the toy. T6. | P2-P4 | Distinct-pair invariant holds. |
| P6 | Commit `tests/data/simulator_real_regions_20.bed` (10 `chr21` rows + the rest elsewhere). Pin the real h5 and FASTA paths. R1, R2, R4. | P1-P5 | `REQUIRE_REAL_DATA=1` run: 0 skipped, real file ≤ 30 s. `tests/data` grows by ~1 KB only. |
| P7 | Mutation sweep M1-M42 on scratch edits (module or library, never committed), reverted. Run the §4.2 cross-check by script | P1-P6 | Every row red **except the listed equivalent mutants** (M14, H1). Table in the PR, with the site column. |
| P8 | After-baselines | P7 | See below. |

P8 targets:
- `make test` unchanged (2 failed / 399 passed / 3 skipped; it does not collect `tests/`).
- `tests/` with `REQUIRE_REAL_DATA=1`: 673 + 4 (`844f227`) + N_new passed, **0 skipped**.
- Each new file ≤ 30 s. `tests/data/` gains only the 20-row BED. Nothing staged except the
  test files, the oracle, `tests/conftest.py` and the BED.

## 9. Open questions for the owner

- **Q1 (propensity tests).** Delete the three partial tests in
  `tests/test_simulator_propensity_denominators.py`? Keep `test_the_rev_tables_are_not_swapped`
  either way (§1.3). The new suite does not depend on the answer.
- **Q2 (F6).** Should `N` use the same both-sites validity as C? The effect is unmeasurable
  on the current region set. The tests stay neutral until you rule.
- **Q3.** Default `make test` does not collect `tests/`, and it depends on PATH (F16).
  Change the `PYTEST_ARGS` default and pin the interpreter, or document both?
- **Q4.** Does "dropped zero-length starts: no test" also exclude the *support* assertion in
  `t5_zero_weight_cause`? The design assumes not, because it asserts no shortfall count. If
  you read it otherwise, the row-level variants go and the cell-level variants stay.
- **Q5.** Real tier: default-on with a visible skip (as designed), or opt-in only? And
  should the acceptance gate `REQUIRE_REAL_DATA=1` also run in CI or the batch container?
- **Q6 (F13).** Which is authoritative for the balance bound: spec `strand_tol`, or the code's
  `min(0.45, max(strand_tol, 1.5/√n_regions))`? `b50ab4f` fixed the formula's vacuous-below-10
  defect (F3) but did not pick between spec and code, so this question is unchanged in kind.
- **Q7 (F14).** Keep `n_short_regions` (and amend the Settled line), or remove it?
- **Q8 (F17).** Is a minus-strand region a supported input? If yes, rule the label defect —
  bytes not swapped, **and** order not reversed, so a decode-only fix still mislabels some
  fragments (M5). Then choose: fix both parts of the flip, or make `count_sample` refuse
  minus-strand regions. If no, no test is written.

## Least sure of

- **h5 order for equal `(start,stop)`** [Verified: review, one case]. M12 relies on it. A
  second case in P2 settles it. If the h5 always puts B first, M12 needs a direct
  `filter_fragments` test on a hand-built array.
- **Minus-strand regions** (F17). No test feeds a minus-strand region, so the flip trap is not
  covered. The owner rules first (Q8).
- **Runtimes and paths.** R1 and R2 runtimes are unmeasured; R4's build step is pinned at
  ~11 s but the rest of its pipeline is not. The T5 cost is scaled. The real h5 path is absent
  from the research record. P4, P6 measure and pin these.
- **Cites not touched by this revision** may still carry pre-`b50ab4f` line numbers; P0's
  baseline re-measurement is the right place to re-walk them exhaustively.
- **Sibling import of the oracle** [Inferred]. P1 confirms it. **Cell barcode read-back:** the
  reader attribute is unnamed.
- **Reviewer measurements are not re-run.** No Python ran in this revision. Cites tagged
  `[Verified: review]` come from the review scratch files. P0 and P2 re-measure them.
- **Resolved in this revision:** the `fetch_array` filters (§1.2), the M2 doctest question
  (review: the doctests do not expose endianness), and the M34 site (`region.py:854-865`).
- **Guard tests (T3 strand guards, `t3_count_guards`, T6 guards).** These are the closest to
  re-assertion. Each turns a silent downstream failure into a loud one, and the requester
  listed them as lost evidence. They are still the first candidates to cut.
