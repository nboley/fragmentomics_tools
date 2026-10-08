# Test design: `background_model/simulator/count_hexamers_rdf.py`

**Status (2026-10-08): v5, reconciled against HEAD `eda1b14`. The tests this design proposed
are now LARGELY IMPLEMENTED** (`tests/test_count_hexamers_rdf.py`, 43 test functions, added in
`45b32ec` and closed out through `02f7f39`) **— this revision is a drift fix, not a new
proposal.** v4 (below) was last synced at `b50ab4f`; 14 commits landed between `b50ab4f` and
`eda1b14` and this file was only partially kept up to date (`2eb09ac` updated the mutation-matrix
status; nothing after that touched this file until now) [Verified: `git log --oneline --
docs/pending/simulator_test_design.md` shows `2eb09ac`, `aeb35b9`, `8eb5dc2` only; `git log
--oneline b50ab4f..eda1b14` lists 14 commits]. The concrete drift this revision fixes:

- **The strand-balance guard this design spent §4.4, F3, F13, M3 and Q6 on was REMOVED**
  (`38e5198`, "doesn't seem useful") — not merely capped. F3/F13/M3/Q6 are VOID; see each
  section below for the replacement text, not a deletion, so a reader who remembers them finds
  out what happened rather than finding them silently gone.
- **The real-data tier (R1, R2, R4) described in §2.4/§2.5 and catalogued in §5 was never
  built.** No `tests/test_count_hexamers_rdf_real.py`, no `tests/conftest.py`, no
  `tests/data/simulator_real_regions_20.bed` exist at `eda1b14` [Verified: `ls tests/conftest.py`
  → does not exist; `ls tests/test_count_hexamers_rdf_real.py` → does not exist; `grep -rn
  "simulator_real_regions\|real_data" tests/` → no matches]. This was proposed design, and it
  is still only that — marked throughout.
- **M12, M15, M24 and M25 — this design's own §4.2b, "the four mutations with no test" — were
  closed in `02f7f39`.** They now have tests, each verified red under its own mutation.
- `make test`'s default target now collects `tests/` (`dc1df6f`, `aefa71e`), contradicting this
  design's "Runs where?" bullet, Q3 and the P8 exit criterion as written below.
- The `propensities()` pairing fix cited at `:838-841` throughout now sits at `:846-852`
  (`count_hexamers_rdf.py`'s docstring grew).
- The C/N validity-gate question this design left open as F6/Q2 is now a closed owner
  decision (`docs/pending/simulator_spec.md`, Settled, 2026-10-08): "not interested in further
  pursuing this." Not re-opened here.

None of the above required touching module code to discover — it is git history plus grep.
Rebased on HEAD `0820409` (parents `844f227`, `a409186`) [Verified: `.git` reflog]. The premise
"the module has zero tests" is **wrong** now: `844f227` added 4 tests
(`tests/test_simulator_propensity_denominators.py`) and fixed F1. Those pin the pairing only.
The rest of this design is still needed. v3 adds the F1 status, the propensity test file
(§1.3), review round 1's contract fixes, and a site column in the mutation matrix. **v4
addresses round 2's review. HEAD moved underneath it, `0820409` → `b50ab4f`** [Verified:
`git log -3`], fixing F2/F3/F13/F4/F18/F19 and shifting the module by +26 lines; every cite
this revision wrote was re-grepped there, but untouched cites may still carry old numbers.
**v5 (this revision) re-grepped the citations named in the bullets above and in the sections
they touch; citations elsewhere in this document were NOT individually re-walked line by line
— see "Least sure of" at the end for exactly what that leaves uncertain.**
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
  loudly. **None of this tier exists at `eda1b14`** [Verified: no `tests/conftest.py`, no
  `tests/test_count_hexamers_rdf_real.py`, no `tests/data/simulator_real_regions_20.bed`] —
  read §2.4/§2.5/§9 Q5 as a still-open proposal, not as implemented behaviour.
- **Tiers.** T0 encoder/frame, T1 strand routing, T2 admission, T3 `count_sample` against a
  brute-force oracle, T4 expectation and propensity, T5 sampler, T6 BED/h5 round trip,
  T7 hygiene, R real data. Every oracle imports nothing from the module. Every test names
  at least one mutation that it must turn red (§4.2).
- **Finding F1 (fixed in `844f227`, owner-approved).** `propensities()` divided the minus
  tables by swapped denominators. The fix pairs `start_rev` with `N_end[perm]` and `end_rev`
  with `N_start[perm]` [Verified: `count_hexamers_rdf.py:846-852`, re-grepped at `eda1b14`;
  was `:838-841` as of `b50ab4f` — the docstring above the pairing grew]. `t4_null_identity`
  (test file: `test_null_identity`, parametrized over the four tables) is the exact regression
  guard. Mutation M41 restores the old pairing and must turn two rows red.
- **Finding F2 (MEDIUM).** The docstrings say invalid windows carry index 0. They do not.
  `hexamer_indices` reads N as A inside the window [Verified: Research A]. A consumer that
  skips the validity gate miscounts into a *neighbouring* hexamer, not into cell 0.
- **Finding F17 (library, owner-gated).** A minus-strand region flip leaves the strand label
  unswapped. No test pins it until the owner rules (Q8).
- **Strand statistics.** For simulator output the unit is the fragment, conditional on the
  realised per-region strand counts. For real data the unit is the region, and no test
  asserts a real-data plus fraction. The real-data excess is overdispersion. Its mechanism is
  unmeasured (§4.4).
- **Runs where? SUPERSEDED.** This bullet, Q3 and F16 described a real defect that has since
  been fixed twice over. `dc1df6f` made default `make test` collect `test/ tests/
  fragmentomics_tools/` together — the Makefile's own comment says the omission "meant 677
  tests never ran under the default target" — and `aefa71e` fixed the resulting interpreter/PATH
  trap (pinning `PYTHON` alone took the suite from 2 failed to 54, because `bedtools`/`bgzip`/
  `tabix` live beside the interpreter, not in it; the target now puts the interpreter's own
  `bin/` on PATH too) [Verified: `Makefile:1-55`, `:229-268`, read at `eda1b14`]. **Today, plain
  `make test` is self-contained and collects this module's tests by default.** Q3 is answered
  and F16 is done; §8's P8 exit criterion below is stale and corrected there.

## 1. Ground truth

### 1.1 Baselines (worktree `background-model-work`, `biomarker_env`)

**Re-measured this revision, at `eda1b14`, by actually running the command** (not taken from
a commit message): `make test` with no `PYTEST_ARGS` override.

| Command | Result | Source |
|---|---|---|
| `make test` at `eda1b14` (single default target, now `test/ tests/ fragmentomics_tools/`) | **2 failed, 1125 passed, 3 skipped, 208.83 s, make exit 1 (pytest found failures; `make` itself errors on that, which is not a hang — see Makefile's own exit-137 check, which did not fire)** | [Verified: ran in this session, full log in `/tmp/maketest_eda1b14.log`] |

The 2 failures are still the known missing-data cases: `test/test_formats.py::
test_slice_encode_big_wig` and `test/test_region.py::test_get_one_hot_encoded_sequence`
[Verified: same run]. This single number **supersedes the two-row table below**, which
described a split that no longer exists: `dc1df6f` and `aefa71e` made default `make test`
collect `tests/` (background_model) together with `test/` (library) and
`fragmentomics_tools/`, self-contained (PYTHON pinned, and the interpreter's own `bin/` put on
PATH so `bedtools`/`bgzip`/`tabix` resolve regardless of the caller's shell). There is no
longer a separate `make test PYTEST_ARGS="tests/ -q"` invocation to run; passing that override
today would just re-run a subset of the same default target.

**History, kept for attribution only — do not run these.** Measured at `a409186`, before the
split was closed:

| Command | Result | Source |
|---|---|---|
| `make test` (default `test/ fragmentomics_tools/`, `--doctest-modules`) | 2 failed, 399 passed, 3 skipped, 92.39 s, exit 2 | [Verified: coordinator log, at `a409186`] |
| `make test PYTEST_ARGS="tests/ -q"` | 673 passed, no skips reported, 218.63 s, exit 0 | [Verified: coordinator log, at `a409186`] |
| `make test` under the wrong python (claude-mcp, py3.13) | exit 2, 11 collection errors | [Verified: coordinator log] (F16, now fixed — see "Runs where?") |

The counts matched the v1 draft at the time. The worktree
CLAUDE.md states 4 failed / 386 passed / 3 skipped (2026-09-27). That line was already stale
against those measurements, and is further stale against the `eda1b14` row above.

The Makefile today pins the interpreter (`PYTHON ?=
.../biomarker_env/bin/python`) and puts its `bin/` on PATH rather than running a bare
`python -m pytest` off the caller's PATH [Verified: `Makefile:229-247`, read at `eda1b14`].
The `FLGC_PYTHONPATH` prepend (F15) is still present and still named a stale directory-naming
concern only (`Makefile:220-229`). Default `make test` now DOES collect `tests/`, superseding
the next sentence, which is history: ~~it never runs `tests/` and never runs the doctests in
`background_model/`~~. Run directly, those doctests pass: `2 passed in 5.94 s`
[Verified: Research A] — and now also run as part of the default `--doctest-modules` target.

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
  `:193-199`) [Verified: grep, still true at `eda1b14`]. **The real, stronger guard that exists
  today is `test_encoder_matches_oracle_all_4096`** (`tests/test_count_hexamers_rdf.py:249`),
  which checks all 4096 hexamers against an independent oracle rather than against the second
  copy of the encoder — stronger because the two copies could in principle drift in step and a
  copy-vs-copy check would not see it. If the comment at `:93` is ever touched, it should cite
  that test by name, not the nonexistent one.
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

**NOT IMPLEMENTED at `eda1b14`.** Everything below this line in §2.4 and all of §2.5 describes
a proposal that was never built: no `tests/test_count_hexamers_rdf_real.py`, no
`tests/conftest.py` (so no `real_data` marker and no `REQUIRE_REAL_DATA` hook), and no
`tests/data/simulator_real_regions_20.bed` exist in the worktree [Verified:
`ls tests/conftest.py tests/test_count_hexamers_rdf_real.py tests/data/simulator_real_regions_20.bed`
all fail; `grep -rln "real_data\|REQUIRE_REAL_DATA" tests/` has no matches]. R1, R2 and R4
are therefore not run anywhere, `make test` cannot skip or fail on them, and the acceptance
criterion "0 skipped under `REQUIRE_REAL_DATA=1`" is unreachable because the env var and the
tests it would gate do not exist. This is a gap, not a removal — nothing here was built and
then taken out — so it stays in the document as a design to pick up, not as implemented
behaviour. Read §5's "R real data" list, §8's P6 and the Q5 open question the same way.

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
test that stays green under its mutation is not done. **Line numbers below are at HEAD
`0820409`/`b50ab4f` and have NOT been individually re-walked for this revision**, except the
rows this revision touched directly (M12, M13, M14, M15, M19, M20, M24, M25 — corrected
below) and the ones named in the header-note bullets. The module gained roughly a dozen lines
of docstring between `b50ab4f` and `eda1b14` (the F1/F2 prose and the strand-check rewrite in
`38e5198`), so most untouched numeric cites in this table are likely off by a small,
non-uniform amount. **Prefer the named function/symbol over the number** — every "module
`:NNN`" cite below names a function in its own row or the row immediately above it, and `grep
-n` on that function is one call. The function definitions and their current lines, pinned
this session: `hexamer_indices` 122, `hexamer_vocabulary` 161, `rc_permutation` 195,
`cut_site_hexamers` 276, `counts_from_hexamers` 313, `FragmentLengthDist` 336,
`fl_end_weight` 446, `uniform_hexamer_counts` 470, `sample_region` 577,
`simulate_fragments_to_bed` 657, `propensities` 801, `count_srdf` 864, `filter_fragments`
1046, `count_sample` 1071 [Verified: `grep -n '^def \|^class ' count_hexamers_rdf.py` at
`eda1b14`].

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
| M12 | admission | library `fragment_array.py:1742-1751` | dedup `:1050` moved before MAPQ | A`(s,e,'+')` mapq 5 then B`(s,e,'-')` mapq 30 (L3 split; L7 strand discriminates since hexamers are equal) | `t2_mapq_filter_precedes_dedup` — **CLOSED in `02f7f39`**, implemented as `test_mapq_filter_precedes_dedup`, verified red. The design's own caveat ("carries a caveat that must be settled first", §4.2b) was resolved: `_build_h5` uses a stable sort, ties keep fixture order, so this IS testable through `from_fragments_h5` |
| M13 | admission | module `filter_fragments`, `:1066` at `eda1b14` (was `:1082`) | `subset_fragment_lengths(l_min, l_max)` (half-open drops 180) | L=180 | `t2_length_bounds`, `r1_…` (r1 not implemented, §2.4) |
| M14 | admission | module `filter_fragments`, admission mask, `:1067` at `eda1b14` (was `:1083`) | `starts_0 <= length` | start at g0+R | **Equivalent mutant (H1): no test can turn this red.** The real reader keeps only `starts < region_stop` (`fragments_h5.py:638-640`), so `starts_0 == fa.length` is unreachable through `from_fragments_h5`; a hand-built RFA with `starts_0 == length` raises `FragmentDoesNotIntersect` inside `filter_fragments` instead (even with `validate_data=False`, since `drop_duplicate_fragments` re-validates). `t2_start_admission_half_open` is kept regardless: it still pins the tile assignment of g0−1, g0, g0+R−1 and g0+R through the real reader. |
| M15 | admission | module `filter_fragments`, admission mask, `:1067` at `eda1b14` (was `:1083`) | midpoint admission | tile straddler | `t2_straddler_counted_in_start_tile` — **CLOSED in `02f7f39`**, implemented as `test_straddler_counted_in_start_tile` (+3 cases), verified red |
| M16 | pad | module `:1140` | `right_pad=l_max` | start g0+R−1, L=180 | `t2_max_overhang_fragment_counted` |
| M17 | expectation | module `:459-461` (docstring formula at `:453`, L4) | `max(min_fl, i-R)` in `fl_end_weight` | R < max_fl, min_fl > 1 | `t4_fl_end_weight_matches_enumeration`, `r2_…` |
| M18 | expectation | module `:561` (L4: `:516` only trips the `:543-548` guard for the wrong reason; the meaningful site masks `w` to the region) | `N_end` over the region only (no flank) | any region | `t4_end_weight_total_equals_region_length_sum`, `t4_uniform_hexamer_counts_matches_enumeration_toy`, `t4_uniform_hexamer_counts_chr6`, `r2_…` |
| M19 | propensity | module `propensities`, divide loop, `:854-860` at `eda1b14` (was `:840-848`) | return `C` (no division) | skewed fixture, tandem block | `t4_propensities_forward_exact`, `t4_null_identity[start_fwd\|end_fwd\|start_rev\|end_rev]` — verified red |
| M20 | propensity | module `propensities`, `:857` at `eda1b14` (was `:846`) | `d > min_expected` becomes `>=` | N == min_expected cell | `t4_propensities_forward_exact` — verified red |
| M21 | sampler | module `:608` | minus `s_tab=r['start_rev']` | asymmetric r | `t5_start_probabilities[minus]` |
| M22 | sampler | module `:607` (L4: the track choice itself, not `:608-609`'s table choice) | minus on `fwd` track | non-palindromic region | `t5_start_probabilities[minus]` |
| M23 | sampler | module `:608` | drop `valid` on starts | N in region | `t5_start_probabilities[plus]` |
| M24 | sampler | module `sample_region`, length-weight product, `:627` at `eda1b14` (was `:616`) | drop one factor of the length weight: `fl.densities`, the end validity mask, or `r_end` | the matching single-cause construction | `t5_zero_weight_cause[f_zero]`, `[non_acgt]`, `[r_zero]` respectively — **CLOSED in `02f7f39`**, implemented as `test_zero_weight_cause[f_zero\|r_zero\|non_acgt]`, each verified red against its own factor only |
| M25 | sampler | module `sample_region`, end lookup, `:626-627` at `eda1b14` (was `:614-616`) | end hex at `i+l-1` | point-mass r_end | **CLOSED in `02f7f39`.** Implemented differently than catalogued: not `t5_point_mass_exact_output` / `t5_length_probabilities` (neither exists under those names), but a single consolidated `test_end_hexamer_offset_is_exact` — point masses on both the start and end hexamer with a real (unstubbed) rng, so the drawn length is forced to one exact value and an off-by-one in the end lookup shifts every draw by 1. Verified red. |
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
| M41 | propensity | module `propensities`, `:851-852` at `eda1b14` (was `:840-841`) | restore the pre-fix pairing: `start_rev` ÷ `N_start[perm]`, `end_rev` ÷ `N_end[perm]` | the tandem block and the null regions | `t4_null_identity[start_rev]`, `t4_null_identity[end_rev]` — verified red |
| M42 | expectation | module `uniform_hexamer_counts`, `:559` and `:567` at `eda1b14` (was `:553`, `:561`) | drop `valid` in `uniform_hexamer_counts` | planted N in a start window | `t4_uniform_hexamer_counts_matches_enumeration_toy`, `t4_uniform_hexamer_counts_chr6` — verified red |

### 4.2a Implementation status — read this before trusting the matrix above

**UPDATED this revision.** Every row now has a test — the four rows that did not
(§4.2b below) were closed in `02f7f39`, after this section (and §4.2b) were written in
`2eb09ac`. §4.2b is kept only as a "closed since" record now; its gap no longer exists.

**The matrix names a test for nearly every row. That was the DESIGN, and is now mostly the
state of the repo.** **24 of 42 mutations have been applied programmatically and verified
red** (18 from the `45b32ec` sweep + `d9c6e90`, plus 6 more from `02f7f39`: M12, M15, M24
×3, M25); the other 18 are unverified — a test exists and passes, but nothing has shown it
would fail against the defect it targets. Treat an unverified row as unknown, not as covered.
This 24/18 split matches `docs/pending/simulator_spec.md`'s Open section, written from the
same facts [Verified: spec, Open, re-read this session].

**Verified red** (sweep at `45b32ec`, re-verified at `d9c6e90`, extended by `02f7f39`): M1,
M2, M3, M4, M7, M8, M9, M10, M12, M15, M19, M20, M24, M25, M26, M31, M33, M37, M38, M41, M42.
M14 is an equivalent mutant by construction and is excluded from both the verified-red and the
unverified counts.

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

### 4.2b The four mutations with no test — CLOSED in `02f7f39`, kept as a record

**This section described a real gap when written (`2eb09ac`), and the gap no longer exists.**
All four were closed one commit later, in `02f7f39`, each verified red under its own mutation
and reverted. Kept here, rather than deleted, for the same reason the spec keeps a "Closed
since this section was last accurate" list: this section sat in two different documents
claiming an open gap for a day after it closed, which is the exact drift this whole exercise
exists to catch.

| Mutation | Test written | What it constructs | Caveat that had to be resolved first |
|---|---|---|---|
| **M12** — dedup moved before MAPQ | `test_mapq_filter_precedes_dedup` | Two fragments sharing `(start, stop)`: A mapq 5, B mapq 30, on *opposite strands* so the surviving one is identifiable (their hexamers are equal, so only strand discriminates). Correct order keeps B; mutated order dedups to A and then MAPQ drops it, losing the pair entirely. | The h5 ordering for equal `(start, stop)` — flagged below as verified on only one case. **Resolved**: `_build_h5` uses a stable sort on `(contig, start, stop)`, so ties keep fixture order deterministically. M12 did not need a hand-built-array fallback. |
| **M15** — midpoint instead of start-in-region admission | `test_straddler_counted_in_start_tile` (+3 cases) | A fragment straddling a tile boundary whose START is in tile *k* and whose MIDPOINT is in tile *k+1*. Asserts it is counted in *k* and absent from *k+1*. | None. |
| **M24** — drop one factor of the length weight | `test_zero_weight_cause[f_zero\|r_zero\|non_acgt]` | Three single-cause fixtures, one per factor of `w[l] = f(l)·r_end·valid`. | A point mass in `start_fwd` pins the HEXAMER, not the position, so collisions with other positions sharing that hexamer are common at small region sizes; both sampler tests had to choose a region-unique hexamer via a `_unique_start` helper. One N invalidates six consecutive cut sites (the window spans `[c, c+6)`), so the `non_acgt` case's control length had to sit 10 away, not 1. |
| **M25** — end hexamer read at `i+l-1` | `test_end_hexamer_offset_is_exact` | A point-mass `r_end` on one hexamer, so the drawn length is deterministic and an off-by-one in the end index shifts it by exactly 1. Implemented as one consolidated test, not the two (`t5_point_mass_exact_output`, `t5_length_probabilities`) this design originally proposed. | None beyond the hexamer-uniqueness point above. |

**One more thing this closure corrected, not caught by the original design:** M12 first came
back *uncaught* against the fixture's initial site. The mutation had patched
`min_mapq=min_mapq` inside `from_fname` — which CLAUDE.md documents as dead code — rather than
the real fetch-time mask (`mapq_vals >= min_mapq`). Re-run at the correct site, M12 was caught.
A mutation that does not express the defect proves nothing about the test it is supposedly
checking, and would have been wrongly reported as a coverage gap had that not been noticed.

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
opposite directions [Verified: `count_hexamers_rdf.py:1020-1022` at `eda1b14` — the overdispersion
numbers are now stated in the "deliberately NO BALANCE CHECK" comment, not in a live guard's
error message; see the void notice below]. The
mechanism is **not measured**. Candidates: within-region clustering, coarser structure, and the
strand-blind dedup that keeps the first h5 row (`fragment_array.py:1050-1052`). Per-region strand
counts are not stored (`count_srdf`, `:971-982` at `eda1b14`), so they need RD-56804 [Unverified]. The simulator
draws strand as an independent binomial per region (`sample_region:607`), so the excess is not reproduced
downstream. The sampler's zero-length drops (`sample_region:630-632`) act on simulator output only.

**VOID as of `38e5198` — there is no "the code's guard" row any more.** The table and the two
paragraphs below it described the strand-balance check this design spent F3/F13/M3/Q6 on. The
owner removed it entirely on 2026-10-07 ("doesn't seem useful"), not merely capped it — see the
status header for the three measured reasons (it could not catch a wholesale swap; it
false-positived on a legitimate 10-region run; rescaling to fix that made it vacuous below 10
regions and needed a cap, and "two corrections to stop being either wrong or dead" was the
owner's own framing for why it was not worth keeping). **Kept below, struck through, as the
historical record** — not because any of it is still true of the code:

~~**Measured false-positive rates** [Verified: Research D scratch simulation, gamma `n_r`
mean ~37; nominal two-sided 3σ = 0.0027]:~~

~~| Rule | R=10 | R=200 | R=2000 |~~
~~|---|---|---|---|~~
~~| fragment-level 3σ, simulator draw | 0.0025 | 0.0025 | 0.0040 |~~
~~| region-level SE 3σ, simulator draw | 0.0060 | 0.0000 | 0.0000 |~~
~~| the code's guard, simulator draw | 0.0000 | 0.0000 | 0.0000 |~~
~~| the code's guard, strand-pure regions, unequal `n_r` | 0.0038 | 0.0141 | 0.0000 |~~

~~The guard bound `1.5/√R` is `3 × 0.5/√R`. It assumes the maximum per-region sd, 0.5, so it
is conservative. For R > 225 it equals the 0.1 floor, about 54 fragment-σ at 73,545 fragments.
`b50ab4f` (F3) caps it: `min(0.45, max(strand_tol, 1.5/√R))`, capped at 0.45 for R ≲ 11 where
the uncapped term would reach/exceed 0.5 and make the check vacuous. At R=10 the bound is now
0.45, not the uncapped 0.474 this design previously quoted.~~ **`38e5198` deleted this bound,
the `strand_tol` parameter on `count_srdf`/`count_sample`, and the capping logic outright — the
false-positive table above has no corresponding code at `eda1b14` and should not be
re-measured against anything.**

**What actually remains, today:** check (1) only — both strand tables non-empty, exact, no
threshold [Verified: `count_hexamers_rdf.py:999-1007`]. `n_plus`, `n_minus` and `plus_frac` are
still computed and returned in `stats` (`:978-981`), just never asserted. The code's own
comment at the deletion site (`:1009-1030`) names the three reasons above and says explicitly:
"Do not reintroduce a balance assertion without naming a failure mode it detects that (1) does
not."

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
- ~~Guard tests keep inputs far from the bound, so they do not freeze the 1.5 constant (an
  owner-level choice, F13). At R=200 the bound is max(0.1, 0.106) = 0.106. A 0.80 plus
  fraction raises and 0.50 passes. A mildly lopsided 0.55 also passes, by design (F3).~~ **VOID
  (`38e5198`): there is no bound and no 1.5 constant left to freeze or avoid freezing.** F13 is
  void outright — it was a conflict between the spec's flat `strand_tol` and the code's capped
  formula, and both sides of that conflict are gone, so there is nothing left to reconcile. The
  only surviving guard test is `t3_one_empty_strand_raises` (implemented as
  `test_one_empty_strand_raises`), which needs no tolerance and so was never affected by this.

## 5. Test catalogue

Synthetic file `tests/test_count_hexamers_rdf.py` — **this part is built**, 43 test functions
organised as `TestT0EncoderAndFrame` .. `TestT7Hygiene` classes rather than as bare `t0_`-/
`t1_`-prefixed functions; a test named `t3_count_sample_matches_bruteforce` below is
`TestT3CountSample.test_count_sample_matches_bruteforce` in the actual file. This revision did
not rename every id in this catalogue to match — where a name differs materially (not just the
missing tier prefix) it is called out in place. Real file
`tests/test_count_hexamers_rdf_real.py` (§2.4) **does not exist** — R1/R2/R4 below are
proposal only. Oracle `tests/cut_site_oracle.py` (not
named `test_*`) is built, 254 lines. Marker and hook in `tests/conftest.py` (§2.5) **do not
exist** — there is no `tests/conftest.py` in this worktree at `eda1b14`, so the `real_data`
marker and the `REQUIRE_REAL_DATA` hook described here are unbuilt along with the real-data
tier itself. Session fixtures:
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
- `t3_one_empty_strand_raises`, over **≤ 9 regions**. Implemented as
  `test_one_empty_strand_raises`. All-plus fragments raise `AssertionError`,
  `match="one strand table is EMPTY"` (check (1), `count_srdf:999-1007` at `eda1b14`, was
  `:994-1002`). This is now the **only** strand-related raise in the function — **VOID
  (`38e5198`): check (2), the balance bound, no longer exists at any line number, so there is
  nothing for check (1) to be reached "unconditionally instead of".** M39 (deleting check (1))
  still turns this test red; that part of the row is unaffected.
- ~~`t3_strand_balance_guard_fires` (parametrised, R = 200)...~~ **VOID, and not merely stale —
  this test does not exist and was never written.** `38e5198` removed the balance check before
  `45b32ec` added the implemented test file, so there was nothing left to test by the time the
  suite was built: `grep -n "balance_guard\|strand_fraction" tests/test_count_hexamers_rdf.py`
  has no matches [Verified, this session]. **The converse half of M3**, that a balanced 20/20
  split passes, is also moot — with no bound, nothing can fail it regardless of balance. Read
  F3/F13/M3/Q6 in §6 and §9 for the full history.
- `t3_count_guards`: implemented much narrower than catalogued, as
  `test_count_guards_missing_sequence` (one case, despite the name) — it tests ONLY the missing
  `'fragment_array'` column raise (`count_srdf:918-924` at `eda1b14`), not the missing
  `'sequence'` column case and not the all-empty-admission case
  (`count_srdf:1031-1042`, "EVERY fragment was removed before counting") that this row and
  M39's test list both name. **Possible coverage gap, reported in the findings below, not
  fixed here.**
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
- `t6_writer_guards`: **implemented much narrower than this row describes.** Only
  `test_writer_guard_gz_path` exists, covering the first case only. The other four designed
  cases (`region_counts` shape mismatch, missing column, `fa.length != stop-start`, sequence
  length mismatch) have no test under any name [Verified, this session:
  `grep -n "region_counts has shape\|fragment_array.length\|sequence is .* b, expected"
  tests/test_count_hexamers_rdf.py` — no matches]. **Possible coverage gap, reported below.**
  Current line numbers for the five raises in `simulate_fragments_to_bed`, re-verified this
  session: `.gz` out_path `:712-717`; `region_counts` shape `:719-723`; missing column (loops
  over `contig,start,stop,fragment_array,sequence`) `:724-726`; `fa.length` mismatch
  `:743-748`; sequence-length mismatch `:749-757` (all were `:702`/`:709`/`:715`/`:733`/`:739`
  as of `b50ab4f`). The `AssertionError` cases are explicit `raise` statements, so `python -O`
  keeps them.
- `t6_fragment_length_dist_guards`: implemented, `test_fragment_length_dist_guards`, all six
  cases present and matching. Each case has `match=` copied from its raise.
  Current lines (`FragmentLengthDist` moved to `:336-428` as a class): non-1D/negative/zero-sum
  counts `:355-361`; `from_dataframe` missing column `:389-391`; empty frame `:394-395`;
  duplicate lengths `:396-401`; `from_srdf` with no `fragment_array` column `:414-418`; no
  fragments `:419-421` (were `:345-350`, `:380`, `:384`, `:386`, `:404`, `:410` as of
  `b50ab4f`). Plus `densify`, a separate test `test_fragment_length_dist_densify`:
  from `{25:1, 27:3}` the densities are 0.25 at 25, 0 at 26 and 0.75 at 27.
- `t6_input_guards`: implemented, `test_input_guards`. `load_sample_dataframe` raises
  `ValueError`, `match="no samples given"` (`:231` at `eda1b14`, was `:226`).
  `uniform_hexamer_counts` raises `ValueError`, `match="WITHOUT fragment arrays"` (`:515-520`
  at `eda1b14`, was `:504`).

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
| Input guards | `t6_writer_guards`, `t6_input_guards`, `t3_count_guards` | **Partially covered.** `t6_input_guards` and `t6_fragment_length_dist_guards` are fully implemented. `t6_writer_guards` and `t3_count_guards` are each implemented for ONE of their several designed cases only — see §5's T6/T3 entries for exactly which. Not every exception type per case exists yet. |
| Emptied vs lopsided strand | ~~`t3_one_empty_strand_raises` (≤ 9 regions), `t3_strand_balance_guard_fires` (R = 200)`~~ | **"Lopsided" is VOID (`38e5198`) — there is no lopsided-strand check, and no test of that name exists.** Only "emptied" is covered, by `test_one_empty_strand_raises`. |
| 10 real regions → h5 with exactly the fragments written | T6 on the toy (`t6_recount_equals_distinct_pairs`, implemented as `test_recount_equals_distinct_pairs`) | **NOT pinned on real data.** `r4_real_region_round_trip_chr21` and the whole real-data tier do not exist (§2.4); the `chr21`-restricted claim this row made is unbuilt, not merely unverified. |

**Run** (biomarker_env first on PATH — but see "Runs where?" in the TL;DR: this is now
automatic, not something the caller must arrange):
- Synthetic: `make test PYTEST_ARGS="tests/test_count_hexamers_rdf.py -q -rs"` still works
  standalone, but default `make test` runs it too, now that `dc1df6f`/`aefa71e` made `tests/`
  part of the default target.
- ~~Real, acceptance: `REQUIRE_REAL_DATA=1 make test PYTEST_ARGS="tests/test_count_hexamers_rdf_real.py -q -rs"`.~~
  **Does not work.** `REQUIRE_REAL_DATA` is read by a hook in `tests/conftest.py`, and
  `tests/test_count_hexamers_rdf_real.py` is the file it would gate; neither exists (§2.4).
- Full suite, verified this session: plain **`make test`** — **2 failed, 1125 passed, 3
  skipped, 208.83 s** (§1.1). There is no separate `tests/`-only invocation left to run; it is
  inside the default target now.

**Runtime budget**

| Item | Cost | Source |
|---|---|---|
| Module import (once per session) | 7.24 s wall | [Verified: Research C] |
| Toy h5 build, ~4 scenarios + 1 for R=200 | 0.15-0.16 s each | [Verified: Research C] |
| Toy `count_sample` (~20 calls), chr6 fetches (T4), doctests | 0.04 s, ~5 ms, 5.94 s each | [Draft measurement; doctests Verified: Research A] |
| T5 recovery at n = 4 × 10⁴ total draws | ~2 s, ~300 MB | [Inferred: scaled from review's 5.26 s, 767 MB at n = 10⁵] |
| Whole synthetic file + oracle, measured end to end | **included in the 208.83 s full-suite number above** — not separately broken out this session | [Verified: this session, §1.1] |
| R1, R2, R4 (h5 open, FASTA open, `count_sample`, `uniform_hexamer_counts`, `chr21` build) | **N/A — not implemented**, so there is nothing to time | superseded; see §2.4 |

The R-row costs below are kept as history only, since they describe a tier that was never
built; do not plan against them.

~~| R1: h5 open + FASTA open + `count_sample` (20 regions) | 0.12 + 0.01 + 1.83 s | [Verified: Research C] |~~
~~| R1 oracle: raw fetch + Python brute force | unknown | [Unverified: time it in P6] |~~
~~| R2: `uniform_hexamer_counts` + Python double loop over ~30,720 bp × 156 lengths | 0.03 s + a few s | [Verified / Inferred] |~~
~~| R4: count/sample/simulate pipeline + build (`chr21` only) + recount | ~11 s build [Verified: review, `realbuild/rb.py`] + rest unknown | [Unverified: time the non-build steps in P6] |~~
~~| **Never, except R4:** an h5 build against the unrestricted real FASTA or chr6 | 76.26 s / 26.2 s | [Verified: Research C] / [Draft measurement]. R4's `chr21`-restricted build is the one exception, at ~11 s (M2). |~~

Budget: **≤ 30 s per file**, measured by the implementer. The synthetic file's share of the
208.83 s full-suite run was not isolated this session; isolating it with `pytest
tests/test_count_hexamers_rdf.py -q --durations=0` is the way to check this budget still
holds. Above that is a finding.

## 6. Findings for the owner (report, do not fix)

| ID | Sev. | Finding | Evidence | Owner approval? |
|---|---|---|---|---|
| F1 | FIXED | `propensities()` divided the minus tables by swapped denominators at `a409186`. `counts_from_hexamers` puts `perm[hex(STOP)]` into `start_rev` (`:331-332` at `eda1b14`, was `:320-321`), so `E[start_rev] ∝ N_end[perm]`. The fix at `844f227` pairs them that way (`:846-852` at `eda1b14`, was `:840-841`, with a comment saying the crossing is on purpose). At `a409186` the real-data effect (800 of 66,649 regions, history) was median 1.2% and max 16% per cell. Found independently by Research B and by the closed loop in the `844f227` message. `t4_null_identity` guards it. M41 restores the old pairing. **Spec wording is no longer open**: `afe3611` added the crossed-pairing table to `simulator_spec.md` ("Which expectation each table divides by") [Verified: spec, read this session, matches the code exactly]. | [Verified: Research B at `a409186`; `git` reflog; `count_hexamers_rdf.py:846-852`] | Done (owner-approved). Spec wording also now done |
| F2 | **DONE — fixed in `c62f4a2`, not just "doc only" as this row assumed** | Docstrings (`hexamer_indices:135-142`, `_hexamers_at:250-260` at `eda1b14`) used to say invalid windows carry index 0. They carry the N-as-A index (`safe = np.where(win==255, 0, win)`, `:155` at `eda1b14`, was `:150`). **This was fixed in three places, not the one or two this row named** — `c62f4a2`'s own message is titled "it was in three places, not one". `_hexamers_at`'s docstring now states explicitly: "This docstring said 'index 0 / AAAAAA' until 2026-10-07. The sibling claim in `hexamer_indices` was corrected first and this one was missed, so the wrong version survived one round of fixing it" [Verified: read, `eda1b14`]. | [Verified: `count_hexamers_rdf.py:135-142`, `:250-264`, read this session] | Done |
| F3 | **VOID — superseded by removal, not by the `b50ab4f` fix this row described** | ~~Fixed in `b50ab4f`. The balance check was `tol = max(strand_tol, 1.5/√n_regions)`, vacuous for n_regions ≤ 9. Now `tol = min(0.45, max(strand_tol, 1.5/√n_regions))`, `\|plus_frac−0.5\| > tol` raises: the check can always fire.~~ **`38e5198` (one day after `b50ab4f`, same stream) deleted the whole balance check, `tol` formula included, rather than keeping the capped version this row describes.** The "whole table swap is invisible to this check" half of the finding is now moot by construction: there is no check for it to evade. | [Verified: `git show 38e5198 -- background_model/simulator/count_hexamers_rdf.py`, read this session] | Void — nothing left to approve |
| F4 | LOW, DONE | `:939-943` (even total) and `:957-963` (pair identities) at `eda1b14` (were `:934-938`, `:954-960`) are tautologies given `counts_from_hexamers`; they cannot fire. **`b50ab4f` corrects the claim in the code's own comment** (it no longer says these "catch broken strand routing", and spells out why, matching this finding) and keeps the checks only as a malformed-`counts` guard (M4). Do not count them as routing protection; M39 drops both sites from its list (M4). Unaffected by the `38e5198` strand-balance removal — these two checks are a different mechanism (fragment-count tautologies, not the 0.5 balance bound) and are still live. | [Verified: read, `eda1b14`] | Done |
| F5 | MED | Any strand label other than `'+'` counts as minus (`counts_from_hexamers:325` at `eda1b14`, was `:314`): `b'+'`, `'.'`, `''`. Check (1) (`count_srdf:999-1007` at `eda1b14`, was `:994-1002`) fires if **either** table is empty — this is now the ONLY strand check (`38e5198` removed the other one; see F3). So an all-bytes frame is caught loudly, but only if `n_counted > 0`: when both are zero the `if n_counted :=` guard skips the check. A mixture with some `'.'` or bytes silently inflates minus. The raw reader yields bytes (§1.2), so the str contract rests on the RFA layer. | [Verified: Research A, re-grepped this session] | Yes, if it becomes a raise |
| F6 | **CLOSED — owner ruled 2026-10-08, do not re-open** | Validity asymmetry. C needs both cut sites valid (`cut_site_hexamers:305` at `eda1b14`, was `:294`). `N_start`/`N_end` each gate only their own site (`uniform_hexamer_counts:556-561`, `:567` at `eda1b14`, was `:547-548`, `:556`). On this region set the effect is unmeasurable: 0 invalid cut windows in 1,373,600 (800 regions); 4 of 66,649 regions hold a non-ACGT base. **The owner's ruling, recorded in `simulator_spec.md`'s Settled section: "the simulator gets to define the probability model… not interested in further pursuing this."** `r` is definitional (drawn and scored from the same `C`/`N`), so the gate choice changes what the simulator *is*, not whether it is right — not an approximation error to fix. | [Verified: Research A/B; spec Settled, read this session] | **No — closed, not open.** Q2 below is answered. |
| F7 | LOW | `propensities()` sets cells with `C>0, N=0` to 0 and raises no error (`:846-848`). | [Verified: grep, 0820409] | Owner, with F1 |
| F8 | LOW | `_hexamers_at` promises `IndexError` (`:242`, `:279`) but wraps a negative `pos` silently. On the research probe's sequence, `_hexamers_at(seq, np.array([-2]))` returned index 283 with `valid` True. That sequence is not recorded, so 283 is not a constant. On another sequence the wrap gave index 0 with `valid` True [Verified: review]. Only the `starts_0 ≥ 0` gate in `filter_fragments` prevents it, so tests cover the gate (T2), not the helper. | [Verified: Research A probe] | No |
| F9 | INFO | `rc_permutation()` returns a shared writable cached array (`:189`). A caller that writes to it corrupts later calls. Proposal: `setflags(write=False)` (engineering) plus a test that a write raises. No test lands unless that fix lands, because today it can only be red or re-assert the hazard. | [Verified: Research A] | No (engineering), but confirm |
| F10 | INFO | Stale comments and docs (§1.4). The module doctests pass but default `make test` never runs them. | [Verified: Research A] | No |
| F11 | LOW | `FragmentLengthDist` casts counts through int64 (`:343`, `:381-382`), so float counts truncate silently. Not tested: either assertion presumes a decision. | [Verified: Research A] | Yes, if changed |
| F12 | INFO | **v1 was wrong here.** The writer guards (`:732-746`) are explicit `raise AssertionError` (`:733`, `:739`), so `python -O` keeps them. The only bare `assert`s are `:175-176` in `hexamer_vocabulary`, which `-O` strips. They run once at build of a derived table, so the exposure is small. | [Verified: Research A] | No |
| F13 | **VOID** | ~~Spec/code conflict, still open after `b50ab4f`. Spec says the guard asserts within `strand_tol`. The code uses `min(0.45, max(strand_tol, 1.5/√n_regions))` — reconcile the formula with the spec's flat `strand_tol`.~~ **There is no formula left on either side to reconcile.** `38e5198` deleted the code's balance check (and the `strand_tol` parameter entirely, from both `count_srdf` and `count_sample`) in the same stream as this finding's own `b50ab4f` fix, one commit later. The spec was updated in the same commit to match. A conflict between two things needs both things to still exist; neither does. | [Verified: `git show 38e5198`, read this session] | Void — Q6 is answered by deletion, not by a ruling |
| F14 | LOW | Spec Settled says dropped zero-length starts get "no counter, no test" (owner 2026-10-06). But `simulate_fragments_to_bed` returns `n_short_regions` (`:737`, `:796` at `eda1b14`, was `:726`, `:785`): a counter exists. Flagged, not tested. | [Verified: Research D, re-grepped this session] | Confirm (Q7) |
| F15 | LOW | Removed-feature code still live: `scripts/run_simulator.py:66` imports `simulator.capture` at module level; `simulator/__init__.py` imports `capture`, `build_predict_lut`, `midpoint_index_arrays`; `Makefile` keeps `FLGC_PYTHONPATH`. Flag as stale; do not fix here. | [Verified: Research D] | Deletion is the owner's call |
| F16 | LOW | `make test` depends on PATH: it runs bare `python -m pytest`. The wrong python gave exit 2 and 11 collection errors. A fix pins the interpreter in the Makefile (engineering). | [Verified: coordinator log] | No (engineering) |
| F17 | MED, LIBRARY — **HALF FIXED, HALF STILL LIVE, re-verified this session** | A minus-strand region flip has **two** defects. **Defect 1 (bytes-vs-str, FIXED in `aefa71e`):** `_switch_plus_with_minus_and_minus_with_plus` (`fragmentomics_tools/fragment_array/fragment_array.py:123-158` at `eda1b14`) used to compare only against the `str` literals `"+"`/`"-"`, so the raw `\|S1` bytes array from `from_fragments_h5` (call at `:1819`) compared all-False and labels were never swapped. `aefa71e` widened the comparison to accept both `str` and `bytes`; the function's own new docstring documents the before/after. **Defect 2 (order, STILL LIVE, NOT addressed by `aefa71e`):** `starts_0`/`stops_0` (and methyl/gc) are reversed with `[::-1]` to flip coordinate order (`fragment_array.py:1814-1815`), but the call that swaps strand labels (`:1819-1821`) does **not** also reverse the label array's order — so after the bytes fix, the labels are swapped correctly but sit at the wrong POSITIONS relative to the now-reordered coordinates. Reproduced fresh this session with the exact fixture this finding originally used (`chrT:500-1000,-` with `(600,700,+)`,`(650,760,+)`,`(800,900,-)`): the current code gives strands `['-','-','+']` against expected `['+','-','-']` — wrong at 2 of 3, matching this finding's pre-existing prediction exactly [Verified: this session, ran `_switch_plus_with_minus_and_minus_with_plus` and the `[::-1]` reversal by hand against the library's actual `from_fragments_h5` code path at `fragment_array.py:1811-1829`]. **This is a real, currently-live defect**, not a stale finding — report it, do not fix it here. `generate_weights_callback` weights are also not reversed [Inferred]. The `U1` coercion runs later. Reached only by minus-strand regions; the current simulator region BED is strandless, so not reachable from this pipeline [Inferred] — but reachable by `ctcf_pileup_run.py`, per `aefa71e`'s own commit message. | [Verified: `git show aefa71e`; this session's repro against `fragment_array.py` at `eda1b14`] | **Yes** (Q8), for the still-live half. No test until ruled |
| F18 | LOW, DONE, **superseded by a bigger change in the same spot** | Fixed in `b50ab4f`; the message no longer said "Strand is clustered within regions". **`38e5198` then deleted the balance-check message entirely** (it was the message for the check F3 removed) rather than merely continuing to reword it — the overdispersion numbers now live only in a code comment at `count_srdf:1012-1022`, not in any raised message, since nothing is raised. | [Verified: read, `eda1b14`] | Done (by removal) |
| F19 | LOW, DONE | Fixed in `b50ab4f`. The all-empty-admission error (`count_srdf:1031-1042` at `eda1b14`, was `:1047-1058`) no longer says "MAPQ removed ALL"; it now names the whole admission chain (MAPQ, dedup, length filter, start-in-region) and keeps MAPQ only as the first thing to check. `t3_count_guards`'s match string must track this wording (L2/L5) — and per §5's T3 entry, no test currently exercises this raise at all, so there is nothing to have tracked yet. | [Verified: read, `eda1b14`] | Done |

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
- **Minus-strand region input** (F17). No test until the owner rules (Q8) — and per F17 above,
  half of this is now a confirmed live defect, not just an untested input.
- ~~Any real-data plus fraction (§4.4). The `1.5` constant itself (F13).~~ **VOID**: the `1.5`
  constant does not exist (`38e5198`); there is nothing left to avoid freezing. Real-data plus
  fraction remains genuinely untested, for the unrelated reason that §4.4 always gave (region
  is the real-data unit, dispersion unmeasured) — that half of the bullet still stands.
- **The index value of an invalid window** (F2 — **now DONE**, see Findings; this bullet is
  about the index VALUE, which is still deliberately untested even though the docstring claim
  about it is now fixed). The old-generation tests and the CLI h5
  path. F5/F7/F11 behaviour, which a test would freeze.
- **The even-total and pair-identity checks** (`:939-943`, `:957-963` at `eda1b14`, were
  `:934-938`, `:954-960`) are tautologies, not
  routing protection (F4, M4) — no test can turn either red, and `b50ab4f`'s own code comment
  now says so. M39 excludes both sites from its list. Unrelated to, and unaffected by, the
  `38e5198` strand-balance removal (F3).
- **Pinned `default_rng` draws.** None anywhere.

## 8. Implementation plan

| Phase | Work | Depends on | Exit criterion | Status at `eda1b14` |
|---|---|---|---|---|
| P0 | Re-measure both baselines at the implementation HEAD, biomarker_env first on PATH. Re-run the review measurements that P2 and P6 cite | — | Numbers recorded. Last measured at `a409186`: 2F/399P/3S and 673P/0S. Expect `tests/` near 677. | **Superseded** — the two-target split this estimated no longer exists; see §1.1 for the single current number (2 failed / 1125 passed / 3 skipped). |
| P1 | `tests/conftest.py` (marker, REQUIRE hook). Oracle module and its self-tests (de Bruijn property, palindromes = 64). Confirm the sibling import. T0, T7. | P0 | Green. The hook turns a forced `real_data` skip into a failure under `REQUIRE_REAL_DATA=1`. | **Oracle and T0/T7 done.** `tests/conftest.py` (marker + hook) **NOT done** — does not exist. |
| P2 | Fixture builders: toy FASTA (asserted properties), BED → tabix → h5 helper, regions. T1, T2, T3. | P1 | T3 exact equality green. The M12 self-check discriminates. | Done. |
| P3 | T4 (§4.3), with M41 run in the same phase | P2 | All four null-identity tables green at HEAD. M41 red on `[start_rev]` and `[end_rev]`. | Done. |
| P4 | Recording rng. T5. Time the recovery test. | P1 | Recovery gap ≥ 12σ under M27, by the T5 arithmetic. | Done. |
| P5 | Closed loop on the toy. T6. | P2-P4 | Distinct-pair invariant holds. | Done. |
| P6 | Commit `tests/data/simulator_real_regions_20.bed` (10 `chr21` rows + the rest elsewhere). Pin the real h5 and FASTA paths. R1, R2, R4. | P1-P5 | `REQUIRE_REAL_DATA=1` run: 0 skipped, real file ≤ 30 s. `tests/data` grows by ~1 KB only. | **NOT done.** No such BED, no real-data test file, nothing real-data-gated in `make test` (§2.4). |
| P7 | Mutation sweep M1-M42 on scratch edits (module or library, never committed), reverted. Run the §4.2 cross-check by script | P1-P6 | Every row red **except the listed equivalent mutants** (M14, H1). Table in the PR, with the site column. | **24/42 done** (not all) — see §4.2a. 18 rows remain named-but-unverified. |
| P8 | After-baselines | P7 | See below. | Superseded, see below. |

P8 targets, **as corrected this revision**:
- ~~`make test` unchanged (2 failed / 399 passed / 3 skipped; it does not collect `tests/`).~~
  **Superseded.** `make test` now collects `tests/` by default and gives **2 failed / 1125
  passed / 3 skipped** [Verified: ran this session, §1.1]. There is no "unchanged" baseline to
  compare against any more, because the thing P8 described (a target that excludes `tests/`)
  no longer exists.
- ~~`tests/` with `REQUIRE_REAL_DATA=1`: 673 + 4 (`844f227`) + N_new passed, **0 skipped**.~~
  **Cannot be run.** `REQUIRE_REAL_DATA` and a dedicated `tests/`-only invocation both
  presuppose infrastructure (§2.4/§2.5) that does not exist.
- Each new file ≤ 30 s: not independently re-measured this session (§5's Runtime budget note).
  `tests/data/` did NOT gain the 20-row BED — P6 was not executed.

## 9. Open questions for the owner

- **Q1 (propensity tests).** Delete the three partial tests in
  `tests/test_simulator_propensity_denominators.py`? Keep `test_the_rev_tables_are_not_swapped`
  either way (§1.3). The new suite does not depend on the answer.
- ~~**Q2 (F6).** Should `N` use the same both-sites validity as C? The effect is unmeasurable
  on the current region set. The tests stay neutral until you rule.~~ **ANSWERED, 2026-10-08**:
  no. Recorded in `simulator_spec.md`'s Settled section: "the simulator gets to define the
  probability model… not interested in further pursuing this." Do not re-ask this.
- ~~**Q3.** Default `make test` does not collect `tests/`, and it depends on PATH (F16).
  Change the `PYTEST_ARGS` default and pin the interpreter, or document both?~~ **ANSWERED by
  action, not by a ruling.** `dc1df6f` changed the default to collect `tests/`; `aefa71e` fixed
  the PATH dependency (pins the interpreter AND puts its `bin/` on PATH, after pinning the
  interpreter alone made things worse — 2 failed became 54 failed on a `bedtools`-off-PATH
  trap). Both halves of this question are done.
- **Q4.** Does "dropped zero-length starts: no test" also exclude the *support* assertion in
  `t5_zero_weight_cause`? The design assumes not, because it asserts no shortfall count. If
  you read it otherwise, the row-level variants go and the cell-level variants stay.
- **Q5.** Real tier: default-on with a visible skip (as designed), or opt-in only? And
  should the acceptance gate `REQUIRE_REAL_DATA=1` also run in CI or the batch container?
  **Still fully open** — unlike Q2/Q3/Q6, nothing has happened here at all; the whole tier is
  unbuilt (§2.4).
- ~~**Q6 (F13).** Which is authoritative for the balance bound: spec `strand_tol`, or the
  code's `min(0.45, max(strand_tol, 1.5/√n_regions))`?~~ **VOID, answered by deletion.**
  `38e5198` removed the balance check from the code, and the spec was updated in the same
  commit. There is no bound left on either side for one to be authoritative over.
- **Q7 (F14).** Keep `n_short_regions` (and amend the Settled line), or remove it?
- **Q8 (F17).** Is a minus-strand region a supported input? **Partially acted on without being
  answered**: `aefa71e` fixed the bytes-vs-str half of the label defect, for the unrelated
  reason that it also sits on the `ctcf_pileup_run.py` path, but did not address, and did not
  claim to address, the order-reversal half — `fragment_strands` still is not run through
  `[::-1]` alongside `starts_0`/`stops_0`, confirmed live this session (F17). The question
  itself is therefore MORE urgent than when written, not less: a half-fix that doesn't mislabel
  byte-strand input as loudly as before could read as "handled" to someone who does not check
  the order. Still rule: fix both parts of the flip, or make `count_sample` (and any other
  minus-strand caller) refuse minus-strand regions. If no, no test is written.

## Least sure of

**This section itself is now partly stale, same lesson as always — updated in place rather
than pretending it was always accurate.**

- ~~**h5 order for equal `(start,stop)`** [Verified: review, one case]. M12 relies on it. A
  second case in P2 settles it.~~ **RESOLVED, `02f7f39`.** Measured directly: `_build_h5` uses
  a stable sort on `(contig, start, stop)`, so ties deterministically keep fixture order. M12
  did not need the hand-built-array fallback this bullet anticipated.
- **Minus-strand regions** (F17). Still true that no test feeds one through
  `count_hexamers_rdf.py` or `count_sample` — the flip trap is uncovered by a test either way.
  **But no longer true that the trap is fully unaddressed**: `aefa71e` fixed half of it
  (bytes-vs-str) for an unrelated caller, and this session's repro confirms the other half
  (order) is still live. The owner still rules first (Q8), but "rules first" now means ruling
  on a half-fixed defect, not an untouched one.
- **Runtimes and paths.** Unchanged — still true. R1 and R2 runtimes are unmeasured because
  R1/R2 do not exist (§2.4), not merely because nobody timed them; R4 likewise. The real h5
  path is still absent from the research record.
- **Cites not touched by this revision may still carry pre-`b50ab4f` line numbers** — true of
  the v4 revision, and now compounded: cites not touched by THIS (v5) revision may carry
  pre-`eda1b14` numbers from anywhere between `b50ab4f` and `eda1b14`, a 14-commit, multi-hundred-line gap.
  This revision re-walked the citations named in the status-header bullets, the strand-balance
  material (§4.4, F3/F4/F5/F13, the T3/T6 guard rows), F1/F2/F6/F14/F17/F18/F19, and the M12/
  M13/M14/M15/M19/M20/M24/M25/M41/M42 mutation rows, plus every top-level function's current
  definition line (listed in §4.2's header note). **Numeric line cites inside mutation rows
  M1-M11, M16-M18, M21-M23, M26-M40 were NOT individually re-walked this session** — the module
  shifted by roughly a dozen lines across that range (mostly docstring growth before line
  ~470, consistent +11-ish after it, per the samples checked); treat any number in those rows
  as approximate and grep the named symbol instead of trusting it.
- **Sibling import of the oracle** [Inferred]. Resolved implicitly — `tests/test_count_hexamers_rdf.py`
  exists and imports `cut_site_oracle` (per its own `test_oracle_is_independent`), so the
  sibling-import mechanism this bullet worried about evidently works. **Cell barcode
  read-back:** still unresolved; the reader attribute is still unnamed.
- **Reviewer measurements are not re-run.** Still true for anything not explicitly re-verified
  in this revision (see the cites bullet above). This revision DID run Python directly for: the
  full `make test` baseline (§1.1), the F17 minus-strand repro (findings table), and several
  `grep -n`/`ls` existence checks (§2.4, §5).
- **Resolved in earlier revisions:** the `fetch_array` filters (§1.2), the M2 doctest question,
  the M34 site.
- **Guard tests (T3 strand guards, `t3_count_guards`, T6 guards).** This bullet called these
  "the first candidates to cut" — and in practice, most of them WERE effectively cut: only one
  case of `t3_count_guards` and one of five `t6_writer_guards` cases actually got implemented
  (§5), and the strand-balance guard test (`t3_strand_balance_guard_fires`) was never written
  at all once its target was removed. Whether that's the right amount of coverage is a question
  for the owner, not settled by this revision — flagged as a possible gap in the findings
  below, not fixed.
- **NEW this revision.** Whether the synthetic suite still meets the "≤ 30 s per file" budget
  (§5's Runtime budget note) was not isolated from the 208.83 s combined `make test` run.
  Whether `t6_writer_guards`'/`t3_count_guards`'s narrower-than-designed coverage (§5) is an
  accepted scope cut or an oversight is unasked — add as a question if the owner wants it
  raised formally.
