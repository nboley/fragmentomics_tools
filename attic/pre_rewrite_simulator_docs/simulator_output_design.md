# Simulator output path: fragment h5 + one metadata file

**Status: DESIGN. Nothing here is implemented. Nothing here has been run.**
No code, test, or timing in this doc was executed by the author of this doc.
**Review: independently reviewed, round 1, verdict B-.** The review's findings are fixed in this revision (H1-H3, M1-M7, L1-L8, and the INFO items). Each fix is tagged where it lands. Owner questions raised by the review are recorded as [Open] in §12, not decided.

Evidence base. The author read no source files. Every claim comes from research summaries:

| Tag | Meaning in this doc |
|---|---|
| [Verified] | The coordinator read the cited file/lines directly this session: `background_model/simulator/count_hexamers_rdf.py` (:89-91, :93-210, :299-414, :456-812), `background_model/preprocess.py:112-189,674`, `background_model/config.py:47-57,81-82,106-108,121-147`, `background_model/store.py:93-98`, `fragmentomics_tools/fragment_array/fragment_array.py:1049,1684-1754`, `~/src/fragments_h5` (`fragments_h5.py:1067,1124-1126,1338-1347`, `fragment.py:630-781`), the first 12 lines of `data/region_sets/quiet_v2_pad1200_repeats_removed_tile1536.bed` (main checkout), a Glob for a `tables` package in biomarker_env (none found). |
| [Verified-B] | Read directly by research agent B (`background_model/simulator/emit.py`, `tests/test_simulator_phase3.py`); coordinator reviewed the classification, not every line. |
| [Verified-R] | Read by the independent reviewer (round 1) this session. The author did not re-run it; the claim cites the reviewer's measurement. |
| [Inferred] | Strong inference from the above. |
| [Unverified] | No evidence. The verifying command or file is named. |
| [Open] | Needs an owner decision. This doc does not assume the answer. |

Authority on simulator rules: `docs/pending/simulator_spec.md`. This doc covers only the output path.

---

## 1. Problem

- [Verified] `count_hexamers_rdf.py` can draw fragments in memory (`sample_region`, :563-633). It can measure parameters (`count_sample` :752-812, `uniform_hexamer_counts` :456-560, `propensities` :636-668, `FragmentLengthDist` :322-414).
- [Verified] Nothing connects measurement to sampling over a region set. No driver exists. Nothing writes draws to disk.
- [Verified] The module has no tests. Only the spec and the module itself mention `count_hexamers_rdf` (grep).
- [Verified] Its docstring (:89-91) cites a pinning test `test_encoder_matches_precompute`. That test does not exist.

Goal. A run over a region set produces exactly:

1. a fragment h5 that the background model's existing ingest consumes unchanged, and
2. one metadata file. From it, a reader six months later can tell the parameters, region set, reference and code that made the run. The reader can recompute the oracle without guesses.

Out of scope (owner ruling 1): zarr store, crop width D, oracle computation. A separate model agent owns those and imports simulator primitives.

---

## 2. What exists

### 2.1 Rewrite hand-off (`count_hexamers_rdf.py`)

| Primitive | Contract | Tag |
|---|---|---|
| Constants | `KMER=6, HEX_HALF=3, NHEX=4096, L_MIN=25, L_MAX=180` (inclusive), `N_LENGTHS=156` | [Verified] :93-150 |
| `hexamer_indices(seq)` | `(fwd int64, rc int64, valid bool)`, length `len(seq)-5`. A=0,C=1,G=2,T=3, big-endian, case folded, non-ACGT -> 255. Invalid windows carry index 0 (same as AAAAAA): callers must gate on `valid`. | [Verified] :93-150 |
| `hexamer_vocabulary()` | `(4096,)` `S6`; `vocab[i]` = 6-mer with forward index i; derived from the encoder | [Verified] :153-178 |
| `rc_permutation()` | `(4096,)` int64 involution from the encoder | [Verified] :186-210 |
| Tables | `TABLE_NAMES=("start_fwd","end_fwd","start_rev","end_rev")`, each `(4096,)` int64 | [Verified] :299-319 |
| `counts_from_hexamers` | plus: `start_fwd[hex(start)]`, `end_fwd[hex(stop)]`; minus: `start_rev[perm[hex(stop)]]`, `end_rev[perm[hex(start)]]` | [Verified] :299-319 |
| `uniform_hexamer_counts(rdf, fasta, fl)` | `{"start": int64, "end": float64}` + meta (`n_regions, n_start_positions, n_start_invalid, end_weight_total`). Serial pysam walk. Refuses a frame with `fragment_array`. Frame: `left_flank=3`, `right_flank=fl.max_fl+3`. Raises on truncated fetch near a contig end. | [Verified] :456-560 |
| `propensities(C, N, min_expected=0.0)` | 4 float64 `(4096,)`; `r=C/N`, 0 where `N<=min_expected`; rev tables divide by `N[perm]` | [Verified] :636-668 |
| `FragmentLengthDist` | `counts` int64, `densities` float64, `min_fl`, `max_fl`, cached CDF. `from_dataframe`, `from_srdf`. Support is data-derived, not forced to [25,180]. | [Verified] :322-414 |
| `sample_region(sequence, region_len, n, *, r, fl, p_plus, rng)` | Returns `(starts_0, lengths, is_plus)`: region-local 0-based starts in `[0, region_len)`, plus block first then minus, unsorted. Needs `left_pad=3`, `right_pad>=183`. No genomic coordinates. A start with no valid length is dropped. A strand with all-zero start weight is skipped silently. So `n_drawn <= n`. | [Verified] :563-633 |
| `filter_fragments(fa, l_min, l_max)` | `drop_duplicate_fragments()` on `(starts_0, stops_0)` (strand NOT in key, `fragment_array.py:1049`); `subset_fragment_lengths(l_min, l_max+1)`; keep `starts_0 in [0, fa.length)` (start-in-region) | [Verified] :727-749 |
| `count_sample(...)` | `init_from_rdf_and_sdf` -> `attach_fragment_arrays(min_mapq, callback=filter_fragments)` -> `attach_sequence(left_pad=3, right_pad=l_max+3)` -> `count_srdf`. Returns only `(counts, stats)`; discards the srdf. | [Verified] :671-724, :752-812 |
| `count_srdf` | Drops fragments with an invalid cut-site hexamer, so `n_counted <= n_after_filters` | [Verified] :671-724 |

- [Verified] **`p_plus` has no producer.** Nothing in the rewrite measures it. See §12, item 1.
- [Verified] `attach_sequence` is not required to run after `attach_fragment_arrays`. It calls `get_sequence`, which pads at the fetch and does not call `expand_regions` (`dataframe.py:1808-1813`, `:1824-1827`; `region.py:853`). The ordering comments at `count_hexamers_rdf.py:21-23` and `:804-806` cite `expand_regions` and are stale (§11). Not verified: that `attach_sequence` runs without error on a frame that already carries fragment arrays. Test it in P1.
- [Verified] `uniform_hexamer_counts` still refuses a frame that carries `fragment_array` (`:500-506`). That statement stays true, and §6.3 keeps the separate frame.
- [Verified] Contig ends: `region.py:853-865` raises when the padded span runs off a contig. Not verified: that `_get_seq` propagates the exception rather than catching it. Read `dataframe.py` `_get_seq` before P1.

### 2.2 Ingest contract (consumer)

| Fact | Tag |
|---|---|
| `_worker_inner` opens `FragmentsH5(h5_path, cache_pointers=False)` | [Verified] `preprocess.py:131-189` |
| `total_fragments = int(h5.fragment_length_counts.sum())`, stored per sample | [Verified] `preprocess.py:140`, `store.py:98`. Downstream use is a depth gate: `preprocess.py:558-563` sets `role=2` ("dropped_low_depth") when `total_fragments < config.min_total_fragments`. Default `20_000_000` (`config.py:92`), excluded from the config hash (`config.py:34`). A grep of `background_model/` finds no other reader. See §10.4. |
| Per tile: strandless margined `Region` (tile +- `config.jitter`, `strand="."`, clamped to contig) | [Verified] `preprocess.py:154-179` |
| `from_fragments_h5(h5, region, min_mapq=config.min_mapq, max_frag_len=config.max_frag_len)`; `min_mapq` default 10 | [Verified] `config.py:81` |
| MAPQ rule: `min(mapq1, mapq2) >= min_mapq` | [Verified] `fragment_array.py:1742-1754` |
| `if config.dedup: rfa.drop_duplicate_fragments()` (default True), dedup on `(start, stop)` | [Verified] `config.py:82` |
| `build_coverage_counts(fl_bands, split_strand=True, return_sparse=True)`; strand required | [Verified]; behaviour without strand [Unverified] |
| `max_frag_len = max(hi for fl_bands)`, exclusive | [Verified] `config.py:106-108` |
| `has_gc/has_strand/has_methyl` auto-detected; GC not consumed by background_model | [Verified] auto-detect at `fragment_array.py:1716-1719` (`return_gc` defaults to `has_gc`). Not consumed: a grep of `background_model/` for `gc` / `return_gc` hits only `simulator/`. That is a grep result, not a proof of absence. |
| Region admission inside `fetch_array` is by OVERLAP: `starts < stop`, `stops > start`, `lengths <= max_frag_len` | [Verified] `fragments_h5.py:638-640`. The model's margined query (tile +- jitter) therefore admits fragments that overlap the margin, including fragments that start before the margin. The ingest does NOT apply start-in-region. `filter_fragments` start-in-region is the simulator-side C rule only. |
| Samples keyed by explicit `library` column of a TSV sample sheet (`library, h5_path, seqrun, endo_category`), not by filename; `h5_path` stored as a plain string, no hash | [Verified] `preprocess.py:112-113,674`, `store.py:93-98` |
| Store identity: `config_hash` over md5 of region BEDs, blacklist, sample sheet, and md5 of the reference `.fai` as a proxy for FASTA identity | [Verified] `config.py:47-57,121-147` |
| Ingest tests use a monkeypatched fake plus one golden h5 built from a BAM; no test hand-writes the h5 schema | [Verified] `test_bg_e2e.py:417-507`, `test_bg_golden_counts.py:170` |

Consequence [Inferred]: the simulator must produce the h5 with the production builder. Hand-writing the schema has no test coverage on the consumer side.

### 2.3 Writer (`fragments_h5`, env pin v2.13.3 at `environment.yml:67`)

| Fact | Tag |
|---|---|
| `build_fragments_h5(input_fname, ofname, fasta_filename=None, ..., read_strand=True, num_processes=None, ..., min_mapq=None, *, build_argv=None)` | [Verified] `fragments_h5.py:1067` |
| Accepts bgzipped + tabix-indexed `.bed.gz`/`.tsv.gz` | [Verified] `fragment.py:630` |
| BED input requires `fasta_filename` (contig lengths + GC) | [Verified] `fragments_h5.py:1124-1126` |
| Index contigs must exist in the FASTA; `contig_lengths` recorded only for contigs present in the index | [Verified] `fragment.py:654-781` |
| BED input forces off `min_mapq`, `include_duplicates`, `store_fragment_end_clipped` | [Verified] same |
| 6 cols -> strand read; 7 cols rejected; 8 cols -> `mapq1, mapq2` from cols 7-8 (int 0-255, else row skipped) | [Verified] same |
| No MAPQ -> 255 sentinel, read back as -1 -> `min_mapq=10` drops everything | [Verified] same |
| Rows skipped (warning) when `stop<=start`, negative coords, or length > `max_tlen` (default 1000); >50% skipped on a contig raises | [Verified] same |
| Col 4 (name), when non-empty, becomes `cell_barcode`; `"."` becomes barcode `"."` | [Verified]; whether the h5 stores/uses it [Unverified] |
| Root attrs: `index_block_size, max_fragment_length, _bam_header, _source_format` (`"TSV"` for BED), `_contig_lengths_str`, `_build_argv` (only if passed), `_build_code_revision` | [Verified] `fragments_h5.py:1338-1347` |
| Output byte-deterministic for a given input | [Unverified] for BED input. Measured 2026-08-24 at v2.12.1 for BAM builds only (coordinator memory). |
| `build-fragments-h5` console script often not on PATH in biomarker_env (exit 127) | [Verified] (coordinator note). Call the Python API. |
| BED is 0-based half-open: row `(start, start+L)` round-trips to `starts_0/stops_0` | [Verified] |

---

## 3. Old generation: reuse / reject

The old output code is `background_model/simulator/emit.py`. It is not deleted (owner deferred). [Verified-B]

**Rule for reuse:** copy the helpers into the new module. Do not import `emit.py`. An import makes the deferred deletion impossible. [Inferred]

| Old item (emit.py unless noted) | Decision | Reason |
|---|---|---|
| 8-col BED `contig start stop . 0 strand 60 60` (`write_bed` :184-207, `_MAPQ=60` :181) | KEEP format, REJECT implementation | 60/60 survives `min_mapq=10`. Per-row Python append to a persistent path is slow and leaves debris. Replace with vectorised text into a temp dir. |
| `LC_ALL=C sort` -> `bgzip` -> `tabix -p bed` (`sort_bgzip_tabix` :210-280) | KEEP the pipeline shape; CHANGE the tools (§6.7) | Intermediates were written next to the target, not a temp dir (violates ruling 2). |
| Binary preflight before the sampling run | KEEP (whatever external tool remains) | Fail before paying for the sampling run. |
| Shell-out to `build-fragments-h5` console script (:660-679) | REJECT | Console script often not on PATH. Use the Python API. |
| Hexamer-string-keyed DataFrames, slot-swap guard, vocabulary reindex (:50-176, owner decision 81) | KEEP the intent | Ruling 5. Port the guard: address columns by name, never by position. |
| `_hash_file`, `_hash_array` (sha256) | KEEP (copy) | Identity without git. |
| `git_blob_sha`, `_to_repo_relative` | KEEP (copy), best-effort tier only | git is not on PATH in the Batch container. |
| Mandatory-vs-best-effort verification tiers; `ManifestMismatch` / `ManifestVerificationIncomplete`; "unresolvable" recorded as a fact, not a pass | KEEP | Same failure policy. |
| JSON manifest, `MANIFEST_VERSION=2` (:399-654) | REJECT container, KEEP fields that still have referents | JSON loses float64 bits unless care is taken; replaced by HDF5 (§7). |
| `predict_lut`, GC bins (:406,449,630,635) | REJECT | Capture model dropped. |
| `marginal_fl` as defined by `weights.py` | REJECT; analogue is `FragmentLengthDist.counts` | Tied to capture normalisation. |
| `pad` field; midpoint / symmetric-flank geometry (`weights.py MAX_FL_HALF`; `precompute.py` symmetric pad :121,149,163) | REJECT | Rewrite frame is asymmetric `left_pad=3, right_pad=183`; admission is start-in-region. |
| `build_region_weights` / `RegionWeights` | REJECT | Capture layer. |
| `sampler._REGION_COUNTS={2560:54,1536:37}` (`sampler.py:217-234`) | REJECT | Fabricated per docs; replaced by ruling 6. |
| `fl_bands` in the manifest (hardcoded `((25,110),(110,180))` at :416) | REJECT | `FL_BANDS` is a model parameter, not a simulator parameter. |
| `per_region_counts`, `rng_seed`, `commit_sha`, script blob sha, region-set / reference sha256 | KEEP (renamed, §7) | Still have referents. |
| `tests/test_simulator_phase3.py`: e2e test without `importorskip` (:899-910), monkeypatched `shutil.which` preflight (:1319-1342) | KEEP the pattern | A missing binary must fail loud. |

Two encoders coexist [Verified-B]: `precompute.py` and the rewrite. Same convention, except the rewrite's LUT also maps lowercase. No pinning test exists. `scripts/count_cut_site_hexamers.py` imports the old encoder.

---

## 4. Reconciliation: what rulings 3/4 told the simulator to persist

Rulings 3/4 (one metadata file; realised arrays) were written when capture/GC modelling existed. The item list below comes from `emit.py` manifest fields and the stale docs, not from the ruling text. [Inferred]

| Item rulings 3/4 covered | Referent now? | Why |
|---|---|---|
| `predict(L, gc)` LUT | **No** | Capture model dropped. |
| ZTNB parameters | **No** | Dropped. |
| flgc model identity / `GCFlDistModel` | **No** | Dropped; no flgc dependency. |
| Dup-histogram sidecar | **No** | No longer an input. |
| GC bin edges | **No** | No GC conditioning. |
| `marginal_fl` | **Changed** | Now `FragmentLengthDist.counts` from the same admitted population as C. |
| `build_region_weights` output | **No** | Capture layer. |
| Per-region counts | **Changed** | Was a fabricated constant table; now `n_observed` from the h5 (ruling 6). Stored per region (§9). |
| Hexamer tables | **Yes, extended** | Four C tables + N (start, end) + four r tables. |
| `pad` | **Changed** | Replaced by `left_pad=3`, `right_pad=183` (asymmetric). |
| Admission rule | **Changed** | Start-in-region, never midpoint. Recorded as a string. |
| `fl_bands` | **No** (moved) | Model concern. |
| Region set / reference sha256 | **Yes** | Plus `.fai` md5, matching the store convention. |
| Seed | **Yes, demoted** | Provenance only (ruling 4). |
| Code identity | **Yes** | Per-module source sha256 (mandatory) + git (best-effort). |
| Self-verification summary | **Yes** | One `self_check` block (§8). |
| `p_plus` | **New** | Did not exist; no producer yet (§12). |
| Store, D, oracle | **Removed** | Ruling 1. |

Accepted consequence of the drop, not a defect: the input f(L) matches real data by construction. The (length, GC) joint does not.

---

## 5. Output contract

### 5.1 Files

Exactly two files in the output directory:

| File | Content |
|---|---|
| `<stem>.fragments.h5` | Realised draws, built by `fragments_h5.build_fragments_h5` from a BED. |
| `<stem>.simmeta.h5` | Everything else: provenance, parameters, realised arrays, self-check summary. |

- No other file survives. BED, `.bed.gz`, `.tbi` live in a temp dir and are deleted. [Ruling 2]
- The h5 cannot point to its metadata. [Inferred] The builder writes its own attrs; `_build_argv` holds only what the caller passes. Linkage is the shared stem plus the h5 sha256 and size recorded in the metadata.
- Consequence: a renamed h5 loses its link by name. The hash still identifies it.
- Hazard: `*.simmeta.h5` is not a fragments h5. A sample sheet that points at it fails at `FragmentsH5` open. [Inferred]

### 5.2 Destination and publish order

Destination is on `/efs`. Treat `/home` as read-only in batch (disputed writability, standing rule).

1. Refuse to start if either final file exists. No silent overwrite.
2. Build everything in a local temp dir (`tempfile.mkdtemp`).
3. Run all mandatory self-checks against the temp h5 (§8). On failure, publish nothing.
4. Copy temp h5 -> `<dest>/<stem>.fragments.h5.tmp`. Re-hash the copy; require equality with the temp hash. `os.replace` to the final name.
5. Write metadata -> `<dest>/<stem>.simmeta.h5.tmp`. `os.replace` to the final name.
6. Delete the temp dir.

- The metadata file's existence is the completion marker. An h5 without metadata is a failed run. [Inferred]
- `os.replace` is atomic only within one filesystem. Hence copy-then-rename across `/tmp` -> `/efs`. [Inferred]
- Refinement over the coordinator proposal: checks run on the temp copy before any publish, so a failed check leaves nothing at the destination.

---

## 6. Driver

New module `background_model/simulator/simulate.py`. CLI via `python -m`. It consumes library entry points only (CLAUDE.md table). Step numbers match the coordinator proposal.

### 6.1 Load regions

- `rdf = RegionDataFrame.from_bed(bed, ref=...)`. `ref` is required.
- `region_index` = row position after load. Whether `from_bed` reorders rows is [Unverified]: read `from_bed`.
- Assert regions are disjoint. [Inferred] Overlapping regions would admit one fragment into two regions. That double-counts in C and breaks the per-region round trip (§8).

### 6.2 One measuring pass (split `count_sample`)

Split `count_sample` into `measure_sample(...) -> (srdf, counts, stats)` plus the existing wrapper. `count_sample` keeps its return value. This is plumbing; it must not change computed results (§14 risk 1).

From the one retained srdf, in one pass, from one admitted population:

| Output | Source |
|---|---|
| C (4 tables, int64) | `count_srdf(srdf)` |
| f(L) | `FragmentLengthDist.from_srdf(srdf)` |
| `n_observed` per region | `len(fragment_array)` after `filter_fragments` |
| padded sequence per region | `srdf["sequence"]` (`left_pad=3`, `right_pad=183`) |
| strand counts | fragment arrays (input to `p_plus`, §12) |

This matches spec §2: stages 1, 2 and 4 share a pass. [Inferred from the coordinator's reading of the spec]

`min_mapq=10` default. Recorded.

### 6.3 Uniform counts

`N = uniform_hexamer_counts(rdf, fasta, fl)` on the region frame **without** fragments (it refuses a frame that has them). Serial pysam walk. N depends on f via `right_flank=fl.max_fl+3` and the float64 end weights. [Verified] signature; [Inferred] that `N_end` is fl-weighted (float64 dtype, `fl` argument, `end_weight_total` meta).

### 6.4 Propensities

`r = propensities(C, N, min_expected=...)`. `min_expected` default 0.0. Any other value changes the sampled data and needs owner approval. Recorded.

### 6.5 `p_plus`

**[Open]** See §12 item 1. The driver takes `p_plus` from an explicit source flag. It has no default until the owner decides.

### 6.6 Sampling loop

For each region in `region_index` order:

```
n_requested = n_observed[i]                         # v1; §9 seam
rng = np.random.Generator(np.random.PCG64(
          np.random.SeedSequence([seed, i, 0])))    # stream 0 = fragments
starts_0, lengths, is_plus = sample_region(seq[i], region_len[i], n_requested,
                                           r=r, fl=fl, p_plus=p_plus, rng=rng)
gstart = region.start + starts_0
gstop  = gstart + lengths
```

- Per-region seed: draws do not depend on region order or on worker count. Stream 1 is reserved for the count draw (§9). [Inferred]
- Assertions, loud: `fl.min_fl >= L_MIN`, `fl.max_fl <= L_MAX`, `len(seq[i]) == region_len[i] + 186`, all `0 <= starts_0 < region_len`.
- Record per region: `n_requested`, `n_drawn`, `n_distinct` (distinct `(gstart, gstop)` among drawn).
- Start serial. The cost of `sample_region` per region is [Unverified]; measure it on a subset before any parallel path. The old ms/region notes do not apply to the rewrite. `parallel_apply` has a fork-deadlock history in this repo.

### 6.7 Emission

1. Accumulate `contig_code, gstart, gstop, is_plus` arrays. Sort with `np.lexsort` by (contig order, start, stop, strand). In-memory sort makes the BED bytes deterministic.
2. Write 8-col BED text vectorised (one `np.savetxt`-style call per chunk, not one Python write per row) to the temp dir. Strand column `+`/`-`; MAPQ cols `60 60`.
3. Compress + index. **Recommendation:** `pysam.tabix_index(path, preset="bed")` instead of `bgzip`/`tabix`/`sort` binaries. Reason: pysam is already a hard dependency (`uniform_hexamer_counts`); CLAUDE.md records PATH-binary failures in sandboxes and Batch (bedtools). Exact pysam behaviour in this env is [Unverified]: test it in P3. Fallback: emit.py's binary pipeline with preflight.
4. `build_fragments_h5(bed_gz, tmp_h5, fasta_filename=ref, read_strand=True, num_processes=<explicit>, build_argv=<driver argv>)`. Pass `build_argv` so the h5 records how it was built. The builder's fork behaviour is [Unverified]; pass an explicit `num_processes` and record it.
5. Assert the temp dir holds only the expected files.

**Name column (col 4): [Open], §12 item 4.** `"."` becomes `cell_barcode "."`. Whether the h5 stores or uses barcodes is [Unverified]. Recommendation: build a tiny h5 with `"."` and with an empty field; keep the form that writes no barcode, if the parser accepts it. Record the choice.

### 6.8-6.10 Self-check, metadata, publish

§8, §7, §5.2. Metadata is written last.

---

## 7. Metadata schema (`<stem>.simmeta.h5`)

### 7.1 Container

| Option | Verdict | Reason |
|---|---|---|
| **HDF5 via h5py** | **Recommend** | Already a hard dependency of the ingest. Lossless int64/float64. Real strings. One file. `h5dump` reads it. |
| pandas `HDFStore` | Reject | Needs pytables; no `tables` package found in biomarker_env. [Verified] Glob |
| Multi-file Parquet | Reject | Violates ruling 3. |
| SQLite | Viable | Lossless single file. Not used elsewhere in the repo. [Inferred] Owner confirms (§12 item 3). |
| npz | Reject | No keyed frames; no string-keyed index. |
| zarr | Reject | Pinned v2, a directory not a file, heavier. |
| JSON | Reject for arrays | float64 round trip needs care; large. Kept only for the provenance text attr. |

### 7.2 Layout

Each "frame" is a group with one 1-D dataset per column, equal length, plus attrs `index` (column name) and `columns` (ordered list).

```
/                       attrs: simmeta_schema_version (int)
                               provenance_json (str, §7.4)
                               self_check_json (str, §8)
                               p_plus (float64)
                               min_expected (float64)
                               seed (int64)          # provenance only
/hexamer                index = "hexamer"; 4096 rows, written in hexamer_vocabulary() order
    hexamer             S6        key
    C_start_fwd, C_end_fwd, C_start_rev, C_end_rev     int64
    N_start             int64
    N_end               float64
    r_start_fwd, r_end_fwd, r_start_rev, r_end_rev     float64
/fragment_length        index = "fragment_length"; rows = fl support
    fragment_length     int64
    count               int64
    density             float64
/contigs                name (vlen str), one row per contig used
/regions                index = "region_index"; one row per region
    region_index        int64
    contig_code         int16    -> /contigs row
    start, stop         int64    (BED, 0-based half-open)
    n_observed          int32    real post-admission count (ruling 6)
    n_requested         int32    what sample_region was told
    n_drawn             int32
    n_distinct          int32    distinct (start,stop) among drawn
    n_readback          int32    from the round trip (§8)
```

- Scalars live as root attrs, not only inside JSON. A reader gets `p_plus` without a JSON parse.
- r is stored even though it is derivable from C and N. Reason (ruling 4): the sampler consumed r; a later change to `propensities` must not change the recorded input. On write, assert `np.array_equal(r_stored, propensities(C, N, min_expected=...))` per table.
- C, N, f counts are int64 or float64 as produced. Accumulation in the driver uses float64 (standing rule).
- `int32` counts assume no region holds more than 2^31-1 fragments. [Inferred] safe for 1536-bp tiles.

### 7.3 Loader (the primitive the model agent imports)

- `load_simmeta(path) -> SimMeta`: DataFrames (`hexamer` indexed by the string, `fragment_length`, `regions` with contig names joined), scalars, parsed provenance and self-check.
- `SimMeta.r_arrays() -> dict[str, np.ndarray]`: the four r tables in current `hexamer_vocabulary()` order. Guard (port of emit.py :50-176 intent):
  - on-disk keys are 4096 unique strings over `ACGT`;
  - their set equals the current vocabulary;
  - reindex by string, not by position;
  - columns addressed by name only.
- `SimMeta.fl()` -> `FragmentLengthDist` rebuilt from stored counts.
- `simmeta_schema_version` mismatch raises. No compat shim.

### 7.4 Provenance JSON (`provenance_json`)

| Block | Fields | Tier |
|---|---|---|
| run | `run_id` (uuid4), `created_utc`, driver argv | mandatory |
| rng | `seed`; construction string `PCG64(SeedSequence([seed, region_index, stream]))`; stream map `{0: fragments, 1: reserved count draw}`; numpy version | mandatory |
| code | `source_sha256` of the bytes of `count_hexamers_rdf.py` and `simulate.py` | mandatory |
| code | git commit sha, dirty flag | best-effort; `null` + reason when git is absent. A recorded null, never a pass. |
| libs | fragmentomics_tools identity; numpy, pandas, pysam, h5py versions | mandatory |
| libs | fragments_h5 identity read from the **output** h5 attr `_build_code_revision` ([Verified] written) and `_build_version` ([Unverified] that it exists) | mandatory for `_build_code_revision` |
| source sample | sample id, resolved h5 path, size, sha256, its `_source_format`, `source_total_fragments = fragment_length_counts.sum()` | mandatory |
| measurement | `min_mapq`, `l_min=25`, `l_max=180`, admission `"start-in-region"`, `left_pad=3`, `right_pad=183`, `KMER`, `HEX_HALF`, `count_srdf` stats (`n_after_filters`, `n_counted`), `uniform_hexamer_counts` meta | mandatory |
| reference | path, sha256 of the FASTA, md5 of `.fai` (store convention, `config.py:47-57`) | mandatory |
| region set | path, sha256, `n_regions`, tile length(s), `region_seq_sha256` (§7.6) | mandatory |
| parameters | `p_plus`, `p_plus_source`, `min_expected`, `n_source = "exact_from_h5"` | mandatory |
| output | h5 filename, size, sha256, `n_fragments` (`fragment_length_counts.sum()`), `_source_format` (expect `"TSV"`), BED name-column choice, `num_processes` | mandatory |

Failure to compute any mandatory field aborts the run before publish.

- NumPy does not promise Generator stream stability across releases. [Inferred] (NEP 19 policy; not checked this session). This is why ruling 4 stores arrays and why the h5 itself is the realised draw. The seed reproduces the draws only under the recorded numpy version.
- Hash cost of the reference FASTA and the source h5 is [Unverified]; time it in P5.

### 7.5 Byte cost (arithmetic, not measured; uncompressed payload, HDF5 overhead excluded)

Hexamer frame, per row:

| Columns | Bytes |
|---|---|
| `hexamer` S6 | 6 |
| 4 x C int64 | 32 |
| `N_start` int64 | 8 |
| `N_end` float64 | 8 |
| 4 x r float64 | 32 |
| **row** | **86** |

`4096 x 86 = 352,256 B`. Of this, r alone is `4096 x 32 = 131,072 B` (the cost of ruling 4's redundancy).

Fragment-length frame: `8 + 8 + 8 = 24 B/row`. At most `N_LENGTHS = 156` rows inside [25,180]: `156 x 24 = 3,744 B`.

Regions frame: `region_index 8 + contig_code 2 + start 8 + stop 8 + 5 x int32 20 = 46 B/row`.

| Region set | Rows | Regions frame | Total with hexamer + FL frames |
|---|---|---|---|
| repeats-removed tile1536 (66,649 regions per spec) | 66,649 | `66,649 x 46 = 3,065,854 B` | `3,065,854 + 352,256 + 3,744 = 3,421,854 B` |
| repeats-kept set (904,975 rows, coordinator figure) | 904,975 | `904,975 x 46 = 41,628,850 B` | `41,628,850 + 356,000 = 41,984,850 B` |

Plus provenance/self-check JSON: size not measured, small relative to the frames. [Unverified]

Realised draws: the fragments h5 **is** the realised draws. The metadata does not duplicate them. h5 size is [Unverified]; record it per run.

### 7.6 Region sequence: hash-pin vs store **[Open]**, §12 item 2

| Option | Cost (arithmetic, not measured) | Property |
|---|---|---|
| Store padded sequence per region | `region_len + 186` B/region; 1536-bp tiles: `1,722 B`; `66,649 x 1,722 = 114,769,578 B` | Self-contained. 2-bit packing cannot hold N / soft-mask losslessly. |
| Same, repeats-kept set, if also 1536-bp tiles ([Unverified] tile length) | `904,975 x 1,722 = 1,558,366,950 B` | |
| **Hash-pin (recommend)** | `region_seq_sha256`: 32 B (sha256 over padded sequences in region order) + FASTA sha256 + `.fai` md5 | Reference is an immutable, content-addressed asset, like the source h5. Reconstruction re-fetches and checks the hash. |

This is a ruling-4 trade. The recommendation treats the reference as an asset, not a recipe. The owner confirms.

---

## 8. Self-verification

Summary statistics only, in `self_check_json`. No arrays. All checks run on the **temp** h5 before publish.

The read-back reuses `measure_sample` from §6.2 on the simulated h5, same `min_mapq=10`, same `filter_fragments`. The check does not re-derive the admission rule. One call gives the read-back srdf, `C_sim`, `f_sim` and `n_readback`.

| # | Check | Tier | Pass rule | Cannot show |
|---|---|---|---|---|
| i | Round-trip exactness. Per region: the read-back distinct `(start, stop)` set equals the drawn distinct set; `n_readback == n_distinct`; strand equal for every non-colliding pair. Totals `n_drawn, n_distinct, n_readback` recorded. | mandatory | exact | Whether the draws follow the intended distribution. For a `(start, stop)` collision across strands, which strand survives (dedup key omits strand). |
| ii | Shortfall: regions with `n_drawn < n_requested`, their count, total shortfall. | report | none | Why a start had no valid length. |
| iii | Recount: `C_sim` vs C per table (Pearson/Spearman, ratio quantiles). | report | **none: no threshold has been derived** | Sampler bias of a size below the noise; any absolute correctness. |
| iv | Length: total-variation distance between `fl.densities` and the read-back length histogram. | report | none | Note: `sample_region` weights length by `end_table x fl.densities` per start [Verified :563-633], so the realised marginal can tilt away from f(L). [Inferred] The TV measures the tilt; it does not judge it. |
| v | `r` stored == `propensities(C, N, min_expected)` bit-for-bit. | mandatory | `np.array_equal` | That `propensities` itself is correct. |
| vi | Output hash: published copy sha256 == temp sha256. | mandatory | equal | |

Check (i) also proves: BED coordinate base, strand column, MAPQ 60/60 survival of `min_mapq=10`, and the sort/index/builder plumbing. [Inferred]

Check (i) holds whatever `fetch_array` admits (overlap or start), because `filter_fragments` applies start-in-region after the fetch. [Inferred]

---

## 9. Count-source seam (ruling 6) and the follow-up

v1:

- `n_observed[i]` = real post-admission count from the source h5 (after MAPQ, dedup, length, start-in-region). It includes fragments that `count_srdf` later drops for an invalid cut-site hexamer.
- `n_requested[i] = n_observed[i]`. `n_source = "exact_from_h5"`.
- No count artifact. The count comes from the h5 at simulation time.

Where the follow-up (Poisson or other fitted draw, not designed here) attaches:

| Element | v1 | Follow-up |
|---|---|---|
| `regions.n_observed` | exists | unchanged |
| `regions.n_requested` | `= n_observed` | drawn from the fitted model |
| RNG | stream 0 only | count draw on stream 1, `SeedSequence([seed, i, 1])` |
| `n_source` | `"exact_from_h5"` | e.g. `"poisson_fit"` |
| provenance | — | `count_model` block (family, fit method, inputs) |
| realised arrays | — | per-region `lambda` (float64; `66,649 x 8 = 533,192 B`, arithmetic) + fitted parameters (ruling 4) |
| sampler | `sample_region(n=...)` | unchanged; already n-parametric |
| schema version | v1 | bump only if columns change |

The count draw does not consume stream 0. So a region whose `n_requested` does not change keeps identical fragment draws (same numpy version). A region whose `n_requested` changes gets different draws, as it must. [Inferred]

Designs that would foreclose the follow-up:

- metadata with only total counts, no per-region `n_observed`;
- the count chosen inside `sample_region`;
- one global RNG stream across regions (a count draw would shift every later region's fragments);
- a separate count file (violates rulings 2, 3, 6).

---

## 10. Interface hazards for the model agent

State only; none is a simulator defect.

1. **Start-in-region truncation.** Only fragments that start inside a region exist in the h5. For an isolated tile, no fragment starts in the gap to its left. Stop-track coverage within about `L_MAX` of the left edge, midpoint coverage within about `L_MAX/2`, and everything in the model's margin (tile +- jitter, `preprocess.py:154-179`) is depleted relative to real data. The oracle and any D must use the same truncated domain. Evidence: lines 1-6 of the 66,649-region BED are isolated tiles; lines 8-10 are contiguous. [Verified] So "tiles are contiguous" (`filter_fragments` docstring) holds only within runs.
2. **Dedup at ingest.** `config.dedup=True` removes drawn `(start, stop)` collisions. Real data had the same dedup before C was measured. An oracle on raw draws differs from one on ingested draws by the collisions. `n_distinct` per region is recorded for this.
3. **Sampler renormalisation.** A drawn start with no valid length is dropped; a strand with zero start weight is skipped. The effective distribution is conditional on these events. The oracle must replicate `sample_region`, not a textbook r-normalised model. [Verified :563-633]
4. **Total fragments.** `preprocess.py:140` takes `fragment_length_counts.sum()`. For a simulated h5 that is the in-region draw count, not the real sample's genome-wide total. `source_total_fragments` is recorded. How the store uses the total is [Unverified].
5. **Strand** is required by `split_strand=True` and is emitted.
6. **GC** in the h5 is computed by the builder from the reference at simulated coordinates. The (length, GC) joint is not matched (accepted).
7. **Missing contigs.** Contigs absent from the tabix index are absent from the h5 (`contig_lengths`). Ingest behaviour for a region on such a contig is [Unverified]. Relevant to small subsets and tests.
8. **Shortfall.** `n_drawn` can be below `n_requested` (§8 ii).
9. **MAPQ.** Draws carry 60/60. A model config with `min_mapq > 60` drops everything. [Inferred]

Oracle generative inputs, all in the two files plus the reference: r (4 tables), f(L), `p_plus`, per-region `n_requested`, region coordinates, padded sequence (reference + coordinates + `region_seq_sha256`), `L_MIN/L_MAX`, pads, admission rule.

---

## 11. Stale references (flagged, not fixed)

| Item | Why stale |
|---|---|
| `docs/pending/simulator_basic_inputs.md` (2026-10-05, "design, nothing implemented") | Built on `predict(L,gc)`, marginal_fl estimator, duphist, **midpoint** admission, the 11,016-row autosome tile2560 set, `counts_source`/`marginal_fl_source` fields. Contradicts the spec on admission and the dropped capture model. One research agent trusted it and reported capture as live; the coordinator checked. |
| `docs/pending/h5_derived_counts_and_fl.md` | Superseded by the doc above, which is itself superseded. Capture/midpoint/count points stale. |
| `docs/pending/simulator_and_fragment_nll.md` | Step 1 duphist, Step 5 counts 54/37, "Measured anchors", `build_region_weights` as binding oracle contract. |
| `docs/pending/simulator_overview.md` | Lists the fabricated constant as the Open state; documents emit's `hex_tables_to_dict`. |
| `HANDOFF_BACKGROUND_MODEL_V2.md`, `HANDOFF_SIM_STUDY_V2.md` (repo root, untracked) | Describe the quarantined attic `scripts/sim_fragments.py` generation. |
| `capture.py` (whole), `weights.py` Step 4, `precompute.py` `MAX_FL_HALF` pad, `sampler.py` (`build_region_weights`, `_REGION_COUNTS`), `count_hexamers.py` (midpoint) | Old generation. |
| `emit.py` `predict_lut` fields (:406,449,630,635), `pad`, `fl_bands` default (:416) | Old generation. |
| `scripts/run_simulator.py`, `scripts/cut_site_oracle.py` | Consume the old emit API. |
| `tests/test_simulator_phase2.py`, `tests/test_simulator_weights.py` | Test the weights/capture layer. |
| `count_hexamers_rdf.py:89-91` | Cites nonexistent `test_encoder_matches_precompute`. |
| `scripts/count_cut_site_hexamers.py` | Imports the old encoder, not the rewrite's. |
| Old oracle input list (`build_region_weights` + predict LUT + marginal_fl) | Reduced to the list at the end of §10. |

---

## 12. Open decisions (owner)

| # | Decision | Options | Recommendation | Why it needs the owner |
|---|---|---|---|---|
| 1 | `p_plus` | (a) measured: `start_fwd.sum() / (start_fwd.sum() + end_rev.sum())` from C (hexamer-valid fragments); (b) measured from fragment strands of all admitted fragments (before hexamer gating); (c) fixed 0.5 | measured from the same admitted population; (a) vs (b) also open | It determines the sampled data (repo rule). Zero extra cost either way. |
| 2 | Region sequence | hash-pin vs store | hash-pin (§7.6) | Ruling-4 trade. |
| 3 | Metadata container | HDF5 vs SQLite | HDF5 | Format lock-in. |
| 4 | BED col 4 | `"."` vs empty | the form that writes no barcode, after a test | Unknown h5 effect. |
| 5 | Old generation | delete now vs later | later is fine; copy emit.py helpers, never import | Owner deferred it. |
| 6 | Parallel sampling | serial vs parallel | serial until P5 measures cost | fork-deadlock history. |
| 7 | Compress/index tool | pysam vs binaries | pysam (§6.7) | Changes a tested pattern. |
| 8 | `min_expected` | 0.0 vs other | 0.0 (module default) | Changes r. |

---

## 13. Testing

The module has no tests today. Run `make test` (never bare pytest; `timeout 3600`; exit 137 = hang, diagnose with `py-spy dump`) for both suites (`test/` library, `tests/` background_model) before and after each phase. Measure baselines at implementation time. This doc quotes none.

Fixture: a small FASTA. The repo has `tests/data/GRCh38.p12...chr6_99110000_99130000.fa.gz`; its contig naming is [Unverified]. Synthetic fragment source: a tiny BED built into an h5 by the production builder.

Rule: never pin a `default_rng` draw. Assert invariants.

| Test | Asserts |
|---|---|
| Encoder pinning | Rewrite encoder == `precompute.py` encoder on uppercase input. Write it, or delete the docstring claim at :89-91. |
| `count_sample` split | `count_sample` output before == after on the fixture, bit-exact. |
| Degenerate r | r nonzero for one start hexamer only -> every drawn plus start carries it. Deterministic. |
| Strand routing | plus draws read `start_fwd`; minus draws route through `rc_permutation` to `start_rev`/`end_rev`. Degenerate tables per strand. |
| Admission | all `starts_0 in [0, region_len)`. |
| Shortfall | a region with zero valid lengths -> `n_drawn < n_requested`, reported, no crash. |
| Contig end | a region within 183 bp of a contig end fails loud. |
| Round trip | §8 (i) on the fixture: exact. |
| MAPQ survival | built h5 read with `min_mapq=10` keeps every row. |
| Metadata round trip | every array `np.array_equal`, float64 bit-exact; scalars exact. |
| Vocabulary guard | permuted, truncated, or duplicated hexamer keys raise; permuted-but-complete keys reindex correctly. |
| Schema version | wrong `simmeta_schema_version` raises. |
| Atomic publish | fault injected after the h5 publish -> no metadata file; fault before -> no files. |
| Overwrite refusal | existing final file -> refuse. |
| Tool preflight | monkeypatched `shutil.which` (if binaries remain) fails loud. |
| E2E | full driver on the fixture. No `importorskip`: a missing dependency fails. |
| Ingest compatibility | run `preprocess._worker_inner` (or its smallest real entry) on the simulated h5. |

---

## 14. Implementation plan

| Phase | Work | Depends on |
|---|---|---|
| P0 | Measure `make test` baselines, both suites. Encoder pinning test (or docstring fix). | — |
| P1 | Split `count_sample` -> `measure_sample` + wrapper. Equality test. | P0 |
| P2 | `simmeta` module: schema, writer, `load_simmeta`, guard, hash helpers copied from emit.py. Tests. | P0 |
| P3 | Emission: sorted arrays -> BED -> bgzip/tabix -> `build_fragments_h5`. Decide col 4 and pysam vs binaries here. Round-trip test. | P0 |
| P4 | Driver, self-checks, publish. E2E + ingest-compatibility tests. | P1, P2, P3; owner answer on `p_plus` |
| P5 | Subset run (first few hundred regions) to time `sample_region`, hashing, build. Then full run on Batch, output to `/efs`. | P4 |

Risks:

1. **`count_sample` split** must not change results. Verify bit-exact C before/after on the fixture.
2. **Builder labels.** fragments_h5 release tags/commits/containers have been mislabelled (coordinator memory). Record identity from the output h5's `_build_code_revision`, not from `importlib.metadata` (editable-install dist-info can be stale).
3. **Builder determinism** for BED input is [Unverified]. If not deterministic, the h5 sha256 is still a valid identity of the published bytes, but a rebuild will not reproduce it.
4. **Col 4 barcode** side effect (§12 item 4).
5. **Contig-end truncation**: `attach_sequence` behaviour [Unverified]; the length assertion catches it.
6. **Memory.** The retained srdf holds padded sequence: `1,722 B/region` for 1536-bp tiles, `114,769,578 B` for 66,649 regions, `1,558,366,950 B` for 904,975 regions if also 1536-bp (arithmetic, not measured; Python object overhead and fragment arrays excluded). v1 holds the whole srdf. For the 904,975 set, a chunked second pass (re-attach sequence per chunk) is the likely fix; not designed here.
7. **Output writability** in Batch: write to `/efs`; run from `cd /tmp` with `PYTHONPATH`; temp dir under local `TMPDIR`.
8. **Parallelism inside `attach_fragment_arrays`** (whether it uses `parallel_apply`) is [Unverified]. A hang shows as exit 137 under `make test`.
9. **Hash cost** of a whole-genome FASTA and a deep source h5 per run is [Unverified].

---

## 15. Self-grade

**B.** The structure follows the rulings, and every number is arithmetic shown or quoted with a source. The grade is capped because nothing was run, and several load-bearing claims are inferences.

Least sure of, in order:

1. **pysam tabix route** (§6.7). I recommend replacing three binaries with `pysam.tabix_index`. Its behaviour in biomarker_env and its compatibility with the builder's BED reader are [Unverified]. If it fails, the emit.py binary pipeline is the fallback.
2. **Check (iv) tilt claim.** I infer from the summary of `sample_region` that end-table weighting can tilt the realised length marginal away from f(L). I did not read the code. If wrong, check (iv) is only a sanity check.
3. **`p_plus` option (a) formula.** It assumes `start_fwd.sum()` and `end_rev.sum()` count the same hexamer-valid plus/minus populations. That follows from the routing at :299-319, but a fragment with one valid and one invalid end may be counted differently. [Unverified]
4. **Disjoint-region assertion.** I assume the region BEDs are non-overlapping. Only 12 lines were read.
5. **Reconciliation table (§4).** The item list comes from emit.py fields and stale docs, not from the text of rulings 3/4.
6. **int32 count columns** and the omitted HDF5 overhead in §7.5.
