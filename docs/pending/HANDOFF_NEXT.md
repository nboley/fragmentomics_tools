# Handoff — what to do next

Forward-looking only. For what was already done and why, see
`docs/pending/dataframe_critical_review.md` §11 (resolution ledger) and
`docs/pending/absolute_capture_probability.md` §13 (closed line of work).

**State as of handoff:** `main` == `origin/main` == `e0d290e`. Everything is
merged and pushed. Library suite 2 failed / 289 passed; background_model suite
396 passed.

---

## 0. Read this before running anything

**There are two test suites and they need two different environments.**

| suite | env | result |
|---|---|---|
| `test/` (library) | `/home/nathanboley/.local/share/mamba/envs/fragtools-test/bin/python` | 2 failed, 289 passed |
| `tests/` (background_model) | `/home/nathanboley/miniconda3/envs/biomarker_env/bin/python` | 396 passed |

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is **required** for both — a stray plugin in
`~/.local` breaks collection otherwise. Running the wrong env against the wrong
suite produces `ModuleNotFoundError` for `lightning`/`zarr` and looks like
broken code. It is not.

**`<env>/bin` must be on `PATH`, not merely used to locate python.** The library
suite shells out to eight binaries — `bgzip`, `tabix`, `bedToBigBed`,
`bigBedInfo`, `bigBedToBed`, `bedGraphToBigWig`, `samtools`, `bedtools`. All are
installed in `<env>/bin/`, but invoking `<env>/bin/python` by absolute path does
**not** put that directory on `PATH`. Omit it and you get **48 failed / 243
passed** instead of 2/289: 26 `FileNotFoundError: 'bgzip'`, 11
`'bedToBigBed'`, and 7 `pybedtools` `NotImplementedError` for `intersectBed` /
`sortBed`. Every one of those reads as a code defect, and none of them is one.

> **Correction (2026-09-24).** An earlier revision of this file gave the command
> without the `PATH` assignment. Its *numbers* were right — 2/289 reproduces
> exactly — but the command as written does not produce them. This is the trap
> `CLAUDE.md` already flags: `bedtools` ships in the conda env's `bin/` and "is
> frequently *not* on `PATH` in sandboxes." Measured, not inferred.

```bash
E=/home/nathanboley/.local/share/mamba/envs/fragtools-test
PATH="$E/bin:$PATH" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $E/bin/python \
    -m pytest test/ --ignore=test/test_dataframe.py -q -p no:cacheprovider
```

---

## 1. Tests: three distinct problems, in priority order

### 1a. Clean-checkout reproducibility — DONE (`59ccb00`)

`test/fragment_array/data/` was **gitignored** (`.gitignore:2: data/`), so the
fixtures lived only in whatever working tree happened to have them, and a fresh
clone errored **20 tests** with `FileNotFoundError: .../small.chr6.bam`. Both the
20 and the gitignore cause were confirmed by cloning to `/tmp` and running it.

Resolved by committing the five files (624 KB) behind a `.gitignore` exception
block mirroring the one already present for `tests/data/` — the same convention
the owner had established for the same reason, rather than a new policy.
Fetch-on-demand was rejected: there is no CI to exercise such a fetch, so a
broken one would stay silently broken until the next clone.

Verified end to end: a clean clone of `59ccb00` now gives **2 failed / 289
passed / 0 errors**, matching the tree that had the files.

### 1b. `test_dataframe.py` — 26 tests, still uncollectable

Its primary fixture (`tss.all_rampage.hg19/hg38.bed.gz`) is **unrecoverable**:
listed in the manifest, but the backing bucket `s3://test-data-manifest-2-2024`
no longer exists and the files are not in the surviving
`karius-biomarker-data-assets` (searched all 742,361 objects).

The assertions pin values derived from that exact file
(`len(rdf) == 18263`, `Counter({0: 9773, 1: 5566, -1: 2924})`), so restoring
coverage means **synthesising a fixture and recomputing the assertions**. That
trades test *meaning* for test *coverage* — the recomputed numbers assert only
"the code does what it currently does". Decide deliberately; see
`dataframe_critical_review.md` §9.4.

`I12` (test references undefined `blacklist_path`) lives in this file and is
blocked behind the same decision.

### 1c. Two pre-existing failures, both missing data

- `test_slice_encode_big_wig` — needs the ENCODE bigwig
- `test_get_one_hot_encoded_sequence` — needs the in-package GRCh38 reference

Not code defects. Leave red or source the files; do not "fix" the tests.

---

## 2. `dataframe.py` cleanup — the main event

The file is ~2300 lines and mixes region algebra, sample joins, fragment-array
plumbing, I/O and labelling. The review found 57 issues; the ones below are
what remains open. **Fix the correctness items before any restructuring** — a
refactor on top of unfixed silent-wrong-answer bugs just relocates them.

### Likely deletable (check callers first — three such layers have already gone)

| id | what |
|---|---|
| `I13` | `attach_num_tss_overlaps` calls `get_tss_intervals`, which is **defined nowhere** — another orphaned `ravel` reference |

Precedent: fragment-BED support (351 lines), the binary-labeling API (146
lines), `ref_path`, and `region_mask` were all deleted after a blast-radius
check showed zero consumers. **Always check `biomarker`, `biomarker-pipeline`
and `biomarker-projects` before deleting, and exclude `.ipynb_checkpoints` and
`.claude/worktrees` from the grep — they duplicate files and inflate counts.**
Also grep `*.ipynb`, not just `*.py`: a `.py`-only search once produced a
wrong "zero consumers" answer here.

### Correctness — silent or structural

| id | what |
|---|---|
| `R3` | empty-intersection coverage returns a wrong-shaped array |
| `I4` | return type varies with result size (plain `DataFrame` when empty, subclass otherwise) |
| `I10` | `__eq__` returns a scalar — breaks pandas semantics and makes instances unhashable |
| `I11` | `center_on_summit` uses `summit <= stop`; half-open convention wants `<` |
| `I16`-`I23` | `overlaps_rdf` `KeyError` on missing contig; `__and__` downcasts subclass; `lift_over` empty/`-1`; `concat([])`; mutation semantics; dead `has_header`; descending `unique_regions` sort |
| `R4`, `R8`, `R10`, `R14` | resize/binning boundary conditions |
| `S3` | `SampleDataFrame.__init__` inspects only `iloc[0]` to decide whether handles are live |
| `S6`-`S9` | opaque `KeyError`; per-group assert; empty-list `IndexError`; no h5 close path |

### Ergonomics

| id | what |
|---|---|
| `I5`, `I6` | unstranded regions rejected by `get_interval_dict`; `from_bed` crashes on an empty file |
| `I7` | `drop_overlapping_regions` hardcodes `assert ref == "hg38"` for no stated reason |
| `R11` | `resize_regions` warns about discards even when none occurred |
| `B8` | mutable class-level list defaults (`_metadata = []`) — latent, all current subclasses reassign |

### Needs the owner — do NOT change unilaterally

`CLAUDE.md` gates anything altering computed results.

| id | what |
|---|---|
| `S5` | `get_sample_count_bounds` computes a **median**, returns it as `mean_fragment_counts`. Name or computation is wrong — unclear which was intended. |
| `S4` | duplicate `sample_id`s **silently dropped** in `FlDist.init_from_sdf` (last wins) — data loss with no warning |

---

## 3. Two smaller loose ends

- **A latent doctest failure.** `get_indices_of_balanced_labels` fails under
  numpy 2.x (`np.int64(1)` where the docstring expects `1`). Invisible because
  the suite does not run `--doctest-modules`.
- **`reset_fragment_array_weights` docstring says "set to zero"; the code sets
  ones.** Zero and one are opposite defaults here. Owner flagged ones as
  correct, so the docstring is what is wrong.

---

## 4. Working notes that will save time

- **Verify before asserting.** Several claims this session — from agents and
  from me — were confidently wrong and had to be retracted in the docs. Reading
  a manifest entry is not the same as checking the bucket exists; a `.py`-only
  grep is not a blast-radius check; a number transcribed from a table is not a
  measurement. Run it.
- **Prove a test fails against the unfixed code.** Extract the old version with
  `git show <sha>:path`, copy the package to `/tmp`, set `PYTHONPATH`. Never
  revert the working tree — another session may be live in it.
- **A grep hit in a `.ipynb` is not necessarily a consumer.** The `.py`-only
  search that missed notebook callers is recorded below, but the opposite error
  is just as easy: `attach_num_tss_overlaps` (I13) matched three times in
  `biomarker-projects`, and all three were **output cells** — `dir()` dumps, not
  calls. Parse the notebook and check `cell["source"]` rather than grepping the
  raw JSON. A tell: those same dumps listed `ref_path` and `set_binary_label`,
  both deleted long before.
- **Bash traps that produced wrong answers here:** `$?` after a pipeline is the
  last command's status, not the one you care about; a grep that times out
  (exit 124) piped to `head` looks exactly like "no matches"; `git tag` sorts
  lexically, so `tail` shows v2.9 above v2.13 — use `sort -V`.
- **The working tree may be shared.** Check `git status` for other sessions'
  uncommitted work before any checkout, stash, or reset. Use a separate
  worktree for anything destructive.
- **There is no CI.** The owner declined it. The suite went 0 → 289 tests this
  session and nothing runs it on a schedule; the deleted `v2.10.1` tag that
  broke `docker build` for everyone is what that drift looks like in practice.
