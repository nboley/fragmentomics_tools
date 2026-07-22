# Fragmentomics Tools - Code Review Report

**Date:** 2026-04-03
**Scope:** ~17K lines of Python across 40 files
**Method:** Automated multi-agent review (research + sonnet review agents)

---

## Table of Contents

1. [Silent Wrong-Result Bugs (Highest Priority)](#1-silent-wrong-result-bugs-highest-priority)
2. [Crash Bugs (NameError/TypeError/AttributeError)](#2-crash-bugs)
3. [Assert-False Blockers](#3-assert-false-blockers)
4. [Missing Return Statements](#4-missing-return-statements)
5. [Deprecated/Removed APIs](#5-deprecatedremoved-apis)
6. [Test Suite Issues](#6-test-suite-issues)
7. [Hardcoded Paths](#7-hardcoded-paths)
8. [Design Issues](#8-design-issues)
9. [API Inconsistencies](#9-api-inconsistencies)
10. [Packaging Issues](#10-packaging-issues)
11. [Nits](#11-nits)
12. [Per-Module Summary](#12-per-module-summary)

---

## 1. Silent Wrong-Result Bugs (Highest Priority)

These produce incorrect results without any error or warning.

### 1.1 Operator precedence corrupts strand detection for all BED data

**File:** `fragment_array/fragment_array.py:1670-1687`

```python
first_in_pair_mask = df.sam1 & 64 > 0
```

In Python, `>` has higher precedence than `&`. This is parsed as `df.sam1 & (64 > 0)` = `df.sam1 & True` = `df.sam1 & 1`. All six bitflag tests in `from_frag_bed()` check the wrong bit. Strand assignment is silently wrong for all reads from frag BED files.

**Fix:** Wrap bitwise operations: `(df.sam1 & 64) > 0`, `(first_in_pair_sam_flag & 16) == 0`, etc.

---

### 1.2 Wrong dispersion index in negative binomial loss

**File:** `bias_correction/model.py:83`

```python
n, p = self.loss_fn.to_natural_param(pred[i], pred[i + 2])
```

For NB loss with `output_tracks_multiplier=2`, the model outputs `2 * len(output_columns)` channels. The dispersion for column `i` should be at `pred[i + len(self.output_columns)]`, not `pred[i + 2]`. With the default 4-column output, this silently uses wrong dispersion channels for columns 0 and 1.

---

### 1.3 `__eq__` compares unsorted data

**File:** `fragment_array/fragment_array.py:491-492, 1600-1601`

```python
self.sort_in_place()   # returns new object, return value DISCARDED
other.sort_in_place()  # same
return numpy.all(self.starts_0 == other.starts_0) and ...
```

`sort_in_place()` (line 446) creates a new sorted object via `_replace()` — it does **not** mutate `self`. The return value is discarded, so `__eq__` compares the original unsorted arrays. Two FragmentArrays with identical fragments in different order compare as **not equal**.

---

### 1.4 `_standardize_name` returns None for chr-prefixed contigs

**File:** `contig.py:182-184`

```python
def _standardize_name(contig):
    if not contig.startswith("chr"):
        return "chr" + contig
    # implicit return None
```

No else branch. `contig_is_autosome("chr1")` -> `_standardize_name("chr1")` -> `None` -> `None in AUTOSOMES` -> `False`. **All `contig_is_*` functions return wrong results for standard chr-prefixed names.**

---

### 1.5 `intersect()` discards computed strand

**File:** `region.py:183`

```python
return type(self)(chrom=self.chrom, start=start, stop=stop, strand=self.strand)
```

The method carefully computes the correct `strand` variable (lines 168-176) handling cases where one region's strand is None, then ignores it and uses `self.strand`. The intersection result gets the wrong strand.

---

### 1.6 `strand_is_set` guards never fire

**File:** `region.py:107, 129`

```python
if not self.strand_is_set:   # method reference, always truthy
```

`strand_is_set` is a regular method, not a `@property`. Without `()`, this evaluates the bound method object, which is always truthy. The strand guards in `five_prime_shift` and `three_prime_shift` never fire.

---

### 1.7 `unique_regions()` sorts descending

**File:** `dataframe.py:869`

```python
ascending = [False, False, False]
```

All columns sorted descending before deduplication. `head(1)` picks the wrong representative when there are ties.

---

### 1.8 Cache key mismatch - cache never hits

**File:** `region.py:279`

```python
if bed_path not in CACHED_READERS:           # checks string key
    CACHED_READERS[(bed_path, self.ref)] = ...  # stores tuple key
```

The presence check uses `bed_path` (string) but entries are stored under `(bed_path, self.ref)` (tuple). Cache never hits; TabixBedReader re-opened on every call.

---

### 1.9 `merge_fragment_arrays` strand-flip logic checks arbitrary set element

**File:** `fragment_array/fragment_array.py:1983-1985`

```python
regions = list(set(getattr(ar, "region", None) for ar in ars))
if make_data_direction_match_strand and (
    len(regions) > 1 and regions[0] is not None
):
```

`regions` is built from a `set` (no defined ordering). `regions[0]` could be `None` even when non-None regions exist.

---

### 1.10 `from_bed12_parts` treats "." as zero CpGs

**File:** `fragment.py:245-252`

In BED format, `"."` means "missing data", not "zero CpGs". Setting `num_cpgs=0` corrupts downstream methylation calculations.

---

### 1.11 `mapq2 = mapq1` instead of `None`

**File:** `fragment.py:173`

When parsing a 2-part attribute string where the second part is a float (GC), the code copies `mapq1` into `mapq2` instead of leaving `mapq2` as `None`.

---

### 1.12 `init_from_fragment_array` silently drops metadata

**File:** `fragment_array/fragment_array.py:1245-1249`

Converting a weighted/strand-annotated `FragmentArray` to `RegionFragmentArray` discards weights, strands, and methylation fields without warning.

---

### 1.13 `parallel_apply` single-worker vs multi-worker paths diverge

**File:** `dataframe.py:302-328`

Same call with different worker counts can return structurally different DataFrames.

---

### 1.14 `iter_region_row()` drops strand and ref

**File:** `dataframe.py:1596`

While `iter_regions()` preserves them. `SampleAndRegionDataFrame.resize_regions()` uses `iter_region_row()` -- fragment array subsets may be wrong for stranded regions.

---

## 2. Crash Bugs

### 2.1 Undefined Symbols (NameError)

| File | Line | Symbol | Context |
|------|------|--------|---------|
| `fragment_array/fragment_array.py` | 1938 | `Fragment` | Not imported; `frag_len_midpoint_dense_array_to_start_stops()` crashes |
| `bias_correction/loss.py` | 98 | `sqrt` | Bare `sqrt()` without import |
| `bias_correction/model.py` | 86, 90 | `logits_to_probs` | Used but not imported in model.py |
| `bias_correction/layers.py` | 70 | `coo_matrix` | Not extracted from `scipy.sparse` |
| `bias_correction/predict.py` | 89 | `self` | Module-level function references `self` |
| `bias_correction/model.py` | 359 | `get_targets_from_region_fragment_array` | Method doesn't exist on `FragmentEndpointsDataset` |
| `public_data_resources/features.py` | 476 | `sdf` | Undefined variable in `load_chrom_hmm_states()` |
| `public_data_resources/gencode.py` | 121 | `tqdm`, `rdf` | Neither imported nor in scope |
| `util/dataclass.py` | 198 | `path_exists_s3_or_local` | Never imported |
| `fragment.py` | 131 | `funcsigs` | Python 2 backport, not available |
| `formats.py` | 105 | `boto3`, `split_bucket_key` | Never imported |
| `formats.py` | 128 | `signal` | Module never imported; crashes in error-reporting path |
| `formats.py` | 197 | `get_subclasses_of` | Should be `_get_subclasses_of` (missing underscore) |
| `formats.py` | 117 | `is_s3_uri` | Should be `_is_s3_uri` (missing underscore) |
| `formats.py` | 53 | `TextIOWrapper` | Not imported; should be `io.TextIOWrapper` |
| `contig.py` | 325 | `load_data_manifest`, `DEFAULT_DATA_MANIFEST_PATH` | Never imported |
| `motif.py` | 406, 411 | `plt`, `plot_weights_given_ax` | Not imported in motif.py |
| `motif.py` | 193 | `SequenceOneHotToWindowGcFraction` | Not imported or defined |
| `dataframe.py` | 1165 | `dataframe_region_mask` | Never imported |
| `dataframe.py` | 1211, 1213 | `FragmentsH5`, `RegionFragmentMatrix` | From commented-out imports |
| `fragment_matrix_math.py` | 141 | `warn` | Should be `warnings.warn` |
| `conftest.py` | 24 | `string` | Module not imported |
| `region.py` | 21 | `sequence` | Bare import of non-standard package |

### 2.2 TypeError / Wrong Arguments

| File | Line | Issue |
|------|------|-------|
| `fragment_array/fragment_array.py` | 1832 | `from_fname()` passes 6 kwargs that `from_fragments_h5()` doesn't accept |
| `analysis/strand_bias.py` | 361 | `calc_tss_cov_bias()` called with 3 args, signature requires 4 |
| `contig.py` | 209 | `contig_is_unplaced()` passes bool to `_standardize_name()` -> `TypeError` |
| `util/dataclass.py` | 269 | `dtypes=` kwarg should be `dtype=` in `pandas.read_table` |

### 2.3 Other Crashes

| File | Line | Issue |
|------|------|-------|
| `fragment_array/fragment_array.py` | 587 | `oversampled()` passes negative `size` to `numpy.random.choice` |
| `fragment_matrix.py` | 95 | `__radd__` infinite recursion with `sum()` -- missing `if other == 0: return self` |
| `bias_correction/data.py` | 152 | `.cuda()` without device detection -- crashes on CPU-only systems |
| `bias_correction/data.py` | 54 | `build_gene_coverage_counts` uses tuple indexing instead of list -- KeyError |
| `bias_correction/model.py` | 294 | All column names `"pred_dist."` (variable `c` never used) -- output unusable |
| `region.py` | 449 | Error message uses `CONTIG_LENGTHS[self.chrom]` (missing ref key) -- KeyError inside error handler |
| `dataframe.py` | 2054 | `expand_regions()` assigns to `self` (no-op) and returns `None` |
| `dataframe.py` | 1567 | `split_on_column()` queries literal string `"column_name"` instead of variable |
| `dataframe.py` | 372 | Empty `RegionDataFrame` init creates duplicate columns |
| `dataframe.py` | 1277 | `_get_fragment_coverage_sum()` accesses `self.id` unconditionally (optional column) |
| `dataframe.py` | 247 | `parallel_apply()` uses `multiprocessing.Lock()` instead of `ctx.Lock()` |
| `dataframe.py` | 2139 | `set_fragment_array_gc_weights()` references undefined `expansion`, `n_workers`, missing `deepcopy` import |
| `uniformity_filter.py` | 64 | `srdf.df` doesn't exist -- `SampleAndRegionDataFrame` IS a DataFrame |
| `uniformity_filter.py` | 30 | Accesses `.region` on `FragmentArray` but only `RegionFragmentArray` has it |
| `motif.py` | 366 | `tf_ids` references `bm.pwm_id` but `BindingModel` has `id` not `pwm_id` |
| `motif.py` | 27-31 | `Pfm` pseudocount breaks its own binary assertion |
| `chromhmm.py` | 10-11 | Imports from `fbio`/`ravel` -- not installed in standard environments |
| `util/dataclass.py` | 242 | `collections.Container` removed in Python 3.10 |

---

## 3. Assert-False Blockers

These are `assert False` statements at the top of otherwise-implemented methods, blocking the functionality:

| File | Line | Method | Impact |
|------|------|--------|--------|
| `fragment_array/fragment_array.py` | 353 | `pct_meth_cpgs` property | Methylation percentage inaccessible |
| `fragment_array/fragment_array.py` | 1052 | `filter_by_methyl()` | Methylation filtering blocked |
| `dataframe.py` | 2050 | `SampleAndRegionDataFrame._resize_region_boundaries()` | Fragment array resize blocked |
| `dataframe.py` | 1860 | `_set_fragment_array_weights_from_pred_record()` | Prediction-based weight setting blocked |
| `dataframe.py` | 588 | `ref_path` property | Reference path inaccessible |

---

## 4. Missing Return Statements

| File | Line | Function | Impact |
|------|------|----------|--------|
| `dataframe.py` | 2054 | `expand_regions()` | Returns `None` (assigns to local `self`) |
| `util/dataclass.py` | 269 | `load_dataframe()` | Returns `None` |
| `public_data_resources/gencode.py` | 116 | `load_gencode_gtf()` | Returns `None` |

---

## 5. Deprecated/Removed APIs

| File | Line | API | Status |
|------|------|-----|--------|
| `motif.py` | 230 | `np.float` | Removed in NumPy 1.24 |
| `fragment_matrix_math.py` | 384 | `numpy.product` | Removed in NumPy 2.0 |
| `util/dataclass.py` | 242 | `collections.Container` | Removed in Python 3.10 |

---

## 6. Test Suite Issues

### 6.1 False-Passing Tests

| File | Line | Issue |
|------|------|-------|
| `test_dataframe.py` | 169 | `numpy.isclose(...).all` accessed as attribute, not called -- always truthy |
| `test_dataframe.py` | 264 | Same issue -- test never actually asserts anything |

### 6.2 Permanently Broken Tests

| File | Line | Issue |
|------|------|-------|
| `test_dataframe.py` | 413-417 | `sdf` fixture has `assert False` -- 5 tests always fail |
| `test_dataframe.py` | 336 | `blacklist_path` undefined (should be `BLACK_LIST_FILE_HG38`) |
| `test_dataframe.py` | 474 | File path string passed to DataFrame constructor instead of `from_bed()` |
| `test_dataframe.py` | 508 | `REPO_DATA_DIR` undefined |
| `conftest.py` | 24 | `import string` missing |
| `test_formats.py` | 12 | Hardcoded `/home/nboley/` path |
| `test_fragment_array.py` | 228 | `res.n_fragments` -- property is `n_frags` |
| `test_fragment_matrix.py` | 125 | Calls `.fragment_matrix` on `FragmentMatrix` (wrong class) |

### 6.3 Test Coverage Gaps

No tests at all for:
- `public_data_resources/` (any module)
- `util/dataclass.py`
- `util/_logging.py`
- `util/liftover.py`
- `analysis/strand_bias.py`
- `analysis/deconvolution.py`
- `bias_correction/` (any module)
- `plot/` (any module)
- `motif.py`

---

## 7. Hardcoded Paths

| File | Line | Path Fragment |
|------|------|--------------|
| `dataframe.py` | 351 | `/scratch/karius/annotation/GRCh38...` |
| `dataframe.py` | 698 | `/home/nboley/src/Ravel/data/...` |
| `bias_correction/model.py` | 52, 73 | `/home/nboley/src/Ravel/...` |
| `bias_correction/train.py` | 32, 64, 87 | `/scratch/karius/`, `/home/nboley/` |
| `bias_correction/predict.py` | 22, 80, 153 | `/home/nboley/`, `/scratch/karius/` |
| `bias_correction/data.py` | 16, 118 | `/home/nboley/`, `/scratch/karius/` |
| `public_data_resources/encode.py` | 7 | `/scratch/ctcf_analysis/CTCF/...` |
| `public_data_resources/gencode.py` | 5, 13 | `/scratch/nboley/`, `/efs/analytics/` |
| `public_data_resources/features.py` | 464 | `/scratch/nboley/test_unet_pytorch/...` |
| `public_data_resources/public_chromhmm.py` | 6 | `/home/nboley/src/fragmentomics_tools/...` |
| `public_data_resources/jaspar.py` | 1731, 1758 | `/scratch/karius/...` |
| `bias_correction/train.py` | 26 | `sys.path.insert` to `/home/nboley/src/biomarker-projects/` |
| `test_formats.py` | 12 | `/home/nboley/src/fragmentomics_tools/test/data` |
| `formats.py` | 37 | `/ssd/fbio_cache` |

---

## 8. Design Issues

### 8.1 `sort_in_place()` is not in-place

**File:** `fragment_array/fragment_array.py:446-470`

The method returns a new object via `_replace()`. The name is actively harmful -- it caused the `__eq__` bug. Should be renamed to `sorted()` or rewritten to actually mutate.

### 8.2 Thread-unsafe global state

| File | Line | Global |
|------|------|--------|
| `region.py` | 276 | `CACHED_READERS` dict |
| `fragment_array/fragment_array.py` | 147 | `already_warned_diff_region_add` flag |
| `contig.py` | 320 | `_CHROMOSOME_Q_ARM_STARTS_HG38` cache |
| `public_data_resources/public_chromhmm.py` | 138 | Module-level dict mutations at import |

### 8.3 `@dataclass` decorator is vestigial on Fragment

**File:** `fragment.py:43-78`

Class manually defines `__init__`, `__eq__`, `__repr__`, and `__slots__`, overriding everything `@dataclass` provides. The decorator serves no purpose.

### 8.4 Module-level I/O at import time

**File:** `contig.py:144-147`

`CONTIGS` and `CONTIG_LENGTHS` are built by reading `.chrom.sizes` files at import time. Missing data files crash the entire package import.

### 8.5 `ChromOrdering` enum duplicated

Defined identically in both `contig.py:212` and `region.py:34`. They're separate objects -- `isinstance` checks across the boundary would fail.

### 8.6 `shift()` defined twice on Region

**File:** `region.py:80` and `region.py:769`

Second definition silently shadows the first. Different parameter names (`offset` vs `n`).

### 8.7 Region validation silently disabled

**File:** `region.py:522-524`

```python
if self.chrom not in CONTIG_LENGTHS[self.ref]:
    pass  # raise ValueError commented out
```

### 8.8 FASTA_PATH duplicated in 3 files

`bias_correction/data.py:16`, `model.py:52`, `predict.py:22` -- all hardcoded to the same developer path. Should be a single configurable constant.

### 8.9 `total_count = 1000` placeholder

**File:** `bias_correction/data.py:73`

Correct computation is commented out. Silently uses wrong normalization for Binomial/NB losses.

### 8.10 `configure_optimizers` learning rate is dead

**File:** `bias_correction/model.py:247`

Lightning calls `configure_optimizers()` with no arguments. The `learning_rate` parameter is always the default.

### 8.11 `SampleDataFrame.__init__` eagerly loads fragment lengths

**File:** `dataframe.py:2212`

Happens on every pandas operation that calls the constructor (slicing, filtering). Should be lazy.

### 8.12 No consistent interface across public_data_resources

Each module follows a different pattern:
- `chromhmm.py`: stateful class with `__getitem__`
- `encode.py`: bare function
- `expression.py`: RegionDataFrame subclass with `load()` classmethod
- `gencode.py`: bare functions
- `jaspar.py`: mix of dicts, generators, functions
- `public_chromhmm.py`: DataFrameBase subclass with `load_default_data()`

### 8.13 `_logging.py` design issues

- `sys.exit()` in worker thread (line 99) is a no-op in non-main threads
- `build_log_parser()` (line 267) parses `sys.argv` as a side effect
- `FileDescriptorLogger.write()` spawns a new thread per write call
- `configure_root_logger_from_args` passes wrong kwargs (`format`/`level`) which are silently ignored

### 8.14 `BindingModel` ndarray subclass loses metadata on slicing

**File:** `public_data_resources/jaspar.py:14-20`

`BindingModel(np.ndarray)` uses `__new__`/`__array_finalize__`. Attributes `id` and `name` are silently lost on slicing.

### 8.15 `chromhmm.py` mutates caller's DataFrame

**File:** `public_data_resources/chromhmm.py:169`

`self.region_dataframe["peak_index"] = ...` permanently modifies the passed-in DataFrame in `__init__`.

### 8.16 `BigBedWriter.close` writes to CWD

**File:** `formats.py:1462`

`smart_open.open("chrom.sizes", "w")` pollutes working directory; will fail in read-only environments.

### 8.17 Uninitialized strand array in `from_frag_bed`

**File:** `fragment_array/fragment_array.py:1681`

`np.empty((df.shape[0],), dtype="U1")` -- uninitialized memory for fragments that are neither plus nor minus strand.

### 8.18 `__add__` behavior inconsistent between base and subclass

`FragmentArray.__add__` (line 429) asserts `max_frag_len` equality. `RegionFragmentArray.__add__` (line 1533) takes `min()` instead.

### 8.19 `_replace()` reimplements dataclass without `@dataclass`

**File:** `fragment_array/fragment_array.py:374`

Manual `init_kwargs` tracking is fragile -- adding a new `__init__` parameter without updating `init_kwargs` silently breaks reconstruction.

---

## 9. API Inconsistencies

### 9.1 `mask()` and `subset()` are identical

**File:** `fragment_array/fragment_array.py:960-968`

Both call `_subset_or_mask()` with the same semantics. Confusing duplication.

### 9.2 `__and__` is concatenation, not intersection

**File:** `dataframe.py:383`

`RegionDataFrame.__and__` concatenates. The `&` operator in Python (and genomics) universally means AND/intersection.

### 9.3 `intersect` vs `intersects` naming

**File:** `region.py:144, 588`

One-letter difference: `intersect()` returns a Region, `intersects()` returns a bool.

### 9.4 `to_line()` is lossy

**File:** `fragment.py:134-147`

Serializes only `chrom, start, stop, mapq1, mapq2, gc, strand`. Fields `cell_barcode`, `num_cpgs`, `num_meth_cpgs` are silently dropped. Round-trip is lossy.

### 9.5 Duplicate implementations

| Item | Location 1 | Location 2 |
|------|-----------|-----------|
| `_resize_region_boundaries()` | `dataframe.py:1352` | `dataframe.py:2027` |
| `GeneExpression` / `TSSs` | `expression.py` | `features.py` |
| `classproperty` | `fragment.py` | `util/dataclass.py` |
| `ChromOrdering` | `contig.py:212` | `region.py:34` |

### 9.6 `to_tsv()` vs `save_as_bed()` inconsistency

`to_tsv()` writes with header. `save_as_bed()` writes without. Names don't make the distinction clear.

### 9.7 `midpoints_0` vs `midpoint_0`

**File:** `fragment_array/fragment_array.py:787, 791`

`midpoints_0` returns per-fragment midpoints (array). `midpoint_0` returns region center (scalar). Nearly identical names, completely different semantics.

### 9.8 `__all__` contains objects, not strings

**File:** `fragment_array/__init__.py:5`

Must be string names. Breaks `import *` and static analysis.

### 9.9 Public API surface gap

`__init__.py` exports only 7 symbols (storage/query layer). `analysis/`, `plot/`, `bias_correction/`, `public_data_resources/` are all invisible.

### 9.10 `get_pfms` fallback logic is inverted

**File:** `motif.py:110-113`

When `default_jaspar=False` (use HOCOMOCO), fallback also tries HOCOMOCO. When `default_jaspar=True` (use JASPAR), fallback tries HOCOMOCO. Backwards.

---

## 10. Packaging Issues

**File:** `setup.py`

| Issue | Impact |
|-------|--------|
| No `install_requires` -- all ~20 runtime deps undeclared | Can't pip install |
| No `find_packages()` call | Packages won't be found |
| `include_package_data=True` but no `package_data` or `MANIFEST.in` | Data files missing from installs |
| No `name`, `version`, `author`, `description` | `pip show` returns nothing |
| External deps `fragments_h5` and `datamanifest` not declared | Silent import failures |

---

## 11. Nits

| File | Line | Issue |
|------|------|-------|
| `fragment_matrix.py` | 278 | Raises `StopIteration` explicitly -- should be `ValueError` |
| `fragment_array/fragment_array.py` | 247 | Docstring typo: "num_comverted_cytosines" |
| `fragment_array/fragment_array.py` | 1445 | Debug `print()` left in `load()` |
| `fragment_array/fragment_array.py` | 430 | Missing `f` prefix on f-string in assert message |
| `fragment_array/fragment_array.py` | 1379 | Missing `f` prefix + typo + wrong variable name in error |
| `fragment_array/fragment_array.py` | 1737 | `!= None` instead of `is not None` |
| `fragment_array/fragment_array.py` | 44 | Duplicate `import numpy` |
| `fragment_array/fragment_array.py` | 825 | Dead `pass` after return |
| `fragment_matrix.py` | 7-22 | Heavy unused imports (matplotlib, seaborn, pysam, etc.) |
| `fragment_matrix_math.py` | 277 | Wrong type annotation (`numpy.array` should be `numpy.ndarray`) |
| `fragment_matrix_math.py` | 320 | Wrong axis name in error message |
| `plot/pairplots.py` | 1 | `import numpy as numpy` -- unconventional alias |
| `util/dataclass.py` | 7-8 | Both `import numpy` and `import numpy as np` |
| `util/_logging.py` | 74 | `assert False, "Unreachable"` should be `raise RuntimeError()` |
| `util/_logging.py` | 196, 255 | Parameter name inconsistency: `format` vs `log_format` |
| `util/liftover.py` | 38 | Silent `None` returns without logging |
| `constants.py` | 22 | `assert` for input validation -- stripped with `-O` |
| `__init__.py` | 35 | `AttributeError` doesn't suggest valid names |
| `motif.py` | 101 | `all_pfms = all_pfms` no-op |
| `loss.py` | 132 | Docstring says "Multinomial" on `NegativeBinomialNLLLoss` (copy-paste) |
| `loss.py` | 88-93 | `validate_args=False` with no comment |
| `bias_correction/data.py` | 27-30 | Regex matched twice unnecessarily |
| `bias_correction/data.py` | 96 | Variable shadowing (`x`) in lambda |
| `bias_correction/train.py` | 1, 14 | `import os` duplicated |
| `formats.py` | 512 | Bare `except:` catches `KeyboardInterrupt`, `SystemExit` |
| `formats.py` | 1052 | `StopIteration` raised outside generator |
| `formats.py` | 499 | `infer_bed_record_class_from_parts` rejects valid 7-9 column BED |
| `jaspar.py` | 184 | Typo: `"clsuter_124"` should be `"cluster_124"` |
| `jaspar.py` | 1669-1678 | Duplicate HTTP request (first response discarded) |
| `jaspar.py` | 1681 | `and False` disables fallback code |
| `encode.py` | 53 | `.head(20000)` undocumented arbitrary cutoff |
| `fragment_array/fragment_array.py` | 556 vs 570 | Inconsistent RNG approach (default_rng vs global state) |
| `analysis/strand_bias.py` | 46-49 | Dead code after return statement |
| `analysis/strand_bias.py` | 296 | `if False:` block -- disabled feature |
| `bias_correction/train.py` | 174-192 | Unreachable code after `return` |
| `predict.py` | 70 | `ppf(0.99)` call outside `hasattr` guard |

---

## 12. Per-Module Summary

| Module | Crash Bugs | Wrong Results | Design/API | Nits |
|--------|-----------|---------------|------------|------|
| `fragment_array/` | 10 | 5 | 8 | 8 |
| `dataframe.py` | 9 | 3 | 5 | 2 |
| `bias_correction/` | 8 | 2 | 6 | 4 |
| `region.py` | 3 | 4 | 4 | 1 |
| `contig.py` | 3 | 1 | 2 | 1 |
| `formats.py` | 5 | 0 | 3 | 3 |
| `motif.py` | 4 | 1 | 2 | 1 |
| `fragment.py` | 1 | 2 | 1 | 0 |
| `public_data_resources/` | 5 | 0 | 5 | 3 |
| `util/` | 3 | 0 | 3 | 2 |
| `analysis/` | 1 | 0 | 0 | 2 |
| `plot/` | 0 | 0 | 0 | 1 |
| `tests/` | 8 | 2 | 1 | 0 |
| `setup.py` | 0 | 0 | 5 | 0 |

---

## Recommended Fix Priority

1. **Operator precedence in `from_frag_bed`** -- silently corrupts strand for all BED data
2. **`_standardize_name` returns None** -- breaks all `contig_is_*` for standard chroms
3. **`sort_in_place` / `__eq__`** -- equality comparison is broken
4. **`__radd__` infinite recursion** -- `sum()` on FragmentMatrices crashes
5. **Sweep `assert False`** -- 5 instances blocking real functionality
6. **Fix undefined symbols** -- ~23 instances that crash specific code paths
7. **Fix missing return statements** -- 3 instances returning None
8. **Centralize hardcoded paths** -- 15+ instances making library non-portable
9. **Fix deprecated APIs** -- 3 instances that crash on modern Python/NumPy
10. **Fix test suite** -- false-passing tests and broken fixtures

---
---

# Implementation Plan

## Approach

Work is organized into **4 phases** ordered by dependency. Within each phase, streams run in **parallel** (each stream touches disjoint files). Each stream agent will:

1. **Fix bugs** listed for its files
2. **Add/fix tests** for its modules
3. **Document for owner review**: `assert False` stubs, broken functions needing discussion, recommended cleanups
4. **Clean up dead code**: remove commented-out imports, `if False:` blocks, unreachable code, unused imports
5. **Refactor as appropriate**: consolidate duplicates, fix naming, improve clarity

### Conventions

- **Hardcoded paths**: Replace with module-scoped config (a `_config` dict or module-level variables with env-var overrides at the top of each file)
- **`assert False` stubs**: Document in a `# TODO(nboley):` comment with context, replace `assert False` with `raise NotImplementedError("description")`
- **Broken functions**: Document with `# TODO(nboley):` and add a brief note to the DECISIONS section at the bottom of this plan for owner review
- **Dead imports from `fbio`/`ravel`**: Migrate to `fragmentomics_tools` equivalents or remove

---

## Phase 1: Foundation

No cross-dependencies between streams. All downstream modules import from these.

### Stream A: `contig.py` + `constants.py`

**Files:** `contig.py`, `constants.py`, `test/test_contig.py`

**Bug fixes:**
- `contig.py:182` -- `_standardize_name` must return `contig` (not None) when already chr-prefixed
- `contig.py:209` -- `contig_is_unplaced` passes bool to `_standardize_name`; fix to `_standardize_name(contig).startswith("chrU")`
- `contig.py:325` -- `get_chromosome_q_arm_starts` uses undefined `load_data_manifest`/`DEFAULT_DATA_MANIFEST_PATH`; document for review
- `constants.py:22` -- Replace `assert` with `if/raise ValueError`

**Dead code cleanup:**
- `contig.py:212` -- `ChromOrdering` enum duplicated with `region.py`; keep in `contig.py`, remove from `region.py` (done in Stream C)
- Remove any dead branches, clarify `_standardize_name` with explicit else

**Tests to add:**
- `contig_is_autosome`, `contig_is_assembled`, `contig_is_unplaced`, `contig_is_alternate` for both "chr1" and "1" forms
- `_standardize_name` edge cases
- `CONTIG_LENGTHS` and `CONTIGS` population for hg19 and hg38
- `get_flattened_genome_offsets`

---

### Stream B: `fragment.py`

**Files:** `fragment.py`, `test/test_fragment.py` (new)

**Bug fixes:**
- Line 131 -- Replace `funcsigs.signature` with `inspect.signature`
- Line 173 -- Fix `mapq2 = mapq1` to `mapq2 = None` in GC parsing branch
- Line 245-252 -- `from_bed12_parts`: change `num_cpgs=0` to `num_cpgs=None` when `parts[10] == "."`

**Dead code cleanup:**
- Lines 43-78 -- Remove `@dataclass` decorator (manually overrides everything); keep `__slots__` + manual `__init__`
- Line 108 -- Remove unused `none_to_inf` helper inside `mapq12_min`
- Clean up `locals()` loop in `__init__` -- replace with direct assignments

**Tests to add:**
- `Fragment` construction, `tlen`/`length`/`midpoint` properties
- `from_line` / `to_line` round-trip
- `from_line_parts` with various field counts (1-attr, 2-attr, 3-attr cases)
- `from_bed12_parts` with methylation data and "." missing data
- `mapq_gte` and `mapq12_min` edge cases (None mapq1, None mapq2, both None)
- `length_and_midpoint_to_start_and_stop` vectorized

---

### Stream C: `region.py`

**Files:** `region.py`, `test/test_region.py`

**Bug fixes:**
- Line 183 -- `intersect()`: change `strand=self.strand` to `strand=strand`
- Lines 107, 129 -- Add `()` to `self.strand_is_set` calls
- Line 279 -- Fix cache key check: `if (bed_path, self.ref) not in CACHED_READERS:`
- Line 449 -- Fix error message: `CONTIG_LENGTHS[self.ref][self.chrom]`
- Line 80 vs 769 -- Remove first `shift()` definition (dead code, shadowed by second)
- Line 21 -- Move bare `from sequence import ...` into a lazy import inside `get_one_hot_encoded_sequence`
- Line 691 -- Remove `__class__.__name__` string check from `__eq__`

**Dead code cleanup:**
- Lines 34-37 -- Remove `ChromOrdering` (import from `contig.py` instead, per Stream A)
- Line 522-524 -- Either restore or remove the commented-out chrom validation
- Line 39 -- Split `CACHED_READERS` into two separate caches with consistent key shapes
- Clean up commented-out import block

**Tests to add:**
- `intersect()` strand propagation (stranded x unstranded)
- `strand_is_set` as method call
- `five_prime_shift` / `three_prime_shift` on unstranded regions (should raise)
- `intersect_with_bed` cache behavior
- `shift()` with positive and negative offsets

---

### Stream D: `formats.py`

**Files:** `formats.py`, `test/test_formats.py`

**Bug fixes:**
- Line 53 -- Change `TextIOWrapper` to `io.TextIOWrapper`
- Line 105 -- Either properly import `boto3`/`split_bucket_key` or remove S3 functions; document for review
- Line 117 -- Fix `is_s3_uri` to `_is_s3_uri`
- Line 128 -- Add `import signal` or remove `VerboseCalledProcessError`; document for review
- Line 197 -- Fix recursive call: `get_subclasses_of` -> `_get_subclasses_of`
- Line 512 -- Change bare `except:` to `except Exception:`
- Line 1052 -- Change `StopIteration` to `ValueError`
- Line 1462 -- `BigBedWriter.close`: write `chrom.sizes` to tempdir, not CWD

**Dead code cleanup:**
- Line 37 -- Replace `DEFAULT_FBIO_CACHE_DIR = "/ssd/fbio_cache"` with env-var config
- Remove unused/broken S3 functions if migration isn't needed (document for review)
- Assess whether `BedReader.extensions` should include "tsv"

**Tests to fix:**
- Line 12 -- Replace hardcoded `/home/nboley/` with `Path(__file__).parent / "data"`

---

### Stream E: `util/` + `__init__.py`

**Files:** `util/dataclass.py`, `util/_logging.py`, `util/liftover.py`, `__init__.py`

**Bug fixes:**
- `dataclass.py:198` -- Remove or replace `path_exists_s3_or_local` call; document for review
- `dataclass.py:242` -- Change `collections.Container` to `collections.abc.Container`
- `dataclass.py:269` -- Fix `dtypes=` to `dtype=`; add `return` statement
- `dataclass.py:97` -- Change `StopIteration` to `ValueError` in `only_one()`
- `_logging.py:99` -- Replace `sys.exit()` with `return` in worker thread
- `_logging.py:74` -- Replace `assert False` with `raise RuntimeError`
- `_logging.py:192` -- Fix kwargs: `format` -> `file_format`/`stream_format`
- `__init__.py:35` -- Add valid names to AttributeError message

**Dead code cleanup:**
- `dataclass.py` -- Remove commented-out `fbio` imports
- `dataclass.py:7-8` -- Consolidate duplicate numpy imports
- `_logging.py` -- Review `FileDescriptorLogger.write()` spawning threads per call
- `_logging.py:267` -- `build_log_parser()` should not parse sys.argv as side effect

**Tests to add:**
- `dataclass.py`: `from_dict`, `to_dict`, `replace`, `from_json_s3_or_local` round-trip, `only_one`, `load_dataframe`
- `_logging.py`: `configure_root_logger`, `FileDescriptorLogger` basic operation
- `liftover.py`: `uniquely_convert_coordinate`, `uniquely_convert_region`

---

## Phase 2: Core

Depends on Phase 1 completion (imports from foundation modules).

### Stream F: `fragment_array/`

**Files:** `fragment_array/__init__.py`, `fragment_array.py`, `fragment_matrix.py`, `fragment_matrix_math.py`, `uniformity_filter.py`, `test/fragment_array/test_fragment_array.py`, `test/fragment_array/test_fragment_matrix.py`, `test/fragment_array/test_fragment_matrix_math.py`

**Bug fixes:**
- `__init__.py:5` -- Change `__all__` to list of strings
- `fragment_array.py:1670-1687` -- Fix operator precedence: wrap all `&` operations in parens before comparison
- `fragment_array.py:446-470` -- Rename `sort_in_place` to `sorted`; update `__eq__` to use return value
- `fragment_array.py:491` -- Fix `__eq__` to use sorted copies
- `fragment_array.py:587` -- Fix `oversampled`: `self.n_frags - n` -> `n - self.n_frags`
- `fragment_array.py:1832` -- Fix `from_fname` to match `from_fragments_h5` signature, or document for removal
- `fragment_array.py:1938` -- Add `from ..fragment import Fragment`
- `fragment_array.py:430` -- Add `f` prefix to assert message
- `fragment_array.py:1379` -- Add `f` prefix, fix `{sub_region{` typo, fix variable name
- `fragment_array.py:1681` -- Initialize strand array with `"."` instead of `np.empty`
- `fragment_array.py:1983` -- Fix set ordering: use `any(r is not None for r in regions)`
- `fragment_matrix.py:95` -- Fix `__radd__`: add `if other == 0: return self` guard
- `fragment_matrix.py:278` -- Change `StopIteration` to `ValueError`
- `fragment_matrix_math.py:141` -- Add `import warnings`; use `warnings.warn`
- `fragment_matrix_math.py:384` -- Change `numpy.product` to `numpy.prod`
- `uniformity_filter.py:64` -- Remove `.df` access (use `srdf` directly)
- `uniformity_filter.py:30` -- Change type hint to `RegionFragmentArray`

**`assert False` stubs to document:**
- `fragment_array.py:353` -- `pct_meth_cpgs`: document methylation percentage feature status
- `fragment_array.py:1052` -- `filter_by_methyl`: document methylation filtering feature status; also note `self.num_meth_cpgs` doesn't exist

**Dead code cleanup:**
- `fragment_array.py:1445` -- Remove debug `print()` in `load()`
- `fragment_array.py:825` -- Remove dead `pass`
- `fragment_array.py:44` -- Remove duplicate `import numpy`
- `fragment_array.py:980-990` -- Remove dead bytes branch in `subset_by_fragment_strand`
- `fragment_array.py:147` -- Replace global `already_warned_diff_region_add` with `warnings.warn`
- `fragment_array.py:1725` -- Replace string type-checking with `isinstance`
- `fragment_array.py:1737` -- `!= None` -> `is not None`
- `fragment_array.py:960-968` -- Document or consolidate `mask()`/`subset()` duplication
- `fragment_matrix.py:253-268` -- Remove `__class__.__name__` string matching fallback
- `fragment_matrix.py:7-22` -- Remove unused imports (matplotlib, seaborn, pysam, etc.)
- `fragment_matrix.py:282-285` -- Simplify unreachable `else: assert False`

**Tests to add/fix:**
- `__eq__` with fragments in different order (regression test for sort_in_place bug)
- `oversampled()` with n > n_frags
- `from_frag_bed` strand detection (regression test for operator precedence bug)
- `__radd__` with `sum([fm1, fm2])`
- `test_fragment_array.py:228` -- Fix `n_fragments` -> `n_frags`
- `test_fragment_matrix.py:125` -- Fix `.fragment_matrix` on wrong class
- Uncomment disabled parametrize cases where feasible

---

### Stream G: `motif.py`

**Files:** `motif.py`

**Bug fixes:**
- Line 406, 411 -- Add `import matplotlib.pyplot as plt` and `from .plot.viz_sequence import plot_weights_given_ax` (lazy)
- Line 230 -- Change `np.float` to `float`
- Line 366 -- Change `bm.pwm_id` to `bm.id`
- Line 27-31 -- Move assertion before pseudocount addition, or change assertion to validate input range
- Line 101 -- Remove no-op `all_pfms = all_pfms`
- Line 110-113 -- Fix inverted fallback logic in `get_pfms`
- Line 193 -- Document or remove `SequenceOneHotToWindowGcFraction` reference
- Line 216 -- Change `"strand" not in dir(record)` to `hasattr(record, "strand")`

**Dead code cleanup:**
- Review `Pfm` vs `BindingModel` naming confusion and document
- Clean up any remaining `ravel` references

---

### Stream H: `dataframe.py`

**Files:** `dataframe.py`, `test/test_dataframe.py`, `conftest.py`

**Bug fixes:**
- Line 2054 -- Fix `expand_regions`: add `return` statement (or `return super().expand_regions(...)`)
- Line 1567 -- Fix `split_on_column`: `f"{column_name} in @values"`
- Line 372 -- Fix duplicate columns in empty init
- Line 869 -- Fix `unique_regions` sort direction: `ascending=[True, True, True]` for coordinates
- Line 247 -- Fix `parallel_apply` Lock: use `ctx.Lock()` instead of `multiprocessing.Lock()`
- Line 1277 -- Guard `self.id` access with `hasattr` check

**`assert False` stubs to document:**
- Line 2050 -- `_resize_region_boundaries` in SampleAndRegionDataFrame
- Line 1860 -- `_set_fragment_array_weights_from_pred_record`
- Line 588 -- `ref_path` property

**Functions to document for review:**
- Line 2139 -- `set_fragment_array_gc_weights`: entirely broken (undefined vars, unused closure)
- Line 1211 -- `_get_fragment_coverage_track`: uses undefined `FragmentsH5`/`RegionFragmentMatrix`
- Line 1165 -- `region_mask`: uses undefined `dataframe_region_mask`
- Line 383 -- `__and__` as concatenation: document for potential rename/removal

**Dead code cleanup:**
- Lines 47-55 -- Remove commented-out `fbio`/`ravel` imports
- Line 184 -- Remove deprecated `from_fname_s3_or_local` or mark clearly
- Lines 783-785 -- Consolidate TF imports (use `fragmentomics_tools.motif` not `ravel`)
- Line 1498 -- Replace `assert False, "UNREACHABLE"` with proper unreachable marker
- Remove duplicate `import logging`
- Normalize `numpy`/`np` usage

**Hardcoded paths to replace with config:**
- Line 351 -- `/scratch/karius/annotation/GRCh38...` in `get_fasta_path()`; delegate to `contig.get_reference_path()`
- Line 698 -- `/home/nboley/src/Ravel/data/...`

**Tests to fix:**
- `conftest.py:24` -- Add `import string`
- `test_dataframe.py:169, 264` -- Change `.all` to `.all()` (call the method)
- `test_dataframe.py:413` -- Document broken `sdf` fixture for review (needs real sample data paths)
- `test_dataframe.py:336` -- Fix `blacklist_path` -> `BLACK_LIST_FILE_HG38`
- `test_dataframe.py:474` -- Use `RegionDataFrame.from_bed()` instead of constructor
- `test_dataframe.py:508` -- Fix or remove `REPO_DATA_DIR` reference

**Tests to add:**
- `expand_regions` return value
- `split_on_column` with various column names
- `unique_regions` ordering
- `parallel_apply` basic operation

---

## Phase 3: Applications

Depends on Phase 2 completion.

### Stream I: `bias_correction/`

**Files:** `bias_correction/data.py`, `layers.py`, `loss.py`, `model.py`, `predict.py`, `train.py`, `__init__.py`

**Bug fixes:**
- `loss.py:98` -- Fix `sqrt`: use `d ** 0.5` or `import math; math.sqrt(d)`
- `loss.py:132` -- Fix docstring: "Multinomial" -> "Negative Binomial"
- `model.py:86,90` -- Add `from torch.distributions.utils import logits_to_probs`
- `model.py:83` -- Fix dispersion index: `pred[i + len(self.output_columns)]` not `pred[i + 2]`
- `model.py:294` -- Fix column names: `["pred_dist." + c for c in self.output_columns]`
- `model.py:359` -- Fix method name or document for removal
- `predict.py:89` -- Remove `self` reference in module-level function
- `predict.py:70` -- Move `ppf(0.99)` inside `hasattr` guard
- `layers.py:70` -- Add `from scipy.sparse import coo_matrix`
- `data.py:54` -- Fix tuple indexing to list: `res[["col1", "col2"]]`
- `data.py:152` -- Replace `.cuda()` with device-aware code
- `data.py:27-30` -- Remove duplicate regex match

**Functions to document for review:**
- `data.py:73` -- `total_count = 1000` placeholder; correct computation commented out
- `model.py:359` -- `predict_from_rdf_and_sdf` calls nonexistent method
- `model.py:247` -- `configure_optimizers` dead learning_rate parameter
- `train.py:173-192` -- Unreachable code after `return`

**Hardcoded paths to replace with config:**
- `data.py:16` -- FASTA_PATH
- `data.py:118` -- blacklist path
- `model.py:52,73` -- FASTA_PATH (deduplicate with data.py)
- `predict.py:22,80,153` -- FASTA_PATH, annotation paths
- `train.py:32,64,87,95` -- All annotation paths
- `train.py:26` -- Remove `sys.path.insert` hack; document external dep for review

**Dead code cleanup:**
- `loss.py:174` -- Document `NegativeBinomialNLLLossOld` status (deprecated? active?)
- `train.py:174-192` -- Remove unreachable code
- `train.py:1,14` -- Remove duplicate `import os`
- `model.py:224` -- Remove dead `self.num_workers = 64`
- `model.py:283-295` -- Remove commented-out MultiIndex approach
- `data.py:96` -- Fix variable shadowing in lambda

---

### Stream J: `analysis/` + `plot/`

**Files:** `analysis/strand_bias.py`, `analysis/deconvolution.py`, `plot/tracks.py`, `plot/strand_bias.py`, `plot/viz_sequence.py`, `plot/pairplots.py`, `plot/__init__.py`

**Bug fixes:**
- `strand_bias.py:361` -- Fix argument count in `calc_tss_cov_bias()` call
- `pairplots.py:1` -- Change `import numpy as numpy` to `import numpy as np`

**Dead code cleanup:**
- `strand_bias.py:46-49` -- Remove dead code after return
- `strand_bias.py:296` -- Remove `if False:` block or document
- `strand_bias.py:256` -- Add None guard for `label_col`
- `tracks.py` -- Check for duplicate `smooth1d` definitions; remove one

**Tests to add:**
- `deconvolution.py`: test `fit_celltype_weights_l2` and `rank_correlation_y_vs_X` with synthetic data
- `strand_bias.py`: basic smoke tests for `calc_tss_cov_bias`, `calc_gene_cov_bias`

---

### Stream K: `public_data_resources/`

**Files:** `public_data_resources/chromhmm.py`, `encode.py`, `expression.py`, `features.py`, `gencode.py`, `jaspar.py`, `public_chromhmm.py`, `__init__.py`

**Bug fixes:**
- `features.py:476` -- Document broken `load_chrom_hmm_states()` for review
- `gencode.py:116` -- Add `return rdf` to `load_gencode_gtf()`
- `gencode.py:121` -- Fix `tqdm` import and `rdf` scoping in `build_transcript_bed_from_gencode_gtf()`
- `public_chromhmm.py:6` -- Replace hardcoded BASE_DIR with `os.path.dirname(__file__)`
- `jaspar.py:184` -- Fix typo: `"clsuter_124"` -> `"cluster_124"`
- `jaspar.py:1681` -- Remove `and False` to re-enable fallback
- `jaspar.py:1669-1678` -- Remove duplicate HTTP request

**Migration from fbio/ravel:**
- `chromhmm.py:10-11` -- Replace `from fbio.dataframe import RegionDataFrame` with `from ..dataframe import RegionDataFrame`; replace `from ravel.learn.transforms import indicator_vect_to_one_hot_mat` with local implementation or document for review

**Functions to document for review:**
- `features.py:476` -- `load_chrom_hmm_states()`: broken script fragment
- `features.py` vs `expression.py` -- Duplicate `GeneExpression`/`TSSs` classes: which to keep?
- `expression.py:169` -- `HematopoieticGeneExpression.load()` returns mis-structured RDF
- `gencode.py:9` -- `load_gencode_genes_rdf` asserts `ref == "hg38"` despite accepting ref param

**Hardcoded paths to replace with config:**
- `encode.py:7` -- `/scratch/ctcf_analysis/CTCF/...`
- `gencode.py:5,13` -- `/scratch/nboley/`, `/efs/analytics/`
- `jaspar.py:1731,1758` -- `/scratch/karius/...`

**Dead code cleanup:**
- `chromhmm.py:169` -- Stop mutating caller's DataFrame in `__init__`
- `public_chromhmm.py:138-156` -- Module-level dict mutations; move to function
- `features.py:285-291` -- Module-level mutations to `CELL_ANATOMY_TO_CELL_EID`

---

## Phase 4: Infrastructure

Can run after any phase (touches only infrastructure files).

### Stream L: `setup.py` + test infrastructure

**Files:** `setup.py`, `conftest.py`

**setup.py fixes:**
- Add `name`, `version`, `author`, `description`
- Add `find_packages()`
- Add `install_requires` with all runtime deps
- Add `package_data` for bundled data files
- Document `fragments_h5` and `datamanifest` as external deps

**conftest.py fixes:**
- Line 24 -- Add `import string`
- Line 73 -- Make `CUDA_VISIBLE_DEVICES` configurable
- Lines 79-84 -- Remove commented-out S3 fixture

---

## Decisions Needed (Owner Review)

Items flagged by agents for discussion before implementation:

| # | File | Item | Question |
|---|------|------|----------|
| 1 | `fragment_array.py:353,1052` | Methylation features (`pct_meth_cpgs`, `filter_by_methyl`) | Implement, stub with NotImplementedError, or remove entirely? Also: `num_meth_cpgs` field doesn't exist -- is the data model incomplete? |
| 2 | `dataframe.py:2050` | `SampleAndRegionDataFrame._resize_region_boundaries` | Was working (updates fragment_arrays) then asserts. Remove assert to enable? Or is there a known issue? |
| 3 | `dataframe.py:1860` | `_set_fragment_array_weights_from_pred_record` | Dead, but `_from_weights_record` version exists. Remove pred_record version? |
| 4 | `dataframe.py:588` | `ref_path` property (needs DataManifest) | Remove property? Or implement differently? |
| 5 | `dataframe.py:2139` | `set_fragment_array_gc_weights` | Entirely broken. Remove or rewrite? |
| 6 | `dataframe.py:1211` | `_get_fragment_coverage_track` | Uses undefined `FragmentsH5`/`RegionFragmentMatrix`. Remove or rewrite? |
| 7 | `dataframe.py:1165` | `region_mask` | Uses undefined `dataframe_region_mask`. Remove? |
| 8 | `dataframe.py:383` | `__and__` = concatenation | Rename to avoid confusion with intersection semantics? |
| 9 | `fragment_array.py:1832` | `from_fname()` | Completely broken wrapper. Remove or update signature? |
| 10 | `formats.py:105` | S3 functions (`path_exists_s3`, etc.) | Need `boto3`/`split_bucket_key`. Remove S3 support or add deps? |
| 11 | `formats.py:128` | `VerboseCalledProcessError` | Needs `signal` module. Keep and fix, or remove? |
| 12 | `bias_correction/data.py:73` | `total_count = 1000` | Correct computation is commented out. Restore it? |
| 13 | `bias_correction/train.py:26` | `sys.path.insert` to biomarker-projects | Remove and make external dep explicit? Or keep as research script? |
| 14 | `features.py:476` | `load_chrom_hmm_states()` | Broken script fragment. Remove? |
| 15 | `features.py` vs `expression.py` | Duplicate `GeneExpression`/`TSSs` | Which module is canonical? Remove the other? |
| 16 | `chromhmm.py:10` | `fbio`/`ravel` migration | `indicator_vect_to_one_hot_mat` -- implement locally or find alternative? |
| 17 | `fragment_array.py:960` | `mask()` vs `subset()` | Identical methods. Keep both or consolidate? |
| 18 | `contig.py:325` | `get_chromosome_q_arm_starts` | Needs `load_data_manifest`. Remove or reimplement? |

---

## Estimated Stream Sizes

| Phase | Stream | Files | Bug Fixes | Tests | Cleanup |
|-------|--------|-------|-----------|-------|---------|
| 1 | A: contig + constants | 3 | 4 | ~10 | 2 |
| 1 | B: fragment | 2 | 3 | ~12 | 3 |
| 1 | C: region | 2 | 7 | ~8 | 4 |
| 1 | D: formats | 2 | 8 | 1 fix | 3 |
| 1 | E: util + __init__ | 4 | 8 | ~10 | 5 |
| 2 | F: fragment_array | 8 | 17 | ~10 | 12 |
| 2 | G: motif | 1 | 8 | 0 | 2 |
| 2 | H: dataframe | 3 | 6 | ~8 | 8 |
| 3 | I: bias_correction | 7 | 12 | 0 | 6 |
| 3 | J: analysis + plot | 7 | 2 | ~4 | 4 |
| 3 | K: public_data | 7 | 7 | 0 | 6 |
| 4 | L: setup + conftest | 2 | 5 | 0 | 1 |
