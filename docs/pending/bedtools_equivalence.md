# bedtools equivalence

How each common `bedtools` command is expressed in the `intervals` API, verified
by running both sides on real data.  This is a reference for anyone porting a
bedtools pipeline, and the specification that `test/test_bedtools_equivalence.py`
tests against.

Verified against **bedtools 2.31.1** and **bioframe 0.8.0** on
964,593 CTCF sites (`ctcf.hg38.bed6.bed`) and 636 hg38 blacklist regions.

## Distance parameter conventions

bedtools is deliberately **not** self-consistent across its overlap-testing and
merge operations.  The API follows both conventions, using two parameters:

| bedtools flag | API parameter | Convention | Book-ended (gap=0) |
|---|---|---|---|
| `window -w P` | `pad=P` | gap **< P** (strict) | `pad=0`: no match; `pad=1`: match |
| `merge -d D` | `min_dist=D` | gap **<= D** | `min_dist=0`: merged; `min_dist=None`: separate |

Both are correct for their respective questions:

- **"Does anything overlap?"** (`intersect`, `window`): book-ended intervals
  share zero bases, so they do not overlap.  `window -w 0` and `intersect -u`
  agree: no hit.
- **"Should these be joined?"** (`merge`, `cluster`): book-ended intervals have
  a gap of 0, and `merge -d 0` says "join when gap <= 0", so they are joined.

A single parameter cannot express both without an off-by-one on one side.  The
old `wiggle` parameter was a third convention matching neither tool.

## Command mapping

All row counts and row contents below were verified by execution.

### `bedtools intersect -u` — boolean overlap mask

```bash
bedtools intersect -a A.bed -b B.bed -u
```

```python
mask = intervals.overlaps(a, b)           # Series[bool], aligned to a
a_overlapping = a[mask.values]
```

Verified: **8,804** CTCF sites overlap a blacklist region (both sides identical).

### `bedtools intersect -wa -wb` — inner join with both sides

```bash
bedtools intersect -a A.bed -b B.bed -wa -wb
```

```python
idx = intervals.overlap_indices(a, b)     # DataFrame[a_pos, b_pos, overlap_bases]
# To reconstruct both sides:
a_hits = a.iloc[idx.a_pos.values]
b_hits = b.iloc[idx.b_pos.values]
```

Verified: **8,804** pairs, row sets **identical** (set comparison, not just count).

### `bedtools intersect -v` — anti-join (A rows with no overlap in B)

```bash
bedtools intersect -a A.bed -b B.bed -v
```

```python
anti = intervals.overlap_indices(a, b, how="anti")
a_no_overlap = a.iloc[anti.a_pos.values]
```

Verified: **955,789** non-overlapping rows (955,789 + 8,804 = 964,593 total).

### `bedtools intersect -f` — minimum overlap fraction

```bash
bedtools intersect -a A.bed -b B.bed -wa -wb -f 0.5
```

```python
idx = intervals.overlap_indices(a, b, min_frac_a=0.5)
```

Verified: **8,791** pairs at `-f 0.5` (both sides identical).

### `bedtools intersect -f -r` — reciprocal overlap fraction

```bash
bedtools intersect -a A.bed -b B.bed -wa -wb -f 0.5 -r
```

```python
idx = intervals.overlap_indices(a, b, min_frac_a=0.5, reciprocal=True)
```

`reciprocal=True` applies `min_frac_a` to **both** A and B (the bedtools `-r`
convention).  Verified: **0** pairs at `-f 0.5 -r` (CTCF sites are small
relative to blacklist regions, so the 50% threshold is never met on the B side).

### `bedtools window -w` — overlap with gap tolerance

```bash
bedtools window -a A.bed -b B.bed -w 10
```

```python
idx = intervals.overlap_indices(a, b, pad=10)
```

`pad=P` matches when gap < P, same as `window -w P`.

Verified: **8,808** pairs at `-w 10` / `pad=10` (both sides identical).
Also verified boundary: `-w 0` = 8,804 (same as strict intersect), `-w 1` =
8,804 + book-ended pairs.

### `bedtools merge -d` — merge overlapping/nearby intervals

```bash
bedtools merge -i A.bed -d 0          # default: merge book-ended
bedtools merge -i A.bed -d 10         # merge gaps <= 10bp
```

```python
merged = intervals.merge(a, min_dist=0)   # default: merge book-ended
merged = intervals.merge(a, min_dist=10)
```

`min_dist=D` merges when gap <= D, same as `merge -d D`.
`min_dist=None` is strictly-overlapping-only (book-ended stay separate).

Verified on CTCF: **950,936** at `-d 0` / `min_dist=0`, **944,178** at `-d 10`
/ `min_dist=10`.  Row sets identical (set comparison on (contig, start, stop)).

Note: bedtools `merge` requires sorted input; our `merge` does not.

### `bedtools closest -d` — nearest neighbour with distance

```bash
bedtools closest -a A.bed -b B.bed -d
```

```python
near = intervals.nearest(a, b)           # DataFrame[a_pos, b_pos, distance]
```

Distance follows the ``bedtools closest -d`` convention exactly:

| Geometry | bedtools `-d` | our `distance` |
|---|---|---|
| overlapping | 0 | 0 |
| book-ended (gap=0) | **1** | **1** |
| gap=1 | 2 | 2 |
| gap=G | G+1 | G+1 |

For overlapping pairs, distance is 0.  For non-overlapping pairs (including
book-ended), distance is gap + 1.  This makes ``distance == 0`` unambiguously
mean "overlapping" — the previous convention reported 0 for both overlapping
and book-ended pairs, which made those two relationships indistinguishable.

This deliberately diverges from ``bioframe.closest``, which returns the raw gap
(0 for both overlapping and book-ended).  Do not "simplify" the distance back
to the raw gap — that reintroduces the ambiguity this change exists to remove.

Note: bedtools `closest` requires sorted input; our `nearest` does not.

Verified column-for-column against bedtools 2.31.1 on synthetic fixtures
covering overlapping, book-ended, and gap=1/10/100 cases.

### `bedtools cluster` — connected component labels

```bash
bedtools cluster -i A.bed
```

```python
labels = intervals.cluster(a)            # Series[int], aligned to a
```

Labels are 0-based (bedtools uses 1-based), but the grouping is identical:
intervals that share a cluster label in bedtools share one in ours.

Verified: **950,936** clusters on CTCF (both sides).

Our API also supports two-frame clustering (`cluster(a, b)`), which bedtools
does not — it computes transitive components across both frames.

### `bedtools subtract -A` — remove entire A intervals that overlap B

```bash
bedtools subtract -a A.bed -b B.bed -A
```

```python
anti = intervals.overlap_indices(a, b, how="anti")
a_surviving = a.iloc[anti.a_pos.values]
```

`subtract -A` removes entire A features that have any overlap with B, keeping
those with no overlap — identical to the anti-join.

Verified: **955,789** surviving rows (both sides identical).

Note: regular `subtract` (without `-A`) performs geometric subtraction — it
clips A intervals where they overlap B.  That is **not covered** by this API
(see "Out of scope" below).

## Out of scope

These bedtools capabilities are deliberately not covered.  Each was considered
and declined in the design (see `interval_api_design.md`).

| Command | Why not covered |
|---|---|
| `map` | Aggregating B's columns onto A is `groupby` at the call site, not API surface |
| `slop` / `flank` | Genome-end clamping; not needed by the library |
| `complement` | Set complement of intervals; not needed |
| `shuffle` | Random interval placement; belongs in a null-model module |
| `fisher` / `jaccard` | Enrichment statistics; a separate module if ever needed |
| `genomecov` | Per-base coverage vectors; not an interval-algebra concern |
| `multiinter` | Multi-file intersection; composable at the call site |
| `annotate` | Multi-file annotation; composable at the call site |
| `subtract` (without `-A`) | Geometric subtraction (clipping coordinates); nothing in the library needs clipped coordinates — overlap *length* is enough.  `scripts/build_inactive_regions.py` does need it and stays on `pybedtools` for that reason |
| `reldist` | Relative distance; belongs in enrichment/null-model module |
