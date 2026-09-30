# Cut-Site Hexamer Counts

`scripts/count_cut_site_hexamers.py` counts de-duplicated cut-site hexamer
frequencies from real fragment h5 files. It is a counting tool, not analysis:
it emits observed counts and matching background (all-candidate) counts, and
the consumer decides what to divide by what. Its output is the real-data
replacement for the simulator's synthetic `build_w6` hexamer tables (decision
78).

## Output

**Location:** `/efs/analytics/nathanboley/background_model/cut_site_hexamers/`

One `.parquet` per sample. (`.npz` files also exist at this path but are not
produced by the committed script.)

### Parquet schema

| Column | Type | Description |
|---|---|---|
| `hexamer` | `string` | The 6-mer string (see orientation below) |
| `table` | `string` | One of `start_fwd`, `end_fwd`, `start_rev`, `end_rev` |
| `band_lo` | `int32` | Fragment-length band lower edge (inclusive) |
| `band_hi` | `int32` | Fragment-length band upper edge (exclusive) |
| `observed` | `int64` | Count from de-duplicated, filtered fragments |
| `background` | `int64` | Count from all candidate fragments in the generative domain |

**Row count:** 262,144 per file (4 tables x 16 bands x 4,096 hexamers).

Provenance metadata is embedded in the Parquet file-level metadata under the
key `count_cut_site_hexamers_meta`. `read_output(path)` returns
`(dataframe, meta)`.

### Hexamer orientation

The `hexamer` column is written 5'->3' along the strand of the fragment:

- `start_fwd` / `end_fwd` -- reference-forward 6-mer at the cut site.
- `start_rev` / `end_rev` -- **reverse complement** of the reference-forward
  6-mer. Do not RC these again when joining.

## Per-Sample Results

These 5 samples were drawn from a 36-row sheet; see *Sample selection* below,
which pins the exact sheet and recipe. Region set:
`quiet_v2_pad1200_repeats_removed_tile2560.bed` (11,505 tiles).

All values below are from the Parquet file-level metadata, verified against
the code's filter pipeline.

| Sample | Fetched | MAPQ removed | De-dup'd | Dup rate | Counted | Elapsed |
|---|---|---|---|---|---|---|
| RD-56138-Lib1 | 5,991,138 | 21.07% | 1,010,217 | 78.64% | 857,675 | 195 s |
| RD-56436-Lib1 | 5,256,294 | 14.39% | 764,320 | 83.02% | 687,063 | 185 s |
| RD-56801-Lib1 | 1,791,578 | 7.42% | 586,316 | 64.65% | 529,810 | 328 s |
| RD-56910-Lib1 | 4,641,928 | 17.30% | 979,412 | 74.49% | 853,407 | 191 s |
| RD-57082-Lib1 | 2,039,921 | 7.81% | 508,574 | 72.96% | 452,365 | 335 s |

**Column definitions:**

- **Fetched** -- fragments returned by `h5.fetch_array` across all regions
  (a fragment spanning two tiles is fetched in both, so this double-counts
  at tile boundaries).
- **MAPQ removed** -- `(fetched - mapq_pass) / fetched`.
- **De-dup'd** -- fragments surviving MAPQ filter + `(contig, start, stop)`
  dedup.
- **Dup rate** -- `dup_removed / mapq_pass`.
- **Counted** -- fragments that additionally pass containment, length range,
  and both-cut-sites-valid checks. Each contributes one start and one end
  count.
- **Elapsed** -- `count_sample()` wall time (the Parquet metadata field
  `stats.elapsed_s`).

## Method

### Filter pipeline (in order)

1. **MAPQ:** `min(mapq_read1, mapq_read2) >= 10` (inclusive `>=`, matching
   `PlumbingConfig.min_mapq`). A file where MAPQ was never carried through
   stores unknown MAPQ as `-1`; the script detects all-removed and aborts.
2. **De-duplication:** collapse fragments sharing `(contig, start, stop)` to
   one molecule. Strand is **not** part of the key (matches
   `FragmentArray.drop_duplicate_fragments()`). The strand-inclusive duplicate
   rate is recorded as a diagnostic.
3. **Containment:** `gstart <= start` and `stop <= gstop` -- fully contained.
   A fragment straddling a tile boundary falls out of both tiles.
4. **Length range:** `[25, 180]` inclusive (from `simulator.weights.L_MIN`,
   `L_MAX`).
5. **Valid cut sites:** both the 5' and 3' hexamer windows must be fully ACGT.

### Cut-site geometry

A fragment at `[start, stop)` has cut sites at positions `start` and `stop`
(not `stop - 1`). `stop` in the h5 is already exclusive, so no +/-1 adjustment.

- Plus strand: `c5 = start`, `c3 = stop`; both index `hex_fwd`.
- Minus strand: `c5 = stop`, `c3 = start`; both index `hex_rc`.

The hexamer at cut site `c` spans `seq[c-3 : c+3]` (3 bases inside the
fragment, 3 outside).

### Background tables

`background` holds the same four hexamer tables computed over every candidate
fragment in the generative domain -- all `(position, length, strand)` where
the fragment is fully contained and both cut sites are valid. Computed in
closed form via cumulative sums, not by enumerating candidates.

### Band stratification

16 half-open bands of width 10 bp over `[25, 181)`. The final band is
`[175, 181)` (6 lengths, not 10). Band edges are recorded in each row, so a
consumer joins on `(band_lo, band_hi)` rather than assuming uniform widths.

## Design Rationale

### String-keyed hexamers (decision 81)

The Parquet key is the literal 6-mer string, not a bare 4,096-element array
ordered by an implicit integer code. A mismatched k-mer ordering between
producer and consumer makes every downstream weight wrong, but `sum(w) = 1`
still holds -- normalisation cannot detect a relabelling. With string keys a
convention mismatch fails to *join* instead of silently misaligning.

### Single vocabulary source

`hexamer_vocabulary()` is defined in exactly one place
(`background_model/simulator/precompute.py`) and imported by both the
simulator's `emit.py` and this script. A second ordering would silently
invalidate every comparison. Verified by grep: no other file defines this
function.

## Sample selection — reproducible, but only against a pinned sheet

The five samples were drawn with stdlib `random`, not `np.random.default_rng`,
because NumPy guarantees stream stability only for the legacy `RandomState`.
The counting script itself contains no RNG and is entirely deterministic; the
seed applies to the *selection*, not the counting.

Verified reproduction — all three elements are required:

```python
random.Random(20260929).sample(sorted(names), 5)
# -> RD-56138-Lib1, RD-56436-Lib1, RD-56801-Lib1, RD-56910-Lib1, RD-57082-Lib1
```

where `names` is the `sample_name` column of the **36-row** sheet preserved at
`/efs/analytics/nathanboley/background_model/cut_site_hexamers/sample_sheet_36row_for_seeded_draw.tsv`.

**Two ways this silently returns the wrong five:**

- **Omit `sorted()`** and the same seed over the same sheet yields a different
  five (only `RD-56910-Lib1` overlaps). Nothing errors.
- **Use the current sheet.** `data/sample_sheets/ibd_quiescent.resolved.tsv`
  now holds **213** rows after a resolver change; the same seed over it draws a
  different five. The seed is reproducible only against a stated sheet version.

The five names above are the authoritative record — prefer them over re-running
the draw.
