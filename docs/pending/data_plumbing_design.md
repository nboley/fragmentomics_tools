# Data Plumbing Design — Background Model v2

Status: DRAFT (EM-authored 2026-08-26; agent spawn path was down — an
adversarial design-review agent MUST review this before implementation).
Requirements source: `BACKGROUND_MODEL_BRIEF.md` §"Data plumbing"; statistical
spec: `background_model_core.py` docstring.

Pipeline shape: `sample sheet -> [preprocess] -> zarr store -> [Dataset] ->
BackgroundModel batches (x, y, mask)`.

## 0. Fixed geometry (from brief; derived constants)

| Name | Value | Note |
|---|---|---|
| `TILE` | 16,384 | model output length L_out |
| `JITTER` | 128 | max |shift| at load |
| `RF_BUDGET` | 2,048/side | receptive-field margin, deliberately ~16.5× the default model's need of (16632−16384)/2 = 124/side — future-proofs deeper architectures for ~25% extra stored sequence |
| `L_TARGET` | 16,640 | TILE + 2·JITTER — stored counts/mask extent |
| `L_SEQ` | 20,736 | TILE + 2·(JITTER + RF_BUDGET) — stored sequence extent |
| `C` | 12 | tracks, canonical order = `DEFAULT_OUTPUT_TRACKS` |

All three extents share the tile's center; even lengths throughout (the
`jitter_matrix` same-parity requirement). Dataset asserts BOTH at init:
`L_SEQ >= model.calc_input_region_size(TILE) + 2*JITTER`, and
`TILE % 2 == L_TARGET % 2 == L_SEQ % 2 == model_input_size % 2 == 0`
(a future config change must fail loudly, not crop asymmetrically).
fl_band tuples `(lo, hi)` are HALF-OPEN `[lo, hi)`: band (40,65) = lengths
40–64, band (120,175) = 120–174; `max_frag_len=175` (exclusive) matches
`subset_fragment_lengths` `<` semantics (fragment_array.py:790).

## 1. Zarr store layout

One store per config hash: `bg_store_<hash8>.zarr/` (zarr v2 format,
zarr-python; pin the installed version in store attrs — pre-implementation
check: version present in `biomarker_env`).

```
/attrs: config_json, config_hash, created_utc, zarr_format_note,
        code_version (git sha), fragmentomics_tools_version
/tiles/                        # T tiles, build order = region order x tiling
  contig      (T,)  str        # zarr fixed-length UTF
  start,stop  (T,)  int64      # CENTER tile coords (not margined)
  strand      (T,)  U1         # region strand ('.' allowed)
  region_id   (T,)  str        # source region; tiles inherit its split
  split       (T,)  uint8      # 0 train / 1 val / 2 heldout_inactive / 3 positive_control
  seq         (T, L_SEQ)   uint8   # ASCII ACGTN, uppercased; chunks (64, L_SEQ)
  mask        (T, L_TARGET) bool   # True = valid (non-blacklist); chunks (64, L_TARGET)
/samples/
  library, seqrun, endo_category, h5_path (S,) str
  role        (S,)  uint8      # 0 train / 1 heldout / 2 dropped_low_depth
  total_fragments (S,) uint64  # measured during preprocess (depth filter input)
/counts/                       # CSR over flat index u = s*T + t
  indptr      (S*T + 1,) int64
  pos         (nnz,) uint16    # position within [0, L_TARGET)
  track       (nnz,) uint8     # 0..11 canonical order
  data        (nnz,) uint16    # count (per-bp counts << 65535)
  # chunks: 1<<20 elements; entries within a (s,t) slice sorted by (track, pos)
/totals/
  N           (S, T, C) uint32 # per-track totals over UNMASKED positions of
                               # the CENTER TILE (not the margin) — the min-N
                               # filter statistic; documented, test-locked
```

Size at 50×5,000: seq 104 MB, mask 83 MB, N 12 MB, CSR ≈ nnz·5 B.
nnz/(s,t) ≈ 6·(frags in bands) ≈ 3–20 k → 0.8–5 GB total. All well within
the sparse budget from the brief.

Why CSR-flat rather than per-(s,t) zarr groups: 250k tiny groups is
pathological for zarr (object count, directory scans). One ragged CSR gives
O(1) random access via `indptr[u:u+2]` + one or two chunk reads.

## 2. Config object + hash

```python
@dataclass(frozen=True)
class PlumbingConfig:
    # inputs (content-addressed in the hash, not by path)
    sample_sheet: str        # TSV path
    region_beds: dict[str, str]   # split_name -> BED path; e.g.
                                  # {"train_pool": ..., "positive_control": ...}
    blacklist_bed: str
    fasta: str
    # geometry
    tile_size: int = 16384; jitter: int = 128; rf_budget: int = 2048
    fl_bands: tuple = ((40, 65), (120, 175))
    # counting
    min_mapq: int = 10; dedup: bool = True; blacklist_expansion: int = 120
    # sample draw — affects which Phase A shards exist -> HASHED
    seed: int = 1337
    n_train_samples: int = 40; n_heldout_samples: int = 10
    # Phase-B-only / consumer params -> recorded in attrs, NOT hashed
    region_fracs: tuple = (0.8, 0.1, 0.1)
    min_total_fragments: int = 20_000_000   # depth filter, default on
    min_N: int = 50                          # Dataset default
```

`config_hash = sha256(canonical_json)` where every file path is replaced by
its content identity: md5 of the sample-sheet TSV, md5 of each BED, md5 of
blacklist, md5 of the FASTA `.fai` (proxy — hashing a 3 GB FASTA per run is
pointless; the .fai pins contig set/lengths). `min_N`, `region_fracs`, and
`min_total_fragments` are EXCLUDED from the hash: they are Phase-B/consumer
parameters, and hashing them would force a full re-preprocess (~1–2 h) on
every threshold sweep. Phase B may be re-run on existing shards with new
values — it rewrites the split/role arrays in place and updates attrs, so
**store attrs (not the hash) version the split state**; consumers must read
the applied values from attrs, never assume defaults. Attrs carry a
`split_version` integer, auto-incremented on every Phase B write; the
Dataset exposes it and every training run MUST record
`(config_hash, split_version)` — that pair, not the hash alone, is the
reproducibility key (a Phase B re-run invalidates artifacts trained against
the prior split_version, loudly rather than silently). Hash + full config
JSON persisted in store attrs AND a sidecar `bg_store_<hash8>.config.json`
(readable without zarr).

Drift rule: opening an existing store recomputes the hash from the persisted
config and compares to the directory name + attrs; any mismatch is a hard
error (no silent append; build a new store).

## 3. Preprocess orchestration

Two phases; work unit = **one sample** (embarrassingly parallel, matches h5
access pattern, bounded memory).

**Phase A — per-sample count shards** (parallel, `ProcessPoolExecutor`,
`n_workers` config, default 8):

```
worker(sample) -> shards/<library>.npz  (+ .done marker)
  blacklist_rdf = RegionDataFrame.from_bed(cfg.blacklist_bed,
      ref=<genome ref derived from cfg.fasta>)                  # once/worker
      # from_bed REQUIRES ref (dataframe.py:417)
  total_fragments = <h5 global fragment-length histogram>.sum() # ONE metadata
      # read — NOT a sum of per-tile counts (double-counts margins, misses
      # untiled genome)
  for tile_batch in batched(tiles, 256):
      for tile:
          # RegionFragmentArray.from_fname is BROKEN upstream: it forwards 5
          # kwargs its callee rejects -> TypeError (fragment_array.py:1865-75
          # vs :1708-17). Call the callee directly:
          rfa = RegionFragmentArray.from_fragments_h5(
                    h5, margined_region, min_mapq=cfg.min_mapq,
                    max_frag_len=175)
          rfa = rfa.drop_duplicate_fragments()          # dedup=True
          tile_bl = blacklist_rdf overlap-query on margined tile extent
          rfa = rfa.mask_overlapping_fragments(tile_bl,
                    expansion=cfg.blacklist_expansion)
          sparse = rfa.build_coverage_counts(cfg.fl_bands, return_sparse=True)
          # returns pandas.Series keyed by (strand, fl_band, coverage_type);
          # each value is SparseIntVector(coords, data, length). Convert:
          #   for key, vec in sparse.items():
          #       c = TRACK_INDEX[key]                    # canonical 0..11
          #       triples += (vec.coords, c, vec.data)
  accumulate (pos, track, data) triples in RAM (one sample ≈ 25–250 MB)
  -> np.savez
```

`margined_region` = tile ± JITTER (counts extent = L_TARGET), with
**`strand='.'` ALWAYS**: `from_fragments_h5` flips data unconditionally for
`'-'`-strand query regions (fragment_array.py:1809) and no parameter disables
it — strandless queries are the only way to guarantee genome-oriented stored
counts. Strand handling lives exclusively in track identity + RC augmentation
(design invariant; test-locked by the golden count test). Fragments are
assigned by endpoint position; a fragment can contribute to two adjacent
margined tiles — correct by construction (each endpoint counted where it
falls).

**Phase B — single-writer assembly** (serial, minutes): concatenate shards
in canonical sample order -> CSR arrays; compute `/totals/N` (center-tile,
masked); apply depth filter -> `role=2`; draw frozen splits (§6); write seq
(one pysam.FastaFile pass) + mask (one blacklist intersect pass, via
`RegionDataFrame.from_bed` + interval ops); write attrs; atomic rename from
`bg_store_<hash8>.building/` to final name.

Resume: Phase A skips samples with `.done` markers (shard dir keyed by
config hash); Phase B is idempotent (rebuilds from shards). h5 failure:
worker writes `<library>.error` with the exception; assembly refuses to run
while any `.error` exists (operator decides: fix or drop sample from sheet).

Runtime estimate (to be measured on 1 sample before the full run): 5,000
`from_fname` region reads per sample; v1 experience ≈ 5–15 min/sample
single-core -> 50 samples / 8 workers ≈ **1–2 h wall**; Phase B ≈ minutes.
Memory ceiling ≈ n_workers × (one sample's shard in RAM) ≤ ~2 GB.

## 4. Sample-sheet builder (separate small tool)

`build_sample_sheet.py` (script, not library API):
1. Read manifest TSV (path = CLI arg); parse `notes` JSON -> library,
   seqrun, disease; key -> h5 manifest key.
2. Join `ENDO_CATEGORY` from the pooled clinical CSV (CLI arg) on library —
   the ONE clinical column that crosses the boundary.
3. Filter to quiescent pool (`ENDO_CATEGORY in {Asymptomatic, Remission}`,
   config default, overridable) and emit TSV:
   `library  h5_path  seqrun  endo_category`.
4. `--sync` flag: `DataManifest.sync_and_get` each key (serial, before
   preprocess — never inside workers) and write the resolved local paths.

Depth filtering does NOT happen here (would require opening h5s); it happens
in Phase B from measured `total_fragments`. The sheet's md5 enters the
config hash, so the sheet is the frozen sample-set version.

## 5. Dataset / loader

```python
class BackgroundTileDataset(torch.utils.data.Dataset):
    def __init__(self, store_path, model_input_size: int, split: str,
                 sample_role: str, min_N: int = 50, train_mode: bool = True,
                 rc_prob: float = 0.5, jitter: int | None = None): ...
```

- Index built once in `__init__` (cheap array ops on `/totals/N`, `split`,
  `role`): `[(s, t)]` with matching split+role and `N[s,t,:].min() >= min_N`
  (ALL tracks pass; masking individual low-N track loss terms is a noted
  future alternative, not in scope).
- `__getitem__(i)`:
  1. CSR slice `indptr[u]:indptr[u+1]` -> densify `y_full (C, L_TARGET)`
     float32; read `mask_full (L_TARGET,)`, `seq_full (L_SEQ,)` uint8.
  2. Draw `j ~ U{-jitter..jitter}` (train) else 0. Crop with
     `jitter_matrix`: `y = jm(y_full, j, TILE)`, `m = jm(mask_full, j, TILE)`,
     `x_tokens = jm(seq_full, j, model_input_size)` — same j, shared center,
     even lengths (asserted).
  3. RC with prob `rc_prob` (train): `x_tokens` -> complement LUT + reverse;
     `y = y[rc_perm, ::-1]`; `m = m[::-1]` (rc_perm from
     `reverse_complement_track_permutation`).
  4. One-hot: `one_hot_encode_sequences([x_tokens.tobytes()])[0].T` ->
     `x (4, L_in)` float32 — the encoder returns `(N, L, 4)`
     (sequence.pyx:124-128), so `[0].T` is REQUIRED; input must be bytes
     (asserted in the encoder), upper/lowercase both handled.
     Return `(x, y, m)` — CPU numpy/torch only, NEVER cuda.
- Worker safety: zarr store handle opened lazily per worker process (opened
  on first `__getitem__`, keyed by PID) — no pickled handles, read-only.
- Val/test: `train_mode=False` -> j=0, no RC. Deterministic.

## 6. Split assignment (frozen at Phase B)

- **Regions**: split at REGION granularity (tiles inherit via `region_id`) so
  adjacent tiles of one region never straddle train/val. RNG =
  `default_rng(seed)`; proportions from config; regions arriving via the
  `positive_control` BED(s) get split=3 unconditionally and never enter
  train/val/heldout draws.
- **Samples**: two ordered steps — no circularity with the depth filter:
  1. PRE-Phase A (no depth data exists yet): draw `n_train + n_heldout`
     samples simple-random (seeded) from the sheet and assign role 0/1 at
     draw time. Undrawn samples never enter the store (preprocess cost
     fixed). The drawn sheet copy is recorded in store attrs.
  2. Phase B: the depth filter only DOWNGRADES — a drawn sample with
     measured `total_fragments < min_total_fragments` moves to role=2
     (dropped). NO replacement draw (replacement would condition selection
     on depth). If too many samples drop, the operator raises
     `n_train_samples` and rebuilds — a loud failure, never a silent
     backfill.
- Consumers select via Dataset args only; no ad-hoc splitting anywhere else.

## 7. Testing

- **Golden count test** (the brief's one guard on fragment_array): tiny
  fixture h5 (2–3 small regions; reuse fragments_h5 test data if compatible,
  else build from a mini BAM) — compare `build_coverage_counts` output per
  track against an independent pysam/pure-numpy recount written in the test.
  Locks: endpoint definitions (first/last/midpoint), band edges inclusive/
  exclusive, dedup semantics, genome-orientation invariant (no strand flip).
- **Unit**: config-hash stability + path-independence; CSR round-trip
  (densify(sparsify(y)) == y); crop-alignment property — for a synthetic
  store where `y := f(seq)` positionally (e.g., count = GC indicator), every
  (j, RC) combination preserves `y == f(x_tokens)` on the overlap (this is
  the anti-P2 test from the training playbook, moved into the loader's
  tests); min-N filter; split determinism given seed; drift rule (hash
  mismatch raises).
- **E2E fixture**: 2 samples × ~20 micro-tiles (tile_size=2048 override) ->
  full preprocess -> Dataset -> one `BackgroundModel._step` forward on each
  loss (CPU) runs finite.

## 8. Module layout (loose; migrates to biomarker later)

```
background_model/            # new top-level dir, NOT inside fragmentomics_tools pkg
  __init__.py
  config.py                  # PlumbingConfig, hashing, drift rule
  store.py                   # layout constants, open/validate, CSR access
  preprocess.py              # Phase A worker + Phase B assembly + CLI
  sample_sheet.py            # builder CLI (§4)
  dataset.py                 # BackgroundTileDataset
tests/
  test_bg_config.py  test_bg_store.py  test_bg_dataset.py
  test_bg_golden_counts.py  test_bg_e2e.py
```

`background_model_core.py` stays at repo root for now (open item #4 in the
brief); `dataset.py` imports its primitives.

## 9. Failure modes considered

| Failure | Handling |
|---|---|
| Preprocess crash mid-run | Phase A `.done` markers; rerun skips finished samples |
| Corrupt/truncated h5 | worker `.error` file; assembly blocked until resolved |
| Config drift vs existing store | hash recompute on open; hard error, never append |
| Partial Phase B | builds under `.building/` dir; atomic rename at end |
| NFS + many small writes | shards are single npz files; zarr written once, serially, in Phase B |
| Model deeper than RF_BUDGET | Dataset init assert (L_SEQ check) fails loudly |
| Sheet edited after store build | sheet md5 in hash -> new store |
| zarr version skew | version pinned in attrs; open-time warning on mismatch |

## Self-grade: B+

Honest deductions: (a) runtime/memory numbers are estimates from v1
experience, not measurements — the design mandates a 1-sample measurement
gate before the full run, but the numbers could be 5× off; (b) the
`build_coverage_counts(return_sparse=True)` output format was not verified
against source in this draft (only the signature) — Phase A pseudocode may
need adaptation; (c) crop-alignment arithmetic (three nested extents sharing
a center, plus strand-aware `Region.get_resize_start` floor behavior) is the
kind of thing that's right in prose and off-by-one in code — mitigated by
the property test, but a reviewer should re-derive it independently; (d) EM
self-authored: no independent design pressure yet.

Top risks, ranked:
1. Crop-alignment off-by-one (mitigation: §7 property test is mandatory,
   written BEFORE the Dataset).
2. fragment_array API surprises (sparse output format, strand flip default)
   — mitigation: golden test first, adapt Phase A to reality.
3. Preprocess throughput far worse than estimated (5,000 region reads/
   sample) — mitigation: 1-sample measurement gate; fallback is batching
   contiguous tiles into fewer, larger `from_fname` reads.
4. Zarr-on-NFS behavior (chunk write patterns) — mitigation: Phase B is
   serial and write-once; shards live on local scratch.

## Review Notes (2026-08-26)

**Verdict**: NEEDS REVISION
**Grade**: B-

### Must-Fix (Critical + High)

| # | Severity | Finding |
|---|----------|---------|
| 1 | Critical | `RegionFragmentArray.from_fname` is broken — passes 5 kwargs to `from_fragments_h5` that the callee doesn't accept (TypeError). `fragment_array.py:1865-1875` vs `:1708-1717`. Call `from_fragments_h5` directly or fix upstream. |
| 2 | Critical | `flip_data_to_match_region_strand=False` does not exist as a functional mechanism. Strand flip is unconditional on `region.is_minus_strand()` (`fragment_array.py:1809`). Use `strand='.'` regions to ensure genome-oriented counts. |
| 3 | High | §6 split ordering contradicts itself: "after depth filter" vs "Selection happens BEFORE Phase A". Depth filter requires Phase A data. Clarify: draw samples pre-Phase A, depth filter in Phase B only downgrades roles. |
| 4 | High | `one_hot_encode_sequences` returns `(N, L, 4)` not `(4, L)`. §5 pseudocode is missing `[0].T`. (`sequence.pyx:124-128`; confirmed by `region.py:816-820`.) |

### Should-Fix (Medium)

| # | Finding |
|---|---------|
| 5 | `min_total_fragments` and `region_fracs` only affect Phase B but are in the config hash — forces full re-preprocessing on sweeps. Exclude from hash (like `min_N`). |
| 6 | `build_coverage_counts(return_sparse=True)` returns `pandas.Series` of `SparseIntVector` objects — Phase A pseudocode doesn't show the conversion to CSR triples. |
| 7 | Blacklist region retrieval per tile in Phase A unspecified — `mask_overlapping_fragments` needs Region objects, source not shown. |

### Verified Correct

- Crop arithmetic (L_SEQ, L_TARGET, TILE centers align under jitter) ✓
- Same-parity: all four lengths even ✓
- `calc_input_region_size(16384) = 16632` with default model ✓
- RC permutation semantics match `y[perm, ::-1]` ✓
- Track ordering in `build_coverage_counts` matches `DEFAULT_OUTPUT_TRACKS` ✓
- uint16 pos fits L_TARGET=16640 ✓
- CSR O(1) random-access design is sound ✓
- Batch convention (x, y, mask) matches `BackgroundModel._step` ✓
- `max_frag_len=max(band_highs)=175` correct (exclusive upper bound) ✓

### Risks

1. Phase A crashes immediately on `from_fname` call (TypeError from broken kwargs passthrough)
2. Implementer guesses wrong on split/depth-filter ordering → train set contaminated
3. First depth-threshold sweep forces full re-preprocessing due to over-inclusive hash

### Key Tradeoffs

- CSR-flat vs per-(s,t) zarr groups: CSR chosen for O(1) access and avoiding 250k tiny groups; trade-off is more complex assembly code in Phase B
- RF_BUDGET=2048 is 16× the default model's RF (124/side): future-proofs for deeper architectures at the cost of ~25% more stored sequence per tile
- All-in-one config hash vs split Phase A/B hashes: simpler but prevents cheap Phase B parameter sweeps

## Revision Log

- **r2 (2026-08-26, EM)** — all r1 findings addressed:
  #1/#2 Phase A now calls `from_fragments_h5` directly (from_fname documented
  as broken) with the `strand='.'` invariant replacing the nonexistent
  flip parameter; #3 §6 rewritten as two ordered steps (pre-Phase-A draw,
  Phase B downgrade-only, no replacement); #4 one-hot `[0].T` + bytes note;
  #5 `region_fracs`/`min_total_fragments` moved out of the hash, attrs
  version split state, Phase B re-runnable on shards; #6 SparseIntVector ->
  triples conversion shown; #7 blacklist RDF loaded once per worker +
  per-tile overlap query; #8 half-open band semantics documented (§0);
  #9 total_fragments from the h5 global histogram (single metadata read);
  #10 parity assert added to Dataset init contract (§0); #11 RF_BUDGET
  16× ratio documented (§0).

### Round 2 (2026-08-26)

**Verdict**: APPROVED WITH CONDITIONS
**Grade**: A-

#### Prior-finding verification

| # | Prior Finding | Status | Notes |
|---|---------------|--------|-------|
| 1 | `from_fname` broken (TypeError) | FIXED | §3 calls `from_fragments_h5` directly; line refs 1865-75 vs 1708-17 verified against source |
| 2 | `flip_data_to_match_region_strand` nonexistent | FIXED | `strand='.'` invariant correct; `fragment_array.py:1809` confirms unconditional flip on `is_minus_strand()` |
| 3 | §6 split ordering contradiction | FIXED | Two ordered steps: pre-Phase-A draw + Phase-B downgrade-only; clear, no contradiction |
| 4 | `one_hot_encode_sequences` shape `(N,L,4)` | FIXED | `[0].T` shown; `sequence.pyx:124-125` confirmed `(num_seqs, seq_length, NUM_BASES)` |
| 5 | `min_total_fragments`/`region_fracs` in hash | FIXED | Excluded from hash; Phase B re-runnable; attrs version split state |
| 6 | SparseIntVector conversion not shown | FIXED | Triples pseudocode correct; `.coords`/`.data` attrs verified (`fragment_array.py:42,48`) |
| 7 | Blacklist region source unspecified | FIXED | `RegionDataFrame.from_bed` loaded once/worker; per-tile overlap query shown |
| 8 | Half-open band semantics | FIXED | §0 documents `[lo,hi)`; matches `subset_fragment_lengths` `<` at line 790 |
| 9 | `total_fragments` source | FIXED | h5 `fragment_length_counts.sum()` matches `FragmentsH5.n_fragments` property |
| 10 | Parity assert | FIXED | §0 documents all-even assertion at Dataset init |
| 11 | RF_BUDGET ratio | FIXED | Documented "16×" in §0 (actual 2048/124 ≈ 16.5×; acceptable approximation) |

#### New findings (round 2)

| # | Severity | Category | Finding | Evidence | Recommendation |
|---|----------|----------|---------|----------|----------------|
| 12 | Low | Accuracy | §3 pseudocode `RegionDataFrame.from_bed(cfg.blacklist_bed)` omits required `ref` parameter | `dataframe.py:417` signature: `from_bed(cls, in_bed_file, ref)` | Add `ref` arg; config already has `fasta` which can supply it |
| 13 | Low | Design | Phase B re-run lacks split-state versioning — re-running with different `region_fracs` silently invalidates prior training artifacts | Two Phase B runs with different fracs produce different splits under the same store hash; no mechanism for consumers to detect staleness | Consider a `split_version` counter in attrs, auto-incremented on each Phase B write |
| 14 | Info | Accuracy | §0 says RF_BUDGET is "16× the default model's need" but 2048/124 = 16.52× | `calc_input_region_size(16384) = 16632`; RF/side = 124; 2048/124 = 16.516 | Trivial; "~16×" or "~16.5×" would be more precise |

#### Summary

- **Must-fix**: None
- **Should-fix**: None
- **Concerns**: #12 (pseudocode `from_bed` missing ref), #13 (split-state versioning), #14 (RF ratio approximation)

All 11 round-1 findings verified as FIXED. No critical, high, or medium issues remain. The design accurately reflects the source code APIs (`from_fragments_h5` signature, `SparseIntVector` attributes, `one_hot_encode_sequences` shape, strand-flip behavior, `subset_fragment_lengths` semantics, `fragment_length_counts` property). Requirements fidelity against `BACKGROUND_MODEL_BRIEF.md` §"Data plumbing" is complete — all locked requirements are addressed.

#### Conditions for approval
1. Implementer must pass `ref` to `RegionDataFrame.from_bed` (finding #12 — straightforward from config's `fasta` field)
2. Consider adding split-state versioning if Phase B parameter sweeps become routine (finding #13 — can be deferred to implementation time)

#### Top 3 risks if shipped as-is
1. Crop-alignment off-by-one (mitigated by §7 property test — must be written BEFORE Dataset)
2. Preprocess throughput far worse than estimated (mitigated by 1-sample measurement gate)
3. Phase B re-run silently invalidating prior training artifacts (mitigated by documentation contract; consider versioning later)
