# Correction Outputs Design — Background Model v2, Phase 2

Status: DRAFT (2026-08-28). Design only — no implementation, no commits.
Requirements source: `BACKGROUND_MODEL_BRIEF.md` §"Correction outputs (the
product)"; statistical spec: `background_model_core.py` module docstring; data
geometry + masked-target contract: `docs/pending/data_plumbing_design.md`
(esp. the reconciliation notes, 2026-08-27). Prior art (do NOT maintain):
`fragmentomics_tools/bias_correction/` + `dataframe.py` weight appliers, whose
S1 strand-mask bug is documented in `BIAS_CORRECTION_REVIEW.md`.

Branch `background-model-v2`, HEAD `e85ff0b`.

---

## 0. Problem statement

The product is a **within-sample bias correction**. Given a trained
`BackgroundModel` (sequence → per-position profile *shape* `p` + dispersion)
and one sample's fragment data, remove sequence-driven technical bias so that
downstream fragmentomics analyses (TF footprinting, v-plots, the flagship CTCF
motif pileup) consume corrected signal. Phase 1 built the training plumbing
(`background_model/`: config, store, preprocess, dataset). **Nothing yet
produces a correction.** This design specifies the two locked output
interfaces and the inference machinery underneath them.

Two interfaces (brief §"Correction outputs"), verbatim requirements:

> (a) per-fragment weights attached to RegionFragmentArray (1 / predicted
> relative rate at each fragment's start/stop/midpoint, per strand × band
> track — the fixed version of v1's
> `_set_fragment_array_weights_from_weights_record`, whose strand-mask OR/AND
> bug is documented in BIAS_CORRECTION_REVIEW.md S1), and (b) per-position
> expected-profile vectors for consumers that divide themselves.
>
> Weight clamping semantics: DEFERRED (explicitly out of scope for now).

Correctness contracts carried from the model design:

1. **N is the plug-in scale.** The observed total `N[sample, window, track]`
   (over unmasked positions) supplies scale; the network predicts only *shape*
   (`predict_profile` returns `probs` that sum to 1 over valid positions per
   track — `background_model_core.py:656-683`). Absolute counts are never
   predicted.
2. **Masked-target contract** (reconciliation note, `data_plumbing_design.md`
   L472-489): the mask defines the sample space (softmax support). The store
   keeps RAW unzeroed counts; **"ANY future consumer reading `/counts/*`
   directly (e.g. Phase 2 correction outputs) MUST apply the mask itself."**
   Phase 2 is exactly that consumer — every N and every weight in this design
   applies the mask by construction.
3. **One coordinate frame everywhere — strandless regions, never flipped**
   (SYSTEM-WIDE invariant). Correction queries build their regions with
   `strand='.'`, *exactly as Phase A preprocessing does*, so
   `from_fragments_h5` NEVER flips: the minus-strand flip + strand-swap
   (`fragment_array.py:1809-1819`, gated on `region.is_minus_strand()`) is never
   entered, `starts_0/stops_0` stay in the forward genomic-local frame,
   `fragment_strands` keep their true `+/−` values, and `is_flipped` stays
   `False` (`:1843`). Same invariant, same rationale as Phase A: a single
   genomic coordinate frame across preprocessing, training, and correction, so
   `gpos = region.start + endpoint_coord` and `track = fragment_strands[f]` are
   always valid. Strand-ORIENTATION of the flagship CTCF pileup is a
   CONSUMER-layer operation — flip the *aggregated per-position profiles* for
   `−`-strand motifs — never a per-rfa flip. The applier ENFORCES this (§5.2):
   it asserts `region.strand in {None, '.', '+'}` AND `not rfa.is_flipped`,
   raising a message that points to this section. (`Region` normalizes a
   strandless `'.'` to `None` — `region.py:538-539`, `strand_is_set` treats
   `None` and `'.'` identically — so the value the applier actually observes on
   a `strand='.'` query is `None`. Accepting `{None, '.', '+'}` is what keeps
   every happy-path strandless call from raising; a literal-`'.'`-only assert
   was wrong and broke every happy-path call, found during implementation and
   fixed in 0cc5f18. The `is_flipped` half is unchanged.)

---

## 1. What exists (verified against source)

### 1.1 The frozen model — `background_model_core.py`

- `BackgroundModel.forward(x)` → `(shape_logits (B,C,L_out), dispersion|None)`.
  Unpadded convs: `L_in == calc_input_region_size(L_out)` (:593-604).
- `predict_profile(one_hot_seq (4,L_in), mask=(L_out,)|None)` → dict:
  - `probs (C, L_out)`: masked softmax, **each track sums to 1 over valid
    positions**; masked positions → probability 0 (`masked_fill(-inf)` then
    softmax, :676-677). Multiply by an observed N for expected counts.
  - `log_dispersion`: `(C,)` [dirichlet_multinomial] / `(C,W)` [nb_offset] /
    `None` [multinomial].
- Track axis `C=12`, canonical order `DEFAULT_OUTPUT_TRACKS` (:126-131) =
  `strand{+,-} × fl_band{(40,65),(120,175)} × coverage{first,last,midpoint}`,
  nested in that order. `index_key_to_track_name` / `track_name_to_index_key`
  round-trip (:112-123). `reverse_complement_track_permutation` (:134-156).
- `calc_input_region_size(L_out)` (:593-604): default hparams
  `kernel_size=32, num_residual_layers=2` ⇒
  `16384 + 2·31 + (31·2 + 31·4) = 16384 + 62 + 186 = 16632`, where the `2·31`
  term is the initial conv AND the shape-head conv (each trims `k−1=31`; the
  source groups them as `2·(k−1)`, :601) and `31·2 + 31·4` is the two dilated
  ResNet blocks. **Margin = (16632 − 16384)/2 = 124 bp/side.** This is
  architecture-dependent and MUST be recomputed from the model, never hardcoded.

### 1.2 The training data-assembly path (the inference path must mirror it)

- **Sequence** (`preprocess.py:480-505`, Phase B): per tile,
  `seq_margin = jitter + rf_budget = 128 + 2048 = 2176`;
  fetch `fasta.fetch(contig, max(0, tile.start − seq_margin), tile.stop +
  seq_margin)`; N-pad left/right at contig edges; `.upper().encode("ascii")` →
  uint8 over `L_SEQ = 20736`.
- **Dataset val-mode crop** (`dataset.py:200-230`, `train_mode=False` ⇒
  `j=0, do_rc=False`): center-crop `seq_full (L_SEQ)` → `x_tokens
  (model_input_size)` via `jitter_matrix(·, 0, model_input_size)`; center-crop
  `mask_full (L_TARGET)` → `(TILE)`; one-hot via
  `one_hot_encode_sequences([x_tokens.tobytes()])[0].T` → `(4, L_in)` float32.
  The `RF_BUDGET=2048` store margin is future-proofing; **the val-mode input
  actually consumed is exactly the tile ± 124 bp** (the model's real receptive
  field), because the crop to `model_input_size=16632` discards the rest.
- **Mask** (`preprocess.py:507-552`): store mask over `L_TARGET`, contig-clamped
  + symmetric blacklist expansion (`blacklist_expansion=120`, approved
  algorithmic change, reconciliation note L448-458). Dataset center-crops it to
  `TILE`.
- **N** (`store.py:173-188` `compute_N_for_tile`): per-track sum over the CENTER
  tile's unmasked positions — `(center_y * center_mask).sum(axis=1)`.
- `one_hot_encode_sequences` returns `(N, L, 4)`; `[0].T` → `(4, L)`; input must
  be **bytes**, upper/lowercase both handled (`dataset.py:224-227`).

### 1.3 RegionFragmentArray weight surface — `fragment_array.py`

Three independent weight vectors and the coordinate each indexes (verified
:794-854):

| coverage_type | weight attribute            | position attribute      | coverage getter                  |
|---------------|-----------------------------|-------------------------|----------------------------------|
| `first`       | `first_covered_base_weights`| `first_covered_bases_0` | `get_first_covered_base_array`   |
| `last`        | `last_covered_base_weights` | `last_covered_bases_0`  | `get_last_covered_base_array`    |
| `midpoint`    | `weights`                   | `midpoints_0`           | `get_midpoint_coverage_array`    |

- `build_coverage_counts(fl_bands, split_strand, return_sparse)` (:880-907)
  returns a `pandas.Series` keyed `(strand, fl_band, coverage_type)` whose
  values are per-position count arrays **already multiplied by these weight
  vectors** (each getter passes its `weights_attr` into `_get_covered_base_array`
  :810). So setting the weight vectors IS the correction — downstream coverage
  automatically reflects it.
- `subset_fragment_lengths` (:785-792): band membership is **half-open**
  `min_frag_len <= len < max_frag_len`. Design mirrors this exactly.
- `fragment_strands` (`U1`, `+`/`-`), `fragment_lengths`, `n_fragments`, and
  `mask(bool_array)` for dropping fragments.
- `from_fragments_h5(in_fragments_h5, region, max_frag_len, min_mapq, ...)`
  (:1708) is the working reader; `from_fname` is broken upstream (forwards
  rejected kwargs — `data_plumbing_design.md` finding #1); the `background_model`
  / `min|max_background_scaling_factor` params on `from_fname` (:1858-1860) and
  the ghost type `SeqToEndpointsMultiResModel` are **dead v1 plumbing** and are
  NOT revived. `DEFAULT_MIN_SCALING_FACTOR=1e-6`, `DEFAULT_MAX_SCALING_FACTOR=
  10.0` (:27-28) are the only concrete clamp precedent.

### 1.4 The S1 bug (must be impossible-by-construction in the new applier)

`dataframe.py:1928-1943` `_set_fragment_array_weights_from_weights_record`:

```python
mask = numpy.zeros(n).astype(bool)
mask = (mask | (fragment_lengths >= fl_lb) & (fragment_lengths <= fl_ub))  # band
if strand in '-+':
    mask = (mask | (fragment_strands == strand))                          # BUG
...
getattr(fragment_array, weights_attr)[mask] = 1./weights[count_indices[mask]]
```

Python precedence makes this `(in_band) OR (on_strand)`, not
`(in_band) AND (on_strand)`. Consequences (BIAS_CORRECTION_REVIEW.md S1):
a fragment on the matching strand but outside the band still gets weighted; a
fragment inside the band gets weighted by **both** strands' tracks; because
per-`(strand,band,cov)` masks overlap, later loop iterations silently overwrite
earlier ones, so the final weight depends on dict/loop order. This is the whole
point of the package, and it silently assigns wrong weights. It also has a
second latent mismatch, but ONLY on the `last` coverage: v1 indexes it by
`stops_0` (:1916) whereas the `last` getter reads `last_covered_bases_0 =
stops_0 − 1` (`fragment_array.py:771`) — an off-by-one. `first` and `midpoint`
coincide exactly (`first_covered_bases_0 == starts_0`, :775; `midpoints_0 ==
midpoints_0`, :779), so `last` is the only divergence; the new design fixes it
by using each getter's own coordinate. A **third**
latent inconsistency: v1's band test `(fl >= lb) & (fl <= ub)` (:1933) is a
CLOSED interval `[lb, ub]`, whereas `subset_fragment_lengths` (:790, and hence
`build_coverage_counts`) uses half-open `[lb, ub)` — a fragment of length
exactly `ub` would be weighted but not counted. The new design uses half-open
throughout (§5.2), matching the coverage getter.

---

## 2. Package layout & public API

New modules under the existing loose `background_model/` package (migrates to
biomarker later with the rest of Phase 1). No changes to
`background_model_core.py` or the store format.

```
background_model/
  inference.py    # region → model-input assembly, tiling, seam-free stitching
  correction.py   # the two public interfaces (a) + (b)
tests/
  test_bg_inference.py    # equivalence-to-Dataset, seam-free, contig-edge
  test_bg_correction.py   # analytic uniform/delta, S1 lock, N-definition
```

`inference.py` imports `predict_profile`, `calc_input_region_size`,
`DEFAULT_OUTPUT_TRACKS`, `jitter_matrix` from `background_model_core`, and
`one_hot_encode_sequences` from `fragmentomics_tools.region` (which re-exports
the Cython implementation in `fragmentomics_tools/sequence.pyx` — the SAME
encoder the Dataset imports and calls; reusing it, not reimplementing it, is the
equivalence mechanism).
`correction.py` imports `RegionFragmentArray` and the track constants.

### 2.1 `inference.py`

```python
@dataclass(frozen=True)
class WindowGeometry:
    tile_size: int          # L_out per window (default TILE = 16384)
    model_input_size: int   # = model.calc_input_region_size(tile_size)
    margin: int             # = (model_input_size - tile_size) // 2  (=124 default)

    @classmethod
    def from_model(cls, model, tile_size: int = TILE) -> "WindowGeometry": ...
    # asserts (model_input_size - tile_size) even → symmetric crop


def build_window_onehot(
    fasta: pysam.FastaFile, contig: str, win_start: int, win_stop: int,
    geom: WindowGeometry, *, contig_len: int | None = None,
) -> np.ndarray:
    """One-hot (4, model_input_size) for window [win_start, win_stop) (len
    tile_size). Fetch fasta[win_start-margin : win_stop+margin], N-pad at
    contig edges, .upper() → bytes → one_hot_encode_sequences([...])[0].T.
    BYTE-IDENTICAL to the val-mode Dataset x for the same tile."""


def build_window_mask(
    contig: str, win_start: int, win_stop: int, geom: WindowGeometry, *,
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: int | None = None,
) -> np.ndarray:
    """(tile_size,) bool valid mask: invalid past contig ends and within
    `blacklist_expansion` bp of any blacklist region. IDENTICAL to the
    center-tile crop of the store's L_TARGET mask (reuses preprocess Phase B
    logic on the TILE extent directly)."""


def iter_windows(start: int, stop: int, tile_size: int = TILE):
    """Yield consecutive (win_start, win_stop) covering [start, stop) on the
    grid anchored at `start`. The final window is clamped so win_stop <= stop
    is NOT required — see §4 edge handling (the last window runs full-length
    and its out-of-range output positions are trimmed by the caller)."""


@dataclass(frozen=True)
class RegionProfile:
    probs: np.ndarray          # (C, L) masked softmax, per-window, concatenated
    mask: np.ndarray           # (L,) bool valid
    window_bounds: list        # [(win_start, win_stop, local_lo, local_hi)]
    coord0: int                # genomic position of probs[:, 0] (== start)
    output_tracks: list[str]


@torch.no_grad()
def predict_region_profiles(
    model, fasta, contig: str, start: int, stop: int, *,
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: int | None = None, tile_size: int = TILE,
) -> RegionProfile:
    """Partition [start, stop) into tile_size windows; per window build the
    one-hot + mask, call predict_profile, and CONCATENATE the per-window probs.
    Seam-free in the SHAPE by construction (§3.3); expected COUNTS carry
    per-window N_w and step by N_w ratios at seams."""
```

### 2.2 `correction.py`

```python
@dataclass(frozen=True)
class WeightClampConfig:
    """Explicit clamp bounds for the inverse-probability weights. There is NO
    default — `apply_fragment_weights` REQUIRES a clamp argument, forcing the
    caller to choose consciously. Clamp SEMANTICS remain a deferred owner
    decision (brief §Correction outputs: clamping DEFERRED); this class only
    provides the mechanism. `WeightClampConfig.identity()` is the explicit opt-in
    for NO clamp — but note that with a TRAINED (peaky) model identity can
    produce UNBOUNDED weights (`1/(probs·L_valid)` as `probs → 0`; §7). v1-parity
    is `WeightClampConfig(min_weight=1e-6, max_weight=10.0)`."""
    min_weight: float | None = None
    max_weight: float | None = None

    @classmethod
    def identity(cls) -> "WeightClampConfig":
        """Explicit NO-clamp opt-in. WARNING: unbounded weights with a trained
        model (§7); clamp semantics remain a deferred owner decision."""
        return cls(min_weight=None, max_weight=None)

    def apply(self, w: np.ndarray) -> np.ndarray:
        if self.min_weight is None and self.max_weight is None:
            return w
        return np.clip(w, self.min_weight, self.max_weight)


@dataclass(frozen=True)
class ExpectedProfile:
    expected: np.ndarray       # (C, L) expected counts; NaN at masked positions
    probs: np.ndarray          # (C, L) the underlying shape
    N: np.ndarray              # (n_windows, C) per-window per-track observed N
    mask: np.ndarray           # (L,) bool
    coord0: int
    output_tracks: list[str]


def expected_profile(
    model, fasta, contig: str, start: int, stop: int,
    observed_counts: np.ndarray, *,                 # (C, L) raw sample counts
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: int | None = None, tile_size: int = TILE,
    masked_fill: float = np.nan,
) -> ExpectedProfile:
    """Interface (b). Per window w: N_w[c] = sum_j observed[c, j]·mask[j]
    (mask applied — enforces the store contract by construction);
    expected[c, j] = N_w[c] · probs[c, j] for valid j, else masked_fill."""


def apply_fragment_weights(
    rfa: "RegionFragmentArray", model, fasta, *,
    fl_bands=((40, 65), (120, 175)),
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: int | None = None, tile_size: int = TILE,
    clamp: WeightClampConfig,          # REQUIRED — no default (§5.2); pass
                                       # WeightClampConfig.identity() for no clamp
    drop_uncorrectable: bool = True,
) -> "RegionFragmentArray":
    """Interface (a). Set rfa.first_covered_base_weights,
    rfa.last_covered_base_weights, rfa.weights to 1/relative_rate at each
    fragment's (first/last/midpoint) endpoint, per (strand, band, coverage)
    track. Uncorrectable endpoints (out-of-band, masked, or off-grid) → 0.
    drop_uncorrectable drops fragments whose three weights are all 0."""
```

---

## 3. Region → model-input assembly & seam-free stitching

### 3.1 Equivalence guarantee to the training-time path

The inference input for a window `[w0, w0+TILE)` is produced by the SAME three
operations the val-mode Dataset performs, composed instead of split across
preprocess+Dataset:

| step | Dataset (val) path | inference path | identical? |
|------|--------------------|-----------------|------------|
| fetch | Phase B: fasta[c, w0−2176, w0+TILE+2176] N-padded → L_SEQ | fasta[c, w0−124, w0+TILE+124] N-padded → L_in | same bytes on the overlap |
| crop | `jitter_matrix(seq_full, 0, 16632)` center-crop of L_SEQ | (no crop — fetched at L_in directly) | same window |
| encode | `one_hot_encode_sequences([x.tobytes()])[0].T` | identical call | identical |

Because the val-mode crop is a **center** crop (`jitter_matrix` with `j=0`,
`Region.get_resize_start` symmetric for even parity) and both extents are even,
the `model_input_size`-length window consumed by the model is exactly `[w0−124,
w0+TILE+124)`. Fetching directly at that extent yields the same ASCII bytes
(same FASTA, same `.upper()`, same N-padding rule), hence the same one-hot.
**The RF_BUDGET margin is never in the model's input on either path**, so its
absence at inference is not a difference. This is TEST-LOCKED (§6, T1): build a
micro-store via `run_preprocess`, then assert `build_window_onehot(...)` is
`np.array_equal` to `Dataset(train_mode=False)[i][0]` for every tile, and
`build_window_mask(...)` equal to the center-crop of the store mask.

### 3.2 Output coordinate mapping

The model is unpadded and trims exactly `margin=124` bp symmetrically from each
side of the fetched input (total trim `2·margin=248` — the sum of the conv/head
trims in `calc_input_region_size`). So the emitted output window of length
`TILE` is the center of the input window `[w0−124, w0+TILE+124)`, i.e. genomic
`[w0, w0+TILE)`, and output position `j ∈ [0, TILE)` corresponds to genomic
position **`w0 + j`**. (Each single output position `j` depends on only a small
**receptive field** of input bases — RF = `32 + 31·2 + 31·4 + 31 = 249` for the
default hyperparameters, centered at input position `j + margin` — NOT on all
`model_input_size` bases; the full-width input is needed only to emit all `TILE`
positions collectively. The coordinate mapping follows from the symmetric
`margin` trim, not from per-position input dependence.) Thus `probs[:, j]` ↔
genomic `w0 + j`, and for the whole region `probs[:, k]` ↔ `start + k`
(`RegionProfile.coord0 = start`).

### 3.3 Seam-free stitching (of the SHAPE)

Adjacent windows `[w0, w0+TILE)` and `[w0+TILE, w0+2·TILE)` each emit outputs
that are exact for their own positions and depend only on their own fetched
sequence. Their input extents overlap by `2·margin = 248` bp (re-fetched
sequence — harmless); their outputs **abut with no overlap and no seam**. Each
window carries its OWN masked softmax (`probs` sums to 1 within that window over
its valid positions) and its OWN plug-in N (§5.1). Stitching = plain
concatenation of per-window `probs` (and, for expected profiles, per-window
`N·probs`). There is deliberately **no cross-window renormalization**: the
network never predicts cross-window relative scale (it predicts within-window
shape only — `background_model_core.py` docstring "Scale / the role of N"), so
concatenation is the mathematically correct decomposition, and it is exact
(TEST-LOCKED §6, T2: profiles over a 2-tile region == two independent
single-window `predict_profile` calls, position-by-position).

Worked arithmetic (default geometry), region `chr1:1_000_000-1_032_768`
(2·TILE):
- windows: `(1_000_000, 1_016_384)`, `(1_016_384, 1_032_768)`.
- window 0 fetch: `chr1[999_876 : 1_016_508]` → 16632 bp → one-hot (4, 16632)
  → `predict_profile` → probs (12, 16384) for genomic `1_000_000..1_016_383`.
- window 1 fetch: `chr1[1_016_260 : 1_032_892]` → probs for
  `1_016_384..1_032_767`.
- fetches overlap on `[1_016_260, 1_016_508)` (248 bp); outputs abut exactly at
  `1_016_384`.

**Scope of "seam-free": the SHAPE, not the scale.** Concatenation is seam-free
for `probs` (interface (a)'s weights and interface (b)'s shape). It is NOT
seam-free for expected COUNTS: each window's `expected = N_w · probs` carries its
own per-window `N_w` scale (§5.1), so a multi-window expected track STEPS by the
ratio `N_{w0}/N_{w1}` at every tile seam — an intentional discontinuity, since
the network predicts within-window shape only and never cross-window scale. Any
consumer comparing expected COUNTS across positions must do so WITHIN a single
window. The flagship CTCF pileup regions (±2 kb about a motif = 4001 bp) fit
inside one `TILE`-length window; such a region MUST be evaluated as a SINGLE
window (anchored so it does not straddle a seam) so that all cross-position count
comparisons are within-window normalized.

---

## 4. Edge handling at contig boundaries

Reuse Phase 1's clamp/pad conventions exactly (`preprocess.py:487-503` for
sequence, `:523-552` for mask):

- **Sequence**: `seq_start = w0 − margin`; fetch `[max(0, seq_start), min(
  contig_len, w0+TILE+margin))`; `left_pad = max(0, −seq_start)`,
  `right_pad = model_input_size − len − left_pad`; prepend/append `"N"`. N
  one-hot-encodes to a UNIFORM `[0.25, 0.25, 0.25, 0.25]` column
  (`sequence.pyx:39-40`), IDENTICALLY at train and inference, which is exactly
  what training saw at contig ends.
- **Mask**: positions past `[0, contig_len)` → invalid; blacklist-expanded zones
  → invalid. Same widened overlap query (`count_start − expansion … +
  expansion`) as Phase B so a blacklist region just outside the window whose
  expanded zone reaches in is still caught.
- **Region ends not on the TILE grid** (`stop − start` not a multiple of
  `tile_size`, or a sub-window region): the final window still runs at full
  `TILE` output length (the model has no shorter mode); `predict_region_profiles`
  emits only the `[start, stop)` slice via `window_bounds` `local_lo:local_hi`.
  The trailing model positions beyond `stop` are computed but discarded. `N` for
  that window is the observed sum over the window's valid **EMITTED/TRIMMED**
  positions — `N_w[c] = sum_j observed[c, j]·mask[j]` over `j` in the window's
  emitted slice (§5.1). Full-window `N` is **unrealizable through this
  interface**: `expected_profile` receives `observed_counts` of shape
  `(C, stop − start)`, so there is no data past `stop` to sum. The window's
  probs still form a full-`TILE` multinomial, but `N` can only be the plug-in
  scale over the positions the caller actually supplied. Consequently the
  flatness identity `sum_{valid} expected == N_w` (T9) holds over the EMITTED
  slice, NOT the full window.
  **HARD REQUIREMENT (Phase-3+ consumers).** Any consumer that aggregates
  expected counts across positions/windows (e.g. the CTCF pileup) MUST use
  SINGLE, UNTRIMMED, GRID-ALIGNED windows — i.e. evaluate over a region whose
  `stop − start` is a multiple of `tile_size` and that does not straddle a seam.
  A trimmed final window's expected slice does NOT sum to its own observed total
  and would misread the scale of any obs/expected or pileup aggregation. This is
  a locked precondition on the aggregation layer, not an option.

`contig_len` is passed through from the caller (or looked up via
`fragmentomics_tools.contig.CONTIG_LENGTHS[ref][contig]`, the same source
`preprocess._contig_length` uses); `None` disables right-clamping (matches
Phase 1 behavior when a contig length is unavailable).

---

## 5. Semantics of the two interfaces

### 5.1 Expected-profile vectors — interface (b)

Definitions per window `w` (valid position set `V_w`, `L_valid = |V_w|`):

- `probs[c, j]`: masked softmax from `predict_profile`, `sum_{j∈V_w} probs[c,j]
  = 1`, `probs[c,j]=0` for `j∉V_w`.
- `N_w[c] = sum_{j∈V_w} observed_counts[c, j]` — the **caller's raw sample
  counts, masked** (the store contract: consumers of raw counts must mask). N is
  per **(sample, window, track)**, exactly the training-time N definition
  (`compute_N_for_tile`) but over the inference window instead of a store tile.
  For a TRIMMED final window `V_w` is the window's **EMITTED** valid positions
  (the `observed_counts` array spans only `[start, stop)`, so full-window `N` is
  unrealizable through this interface — §4). Grid-aligned windows are untrimmed,
  so aggregation consumers that follow the §4 HARD REQUIREMENT never see a
  trimmed `N`.
- `expected[c, j] = N_w[c] · probs[c, j]` for `j∈V_w`.

**Masked positions in the output vector: NaN** (`masked_fill=np.nan`, the
default). Justification: at masked positions `probs=0` ⇒ `expected=0`, but the
raw observed count there may be **nonzero** (a blacklist-spanning fragment
survives Phase-A fragment masking and can deposit a midpoint on a masked
position — reconciliation note L484-489). A consumer computing `observed /
expected` would then hit `nonzero / 0 = ±inf`, silently poisoning a pileup. NaN
propagates through any arithmetic and forces the consumer to treat masked
positions as "no support," which is the correct semantics. (`0` would falsely
read as "expected nothing here"; omission would misalign coordinates for a
fixed-length vector. NaN is the only choice that is both length-preserving and
un-ignorable.) The `masked_fill` arg lets a consumer who has already masked its
own observed vector request `0.0` instead.

Within a window, `sum_{j∈V_w} expected[c,j] = N_w[c] = sum_{j∈V_w}
observed[c,j]`, so the corrected residual `observed/expected` averages to 1 over
each window's support — the flatness property the CTCF pileup checks. This
identity holds over the EMITTED slice `V_w` (a trimmed final window's emitted
positions do not sum to a full-`TILE` total); the §4 HARD REQUIREMENT — single,
untrimmed, grid-aligned windows for any expected-count aggregation — is what
guarantees the pileup's `V_w` equals the full window support.

### 5.2 Per-fragment weights — interface (a)

**Relative rate & weight.** For track `c` in window `w`, the model's predicted
per-position rate is `probs[c,j]`; the uniform-null baseline rate over the
window's support is `1/L_valid`. The **predicted relative rate** is therefore

```
relrate[c, j] = probs[c, j] · L_valid
weight[c, j]  = 1 / relrate[c, j] = 1 / (probs[c, j] · L_valid)      (j ∈ V_w)
```

This equals `mean(pred)/pred` against the masked softmax, so it satisfies the
required analytic identity: a model with uniform logits gives `probs[c,j] =
1/L_valid` ⇒ `relrate = 1` ⇒ **weight = 1 everywhere unmasked**. (v1
provenance: the two v1 functions disagree with each other. The dead
`_from_pred_record` (:1875, `assert False`) computed `means.mean()/means`
(:1902) = `mean(pred)/pred` = `1/(L_valid·probs)` and assigned it DIRECTLY
(:1905) — the CORRECT v2 direction, INCLUDING the `L_valid` normalization. The
live `_from_weights_record` (:1940) instead applied `1./record[...]` on a
"pred_dist" vector = `1/probs`, DROPPING the `L_valid` factor. v2 uses the dead
function's formula: weight = `1/relrate = 1/(L_valid·probs)`, the
inverse-probability weight that flattens the corrected coverage.)

**Tiling contract (correctness invariant).** Weights are computed with windows
anchored at `region.start` and a FIXED `tile_size`; both `probs` (the
masked-softmax denominator) and `L_valid` are window-relative, so a fragment's
weight is a function of `(region, fragments, model, tile_size)`. Locked
guarantee: identical `(region, fragments, model)` at the same anchor and
`tile_size` ⇒ IDENTICAL weights, bit-for-bit — TEST-LOCKED (§6, T11,
two-callers-identical-weights). **Limitation:** because `probs·L_valid` is
window-relative, weights are NOT position-intrinsic — the same fragment tiled
under a different anchor/window receives a different weight. This is adequate for
the product (within-sample correction, where one consistent tiling is applied to
the whole sample), but weights MUST NOT be compared across regions tiled
differently.

**Which endpoint drives a fragment's weight, per coverage-type** — each fragment
receives up to three weights, one per coverage vector, each indexed by the
coordinate that coverage getter actually reads (§1.3), fixing v1's
`starts_0/stops_0` mismatch:

| coverage | weight attr set             | endpoint coordinate     | track selected                       |
|----------|-----------------------------|-------------------------|--------------------------------------|
| first    | `first_covered_base_weights`| `first_covered_bases_0` | `(strand_f, band_f, "first")`        |
| last     | `last_covered_base_weights` | `last_covered_bases_0`  | `(strand_f, band_f, "last")`         |
| midpoint | `weights`                   | `midpoints_0`           | `(strand_f, band_f, "midpoint")`     |

where `strand_f = rfa.fragment_strands[f]` and `band_f` is the unique band
`[lo,hi)` with `lo <= len_f < hi` (half-open, matching `subset_fragment_lengths`
:790), or **none**.

**Coordinate-frame precondition (invariant §0.3).** Before anything else the
applier asserts `rfa.region.strand in {None, '.', '+'}` AND `not rfa.is_flipped`,
raising otherwise with a message pointing to §0.3 / this section. (`Region`
normalizes `'.'` → `None` at `region.py:538-539`, so a `strand='.'` query is
observed here as `None`; accepting `{None, '.', '+'}` is what keeps the
strandless happy path from raising — a literal-`'.'`-only assert was wrong,
found during implementation and fixed in 0cc5f18.) This makes the
minus-strand `is_flipped` frame unreachable: `from_fragments_h5` flips
`starts_0/stops_0` into a reversed region-local frame and SWAPS `fragment_strands`
(+↔−) for a `−`-strand region (`fragment_array.py:1809-1819`, `:1843`), under
which `gpos = region.start + endpoint_coord` and `fragment_strands[f]` would both
be wrong. Correction always queries strandless (`.`) regions, so both stay in the
forward genomic frame; strand-oriented pileups flip at the CONSUMER aggregation
layer, never here (§0.3).

**S1 made impossible-by-construction.** Each fragment maps to **exactly one**
strand (its own) and **exactly one** band (bands are disjoint half-open
intervals), hence exactly one track per coverage-type. The applier never builds
an OR/AND boolean mask; instead it computes, vectorized over fragments:

```
assert rfa.region.strand in {None,'.','+'} and not rfa.is_flipped  # invariant §0.3 (Region: '.'→None)
band_idx[f]  = unique b with bands[b].lo <= len_f < bands[b].hi, else -1
               # explicit per-band half-open test — NOT bare searchsorted, which
               # would map an inter-band GAP length onto a neighbouring band
strand_ok[f] = fragment_strands[f] in {"+","-"}
# genomic endpoint → (window index, local position); off-grid → invalid
for cov in ("first", "last", "midpoint"):
    gpos   = rfa.region.start + endpoint_coord[cov][f]        # per fragment
    win, j = locate(gpos)                                      # window & local
    valid  = strand_ok & (band_idx >= 0) & mask[win][j] & in_grid
    # guard BEFORE the gather (F11): bands[-1] is python negative indexing —
    # it silently selects the LAST band for out-of-band fragments
    trk    = TRACK_INDEX[(strand_f, bands[band_idx], cov)] if band_idx >= 0 else NO_TRACK
    w      = 1 / (probs[win][trk, j] · L_valid[win]) where valid
    weight_attr[cov][f] = clamp(w) where valid else 0
```

Because `band_idx` and `strand_f` uniquely determine the track, no fragment is
ever touched by two tracks and no overwrite ordering exists. A fragment on the
matching strand but **out of band** (`band_idx == -1`) gets 0 — the exact case
v1's OR bug mis-weighted. TEST-LOCKED (§6, T4).

**Masked / off-grid endpoint handling: weight 0 (drop), keeping v1's policy.**
If a fragment's endpoint falls on a masked position (`mask[win][j]` false, where
`probs=0` so the weight would be `1/0`), or outside the tiled grid, that
coverage-type weight is 0. `drop_uncorrectable=True` then drops fragments whose
three weights are all 0 (the same `mask(weights>1e-6 | first>1e-6 |
last>1e-6)` filter v1 used, :1943). Justification: a fragment endpoint with no
model support cannot be bias-corrected; leaving it at weight 1 would inject
uncorrected signal into a "corrected" track, and leaving it at `inf` is
nonsense. This is consistent with the brief ("fragments with endpoints on masked
positions are dropped (weight 0), as in v1", brief §Model L45-46).

**Clamping hook — REQUIRED argument, no default (justified).** `WeightClampConfig`
is applied as the LAST step, and `apply_fragment_weights` REQUIRES it — there is
NO default, so a caller must consciously pick a clamp. Rationale: (1) the brief
lists clamping semantics as explicitly DEFERRED / out of scope, and a silent
no-clamp default would bake an un-approved algorithmic choice into scientific
code (per repo policy: algorithmic changes require approval); (2) an identity
default is not merely "deferred" — it is the maximally-UNSTABLE resolution,
because `weight = 1/(probs·L_valid)` is UNBOUNDED and a trained (peaky) model
drives valid-position weights arbitrarily large (§7 failure table; T5 itself
notes weights "→ large elsewhere"); (3) there is no single "v1 parity" value to
inherit — v1 shipped two contradictory clamps (`[0.5, max]` in `predict.py` vs
`[1e-6, 10]` via `from_fname`, review S6). `WeightClampConfig.identity()` is the
explicit opt-in for no clamp (and makes the analytic `weight == 1` test exact);
v1-parity is one line (`WeightClampConfig(1e-6, 10.0)`); the concrete clamp
semantics remain a deferred owner decision.

**Dispersion head contribution: none, this phase.** Both interfaces are pure
functions of `(probs, N)`. The dispersion head (`log_dispersion`) is unused —
weights and expected counts do not read it, and a `multinomial`-trained model
(which has no dispersion head at all) is fully supported. Gamma-based confidence
weighting of the correction is a listed FUTURE item (brief §Evaluation L106-108;
non-goal §8), not this phase. The correction therefore works identically for a
model trained under any of the three losses (it consumes only the shape head).

---

## 6. Test plan (fully exercisable with an UNTRAINED model)

Phase 2 machinery is statistical-quality-agnostic (that is Phase 3). Every test
below runs on an untrained or forced model. Suite baseline before this work:
143 passed / 0 skipped (`background_model` memory note, @ `e85ff0b`).

**Equivalence & geometry**

- **T1 (equivalence to Dataset).** Build a micro-store via `run_preprocess`
  (reuse the E2E fixture: 2 samples, tile_size override, real
  `tests/data/golden.*.frag.h5`). For every tile, assert
  `build_window_onehot(fasta, contig, tile.start, tile.stop, geom)` is
  `array_equal` to `BackgroundTileDataset(train_mode=False)[i][0].numpy()`, and
  `build_window_mask(...)` equals the center-crop of `tiles/mask[t]`. This is the
  single most important test — it proves the store-free inference input == the
  training input.
- **T2 (seam-free stitching).** For a random untrained model,
  `predict_region_profiles(model, fasta, c, a, a+2·TILE)` `probs` equals the
  concatenation of two standalone `predict_profile` calls on windows `[a, a+TILE)`
  and `[a+TILE, a+2·TILE)`, position-by-position (`allclose`, atol 0). Proves no
  cross-window leakage / renormalization.
- **T3 (coordinate mapping).** `RegionProfile.coord0 == start`; window_bounds
  tile `[start, stop)` with no gaps/overlaps; sub-grid `stop` trims correctly.

**Analytic cases (forced model)**

- **T4 (uniform ⇒ weights 1, expected N/L).** Force uniform logits (zero the
  `shape_head` weight+bias, or monkeypatch `forward` to return zeros). Then for a
  fragment array with in-band, unmasked, on-grid endpoints, `apply_fragment_
  weights` sets every weight to `1.0` (atol 1e-6); `expected_profile` gives
  `expected[c,j] == N_w[c]/L_valid` at valid j and `NaN` at masked j. This is the
  task's required analytic anchor.
- **T5 (delta ⇒ reciprocal).** Force all shape mass onto one position p
  (large logit at p): `weight` at p == `1/(1·L_valid)` and → large elsewhere;
  verifies the reciprocal and the `L_valid` normalization directly.

**S1 regression lock**

- **T6 (strand×band is AND + band membership + minus-strand refusal).** All on a
  STRANDLESS (`.`) region (invariant §0.3). Fragment array whose true
  `fragment_strands`/lengths are: (a) `+`, len 50 (band 0); (b) `+`, len 200
  (out of all bands); (c) `−`, len 130 (band 1); all endpoints unmasked/on-grid.
  Assert: (a) nonzero weights ONLY from `(+, (40,65), ·)` tracks; (b) weight 0
  for all three coverage types and dropped; (c) from `(−, (120,175), ·)` only.
  Under v1's OR logic (b) would be wrongly weighted and (a) overwritten by the
  `−` track — fails on v1 code, passes here.
  **Band-membership cases** (explicit `lo <= len < hi`, §5.2): lengths
  `39, 65, 100, 119, 175` each map to NO track (weight 0, dropped) — below band
  0, band 0's open upper edge `[40,65)`, the inter-band gap `[65,120)`, still in
  the gap, and band 1's open upper edge `[120,175)` respectively; lengths
  `40, 64, 120, 174` map to the in-band boundaries (band 0 lo, band 0 hi−1, band
  1 lo, band 1 hi−1) and receive nonzero weights. A bare `searchsorted` would
  mis-assign the gap/edge lengths to a neighbouring band — this locks the
  membership check.
  **Minus-strand-region refusal** (invariant §0.3): build the SAME fragments over
  a `−`-strand region via `from_fragments_h5` (`is_flipped=True`); assert
  `apply_fragment_weights` RAISES the frame precondition before any mapping, AND
  assert the strandless (`.`) path over the identical genomic span produces the
  correct genomic endpoint positions (`gpos = region.start + endpoint_coord`).

**Golden comparison vs v1 (only where v1 was correct)**

- **T7 (single-track reciprocal parity).** v1's OR/AND selection bug collapses
  when the strand/band selection is unambiguous. NOTE (implementation, 0cc5f18):
  this is NOT a literal single-`output_track` model — the applier gathers `probs`
  by `TRACK_INDEX[(strand, band, cov)]`, which spans the full 12
  `strand×band×coverage` tracks, so a 1-track model would `IndexError`. It is
  instead realized with the full **12-track** model driven by fragments that are
  all `'+'` / band-0, which pins the gather to one unambiguous track per
  coverage type. Assert our per-endpoint weight equals the reference reciprocal
  `1/(probs·L_valid)` computed inline from a fixed `probs` (no v1 import). This is a value check on
  the numeric core — the `mean/pred = 1/(L_valid·probs)` formula that v1's *dead*
  `_from_pred_record` already applied CORRECTLY (:1902; L3/§5.2) — in the one
  regime where the strand/band selection is unambiguous. It documents that the
  correct reciprocal is preserved, while v1's selection logic and the LIVE
  `_from_weights_record`'s dropped-`L_valid` error (`1/probs`, :1940) are fixed.

**Untrained-model E2E**

- **T8.** frag h5 + FASTA → `RegionFragmentArray.from_fragments_h5` over a real
  fixture region → `apply_fragment_weights` (untrained model) → `build_coverage_
  counts` runs finite and non-negative; `expected_profile` over the same region
  runs finite (NaN only at masked). One pass per loss type (`multinomial`,
  `dirichlet_multinomial`, `nb_offset`) to prove dispersion-head presence/absence
  is irrelevant to the correction.

**Property tests**

- **T9.** `sum_{valid j} expected[c,j] == N_w[c]` per window/track (flatness
  identity). Weights are strictly positive at valid in-band endpoints and exactly
  0 at masked/out-of-band ones. `WeightClampConfig.identity()` leaves weights
  unchanged; `WeightClampConfig(1e-6,10.0)` bounds them.
- **T10 (`locate` property test — top-risk 2 / self-grade c).** Directly exercise
  the genomic↔(window, local) mapping used by `apply_fragment_weights`:
  (i) random genomic positions in `[start, stop)` round-trip
  `gpos → (win, j) → win_start + j == gpos`; (ii) off-grid starts (region.start
  NOT a `tile_size` multiple) still map correctly (anchor = region.start);
  (iii) fragments whose first/last/midpoint endpoints STRADDLE a window boundary
  land in the correct distinct windows; (iv) a sub-grid `stop` (final window
  trimmed) maps in-range positions and rejects positions `≥ stop` as off-grid;
  (v) minus-strand-region REFUSAL — construct a `−`-strand region rfa
  (`is_flipped=True`) and assert `apply_fragment_weights` raises the invariant
  §0.3 frame precondition before any mapping.
- **T11 (tiling contract — two callers, identical weights).** Run
  `apply_fragment_weights(rfa, model, fasta, clamp=WeightClampConfig.identity())`
  twice on the SAME `(region, fragments, model, tile_size)` (anchor
  `region.start`); assert the three weight vectors are bit-for-bit identical
  (`array_equal`). Locks the §5.2 tiling contract (weights are a deterministic
  function of the tiling; window-relative, not position-intrinsic).

---

## 7. Failure modes

| Failure | Handling |
|---|---|
| Model/geometry mismatch (fetched seq len ≠ `model_input_size`) | `WindowGeometry.from_model` derives sizes from the model; `build_window_onehot` asserts output length == `model_input_size` before `predict_profile`. |
| Region near / past contig edge | N-pad sequence, mask out-of-contig positions (§4); never raises. |
| `N_w[c] = 0` (no observed counts in window/track) | `expected[c,·] = 0` at valid j (NaN at masked). Weights are **independent of N** (functions of `probs` only), so `apply_fragment_weights` is unaffected. Consumers dividing `obs/exp` see `0/0 = NaN` — correct "no data" signal. |
| All-masked window (`L_valid = 0`) | Softmax over empty support is undefined (`predict_profile` masked_fill of all `-inf` → NaN). Guard explicitly: if `mask.sum()==0`, skip `predict_profile`, set that window's `probs = 0`, `expected = NaN`, all weights 0. TEST-LOCKED. |
| Fragment endpoint off the tiled grid or on a masked position | weight 0 for that coverage type (§5.2); fragment dropped if all three are 0. |
| Trained (peaky) model + `WeightClampConfig.identity()` (unbounded weights) | `weight = 1/(probs·L_valid)` is UNBOUNDED; a near-zero `probs` at a valid position injects an enormous weight into the "corrected" coverage. `clamp` is a REQUIRED arg with NO default (§5.2), so the caller must opt into this explicitly via `identity()`; production callers pass finite bounds (e.g. `WeightClampConfig(1e-6, 10.0)`). Clamp semantics are a deferred owner decision. |
| `fragment_strands is None` (strandless rfa) | Cannot assign strand-split tracks; raise a clear error (the model's tracks are strand-specific). Documented precondition: `apply_fragment_weights` requires a stranded rfa. |
| Minus-strand / `is_flipped` rfa (built from a `−`-strand region) | Rejected up front: the applier asserts `region.strand in {'.', '+'}` AND `not rfa.is_flipped` (invariant §0.3) and raises with a message pointing there. Correction queries use strandless (`.`) regions so the frame is never flipped; strand orientation happens at the consumer aggregation layer. TEST-LOCKED (§6, T6 minus-strand-refusal case + T10 (v)). |
| Region shorter than one window | Single window, output trimmed to `[start, stop)`; N over the trimmed valid positions' window (§4). |
| Blacklist rdf not supplied | Mask = contig-clamp only (no blacklist zones). Documented; the caller is responsible for passing the same blacklist used in training for a faithful correction. |
| `contig_len is None` AND region runs past the true contig end | Right-clamping is disabled, matching Phase 1. `pysam.fetch` returns a short string → N-padded, but those past-contig positions are NOT masked invalid (no length to clamp against). The model then predicts on uniform-`[0.25, 0.25, 0.25, 0.25]` (N) one-hot columns there (`sequence.pyx:39-40`). Mitigation: pass `contig_len` (or `ref` so it can be looked up) whenever a region may reach a contig end; documented precondition. |

---

## 8. Explicit non-goals

- **Weight clamping semantics** — deferred (brief); the clamp is a REQUIRED
  argument (no default), `WeightClampConfig.identity()` is the explicit no-clamp
  opt-in, and the concrete bounds remain a deferred owner decision (§5.2, §7).
- **Gamma-based confidence weighting** — FUTURE; dispersion head unused this
  phase (§5.2).
- **Genome-wide batch runner / performance engineering** — this design is the
  per-region correctness core; batching, caching, and parallel sharding are
  separate.
- **Phase 3 evaluation** — the CTCF pileup execution, QQ-uniformity, and
  corrected-residual flatness measurement are consumers of these outputs, not
  part of this phase. (The machinery here must merely *run* on an untrained
  model; statistical quality is Phase 3.)
- **Reviving v1** (`bias_correction/`, `from_fname` `background_model` param,
  `SeqToEndpointsMultiResModel`) — reference only.
- **Changes to `background_model_core.py` or the store format** — out of scope.

---

## 9. Self-grade & top risks

**Self-grade: A−.** The design pins every interface against verified source
(exact `predict_profile` output contract, the coverage-getter coordinate table,
the S1 precedence bug, the Dataset val-mode crop that grounds the equivalence
guarantee), gives worked geometry arithmetic (124 margin, 16632 input, seam
abutment coordinates), makes S1 impossible-by-construction rather than merely
tested, and carries the masked-target contract into both N and weights.
Deductions: (a) the equivalence guarantee is argued from the crop being a
symmetric center-crop — correct in prose, but off-by-one-prone in code; T1 is
mandatory and must be written first. (b) The window **anchor** for
`apply_fragment_weights` (region.start) is a choice, not forced by the math;
different anchors put a position in a different window and thus under a different
window's shape. This is now LOCKED as a correctness invariant (§5.2 Tiling
contract: anchor `region.start`, fixed `tile_size`, identical inputs ⇒ identical
weights, T11), with the window-relative Limitation stated explicitly. (c)
`locate(gpos)` window/local mapping is new index arithmetic not present in Phase
1; it has its own property test (§6, T10) beyond T3.

**Top risks (ranked):**
1. **Equivalence off-by-one** between `build_window_onehot` and the val-mode
   Dataset input (margin parity, N-pad boundary). Mitigation: T1 written before
   the builder, byte-exact.
2. **Weight-application anchor/coordinate mapping** (`locate`, genomic→window
   local, off-grid detection) — the successor to S1's bug class. Mitigation:
   S1-lock T6 + the dedicated `locate` property test T10; index by the getter's
   own coordinate.
3. **All-masked / N=0 numerics** (softmax NaN, `0/0`) reaching a consumer
   unguarded. Mitigation: explicit guards + T9/failure-mode tests; NaN chosen so
   failures are loud, not silent.
4. **Anchor-dependent stitching for multi-window pileups** — if consumers tile
   inconsistently, per-window N/shape boundaries move. Mitigation: the tiling
   contract is now a LOCKED correctness invariant (§5.2, T11) — anchor
   `region.start`, fixed `tile_size`, weights window-relative (documented
   Limitation); expected counts compared only within a single window (§3.3).

---

## Review outcome (2026-08-28)

Independent adversarial review (fresh Opus, clean context) against source:
**verdict A−**, 0 Critical, 1 High, 3 Medium. All addressed in this revision:
- H1 (§3.2): corrected the receptive-field explanation — a single output
  position depends on RF=249 input bases, not `model_input_size`; the coordinate
  mapping follows from the symmetric 124-bp margin trim (conclusion unchanged and
  independently verified correct by the reviewer).
- M1 (§1.1): clarified the `2·(k−1)` trim term = initial conv + shape head.
- M2 (§1.4): added v1's closed-vs-half-open band-boundary inconsistency as a
  third latent bug the new design fixes.
- M3 (§2.1): noted `one_hot_encode_sequences` is the Cython `sequence.pyx`
  implementation re-exported by `region`.
- L3 (§5.2, T7): corrected the v1-provenance note — v1's dead `_from_pred_record`
  (:1902) already computed the CORRECT direction and `L_valid` normalization
  (`means.mean()/means = 1/(L_valid·probs)`, assigned directly at :1905); the
  LIVE `_from_weights_record` (:1940) is the one that dropped the `L_valid`
  factor (`1/probs`). [This r1 note was itself mischaracterized in the first
  pass and is corrected here per EM-gate F8; see Revision r2.]
- L4 (§4): documented that the flatness identity holds per full window, not per
  trimmed emitted slice.
- L5 (§7): added the `contig_len is None` near-edge failure mode.

The reviewer independently verified (I1–I11): all 19 line references, the
equivalence guarantee byte-path (`get_resize_start(0,20736,16632)=2052`
symmetric → genomic `[w0−124, w0+TILE+124)`), the coordinate mapping, seam-free
arithmetic, the weight-surface table, the S1 precedence analysis, `from_fname`
brokenness, all brief quotes and the masked-target contract, and full coverage
of the 8 required design points.

## Completion status

Design only, reviewed (A−, findings addressed). Next: on user approval,
implement `inference.py` + `correction.py` + tests — **T1 (equivalence to the
val-mode Dataset input) written first**, then T6 (S1 lock) and T4 (uniform ⇒
weights 1) before the appliers.

---

## Review Notes (EM gate) — 2026-08-28

Independent adversarial gate (fresh Opus, clean context, read-only on source;
this doc is the only file edited). Re-derived the equivalence arithmetic and the
weight math from source; verified the seven prior fixes; then dug where the
internal review did not. **This gate does not pass the design at A−.**

### Verdict: B+ — 1 High, 6 Medium, 3 Low. Below the A− implementation gate.

The design's core is genuinely strong: the equivalence guarantee is byte-exact
and correctly test-locked, S1 is impossible-by-construction, and the
getter-coordinate choice is not just "the getter multiplies weights" but is
confirmed correct against the *training targets* (the store's Phase-A counts are
themselves built from `build_coverage_counts` — `preprocess.py:211` — so the
model's "last" track was trained on `stops_0-1` positions, exactly what the
applier indexes). But the design ships a **High** strand-frame correctness gap
in the exact "successor-to-S1" mapping it claims to have neutralized, its own
#1-ranked successor risk (`locate`) has **no concrete test in the enumerated
plan**, the band→track mapping sketch is under-specified for the inter-band gap
the brief's half-open semantics create, and it carries a repeated factual error
about the encoder plus an explosive default clamp. These require a revision
round, not just verification.

### Findings

| # | Sev | Section | Finding | Evidence | Recommendation |
|---|-----|---------|---------|----------|----------------|
| F1 | **High** | §5.2, §7 | Minus-strand / `is_flipped` rfa is unhandled. `from_fragments_h5` FLIPS `starts_0/stops_0` into a reversed region-local frame and SWAPS `fragment_strands` (+↔−) for a minus-strand region, setting `is_flipped=True`. The applier's `gpos = region.start + endpoint_coord` (§5.2 L483) and `track = fragment_strands[f]` assume an unflipped forward frame, so for a minus-strand region every endpoint maps to the wrong genomic position and the wrong strand track. The flagship is a **strand-oriented** motif pileup, so minus-strand motif regions are first-class. No precondition, no test. | `fragment_array.py:1809-1819` (flip + strand swap), `:1843` (`is_flipped`); design never reads `rfa.is_flipped`; brief §PURPOSE L24-29 (strand-oriented pileup) | Add hard precondition `region.strand in {'.','+'}` with an assert, OR map `is_flipped` local→genomic explicitly; add a minus-strand-region test to the S1 lock. |
| F2 | Med | §6, §9c | The `locate()` property test — the design's own #1 successor risk (top-risk 2, self-grade c) — is promised in prose but ABSENT from the enumerated §6 plan (T1–T9 contain no `locate` test). The most-cited risk has no concrete spec. | §9c "needs its own property test beyond T3"; top-risk 2 "a dedicated `locate` property test"; §6 lists none | Add an explicit `locate` property test (random genomic↔(win,local) round-trip, off-grid, boundary-straddling, sub-grid `stop`, AND minus-strand region). |
| F3 | Med | §5.2 | Band→track via `searchsorted over disjoint bands` (L479) is under-specified for NON-contiguous bands. `(40,65)` and `(120,175)` leave a gap `[65,120)` and open ends; a length in the gap (e.g. 100), exactly `65`/`119`, or `<40`/`≥175` must yield `-1`, but `searchsorted` alone maps a gap value to a neighbouring band index. Needs an explicit `lo<=len<hi` membership check. T6 tests 50/200/130 but NO gap/boundary value — exactly the "neither band" case the task flags. | design L479 vs L470-471; `FL_BANDS` half-open `background_model_core.py:104`; `subset_fragment_lengths` half-open `fragment_array.py:788,790` | Specify per-band `lo<=len<hi` membership; add T6 cases at len 65, 100, 39. |
| F4 | Med | §4, §7 | "N one-hot-encodes to an all-zero column (no base)" is FALSE. The encoder maps `N→[0.25,0.25,0.25,0.25]`. Stated twice and used as justification. Equivalence itself survives (both paths share the encoder), but the model-input semantics are wrong; an implementer hand-rolling padding from the prose ("all-zero") could diverge, and §7 L615's "predicts on all-zero (N) columns" is likewise wrong. | design L379-380, L615 vs `fragmentomics_tools/sequence.pyx:39-40` (`'X'`/`'N'`→`[0.25,...]`) | Correct to "N→uniform 0.25 column, identically at train and inference"; keep the (correct) equivalence conclusion. |
| F5 | Med | §5.2, §8 | Identity-default clamp + unbounded `1/(probs·L_valid)` ships an explosive default. A trained/peaky model assigns near-zero probability to valid positions ⇒ enormous weights injected into "corrected" coverage (T5 itself notes weights "→ large elsewhere"). "Deferred" is better served by a REQUIRED explicit clamp arg than a silent no-clamp default; identity IS a resolution ("no clamp"), and it is the maximally-unstable one. | design L257-260, L507-515; T5 L560-561 | Make `clamp` a required arg (no default), or add unbounded-weight explosion to §7 failure modes with a caller warning. |
| F6 | Med | §3.3 | "Seam-free by construction" is true only for the SHAPE (`probs`). Expected COUNTS carry per-window `N_w` scale that is intentionally discontinuous at every tile seam; a multi-window expected track steps by `N_w0/N_w1` at each boundary. §5.1/§4-L4 are aware per-window, but the "seam-free" headline over-claims for interface (b). | design L348, L356 vs L433-435, L391-397 | State explicitly that expected counts are within-window normalized (seams carry `N_w` steps); confirm/require flagship regions fit one window. |
| F7 | Med | §5.2, §9b | Per-fragment weights are TILING-DEPENDENT: `weight=1/(probs·L_valid)` where both `probs` (masked-softmax denominator) and `L_valid` are over the window's support, so the SAME fragment gets different weights under different anchors/overlapping regions. Weights are not position-intrinsic. Self-grade (b) notes it but leaves it "not fully specified" — for flagship reproducibility/cross-site comparability this must be a locked contract, not a note. | design L444-445, L490-491; self-grade (b) L648-652 | Lock the tiling contract (anchor=`region.start`, fixed `tile_size`) as a correctness invariant; test that two callers on the same region get identical weights. |
| F8 | Low | §5.2 L452, revision L3 | The L3 "fix" is itself wrong and self-contradictory. Dead `_from_pred_record` computes `means.mean()/means` = mean/pred = `1/(L_valid·probs)` — the CORRECT weight direction, not "the reciprocal of the correct weight." Contradicts the doc's own T7 note (L579 "mean/pred reciprocal that `_from_pred_record` was structured around"). No impl impact (dead code), but the internal review's fix mischaracterizes source. | `dataframe.py:1902` (`weights = means.mean()/means`), `:1905` (assigned directly) vs design L452, L686-687 | Correct L3: `_from_pred_record` had the right direction/normalization (it MATCHES v2's formula); the live `_from_weights_record` is the one dropping the `L_valid` factor (`1./probs`, `:1940`). |
| F9 | Low | §1.4 | Overstates the v1 coordinate mismatch. `first_covered_bases_0==starts_0` and `midpoints_0==midpoints_0` are IDENTICAL to v1's coords; only `last` differs (`stops_0` vs `stops_0-1`, an off-by-one). Doc frames it as a blanket starts/stops-vs-first/last divergence across all three. | `fragment_array.py:771` (`stops_0-1`), `:775` (`starts_0`), `:779` vs `dataframe.py:1916`; design L150-152 | Narrow the claim to the `last` off-by-one. |
| F10 | Low | §6 | Stale baseline: cites "131 passed / 0 skipped (`background_model` memory note)". The memory note at HEAD `e85ff0b` records **143 passed / 0 skipped**. | memory `background_model.md` (143 passed, e85ff0b); design L531 | Refresh to 143. |

### Verified claims (spot-checked against source, file:line)

- `predict_profile` masked-softmax contract: `masked_fill(~mask,-inf)` then `softmax`, per-track sum-to-1 over valid positions, masked→0 — `background_model_core.py:669-683`. ✓
- Geometry: `calc_input_region_size` `:593-604`; defaults `kernel_size=32`, `num_residual_layers=2` `:513-514`; trunk = init conv + 2 dilated blocks (dil 2,4) + shape head, all `padding=0` `:560-575` ⇒ input 16632, **margin=124**; RF=249 checks out. ✓
- Track set: `DEFAULT_OUTPUT_TRACKS` nested strand×band×coverage, C=12 `:126-131`; `FL_BANDS=((40,65),(120,175))` `:104`; RC permutation `:134-156`. ✓
- Dataset val-mode: `j=0, do_rc=False` `dataset.py:258-260`; center-crop seq→`model_input_size`, mask→`TILE` `:200-207`; `y*=m` masked-zeroing `:222`. ✓
- **Equivalence re-derived byte-exact**: store fetch `[tile.start-2176, tile.stop+2176]`→L_SEQ, N-pad, `.upper()` `preprocess.py:483-503`; Dataset center-crop resize_start `(20736-16632)/2=2052` ⇒ x spans genomic `[tile.start-124, tile.start+TILE+124)`; direct fetch at that extent yields identical bytes incl. contig-edge N-pad. Matches the claimed `[w0-124, w0+TILE+124)`. T1 correctly locks it. ✓
- N definition: `compute_N_for_tile` = center-tile masked sum `(center_y*center_mask).sum` `store.py:184-188`. ✓
- Coverage surface: getters/coords + half-open `subset_fragment_lengths` `fragment_array.py:770-792, 818-850`; `build_coverage_counts` keyed (strand,band,cov) `:880-907`. ✓
- **Store targets share the applier coordinate**: Phase-A counts built via `rfa.build_coverage_counts(...)` `preprocess.py:211` — closes the loop the design only half-argues. ✓
- S1 bug: `&` binds tighter than `|` ⇒ mask=(in_band) OR (on_strand) `dataframe.py:1933-1935`; closed interval `<= ub` `:1933`; coord `starts_0/stops_0` `:1916`; drop filter `:1943`. ✓
- `from_fname` broken: forwards `background_model`/scaling/`include_fragment_strand`/`flip_...` `:1865-1874` that `from_fragments_h5` `:1708-1717` does not accept. ✓ `DEFAULT_MIN/MAX_SCALING_FACTOR=1e-6/10.0` `:27-28`. ✓
- `from_fragments_h5` auto-populates strands via `has_strand` `:1741,1793-1794` (so a stranded rfa is obtainable — but see F1 for the minus-strand-region flip). ✓
- **Genuine strength (verified sound):** per-coverage-type independent `locate` correctly handles a fragment whose first/last/midpoint endpoints fall in DIFFERENT windows or different masked states — each endpoint gets its own (window, track, mask). §5.2 pseudocode. ✓
- Brief fidelity: §"Correction outputs" quote verbatim `BACKGROUND_MODEL_BRIEF.md:56-63`; masked-endpoint→weight-0 drop `brief:45-46`; clamping DEFERRED `brief:63`; PURPOSE within-sample `brief:12-29`. No covert clamping resolution beyond the identity default (see F5). ✓

### Top 3 issues (ranked)

1. **F1 — minus-strand / `is_flipped` frame (High).** A real correctness hazard in the exact coordinate-mapping class the design claims impossible-by-construction, hitting the strand-oriented flagship, with no precondition and no test. Must be closed before implementation.
2. **F5 — explosive identity-default clamp (Med, high blast radius).** The default configuration can silently poison the corrected pileup with unbounded `1/probs` weights. Either require an explicit clamp or promote to a documented failure mode.
3. **F2 + F3 — the successor-risk mapping is under-tested/under-specified (Med).** The design's own #1 risk (`locate`) has no enumerated test, and the band→track mapping does not concretely handle the inter-band gap / half-open boundaries the task specifically called out.

### Note on the internal A− fixes
H1/M1/M3/L4/L5 verified correct. **L3 is wrong** (F8): the internal fix inverted the provenance and now contradicts T7. M2 is directionally right but overstated (F9).

### Revision r2 (post EM gate)

Each finding → one-line disposition (findings table above unchanged):

- **F1 (High)** — DONE. Added system-wide invariant §0.3 (correction uses
  strandless `.` regions, `from_fragments_h5` never flips); §5.2 applier asserts
  `region.strand in {'.', '+'}` AND `not rfa.is_flipped` pointing to §0.3;
  strand-orientation moved to the consumer aggregation layer; §7 row + T6
  minus-strand-refusal case added.
- **F5 (Med)** — DONE. `clamp` is now a REQUIRED arg (no default);
  `WeightClampConfig.identity()` is the explicit opt-in (docstring: unbounded
  weights with a trained model, semantics a deferred owner decision); §5.2 prose
  rewritten; §7 unbounded-weight-explosion row + §8 non-goal updated.
- **F2 (Med)** — DONE. Added T10 `locate` property test (round-trips, off-grid
  starts, boundary-straddling endpoints, sub-grid stop, minus-strand refusal).
- **F3 (Med)** — DONE. §5.2 pseudocode uses explicit per-band `lo <= len < hi`
  (NO-TRACK for gap/out-of-range); T6 extended with 39/65/100/119/175 (no track)
  and 40/64/120/174 (in-band boundaries).
- **F4 (Med)** — DONE. Corrected both occurrences (§4, §7): N → uniform
  `[0.25×4]` column (`sequence.pyx:39-40`), identical at train/inference;
  equivalence conclusion kept.
- **F6 (Med)** — DONE. §3.3 retitled "(of the SHAPE)"; added scope paragraph:
  expected counts are within-window normalized and step by `N_w` ratios at
  seams; flagship ±2 kb regions fit one window and require single-window
  evaluation for cross-position count comparisons.
- **F7 (Med)** — DONE. §5.2 tiling contract locked as a correctness invariant
  (anchor `region.start`, fixed `tile_size`, identical inputs ⇒ identical
  weights) with T11; window-relative Limitation stated; self-grade (b)/(c) and
  top-risks 2/4 updated.
- **F8 (Low)** — DONE. Corrected L3/§5.2/T7: dead `_from_pred_record` (:1902)
  had the CORRECT direction incl. `L_valid`; live `_from_weights_record` (:1940)
  dropped `L_valid` (`1/probs`); T7 wording reconciled.
- **F9 (Low)** — DONE. §1.4 narrowed to the `last` off-by-one (`stops_0` vs
  `stops_0-1`); `first`/`midpoint` noted identical.
- **F10 (Low)** — DONE. §6 baseline refreshed to 143 passed / 0 skipped
  @ `e85ff0b`.

### Re-review r2 (EM gate)

Verification of the fix round (fresh Opus, clean context, read-only on source;
this doc the only file edited). Every fix re-checked against source with
file:line evidence; searched for new contradictions the edits may have
introduced; confirmed the deferred clamping SEMANTICS were not quietly resolved.

**Verdict: A− — clears the implementation gate. 0 Critical, 0 High, 0 Med; 1
Info (F11). All ten r1 findings FIXED and source-grounded; no fix introduced a
new error.**

#### Per-finding re-verification

| # | r1 Sev | r2 Verdict | Section | Source evidence |
|---|--------|-----------|---------|-----------------|
| F1 | High | **FIXED** | §0.3, §5.2, §7 | Flip gated on `region.is_minus_strand()` `fragment_array.py:1809`; strand-swap `:1816-1819`; `is_flipped=region.is_minus_strand()` `:1843`; `is_minus_strand()` true only if strand set AND `== "-"` `region.py:789-791`; `self.is_flipped` attr exists `:282`; independent flip op toggles `is_flipped` + swaps strands `:706-719` — so a `'+'` rfa CAN be `is_flipped=True`, making the dual assert (`region.strand in {'.','+'}` AND `not is_flipped`) NON-redundant and correct. Invariant matches Phase A: `preprocess.py:163` "CRITICAL: strand='.' ALWAYS", `:185` `strand="."`, `:188`. §0.3/§5.2/§7 asserts are byte-consistent. |
| F2 | Med | **FIXED** | §6 T10 | New T10 enumerates round-trip, off-grid starts, boundary-straddling endpoints, sub-grid `stop`, and minus-strand refusal. Successor-risk mapping now has a concrete spec. Numbering clean (T1–T11, no collisions). |
| F3 | Med | **FIXED** | §5.2, §6 T6 | `FL_BANDS=((40,65),(120,175))` `background_model_core.py:104` (half-open, non-contiguous gap `[65,120)`). Pseudocode now uses explicit per-band `lo<=len<hi`. Re-derived every T6 boundary: NO-track 39/65/100/119/175 and in-band 40/64/120/174 all correct against half-open bands + `subset_fragment_lengths` `< max` `fragment_array.py:790`. |
| F4 | Med | **FIXED** | §4, §7 | `'N':[0.25,0.25,0.25,0.25]` `sequence.pyx:40` (and `'X'` `:39`). Both prior "all-zero" claims corrected to uniform-0.25; equivalence conclusion retained. |
| F5 | Med | **FIXED (deferral preserved)** | §2.2, §5.2, §7, §8 | `clamp: WeightClampConfig` is REQUIRED (no default) — valid Python (keyword-only, may follow defaulted keyword-only args). `identity()` → `(None,None)` → `apply` returns `w` unchanged. §7 explosion row present. Confirmed this only supplies MECHANISM + forces a conscious choice; it does NOT resolve clamp semantics (no covert algorithmic decision) — correctly stays within the deferred scope, does not overstep the owner call. |
| F6 | Med | **FIXED** | §3.3 | Retitled "(of the SHAPE)"; scope paragraph states expected COUNTS step by `N_w0/N_w1` at every seam and flagship ±2 kb (4001 bp) fits one `TILE` (16384). Consistent with §2.1 docstring ("Seam-free in the SHAPE… COUNTS carry per-window N_w") and §5.1. |
| F7 | Med | **FIXED** | §5.2, §9, §6 T11 | Tiling contract locked (anchor `region.start`, fixed `tile_size` ⇒ bit-identical weights) + window-relative Limitation; T11 two-callers test added; self-grade (b)/(c) and top-risks 2/4 updated coherently. |
| F8 | Low | **FIXED** | §5.2, T7 | Dead `_from_pred_record`: `assert False` `dataframe.py:1876`; `means.mean()/means` `:1902` (= `(1/L)/pred = 1/(L_valid·probs)`), assigned DIRECTLY `:1905` — CORRECT direction incl. `L_valid`. Live `_from_weights_record`: `1./weights[...]` on `pred_dist` `:1936,:1940` = `1/probs`, drops `L_valid`. Doc now states both correctly; T7 wording reconciled. |
| F9 | Low | **FIXED** | §1.4 | `last_covered_bases_0 = stops_0-1` `:771`; `first_covered_bases_0 = starts_0` `:775`; `midpoints_0` `:779`; v1 last coord `stops_0` `dataframe.py:1916`. Claim narrowed to the `last` off-by-one; first/midpoint noted identical. |
| F10 | Low | **FIXED** | §6 | Baseline now "143 passed / 0 skipped @ e85ff0b", matching the memory note. |

#### No new contradictions introduced
- §0.3 invariant ↔ §5.2 pseudocode assert ↔ §7 row: all three read
  `region.strand in {'.','+'}` AND `not rfa.is_flipped`, consistent.
- Test numbering T1–T11 unique; F2→T10, F7→T11; T6/T9 extensions self-consistent.
- §3.3 SHAPE-only seam wording agrees with §2.1 docstring and §5.1.
- Required-`clamp` signature is legal Python (keyword-only w/o default after
  defaulted keyword-only args).
- `'+'`-region-can-be-`is_flipped` (via `:706-719`) makes the dual assert
  non-redundant — the fix is stronger than a bare strand check, not weaker.

#### New finding

| # | Sev | Section | Finding | Evidence | Recommendation |
|---|-----|---------|---------|----------|----------------|
| F11 | Info | §5.2 pseudocode | The vectorized sketch computes `trk = TRACK_INDEX[(strand_f, bands[band_idx], cov)]` UNCONDITIONALLY. For an out-of-band fragment `band_idx == -1`, `bands[-1]` is Python negative indexing → silently the LAST band `(120,175)`, not an error. Harmless in the sketch (result discarded by `valid = … & (band_idx>=0) & …`), but a latent trap: a naive vectorized port that gathers `probs[…, trk, …]` before masking, or that later reuses `trk`, would mis-attribute out-of-band fragments to band 1. | design §5.2 L559–561; `FL_BANDS` `background_model_core.py:104`; `valid` gate L558 | In the impl, clamp/guard `band_idx` before the track gather (e.g. skip `band_idx<0` rows, or use a sentinel track index) rather than relying on `bands[-1]` + downstream masking. Pseudocode-level only; no design-decision impact. |

#### Verified fresh against source (r2 spot-checks)
- `fragment_array.py`: flip/swap/`is_flipped` `:1809-1819,:1843`; `is_flipped`
  attr `:282`; reverse op `:706-719`; getters `:771,:775,:779`;
  half-open `subset_fragment_lengths` `:788,:790`. ✓
- `region.py:789-791` `is_minus_strand` semantics. ✓
- `sequence.pyx:39-40` N/X → uniform 0.25. ✓
- `dataframe.py`: dead-fn `:1876,:1902,:1905`; live-fn `:1916,:1933-1935,:1936,
  :1940,:1943`. ✓
- `background_model_core.py:104` FL_BANDS; `:126-131` track nesting (C=12). ✓
- `background_model/preprocess.py:163,:185,:188` Phase-A strandless invariant. ✓

**Grade: A−.** The fix round is clean and source-grounded; the High (F1) is
genuinely closed by an invariant that matches Phase A, the successor-risk mapping
now has concrete tests (T10/T11) and a source-correct band-membership check, and
the deferred clamping semantics are preserved (required arg + `identity()` opt-in,
no covert resolution). Only an Info-level pseudocode nit (F11) remains — an
implementation-note, not a design gap. Cleared for implementation, with T1/T6/T4
first as the doc already prescribes and F11 folded into the applier port.

### Implementation review r1 — grade A− @ `0cc5f18`

Review of the shipped `inference.py` + `correction.py` (the 12-track applier +
`expected_profile`) against this design. Gate cleared. Four findings; all fixed
in the Scope A commit `d24ac43` (code + tests) and reconciled into this doc in
the companion Scope B commit.

| # | Sev | Finding | Fix |
|---|-----|---------|-----|
| 1 | Med | `correction.py` `elig` omitted the strand gate: an in-band/in-grid/unmasked fragment whose strand ∉ `{+,-}` kept `trk = -1` and silently gathered `probs_w[-1]` (the LAST track) — the F11 guard-before-gather class, on the strand axis. | `elig` now ANDs `strand_ok = (strand ∈ {+,-})`, so such a fragment gets weight 0 on every coverage type; neighbours unaffected. Negative test added (directly-built rfa, strand `'.'`). |
| 2 | Low | `fl_bands` default duplicated the literal `((40,65),(120,175))`. | Default now derives from `preprocess.FL_BANDS` (single source; a divergence fails loud). |
| 3 | Low | No explicit precondition that the model carries all 12 tracks; a smaller model would `IndexError` deep in the gather. | `assert len(model.output_tracks) == len(TRACK_INDEX)` up front, pointing at the 12-track constraint. |
| 4 | Low | The minus-strand refusal tests set `is_flipped` via the constructor kwarg, not the real `from_fragments_h5`/`reverse_strand` path. | Added a refusal test that builds the rfa via `from_fragments_h5` over a `−`-strand region (`is_flipped=region.is_minus_strand()`, `fragment_array.py:1843`) and asserts the applier refuses it. |

**Design divergences reconciled here (verdicts final):**
- **§0.3 + §5.2 frame precondition** corrected to `region.strand in {None, '.',
  '+'}` — `Region` normalizes `'.'` → `None` (`region.py:538-539`), so the
  literal-`'.'`-only assert was wrong and broke every happy-path call (fixed in
  `0cc5f18`). The `is_flipped` half is unchanged.
- **§4 + §5.1 expected-profile `N`** computes over the EMITTED/TRIMMED slice, not
  the full window — full-window `N` is unrealizable through the interface
  (`observed_counts` spans only `[start, stop)`). The flatness identity holds
  over the emitted slice. Added a HARD REQUIREMENT: Phase-3+ consumers
  aggregating expected counts (e.g. CTCF pileup) MUST use single, untrimmed,
  grid-aligned windows.
- **§6 T7** is realized with a full 12-track model (all-`'+'`/band-0 fragments,
  unambiguous selection); a literal single-track model is incompatible with
  `TRACK_INDEX`.

Suite after Scope A: **186 passed / 0 skipped** (was 184).
