# Background Model v2 — Consolidated Brief

Status date: 2026-08-25.  This document is the single source of truth for the
project's purpose and locked decisions.  It supersedes the conversational
history that produced it.  Companion files:

- `BIAS_CORRECTION_REVIEW.md` — critical review of v1 (`fragmentomics_tools/bias_correction/`), grade D
- `background_model_core.py` — v2 core model (repo root; its docstring records the statistical design)
- `tests/test_background_model_core.py` — 37 passing tests (env: `biomarker_env`)
- `COORDINATION.fragmentomics_tools.md` — process state (gitignored, not project doc)

## PURPOSE — do not drift from this

A **within-sample background model to regress away technical effects.**
Sequence-driven fragmentation bias is predicted from DNA sequence alone and
divided out of a single sample's observed fragment data.  The corrected
signal is what downstream fragmentomics analyses (TF footprinting, v-plots)
consume.  Genetic variants and active regulatory processes are the residuals
we want to PRESERVE — they must never be absorbed into the null.

This is NOT a cohort-level disease test.  Deviation tests / QQ calibration
are validation machinery for the correction, not the product.

**Flagship acceptance test: the CTCF motif pileup.**  Aggregate,
strand-oriented fragment-endpoint / coverage profiles centered on CTCF motif
sites (v1's `build_ctcf_rdfs` test set), before vs after correction.  The
corrected pileup should show the footprint + flanking nucleosome phasing
cleanly, with sequence-bias artifacts removed; away from motifs the
corrected residual should be flat.

## Locked decisions

### Model (implemented in background_model_core.py)
- One conv trunk (unpadded, dilated ResNet blocks), two heads: shape
  (per-bp logits, masked softmax per tile) + dispersion (per-window).
- Three switchable losses: `multinomial` (baseline), `dirichlet_multinomial`
  (exact, tile-level gamma), `nb_offset` (pseudo-likelihood, 256bp-window
  dispersion).  DM and offset-NB are near-equivalent (conditional
  factorization); BOTH are implemented and will be compared empirically.
- Observed N per (sample, tile, track) is the plug-in scale (conditioning /
  offset).  The network never predicts absolute counts (v1's fatal flaw).
- Dispersion is sequence-indexed so it generalizes to unseen regions;
  locus-indexed empirical dispersion was REJECTED (absorbs biology).
- Per-sample training targets (merged counts carry no variance info).
- Blacklist positions masked out of likelihood AND totals; fragments with
  endpoints on masked positions are dropped (weight 0), as in v1.
- Track naming: `strand_{s}__fl_{lo}_{hi}__coverage_{first|last|midpoint}`,
  strands +/-, fl bands (40,65) and (120,175).  Legacy names dead.
- Augmentation: jitter (stored margin, `jitter_matrix` at load) +
  reverse-complement (`reverse_complement_track_permutation`).  Both ON for
  training.
- log_dispersion_init = 7.0 (near-multinomial at init).
- SpatialDropout = Dropout1d on (B, C, L) (v1's Dropout2d dropped positions —
  approved bug fix).

### Correction outputs (the product) — to be designed/implemented
- BOTH interfaces: (a) per-fragment weights attached to RegionFragmentArray
  (1 / predicted relative rate at each fragment's start/stop/midpoint, per
  strand x band track — the fixed version of v1's
  `_set_fragment_array_weights_from_weights_record`, whose strand-mask OR/AND
  bug is documented in BIAS_CORRECTION_REVIEW.md S1), and (b) per-position
  expected-profile vectors for consumers that divide themselves.
- Weight clamping semantics: DEFERRED (explicitly out of scope for now).

### Data plumbing (requirements final; design not started)
- Two stages: one-time resumable preprocess -> zarr store; thin Dataset reads.
- Store: sequence as byte tokens with margin (jitter 128 + max receptive
  field); per-sample per-track bp counts, SPARSE (dense is ~600GB at scale,
  sparse ~2.5-12GB); blacklist mask; cached N totals; keyed by config hash
  (region_set_ver, sample_set_ver, fl_bands, tile_size, margin, filters).
- Geometry: 16,384 bp tiles / +-128 bp jitter margin / 256 bp genomic
  dispersion windows (fl bands are tracks, not windows).
- min-N threshold: 50 per (sample, tile, track), config, sweep later.
- Scale target: ~50 samples x ~5,000 tiles initially.
- Splits: samples ~40 train / ~10 held-out (SIMPLE RANDOM — flowcell
  stratification explicitly rejected); regions ~80/10/10
  train/val/held-out-inactive; positive controls (CTCF, marker genes) fully
  excluded from training.  Assignments frozen in the store manifest.
- Samples: IBD cohort manifest
  `~/src/biomarker-projects/tf_binding_site_classification/projects/ibd/manifests/ibd.data_manifest.tsv`
  (DataManifest v3, 763 frag h5s, external S3 records, notes JSON carries
  round/experiment/assay/seqrun/disease).  Access via
  `DataManifest(path).sync_and_get(key).path`; sync before preprocess, not
  inside workers.  DataManifest integration stays OUT of this layer — caller
  provides a sample sheet.
- Sample sheet: 4-5 columns only (sample_id/library, h5 key, seqrun
  [informational], endo_category).  Quiescent pool = ENDO_CATEGORY in
  {Asymptomatic, Remission} (one-column join from the pooled clinical CSV;
  the full clinical join is downstream's business;
  `scripts/build_ibd_metadata.py` is stale — do not reuse).  Quiescent-only
  training is a config default, not a pillar (correction is within-sample).
- Depth filter: drop unusually low-fragment-count samples (config, default
  on).  Fragment source: frag h5s via existing `fragment_array` machinery,
  guarded by ONE golden test (fragment_array counts vs independent pysam
  count on a few small regions).
- Package layout: stays loose in this repo for now; moves into biomarker
  repo once stable and tested.

### Evaluation plan (after plumbing)
1. Bake-off: train all three losses on the same store; on held-out samples x
   held-out inactive regions, QQ-uniformity of window p-values
   (`beta_binomial_window_pvalues` / `nb_window_pvalues`), per track,
   stratified by accessibility class.  Corrected-residual flatness
   (obs/expected ~= 1) at held-out inactive regions.
2. CTCF motif pileup before/after correction (flagship, see PURPOSE).
3. gamma's eventual role: confidence weighting of the correction (ties into
   the deferred clamping decision).

## Open items
1. "Inactive" training-region definition (user: later).  v1 precedents:
   expression < 0.1 genes; DHS minus CTCF.
2. Weight clamping semantics (deferred).
3. Exact quiescent-pool size (unknown until the one-column join runs).
4. Where background_model_core.py finally lives.

## Next steps (in order)
1. Design agent: data plumbing (preprocess -> zarr store -> Dataset) per the
   locked requirements above.  Design doc, review to A-, then implement.
2. Implement correction outputs (fragment weights + vectors) with the S1 bug
   fixed and tested.
3. Run bake-off + CTCF pileup.

## Practical notes for a fresh session
- Nothing is committed: background_model_core.py, tests/, conftest.py,
  BIAS_CORRECTION_REVIEW.md, this file are all untracked on main.  Decide
  worktree/commit strategy early.
- Test env: `conda activate biomarker_env`;
  `python -m pytest tests/test_background_model_core.py` (37 tests, ~20s).
- v1 package `fragmentomics_tools/bias_correction/` is untouched legacy —
  reference only, do not maintain.
