# CLAUDE.md — fragmentomics_tools

Conventions and traps for anyone (human or agent) working in this repo.
Everything below was verified against the code; line numbers rot, so re-check
those, but treat the *rules* as binding unless the owner says otherwise.

## Environment & tests

- Test/run env: `/home/nathanboley/miniconda3/envs/biomarker_env/bin/python`
  (torch 2.5.1 CUDA-12.4 build, lightning, zarr 2.18.3, numcodecs 0.13.1).
- **Run the suite via `make test`, never bare `pytest`.** It wraps the run in
  `timeout --signal=KILL 3600`. This suite can *wedge* rather than fail: a
  fork deadlock in `parallel_apply` once ran 12 hours unnoticed, emitting
  nothing at all, because `pytest -q | tail` never reaches EOF if the process
  never exits. Override with `make test TEST_TIMEOUT=7200`, and select a
  different suite with `make test PYTEST_ARGS="tests/ -q"`.
  A kill shows as exit 137 and means it HUNG — that is a finding to diagnose
  (`py-spy dump --pid <pid>`), not a flake to re-run.
- Two suites exist and are easy to confuse: `test/` is the library suite
  and `tests/` is `background_model`. **The default `make test` collects
  both** (`PYTEST_ARGS` is `test/ tests/ fragmentomics_tools/`), plus the
  `fragmentomics_tools/` doctests.
  Last measured baselines, both PRE-DATING the 2026-10-09 merge of main into
  the simulator branch, so re-measure: `test/ fragmentomics_tools/` alone gave
  **2 failed / 398 passed / 3 skipped / 0 xfailed**, total 403 (2026-10-07, at
  the autoflip-removal commit); the default target on the simulator branch gave
  **2 failed / 1009 passed / 3 skipped** (2026-10-09, before the merge).
  **Reconcile the TOTAL first** when two measurements disagree: it is
  invariant under environment, so a matching total means you are looking at an
  environment difference while a differing total means tests went uncollected.
  Both failures are missing data, not defects: `test_slice_encode_big_wig`
  needs an ENCODE bigwig, `test_get_one_hot_encoded_sequence` needs the
  in-package GRCh38 reference; the 3 skips are Region doctests needing the
  optional `fbio`. A few `self.log()`-without-Trainer warnings in `tests/`
  are expected and harmless.
  These numbers move with almost every commit, so **measure them yourself
  before and after your change** rather than quoting this line — the `tests/`
  figure sat at 196 long enough that the gap to reality reached 298 tests,
  which makes a stale baseline useless as the regression check it exists to be.
- **`make test` also runs `--doctest-modules`.** Docstring examples are tests.
  Turning this on found three LIVE defects that the suite and a five-reviewer
  static pass had all missed, so treat a failing doctest as a real signal
  rather than doc drift. Two packages are excluded because they import
  optional deps (`datamanifest`, `fbio`) missing from the test env.
- **Do not pin a `np.random.default_rng` draw in a doctest.** NumPy only
  guarantees stream stability for the legacy `RandomState`; `Generator`
  streams may change between releases, and one doctest already broke that
  way. Assert the invariant (count, membership) instead of the exact draw.
- Run the suite **before and after** any change. A new test that fails against
  production code is a *finding to report*, not something to patch away.

## Use the library — do not hand-roll genomic logic

`dataframe.py` / `region.py` / `fragment_array/` already implement the
interval, region and fragment machinery. Reimplementing it by hand is a
recurring failure mode here: it produces two implementations of one rule that
are free to drift apart at boundaries (half-open vs inclusive, padding,
strand), and the divergence fails *silently*.

Sanctioned entry points:

| Need | Use |
|---|---|
| Load a BED | `RegionDataFrame.from_bed(path, ref=...)` — `ref` is **required** |
| Merge several BEDs | `RegionDataFrame.from_beds_merged(...)` |
| Blacklist / exclusion filtering | `.drop_overlapping_regions(other_rdf)` |
| Join on overlap | `.join_on_overlap(other)` — returns whole A intervals, NOT geometric intersections |
| Resize / pad regions | `.expand_regions(...)`, `.resize_regions(...)` |
| Attach fragments to regions | `SampleAndRegionDataFrame.attach_fragment_arrays(...)` |
| Per-region fragment loading | `RegionFragmentArray.from_fragments_h5(...)` |

**Legitimate escape hatch, with a condition:** some interval ops go through
`pybedtools`, which needs the `bedtools` binary on `PATH`. It ships in the
conda env's `bin/` but is frequently *not* on `PATH` in sandboxes and AWS Batch
containers, and this has caused real failures. If that forces a hand-rolled
fallback, **write down why in the code** — an unexplained divergence reads as
an accident and gets "fixed" later by someone without the context.

## Superseded code — do not build on it

`fragmentomics_tools/bias_correction/` is **v1 and superseded**. See
`BIAS_CORRECTION_REVIEW.md` for the full findings. Known-broken or misleading:
targets merged across samples (destroys the between-sample variance the model
needs), hardcoded `total_count=1000`, magic NB constants, `.cuda()` calls
inside `Dataset.__getitem__`, and hardcoded `/scratch/...` and `/home/nboley/...`
paths that no longer exist.

The live replacement is `background_model_core.py` plus the `background_model/`
package (config, store, preprocess, dataset, inference, correction), developed
on branch `background-model-v2` with designs in `docs/`.

## Frozen statistical core

`background_model_core.py` is the statistical specification — its module
docstring is authoritative. **Changes to statistical semantics require explicit
owner approval**: the likelihoods, masking rules, per-count loss normalization,
the `log_dispersion_init` offset, and the mask/`lgamma` guard ordering. Consume
its primitives (`jitter_matrix`, `reverse_complement_track_permutation`,
`predict_profile`, the losses) rather than reimplementing them.

Engineering work — plumbing, config, containers, tests, I/O layout — is fine
without a sign-off. Anything that changes computed results is not.

## Writing PyTorch / Lightning code

`docs/pending/pytorch_canonical_research.md` is a 21-rule reference compiled
from official docs, community style guides and published agent rulesets, each
rule carrying a source URL and a bad/good contrast. It is scoped to a *research*
codebase — rules justified only by production-serving concerns are marked as
such, so don't import them here. It is also explicit about its own gaps
(couldn't verify the Lightning style-guide page; most `torch.compile` guidance
targets 2.6+ while we pin 2.5.1; sources conflict on the `weights_only`
default), so treat it as a strong prior, not scripture.

Worth reading before writing model or training code, and worth checking a review
against. Its Tier-1 "silent correctness" rules are the ones that matter: applied
to this repo they found a real latent bug — `predict_profile` left the module in
`eval()` with nothing restoring it, which `BatchNorm1d` makes mode-sensitive
*regardless of dropout rate*, so the `--dropout 0.0` convention would not have
saved us. Fixed in `e8b7ca5`. The other Tier-1 rules passed, which is the more
useful half of the result: `.cpu().numpy()` only inside inference entry points,
and `4 ** torch.arange(...)` inside `register_buffer` rather than rebuilt per
forward.

## Known traps (each cost real debugging time)

- **`RegionFragmentArray.from_fname` is broken** — it forwards kwargs the
  callee does not accept and raises `TypeError`. Use `from_fragments_h5`.
- **`Region(strand=".")` normalizes `.strand` to `None`.** Asserting
  `strand == "."` therefore fails on the ordinary strandless path. Accept
  `{None, ".", "+"}`.
- **Minus-strand regions are NOT flipped on construction.** `from_fragments_h5`
  always returns data in genomic order with `is_flipped=False`, regardless of
  the region's strand. Orientation is deferred to the consumer layer via
  `reverse_strand()` or `make_data_direction_match_strand()`. The correction
  applier requires unflipped input — query strandless or plus-strand, then
  orient at the aggregation layer.
- **`_switch_plus_with_minus_and_minus_with_plus` compares `str` against
  `|S1` bytes.** The `from_fragments_h5` call site that used it was removed
  (autoflip removal), but the function survives for `reverse_strand()` and
  `strand_bias.py`. Those callers receive `dtype="U1"` arrays (normalised by
  `__init__`), so the string comparison works. If you add a new caller that
  passes raw h5 byte data, the comparison will silently match nothing.
- **`SparseIntVector` is a misnomer** — only `coords` are ints; `data` keeps
  its dtype and densifies as `values.dtype`, so fractional correction weights
  survive. Do not "tidy" it to match its name; that would floor every weight
  < 1 to zero and make corrected pileups quietly wrong. See its docstring.
- **`reverse_complement_track_permutation` is NOT dead code** — `ctcf_pileup_run.py`
  uses it to orient minus-strand CTCF sites, so do not delete it when refactoring
  the track models.
- **fl bands are half-open** `[lo, hi)`: `(40, 65)` captures 40–64.
- **Never identify tqdm's monitor thread by name.** `TMonitor` is named
  `tqdm_monitor` only from tqdm **4.69.0**; the pinned env has 4.67.1, where it
  is an anonymous `Thread-N`. Two `parallel_apply` guard tests asserted the
  name and so failed here while passing wherever they were written. Use the
  `monitor` attribute tqdm stores on the class that built the bar. Related:
  tqdm creates a new monitor only when the inherited one is absent or dead, so
  a subclass shares the base's unless you clear it first.
- **Store/zarr pinning**: the zarr store is v2 format; `zarr==2.18.3` with
  `numcodecs==0.13.1`. Newer numcodecs privatized symbols zarr 2.18 imports,
  which breaks at import time — pin both together.

## Working agreements

- Never commit `COORDINATION.*.md` / `STATUS.*.md` (gitignored per-worktree
  process state) or stray `*.csi` index files.
- Large artifacts (stores, checkpoints, pileup arrays) live on EFS or S3 and
  are never committed.
- **Analysis figures ARE committed** (owner decision, 2026-09-24), narrowing
  the rule above, which previously listed plots as never-committed.
  `docs/pending/*.md` embed their figures by relative path, and an uncommitted
  figure means every one of those image references is broken for anyone who
  has only the repo — the analysis doc alone carries 11. Scale is modest:
  `docs/pending/training_analysis_plots/` is ~2.7 MB for 18 PNGs.
  Still never committed: training runs' own output (`lightning_logs/`,
  `checkpoints/`, per-epoch metrics), which belongs on EFS under the run
  directory. The distinction is *curated figure that a document references*
  versus *raw artifact a run emitted*.
