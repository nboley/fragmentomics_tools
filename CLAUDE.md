# CLAUDE.md — fragmentomics_tools

Conventions and traps for anyone (human or agent) working in this repo.
Everything below was verified against the code; line numbers rot, so re-check
those, but treat the *rules* as binding unless the owner says otherwise.

## Environment & tests

- Test/run env: `/home/nathanboley/miniconda3/envs/biomarker_env/bin/python`
  (torch 2.5.1 CUDA-12.4 build, lightning, zarr 2.18.3, numcodecs 0.13.1).
- Run the suite from the repo root: `python -m pytest tests/ -q`.
  Baseline **447 passed, 0 skipped** (measured 2026-09-25 @ f6c5246). A few
  `self.log()`-without-Trainer warnings are expected and harmless.
  This number moves with almost every commit, so **measure it yourself before
  and after your change** rather than quoting this line — it said 196 for long
  enough that the gap to reality reached 251 tests, which makes it useless as
  the regression check it exists to be.
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
| Intersections | `.intersect_with_rdf(other)` |
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
- **Minus-strand regions arrive flipped.** `from_fragments_h5` reverses
  coordinates and swaps strands for minus-strand regions (`is_flipped`). The
  correction applier deliberately *refuses* flipped/minus input: query
  strandless, then orient at the aggregation layer (reverse the position axis
  and permute tracks). Getting this wrong silently destroys strand asymmetry.
- **`SparseIntVector` is a misnomer** — only `coords` are ints; `data` keeps
  its dtype and densifies as `values.dtype`, so fractional correction weights
  survive. Do not "tidy" it to match its name; that would floor every weight
  < 1 to zero and make corrected pileups quietly wrong. See its docstring.
- **fl bands are half-open** `[lo, hi)`: `(40, 65)` captures 40–64.
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
