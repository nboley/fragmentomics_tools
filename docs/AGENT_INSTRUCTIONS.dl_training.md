# Agent Instructions: DL Training & Debugging (Background Model v2)

Operational playbook for agents that train or debug the background model
(`background_model_core.py`).  Inject Section 4 verbatim into every
training/debug agent prompt; Sections 1–3 are the reference it points at.

Provenance: compiled from the project code + standard DL-debugging practice
(Karpathy's training recipe, PyTorch Lightning debugging docs).  A web pass
for published ML-agent prompt collections was deferred (spawns rate-limited);
if done later, append findings — do not rewrite this file.

Environment: `conda activate biomarker_env`; torch 2.5.1, lightning 2.6.x.
CPU smoke tests: `python -m pytest tests/test_background_model_core.py` (37
tests, ~20s — run them FIRST after any change to the core file).

---

## 1. Sanity ladder — run in order; no long run starts until all pass

**L0. Unit tests pass** on the current working tree (command above).

**L1. Geometry round-trip** (already a unit test; re-verify with the real
config): `model(x).shape[-1] == L_out` for `x` of length
`calc_input_region_size(L_out)`.  Off-by-margin here silently shifts
sequence↔target alignment — see P2 in the playbook.

**L2. Loss at init ≈ theoretical value.**  With `log_dispersion_init=7.0`
(γ≈1100, near-multinomial) and random init, per-count NLL should be close to
the uniform-profile value:

```python
# expected multinomial NLL/count for a uniform profile over L_valid positions
expected = np.log(mask.sum())              # per track, per count
# observed: model._step(batch, "t") on an untrained model
```

Tolerance ~±10% (init logits aren't exactly uniform).  DM and nb_offset must
sit within a few % of the multinomial value at init — if DM is far off,
suspect the concentration head init or the lgamma terms.

**L3. Overfit one batch.**  Train on a single batch (Lightning:
`overfit_batches=1`, or a manual loop) for ~500–2000 steps.
IMPORTANT: for count likelihoods the floor is NOT zero — it is the batch's
empirical entropy.  Compute the floor explicitly and require convergence to
within a few % of it:

```python
p_hat = y / y.sum(-1, keepdim=True)          # empirical profile per track
floor = -(y * torch.log(p_hat.clamp_min(1e-12))).sum(-1) / y.sum(-1)  # per-count
```

Failure to approach the floor ⇒ alignment bug, capacity, or LR — go to P2.

**L4. Masked-position invariance.**  Corrupt `x` and (zero-count) `y` at
masked positions with garbage; loss and gradients must be bit-identical.
Also assert no parameter gradient is nan/inf with a mask containing a fully
masked 256bp stretch (unit test exists; rerun with the real tile size).

**L5. Augmentation round-trip.**  Data-side only (the model is not
RC-equivariant): applying RC twice to a batch is the identity
(`perm[perm] == range`, double-reverse); jittered sequence and jittered
targets are cropped with the SAME offset and the SAME strand argument.
Property test: for jitter j, the fragment counts in the cropped target equal
a direct recount of the window shifted by j.

**L6. Fixed-seed reproducibility.**  Two 50-step runs with
`L.seed_everything(seed, workers=True)` and `deterministic=True` produce
identical loss curves.  If not: dataloader worker seeding, cudnn benchmark,
or nondeterministic ops (P8).

**L7. Throughput + checkpoint/resume.**  Measure steps/s on ~200 steps;
record it.  Save a checkpoint, resume, confirm loss continues (not resets).
Only now start a long run.

## 2. Diagnostics to build in (before the first real run)

- **Per-track loss**: losses currently return a scalar mean.  Log per-track
  means (detached) alongside — a single broken track (e.g., one strand after
  an RC bug) is invisible in the aggregate.
- **Dispersion trajectory**: log mean/p10/p90 of pooled `log_dispersion` per
  epoch.  Expected shape: starts at ~7.0, drifts DOWN as real overdispersion
  is learned.  Collapse toward -inf/0 or explosion upward = pathology (P4).
- **Grad norm**: log `torch.nn.utils.clip_grad_norm_` return value via
  `on_before_optimizer_step` (also sets clipping; start with
  `gradient_clip_val=1.0` and log how often it binds).
- **N histogram**: distribution of per-(sample,tile,track) totals actually
  entering batches — confirms the min-N=50 filter is active and reveals
  depth outliers.
- **Throughput + data-wait fraction**: `profiler="simple"` on short runs;
  a data-starved GPU points at zarr chunking/worker count, not the model.
- **Lightning debug flags** (use during bring-up, OFF for real runs):
  `fast_dev_run=True` (wiring), `overfit_batches=1` (L3),
  `detect_anomaly=True` (nan localization; ~10x slower),
  `limit_train_batches`/`limit_val_batches` for short experiments.

## 3. Failure playbook — symptom → ranked causes → discriminating test

**P1. NaN/inf loss or grads.**
1. Masked-lgamma gradient leak (0·digamma(0) paths) — the guards use
   `masked_fill` safe values; any refactor that reorders mask application
   reintroduces it.  Test: L4.  Localize: `detect_anomaly=True`.
2. α = γ·p underflow in DM (tiny p × small γ): `lgamma(α)→inf`.  Test: print
   min α over a failing batch; if <1e-6, floor p or bound log γ.
3. `exp` overflow in `log_r`/γ (dispersion head diverging).  Test: dispersion
   trajectory log (§2).
4. LR too high (loss spikes then nan).  Test: 10x lower LR reproduces?
Order matters: (1) and (2) are project-specific and more likely than (4).

**P2. Loss plateaus at the init/uniform value (L3 fails).**
1. **Sequence↔target misalignment** — THE classic silent bug in this
   architecture: sequence carries margin = receptive field + jitter, targets
   carry jitter margin only; any crop-offset error (including strand-aware
   jitter applied to one side only) leaves targets uncorrelated with the
   receptive field.  Discriminating test: train on targets derived from a
   trivial sequence function (e.g., GC content) — if that also plateaus, the
   alignment is broken, not the biology.
2. Dead activations / bad init (all-LeakyReLU trunk makes this rare).
3. LR far too low/high.  Sweep 1e-5..1e-3 on the single batch.

**P3. Val loss >> train loss or diverges.**
1. Real overfitting (expected eventually at 40 samples × 4k tiles): dropout,
   early stop on val.
2. Augmentation asymmetry: jitter/RC ON in val loader (must be OFF/center) —
   makes val systematically harder.  Check loader configs.
3. Val regions systematically different (split leakage or accessibility
   skew).  Check split manifest.

**P4. Dispersion pathology (DM/nb_offset only).**  γ→0 (everything
"explained" as overdispersion, shape head goes lazy): raise
`log_dispersion_init`, or add a mild prior/weight-decay on the dispersion
head.  γ→∞: no between-sample variance learned — verify targets are truly
per-sample (merged targets have no variance to learn; brief §per-sample).

**P5. Loss oscillates.**  LR × batch-size interaction; per-count
normalization means deep tiles don't dominate, but batches mixing very low-N
tiles raise variance — check N histogram; consider min-N raise before LR
surgery.

**P6. Throughput collapse / GPU idle.**  Zarr chunk misalignment with
(sample,tile) access pattern; too few workers; one-hot/augmentation done on
GPU tensors in workers (must be CPU numpy in workers, `.to(device)` in the
module).  NEVER call `.cuda()` inside `Dataset.__getitem__` (v1 bug).

**P7. OOM.**  Activation memory ~ B × 512 × L_in × n_layers.  Levers in
order: batch size; n_kernels; tile length (requires re-preprocess — last
resort).  Confirm no retained graphs (loss logged detached).

**P8. DDP issues** (single-node multi-GPU later): unused-parameter errors —
the multinomial config already skips building the dispersion head; if a
future config leaves any head unused, prefer not building it over
`find_unused_parameters=True`.  Hangs at start: NCCL_P2P_DISABLE=1 was
needed in v1 on this hardware (train.py:3).

**P9. Nondeterminism.**  `seed_everything(workers=True)`,
`deterministic=True`, fixed val loader order.  Accept cudnn nondeterminism
only for throughput runs, never for A/B loss comparisons.

## 4. AGENT PROMPT BLOCK — paste into every training/debug agent prompt

```
## Training discipline (non-negotiable)
1. Read docs/AGENT_INSTRUCTIONS.dl_training.md. Run the CPU unit tests
   (pytest tests/test_background_model_core.py, env biomarker_env) before
   and after ANY edit to background_model_core.py.
2. Never start a run longer than 10 minutes until the Sanity Ladder
   (doc §1, L0–L7) passes. Record each rung's result in your report.
3. The overfit-one-batch floor is the batch's empirical entropy, NOT zero.
   Compute it; "loss went down a lot" is not a pass.
4. Change ONE variable per run. Every run gets: a name, the git sha, the
   config diff from the previous run, and its metrics directory. No
   uncommitted mystery states — commit or stash before switching.
5. Bind every claim to a logged metric ("val_loss plateaued at 6.91 =
   uniform value, see run X") — never to an impression.
6. Diagnose by discriminating experiment (doc §3), not by shotgun edits.
   State the hypothesis, the experiment, the expected result under each
   branch — then run it.
7. Numerical rules for this model: do not reorder mask/lgamma guards; do
   not remove the log_dispersion_init offset; keep per-count loss
   normalization. Changes to any statistical semantics require owner
   approval — they are documented in the module docstring (the spec).
8. Data rules: per-sample targets only (never merged); jitter/RC OFF in
   val; no CUDA in dataloader workers; same crop offset for sequence and
   targets.
9. Budget: if the same failure survives 3 discriminating experiments, or
   a run must exceed the agreed wall-clock, STOP and report state +
   hypotheses ranked. Do not babysit a sick run to "see if it recovers".
10. Report format: sanity-ladder table, runs table (name/sha/config
    delta/result), current best, open anomalies with evidence.
```

## 5. Sources

- Karpathy, "A Recipe for Training Neural Networks"
  (https://karpathy.github.io/2019/04/25/recipe/) — sanity ladder ordering,
  overfit-one-batch, "one change per run".
- PyTorch Lightning debugging docs
  (https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html) —
  fast_dev_run / overfit_batches / detect_anomaly / profiler flags.
- `background_model_core.py` module docstring + loss docstrings — the
  statistical spec all numerical rules derive from.
- `BIAS_CORRECTION_REVIEW.md` — v1 failure modes (CUDA-in-workers, magic
  constants as instability scars, merged-target flaw) that seeded P1/P4/P6.
- v1 `bias_correction/train.py:3` — NCCL_P2P_DISABLE precedent (P8).
```
