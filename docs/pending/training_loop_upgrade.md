# Training Loop Upgrade — LR Schedule, Divergence Recovery, Checkpoint Retention

**Date:** 2026-09-23
**Branch:** `background-model-v2`. Designed at HEAD `27a6c9e`.
**Status:**

| Phase | State |
|---|---|
| 1 — checkpoint retention, `LearningRateMonitor` | **DONE** — `7cf3c22` |
| 2 — `ReduceLROnPlateau`, per-group `min_lr`, derived patience | **DONE** — `e8fecf4`, fixes in `b6e0be1`; impl-review A- |
| 3 — divergence recovery | **DONE** — `f30f09b`, fixes in `f598a36`, LR floor clamp in `0e1473c`; impl-review **A** (r1 and r2 each returned A- while missing a real defect). 395 tests passing |
| 4 — acceptance run | IN FLIGHT — `phase4_hybrid_lr2e-3_recovery`, submitted 2026-09-24; see §7 |

§5 was decided by the owner on 2026-09-23 (lr_patience 4, max_lr_reductions 3,
1 for test runs). §5.0's formula was **corrected after implementation review** —
read the correction block there, not the original reasoning below it.
**Author note:** written by the EM directly. Two `design`-agent delegations died
to API rate limits; rather than burn a third attempt, this was written from
first-hand context. Every code claim below was verified against source at
`27a6c9e` — see §8 for what is verified versus inferred.
**Reconciliation note (2026-09-24):** §0 and the correction blocks at §4.3, §4.4
and §5.1 were likewise written by the EM directly, after *three* consecutive
`implement`-agent delegations died to API rate limits across a 5.5-hour backoff.
Same reasoning: at three failures the retries cost more than the work. Claims in
those blocks were verified against source at `0e1473c`.

---

## 0. Implementation Status — what Phase 3 actually shipped

Added 2026-09-24, reconciling this document against the built code. **Where the
implementation diverged from the design and the divergence was an improvement,
that is said plainly below rather than back-written into the original text** —
a future reader needs to know which decisions were revised under evidence, and
which parts of this design were simply wrong. The two substantive corrections
are marked inline at §4.3 (the LR ladder) and §5.1 (floor vs. counter).

Shipped in `f30f09b` (initial), `f598a36` (seven review findings), `0e1473c`
(LR floor clamp, owner-approved as it affects computed results).

Two defects were found *after* the initial implementation, both at the seam
between two mechanisms rather than inside either one, and both found by
execution rather than review — two review rounds read the relevant code and
cleared it. See §4.3 and §4.4.

**New `summary.json` outputs.** The owner cross-references these against the run
table in `docs/pending/training_analysis.md` §7.7; terminology here matches that
section.

| Field | Meaning |
|---|---|
| `recoveries[]` | One record per attempt: `attempt`, `epoch`, `pre_divergence_best`, `diverged_value`, `recovery_lr_factor`, `checkpoint` |
| `recoveries[].floor_clamped_groups` | `[{group, requested_lr, clamped_lr}]` — present **only** when the §4.4 clamp actually binds; absent, not empty, otherwise |
| `stop_reason` | Adds `diverged_unrecovered` — budget exhausted, *or* an attempt that trained zero epochs |
| `best_val_loss` | Tracked **globally across all attempts** (§4.5), so not necessarily the final attempt's best |

**Naming trap:** top-level `lr_factor` is the **plateau** factor; nested
`recoveries[].recovery_lr_factor` is the **recovery** factor. Both land in
`summary.json`. The latter was renamed from `lr_factor` precisely because
conflating the two misreads the ladder.

**Zero-epoch guard.** `max_epochs` is absolute while `current_epoch` is restored
from the checkpoint, so a recovery resuming near the cap can train **zero**
epochs. Untracked, `_determine_stop_reason` found no fired guard and returned
`completed` — reporting a diverged run as a clean finish, reinstating exactly
the gap `stop_reason` exists to close. `_RecoveryEpochTracker` now counts
`on_train_epoch_start` and forces `diverged_unrecovered` instead.

**New CLI:** `--max-recoveries` (default 3, 0 disables), `--recovery-factor`
(default 0.5).

---

## 1. Problem

Three symptoms from the eight-run v3 LR sweep, all traceable to one missing
mechanism:

| Run | Symptom | Diagnosis |
|---|---|---|
| `lrsweep_hybrid_lr2e-3` | best 7.5302 @ ep13, **diverged ep15** (→9.62) | LR fine early, too high late |
| `lrsweep_hybrid_lr1e-3` | best 7.5314 @ ep8, then degrades 15 epochs | no decay to settle into the minimum |
| `lrsweep_cnn_lr2e-3` | still descending at ep39/40 | under-trained at a constant LR |

**There is no LR schedule anywhere.** All three `configure_optimizers`
implementations (`background_model_core.py:771`, `:998`, `:1275`) return a bare
`torch.optim.Adam`. LR is constant for the entire run.

Two further problems compound it:

- **Divergence discards the run.** `DivergenceStop` sets `trainer.should_stop`.
  Everything after the last kept checkpoint is lost and an operator must
  hand-retune and resubmit. In this project that has repeatedly meant hours of
  idle GPU.
- **Only 3 checkpoints survive.** `save_top_k=2, save_last=True` — verified on
  disk: `lrsweep_cnn_lr2e-3/checkpoints/` holds ep36, ep39, last, and nothing else.

## 2. Goals / non-goals

**Goals.** Decay LR automatically; recover automatically from divergence rather
than dying; retain every epoch's checkpoint, cross-referenceable to the results
table in `training_analysis.md`.

**Non-goals.** No architecture change. No change to any likelihood, masking
rule, loss normalization or guard ordering. No change to detection *thresholds*
(`factor=1.10`, `stall_patience=5`).

**This change will alter computed results** — that is intended and
owner-approved. §7 defines how we tell improvement from regression.

## 3. Recommended LR schedule: `ReduceLROnPlateau`

**Recommendation: `ReduceLROnPlateau(mode="min", factor=0.5, patience=P)` on
`val_loss`.** See §5 for `P`.

Justification against the alternatives:

| Option | Fit to evidence |
|---|---|
| **ReduceLROnPlateau** | Reactive; needs no knowledge of run length. Directly addresses symptoms 2 and 3, and softens 1. |
| Cosine / cosine+warmup | Needs a meaningful `max_epochs`. **Our runs rarely reach it** — they stop on early-stopping, divergence or stall. A cosine schedule scaled to 40 epochs on a run that ends at 16 has barely decayed, so the mechanism silently does nothing. |
| OneCycle | Same `max_epochs` dependence, plus its own hyperparameters — it would need its own sweep, which defeats doing this before the NB experiment. |

The `max_epochs` argument is decisive: of the 8 sweep runs, only `cnn_lr2e-3`
reached the cap. A schedule keyed to wall-clock progress is the wrong shape for
runs that terminate on a metric.

### 3.1 The param-group hazard

`configure_optimizers` does **not** always return a single-LR optimizer. Both
the KEN and hybrid paths build explicit param groups with *different* LRs and
weight decays — e.g. `background_model_core.py:1275` returns groups for
`embed` (lr, wd=wd), `main` (lr, wd=0) and optionally `dispersion`
(lr × `dispersion_lr_scale`, wd=0).

Any LR manipulation must **scale each group multiplicatively and preserve the
ratios between them.** Setting a scalar LR across groups would silently destroy
the `dispersion_lr_scale` relationship — a change to optimization semantics
that would not announce itself. PyTorch's schedulers already operate per-group,
so the scheduler path is safe; the *recovery* path (§4) is where this must be
done by hand and is easy to get wrong.

## 4. Divergence recovery

### 4.1 Mechanism: outer retry loop, not in-callback surgery

Two implementations were considered:

**(a) In-callback restore.** `DivergenceStop` loads state dicts into
`pl_module` and `trainer.optimizers` mid-`fit`.
**(b) Outer loop.** The guard stops the fit as it does today; `run_training`
detects `stop_reason.reason == "diverged"`, and re-invokes `trainer.fit(...,
ckpt_path=<best>)` with a reduced LR.

**Recommend (b).** It reuses Lightning's own resume machinery — which correctly
restores model weights, optimizer state, epoch and global step — instead of
reimplementing it. It is testable without a GPU, and it cannot corrupt trainer
internals mid-epoch. (a) is more "immediate" but does state surgery on a
running trainer, and `trainer.fit()` is already the supported resume path
(`train.py:795` already passes `ckpt_path=cfg.resume_from`).

### 4.2 Optimizer state — the non-obvious part

Adam's moment estimates are poisoned by the diverged step, so restoring weights
alone would likely re-diverge immediately.

**This resolves itself under (b):** the checkpoint was written at the *best*
epoch, before divergence, and Lightning checkpoints include `optimizer_states`.
Resuming from it restores *healthy* pre-divergence moments. No zeroing or
special handling is needed — and deliberately zeroing them would be worse, as
Adam would need to re-accumulate second-moment estimates from scratch.

### 4.3 Applying the LR reduction on resume

**The subtlety that will bite:** Lightning restores optimizer state *including*
`param_groups[i]["lr"]`. So simply passing a lower LR into the model's hparams
will be **overwritten** by the restore. The reduction must be applied *after*
the optimizer state is loaded.

Mechanism: a small callback (`ApplyLRReduction`) on `on_train_start` that
multiplies each `param_group["lr"]` by a reduction factor — preserving the
inter-group ratios required by §3.1.

> **CORRECTED 2026-09-24 — "the cumulative factor" was wrong, and was falsified
> by measurement.** This section originally said to multiply by the *cumulative*
> factor, i.e. `recovery_factor ** N` relative to the ORIGINAL LR. That is only
> correct while the resumed checkpoint still holds the original LR. It does not,
> as soon as recovery *works*: when an attempt improves the global best, the
> best-checkpoint pointer moves onto a checkpoint saved *during* that attempt,
> whose stored optimizer LR is **already reduced**. Lightning restores that
> reduced LR, and the from-original factor was then applied on top of it.
> Measured: **1.25e-3 where 2.5e-3 was intended** — the error compounds, and it
> fires precisely when the feature is succeeding.
>
> **As built:** the loop tracks `best_recovery_level` — how many recovery
> reductions are already baked into the resumed checkpoint — and applies
> `recovery_factor ** (recovery_count - best_recovery_level)`, a factor relative
> to *the resume checkpoint's stored LR*, chosen so the result targets
> `recovery_factor ** N × original_lr`.
>
> **That target is not an absolute guarantee, and must not be stated as one.**
> A `ReduceLROnPlateau` reduction that fires *inside* a recovery attempt which
> then improves the global best is likewise baked into the checkpoint, and is
> **not** counted by `best_recovery_level`. The invariant that actually holds is
> the clamp in §4.4: recovery never drives any group below its `min_lr`.

> **Do not let `min_lr` bound the recovery budget** (learned during the Phase-2
> review, 2026-09-23). Once every param group sits at its `min_lr`,
> `ReduceLROnPlateau` still fires its no-op reduction and **resets
> `num_bad_epochs` to 0**, cycling indefinitely. The floor therefore cannot
> *report* that it refused a reduction — it silently absorbs the event. Phase 3
> must count recoveries with its own explicit counter (§4.4 `max_recoveries`)
> and must not infer "no further reduction possible" from the LR failing to
> move. See the `_with_lr_schedule` docstring in `background_model_core.py`.

### 4.4 Bounds and termination

| Parameter | Value | Rationale |
|---|---|---|
| `max_recoveries` | 3 | Each halving costs an epoch of re-tread; after 3 the LR is 1/8 of start and further recovery is unlikely to find new minima. CLI `--max-recoveries`; 0 disables recovery. |
| reduction per recovery | 0.5 | Matches the scheduler's factor; one mechanism, one constant. CLI `--recovery-factor`. |
| **LR floor** | recovery clamps at `min_lr` | **ADDED 2026-09-24, owner-approved.** Recovery may *reach* each group's `min_lr` but never pass it. See below. |
| on budget exhausted | stop, `reason="diverged_unrecovered"` | Must terminate. A recovery loop that never ends is worse than today. |

Recovery must also be **monotonic in progress**: if a recovery attempt fails to
beat the pre-divergence best within its own budget, that counts against
`max_recoveries` — otherwise a run that diverges every other epoch could loop
until the job timeout.

**The floor clamp (added after implementation review, owner-approved).** The
ladder and the floor previously agreed only by the numerical *coincidence*
`recovery_factor == lr_factor` and `max_recoveries == max_lr_reductions`.
Nothing enforced it, and `--recovery-factor` is now CLI-exposed, so retuning it
would have broken the agreement silently. Unclamped, recovery could push the LR
*below* `min_lr`, where `ReduceLROnPlateau` goes **permanently inert**:
`new_lr = max(old × factor, min_lr)` then exceeds `old_lr`, the
`old_lr - new_lr > eps` assignment guard blocks the write, and `num_bad_epochs`
resets on every step. That is the same "a floor cannot report a refusal"
pathology described in §4.3 and §5.1 — reintroduced through a second door.

The floor is read from the **live** `ReduceLROnPlateau`'s `min_lrs` via
`trainer.lr_scheduler_configs`, and deliberately **not** recomputed as
`cfg.lr_factor ** cfg.max_lr_reductions`. This is the load-bearing decision: a
second copy of that formula is free to drift from `_with_lr_schedule`'s, and
exactly that kind of duplication is what produced the §4.3 bug. One
authoritative source, no second copy.

A clamp that binds is **reported, not silent** — `recoveries[].floor_clamped_groups`
(§0), present only when it actually binds.

### 4.5 Callback state interactions

This is where a naive implementation breaks. Under (b) each `fit` constructs
fresh callbacks, which resolves most of it — but the consequences must be
deliberate, not accidental:

| Callback | State | Consequence on resume | Required |
|---|---|---|---|
| `EarlyStopping` | `wait_count`, `best_score` | Fresh instance → counter resets | **Desired.** Prevents early-stopping *during* recovery, the failure mode that would defeat the whole mechanism. |
| `DivergenceStop` | `best`, `_last_val`, `_stall_count` | Fresh → `best=None`, re-established from first post-resume epoch | Correct: the new `best` should reflect the restored checkpoint, not the diverged trajectory. |
| `ModelCheckpoint` | `best_model_score` | Fresh instance would lose the global best | **Must be handled.** The outer loop has to track the best across attempts, or a later, worse attempt could be reported as the run's result. |

That last row is the one real trap: `summary.json`'s `best_val_loss` comes from
`trainer.checkpoint_callback.best_model_score` (`train.py:~790`), which under a
fresh trainer reflects only the final attempt.

## 5. Patience — RESOLVED BY OWNER 2026-09-23

**Decision (supersedes the analysis below):**

> Wait **4 epochs** for an LR reduction. Reduce the LR a **maximum of 3 times**
> (configurable — test runs want **1** reduction, still with patience 4). After
> the reductions are exhausted, if val loss doesn't improve, **stop training**.

This is better than the reading I recommended, because it makes termination
**derived from the LR schedule** instead of an independent magic number.

### 5.0 What this means concretely

| Knob | Value | Note |
|---|---|---|
| `lr_patience` | 4 | epochs without improvement before an LR cut |
| `max_lr_reductions` | 3 (default), 1 for test runs | **must be CLI-configurable** |
| LR factor | 0.5 | §4.4 |
| `EarlyStopping.patience` | **derived** = `(lr_patience + 1) × max_lr_reductions + lr_patience` | not set independently; see the correction note below |

> **CORRECTED 2026-09-23 after implementation review — the formula below was
> WRONG.** It assumed `ReduceLROnPlateau` and `EarlyStopping` count bad epochs
> identically. They do not: the scheduler reduces on `num_bad_epochs >
> patience` (strict `>`, so it fires after `lr_patience + 1` bad epochs) while
> `EarlyStopping` stops on `wait_count >= patience`. Verified in a real
> Lightning loop: with the original formula, reductions landed at epochs 6/11/16
> and EarlyStopping also fired at epoch 16 — **the final LR reduction received
> zero training epochs**, so the owner's "3 reductions, then stop" policy was
> not honoured.
>
> **Correct formula:** `patience = (lr_patience + 1) × max_lr_reductions +
> lr_patience`. Defaults → `5×3 + 4 = 19`. Test runs (1 reduction) → `5 + 4 = 9`.
>
> Lesson: the original tests asserted the *arithmetic* (`4 × 4 = 16`) but never
> that the arithmetic produced the intended *behaviour*. Any test for this must
> run an actual Lightning loop and assert the final LR level gets `lr_patience`
> epochs before stopping.

**Original (incorrect) reasoning, retained so the error is not repeated.**
`ReduceLROnPlateau` resets its bad-epoch counter after each reduction;
`EarlyStopping` resets its counter only on improvement. In the worst case (no
improvement at all): reductions fire at epochs 4, 8, 12, then 4 further barren
epochs take us to 16 — which is exactly `4 × (3 + 1)`. If improvement occurs,
both counters reset together and they stay in step. So a single derived value
reproduces the requested policy without a second tunable. With
`max_lr_reductions=1` it gives `patience=8`.

**Implementation note — capping the reductions.** Rather than counting
reductions by hand, set `min_lr = initial_lr × factor^max_lr_reductions`
(e.g. `lr/8` for 3 halvings). `ReduceLROnPlateau` will then refuse to go below
it. **`min_lr` must be a per-group list**, computed from each group's own
initial LR, or the `dispersion_lr_scale` ratio is destroyed at the floor
(see §3.1).

### 5.1 Open interaction — two mechanisms reduce the same LR

The plateau scheduler cuts LR, **and** divergence recovery (§4.4) also cuts LR.
Both are bounded, but their budgets are currently independent, which raises a
question the implementation must answer explicitly:

- Do recovery-driven cuts count against `max_lr_reductions`?
- Does the `min_lr` floor also bind recovery, or can recovery go below it?

**Original proposal:** share the **floor** (`min_lr` binds both), keep the
**counters separate**.

**REVISED 2026-09-23 — the floor is the wrong primitive.** Review verified by
execution that once all groups sit at `min_lr`, `ReduceLROnPlateau` still fires
its (no-op) reduction and **resets `num_bad_epochs` to 0**, cycling
indefinitely without ever signalling that nothing changed.

A floor **cannot report a refusal.** Concretely, with recovery layered on top:
the scheduler exhausts its 3 cuts and bottoms out at `lr/8`; the model diverges;
recovery restores the best checkpoint and asks to halve the LR; the floor
silently blocks it; the run resumes at *exactly the LR that just diverged*,
diverges again, and burns all 3 recovery attempts without the LR ever changing.
`summary.json` would report three recovery attempts, implying three
progressively gentler configurations were tried. None were. That is precisely
the silent-misreporting class the `stop_reason` work exists to eliminate.

**Phase 3 must therefore replace the implicit `min_lr`-as-cap with an explicit
reduction counter**, so that a refused reduction is countable and reportable,
and record the LR trajectory plus any refusals in `summary.json`.
`LearningRateMonitor` (Phase 1) makes a flat LR *visible* in `metrics.csv`, but
only to someone who goes looking — the summary would still mislead.

> **RESOLVED 2026-09-24 — as built, Phase 3 did BOTH, and they answer different
> questions.** "Replace the floor with a counter" was not quite the right frame:
> the two mechanisms bound different things and both are needed.
>
> - The explicit `recovery_count` bounds **attempts**. A floor cannot report a
>   refusal, so it can never bound them. This is what `max_recoveries` enforces.
> - The `min_lr` clamp (§4.4) bounds **depth**. Without it recovery walks below
>   the floor and inerts the scheduler.
>
> Answering this section's two original questions directly: recovery-driven cuts
> do **not** count against `max_lr_reductions` (separate counters, as proposed),
> and the `min_lr` floor **does** bind recovery (shared floor, as originally
> proposed) — but it binds *explicitly and reportably* rather than by silent
> absorption, which is what the revision was reacting to.
>
> The exact scenario this section warned about — "recovery asks to halve the LR,
> the floor silently blocks it, the run resumes at precisely the LR that just
> diverged, and `summary.json` reports three attempts implying three gentler
> configurations were tried when none were" — is now surfaced explicitly by
> `recoveries[].floor_clamped_groups`, which records `requested_lr` alongside
> `clamped_lr` for every group where the clamp bound.

### 5.2 Original analysis (retained for context)

Owner said: *"Given this let's use a more aggressive patience — say 4."*

There are **three** distinct patience knobs, and "4" is coherent with more than
one. They interact, so this cannot be resolved by picking the nearest one.

| Knob | Current | Effect |
|---|---|---|
| `EarlyStopping.patience` | 5 | epochs without improvement before the run stops |
| `DivergenceStop.stall_patience` | 5 | consecutive bitwise-identical val_loss before declaring collapse |
| *(new)* `ReduceLROnPlateau.patience` | — | epochs without improvement before cutting LR |

**The binding constraint:** `EarlyStopping.patience` must be meaningfully
**greater** than the scheduler patience. Otherwise the run early-stops before a
reduced LR has any chance to show benefit, and the schedule is inert. Standard
practice is roughly 2–3×.

| Reading | Interpretation | Consequence |
|---|---|---|
| **A** | `EarlyStopping` 5 → 4 | Then scheduler patience must be 1–2. Very tight: an LR cut gets ~2 epochs to prove itself. Most literal reading of "5 → 4". |
| **B** | scheduler patience = 4 | Then `EarlyStopping` must rise to ~10–12. "More aggressive" = cut LR sooner. Coherent with "given [recovery]" — recovery makes aggressive LR policy safe. |
| **C** | `stall_patience` 5 → 4 | Orthogonal to the LR work; collapse detection only. Unlikely to be what was meant. |

I recommended B (scheduler 4, EarlyStopping 12). The owner's actual answer was
close to B on the LR knob but replaced the arbitrary `EarlyStopping=12` with a
*derived* termination rule — see §5.0. My proposed 12 was a guess; the derived
16 (for 3 reductions) follows from the policy itself.

## 6. Checkpoint retention

**Change:** `save_top_k=2` → `save_top_k=-1` (retain every epoch), keep
`save_last=True` and the existing filename template
`"{epoch}-{step}-{val_loss:.4f}"`.

**Storage cost — stated explicitly, since this is shared EFS:**

| Architecture | Per ckpt | × 40 epochs |
|---|---|---|
| hybrid (128k/3L) | 21.6 MB | ~864 MB/run |
| cnn (128k/1L) | 7.1 MB | ~284 MB/run |
| ken (512k/2L) | ~7 MB | ~280 MB/run |

A sweep of 8 hybrid runs is ~7 GB. Tolerable, but it grows linearly with every
future sweep and nothing currently prunes it.

**Owner decision 2026-09-23: save all, no pruning for now.** Accept the growth;
revisit if EFS pressure materialises. **Pruning is an acknowledged deferred
item, not an oversight** — recorded in the EM's persistent memory so it
resurfaces rather than being silently forgotten.

**Cross-referencing to `training_analysis.md`.** The results table keys on run
name. Checkpoints already live at `runs/<run_name>/checkpoints/`, and the
filename carries epoch, step and val_loss — so run → epoch → checkpoint is
already resolvable. Two gaps to close:

1. **Log LR per epoch.** With a schedule and recovery, LR varies within a run,
   and `metrics.csv` does not currently record it. Add Lightning's
   `LearningRateMonitor`. Without this, a checkpoint's LR is unrecoverable
   after the fact.
2. **Record recovery events in the summary.** `summary.json` records *why* a
   run stopped. A recovered run must not look identical to a clean one — that
   is precisely the class of silent-misreporting bug the `stop_reason` work
   (`2d4cc95`) was added to prevent. Add a `recoveries` list with, per event:
   epoch, pre-divergence best, val_loss that triggered it, and the new LR.

## 7. How we will tell improvement from regression

Baseline on `sim_store_v3_A` (anchors verified, provenance in
`simulation_v3/A/oracle.json`): oracle 7.523195, uniform 7.612705, gap
0.089511 nats.

| Config | Best val | % bias captured |
|---|---|---|
| `lrsweep_hybrid_lr1e-3` | 7.531432 | **90.8%** |
| `lrsweep_hybrid_lr2e-3` | 7.530183 | 92.2% (diverged ep15) |

**Acceptance:** re-run hybrid at lr=2e-3 on `v3_A` with the schedule and
recovery enabled. Success = it does **not** end in an unrecovered divergence,
**and** best val ≤ 7.5302 (i.e. at least matches what the diverged run reached
before blowing up, now as a trustworthy stable result).

This is a strong test: lr=2e-3 is precisely the configuration that diverges
today, so it exercises schedule and recovery together on a known failure.

**Regression signals:** best val worse than 7.5314 (the stable lr=1e-3 result);
or recovery firing on runs that never diverged before; or `cnn_lr2e-3`
converging to worse than 7.5527.

## 8. Verified vs inferred

**Verified against source at `27a6c9e`:** all three `configure_optimizers`
return bare `Adam` with no scheduler (`:771`, `:998`, `:1275`); param groups
carry distinct LRs/weight decays; `ModelCheckpoint(save_top_k=2,
save_last=True)` and the resulting 3-file retention (confirmed on disk);
`EarlyStopping(monitor=val_loss, mode=min, patience=cfg.patience,
min_delta=0.0)`; `trainer.fit(..., ckpt_path=cfg.resume_from)`;
`DivergenceStop` state fields and `stop_reason` payloads; `deterministic=True`
and `gradient_clip_val=1.0`; the 8 sweep results and the oracle anchors.

**Inferred, not yet verified — must be confirmed during Phase 1:**
that Lightning checkpoints include `optimizer_states` and that resuming
restores `param_groups[i]["lr"]` from them (§4.3 depends on this; if the LR is
*not* restored, the reduction becomes simpler, not harder); that constructing a
fresh `Trainer` per recovery attempt resets callback state as described in §4.5.

## 9. Phased implementation

Ordered so the riskiest piece is isolated and independently validatable.

| Phase | Scope | Risk | Validation |
|---|---|---|---|
| **1** | Checkpoint retention (`save_top_k=-1`), `LearningRateMonitor`, LR logged to `metrics.csv`; **verify the two §8 inferences** | Low — config only | Suite green; a short run writes per-epoch ckpts and an `lr` column |
| **2** | `ReduceLROnPlateau` in all three `configure_optimizers`; `--lr-patience` (4) and `--max-lr-reductions` (3) CLI-plumbed; `EarlyStopping.patience` derived per §5.0; per-group `min_lr` | Medium — touches frozen core | Unit-test that per-group LR *ratios* survive a reduction AND at the `min_lr` floor (§3.1, §5.0); test the derived patience for both 3 and 1 reductions; short run shows LR stepping down |
| **3** | Divergence recovery outer loop + `recoveries` in summary | **High** — control flow, cross-attempt best tracking | Unit-test the loop with a synthetic diverging metric; confirm `best_val_loss` tracks the global best across attempts (§4.5) |
| **4** | Acceptance run: hybrid lr=2e-3 on `v3_A` | — | §7 criteria |

Phase 2 is the only one touching the frozen core, and it is separable from
Phase 3 — if recovery proves troublesome, the schedule alone still addresses
symptoms 2 and 3.

## 10. Failure modes

| Failure | Detection |
|---|---|
| Scheduler destroys `dispersion_lr_scale` ratios | Phase-2 unit test asserting ratios post-reduction |
| Scalar `min_lr` flattens per-group LRs at the floor | Phase-2 test asserting ratios hold *at* `min_lr` (§5.0) |
| Derived `EarlyStopping.patience` drifts from the LR policy | Phase-2 test pinning `patience == (lr_patience + 1) × max_lr_reductions + lr_patience` for both 3 and 1, **plus** a real-Lightning-loop test that the final LR level gets `lr_patience` epochs (arithmetic-only tests missed the off-by-one — §5.0) |
| Recovery loops until job timeout | `max_recoveries=3`; wall-clock vs epoch count in summary |
| Early stopping fires during recovery | Fresh callback per attempt (§4.5); assert in Phase-3 test |
| `best_val_loss` reports a later worse attempt | Cross-attempt best tracking (§4.5); Phase-3 test |
| LR reduction silently overwritten by ckpt restore | §4.3; assert post-resume LR in Phase-3 test |
| Recovered run indistinguishable from clean run | `recoveries` in `summary.json` (§6) |
| EFS fills from retained checkpoints | Storage table §6; retention policy still an open item |

## 11. Self-assessment

**Grade: B+.**

Strong on hazard identification — the optimizer-state resolution (§4.2), the
param-group ratio trap (§3.1), the LR-overwritten-on-restore subtlety (§4.3)
and the cross-attempt best-tracking trap (§4.5) are the four things that would
otherwise have surfaced as bugs, and each has a named test.

Held below A- by three things. The two Lightning behaviours in §8 are
**inferred rather than verified**, and §4.3 depends on one of them. The
patience question (§5) is unresolved and blocks implementation. And the
checkpoint retention policy is identified as needed but not specified — it is
deferred rather than designed.

No GPU should be spent on Phase 4 until §5 is answered and the §8 inferences
are confirmed in Phase 1.
