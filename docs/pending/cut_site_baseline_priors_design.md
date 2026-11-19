# Cut-site model: additional simple baseline priors (design)

Date: 2026-10-10 · Branch `worktree-cut-site-model` @ `9aa8ef5`+ · Status: **A-, APPROVED (round-3 review, §9). Revision 4 reconciles this document with Phase 0 and Phase 0b.1 as implemented — see "Implementation status" below and the revision-4 log entry at the end of §"Revision log".**

### Implementation status (revision 4)

- **Phase 0 — COMPLETE.** Gate G0 PASSED, all 8 real checks (commit `2a775ce`). Reviewed A- (`implementation_review_phase_0`, agent `2f724ba2-ecc`).
- **Phase 0b.1 — COMPLETE.** Gate G0b PASSED, all 6 items (commit `9aa8ef5`). Reviewed A- (`implementation_review_phase_0b`, agent `f5d57904-005`). This is the "approved change set" of §0.4: the estimator fix (A), the `m+L` registration fix (B), and the counter (S3, `count_tables.py`) are landed and in production. No `hex_adapter.py` was built (§0.4 C, revised below).
- **Phase 0b.2 onward — NOT STARTED.** This document is the input to Phase 0b.2 (count N0 at `w*` by train-CV and score the new anchor). Nothing past P0b.1 in §5 has run.

Author: synthesis agent. I worked only from an evidence packet. Research agents, a
measurement agent and the coordinator assembled and cross-checked it this session. I did
not open source files or run code myself. **Revision 3 exception:** the reviser read
`model.py:115-309, 395-409, 634-647`, `simulator/measure.py:150-560`,
`simulator/draw.py:70-187`, `stage1_hex_baseline.py:215-232` (plus a grep for lines 187/313)
and `tests/test_hex_registration.py:1-305` directly. Pointers tagged `[Verified: … , r3]`
come from that reading. The reviser ran no code. **Revision 4 exception:** the reviser read
`COORDINATION.cut-site-model.md`'s gate log and "Design-Implementation Divergences" section
only, not the implemented source. Every revision-4 number is tagged
`[Verified: COORDINATION gate <name>]` rather than a file pointer; re-verify against the
source at `9aa8ef5` before relying on a line number this revision did not itself re-check.

**Tag legend.** `[Verified: ptr]` = a research agent, the measurement agent or the
coordinator examined the evidence this session at `ptr`. I did not re-open it.
`[Inferred: basis]` = derivation from verified items, with the arithmetic shown.
`[Unverified: cmd]` = no evidence yet; `cmd` settles it. File pointers are relative to the
worktree root `W=/home/nathanboley/src/fragmentomics_tools/.claude/worktrees/cut-site-model`.
Line numbers rot. Re-check them before you edit.

---

## 0. Premise state, headline, owner decisions

### 0.1 Premise state

The brief describes the prior's structure and the published prior anchor correctly. The code
contradicts four of its other premises:

1. **"`train.py --validate-only --zero-correction` gives the test numbers."** Wrong. It
   evaluates only the VAL split. The published test prior 12.142214 (now RETIRED as an
   anchor by rulings 5-6, §0.4) comes from no single sanctioned flag. [Verified: train.py:169-176 builds train+val only; score_fragments.py:407-418 reproduces val only]
2. **"`unif+p_L` is a column the pipeline produces."** Wrong. No code produces it.
   [Verified: grep by research agent A over train.py, score_fragments.py, stage1_hex_baseline.py, model.py]
   You can produce it through the sanctioned path with a uniform-table `.npz` (§4.2).
3. **"Real data has 1,123,884 fragments."** That number overstates what a counted prior uses.
   It counts the 1536-bp jittered window. The prior counts only the centred crop: 749,554
   scored in total, 599,412 in TRAIN. [Verified: measurement agent T1/T3 output; per_fragment_prior.tsv.summary.json]
4. **"min_N=10 drops ~2% of regions."** Wrong. It drops 5.4-5.7% in every split.
   [Verified: measurement agent T1] COORDINATION now carries 5.4-5.7% (D5 closed).
   [Verified: COORDINATION.cut-site-model.md "Remaining work" item 0, r3]

The brief also omitted one assumption: the current tables already use a smoothing
estimator, and that estimator distorts rare hexamers (§1.3 F1). The docs say "no smoothing
scheme decided". That is stale. [Verified: stage1_hex_baseline.py:224-229] The owner has
since APPROVED its replacement (ruling 5, §0.4).

### 0.2 Headline recommendation

- **Build two thin seams, then land ONE approved change set.** S2 is a reduction wrapper
  that gives val AND test region-weighted NLL from the sanctioned per-fragment output. S3 is
  a new table counter, built directly in the model's own hexamer convention, drawing only
  `FragmentLengthDist`/`fl_end_weight` from `simulator/measure.py` **(revision 4: no adapter
  — none was needed, §0.4 C)**, that writes a conforming `.npz` with provenance. The only
  model code that changes is the approved 3' registration fix (§0.4 B). Neither seam edits
  an owned file. **Both S2 and S3 are now BUILT** (Phase 0 and Phase 0b.1 respectively; see
  "Implementation status" at the top of this document).
- **First deliverable (Phase 0):** a measured real-data NLL table for `uniform`, `unif+p_L`
  and the OLD `prior` on val and test, through `score_fragments.py --zero-correction`, plus
  the identity tests that prove the reduction. This run happens at `bad981d` code, BEFORE
  the change set, because the old prior is not reproducible after it (§0.4).
- **N0 is no longer a candidate. It is the approved new prior (rulings 5+6).** N0 replaces
  stage1's `(c+1)/(occ+4096)` estimator, whose distortion is a measured BIAS of
  0.016-0.019 nats per table, cut-weighted (§3.5, §2.5). Each fragment reads two tables, so
  the second-order cost is ~0.03-0.04 nats/fragment [Inferred] — 2-3x the entire
  estimation-noise budget. That figure is UPPER-BOUND flavoured: it assumes the unsmoothed
  rate is the right target. N0 is built at `m+L`, scored once, and its score becomes the new
  `prior` anchor. Every later candidate is measured against THAT anchor. N4 initialises from
  N0's tables, so N0 must exist first.
- **Then N4 (rank 2, the lead CANDIDATE).** N4 is a region-weighted maximum-likelihood
  calibration of N0's four tables plus `p_L`. It has the same parameter count (16,535) and
  goes through the evaluator with zero code change. The depth confound motivates it (§3.1);
  F1 no longer does, because N0 absorbs it. No measurement yet shows a gain.
- **Next:** N1 (also count the 299,810 TRAIN jitter-margin fragments; bounded at <=0.0046
  nats). M2 (length-band-conditional hexamer deltas) and M1 (flank PWM) follow only if the
  Phase-0c residual maps show signal against the NEW prior's *expected* counts. They need a
  model-side seam (S1), and S1 needs owner approval.
- **Honest expectation, reconciled with F-03.** The 0.02-0.08 range is a judgement for the
  WHOLE stacked set N0 + N1 + N4 + passing M candidates, measured against the OLD prior.
  [Unverified: Phase 0b-2 tables] N0's ~0.03-0.04 sits INSIDE that range, not on top of it.
  It is the largest single expected component. The range's floor (0.02) sits below N0's
  bound on purpose: the bound is upper-bound flavoured, and shrinkage keeps part of the
  distortion to buy back noise (§3.5). [Inferred] So the candidates AFTER N0 share only what
  is left of the range once N0's realised gain is subtracted. Do not quote 0.02-0.08 as
  their gain against the new anchor; this document makes no separate estimate of that
  remainder. Neither number is a share of the 0.145 correction gain any more:
  that gain is RETIRED with its two endpoints (§0.4). Gains will not add up. Measure them
  stacked.
- After the approved change set, N0 at `m+L` is the default prior. Every LATER candidate is
  additive and evaluated side by side against it. A further swap of the production prior
  needs a separate owner approval.

### 0.3 Decisions requested from the owner

| # | Decision | Why it is needed |
|---|---|---|
| D1 | Approve seam S1 (`prior_terms.py`, about 15 lines in `model.py`, one flag each in `train.py` and `score_fragments.py`; the default stays bit-identical) | All M-layer candidates need it. You can defer it until Phase 0 shows an M candidate has signal (gate G3). |
| D2 | ~~Acknowledge the stage1 smoothing finding~~ **CLOSED by ruling 5 (estimator fix approved).** The relay to the stage1 owner (session 8d207c5986) is now mandatory, not optional: it must carry the retirement of stage1's counting path (§0.4). | stage1 is owned elsewhere this phase. This design does not edit it. |
| D3 | Strand pooling (N2) touches the "four untied tables" design (cut_site_fragment_model.md:143-151). | N2 is a DIAGNOSTIC only until you sign off. |
| D4 | Approve an additive, default-off `return_terms` on `score_fragments.region_weighted_mult_nll`. | The paired bootstrap needs per-region terms. |
| D5 | ~~Correct COORDINATION "~2% min_N drop"~~ **CLOSED: COORDINATION now says 5.4-5.7%.** | §0.1 item 4. |

Closed decisions that this document does not re-open: `FL_BANDS=((25,110),(110,180))`;
`L_max=180` INCLUSIVE; the frozen statistical core (`background_model_core.py`); four
untied tables with no RC tying; the retired 92-sample pooled pipeline (decision 189); the
L2b interior branch; `tile_size` as a required kwarg; NON-NN candidates only this phase
(ruling 2); **owner rulings 5, 6 and 7 of 2026-10-10 (§0.4)**. [Verified:
COORDINATION.cut-site-model.md "Owner rulings, 2026-10-10 (second set)", r3]

The 3' `m+L` vs `m+L-1` question is **no longer open.** Round 1 (F-05) recorded it as
unresolved. Ruling 6 resolves it in favour of `m+L` (§0.4 B).

### 0.4 Approved change set (owner rulings 5-7) and its sequencing constraint

**Read this before anything in §3-§7.** Three approved changes land TOGETHER as one change
set. Each one alone either breaks the prior or makes a later fit look better than it is.

**A. Estimator fix (ruling 5, APPROVED).** `stage1_hex_baseline.py:227` computes
`rate=(counts+alpha)/(hex_occurrences+alpha*n_hex)` with `alpha=1.0` [Verified:
stage1_hex_baseline.py:224-229, r3] and `n_hex` = 4^6 = 4096 [Inferred: matches the
`(c+1)/(occ+4096)` that §1.1 verified]. This is a category error. `hex_occurrences[h]`
is a PER-HEXAMER opportunity count, so `counts[h]/occ[h]` is a RATE, not a share of a
multinomial. The `+alpha*K` term is Laplace smoothing for a K-category share, and it does
not belong in a per-hexamer denominator. [Verified: COORDINATION ruling 5 and gate
em_verify_stage1_smoothing] The replacement:
1. Clean rate `r(h) = C(h)/N(h)` from `simulator.measure.propensities`. That function
   applies NO smoothing. Its only guard sets `r=0` where `N <= min_expected`, to avoid a
   divide by zero. [Verified: measure.py:500-560, r3]
2. Shrinkage `(C + w·q)/(N + w)`, with `q` a backoff rate on the cuts-per-occurrence scale
   and `w` chosen on train-CV.
3. Normalise each table to sum 1. The loader does NOT renormalise: it takes
   `np.log(np.maximum(data[key], 1e-30))` and nothing else. [Verified: model.py:399-404, r3;
   Q6] So table scale lands in the model and the gauge matters (§1.1, §3.1).

Measured value: cut-weighted var/2 = 0.0160-0.0189 per table, ~0.03-0.04 nats/fragment
[Verified: check2.log:15-18; Inferred: ×2 tables]. Upper-bound flavoured, because it
assumes the unsmoothed rate is the right target. This is N0 (§3.5).

**B. 3' registration fix to `m+L` (ruling 6, APPROVED).** `CUT_OFFSET_IN_KMER=3` means a cut
at base `s` takes the 6-mer `s-3..s+2`, with the junction between `s-1` and `s` mid-k-mer.
[Verified: model.py:119-127, r3] A fragment at `m` of length `L` occupies `m..m+L-1`. Its 3'
junction is `m+L-1|m+L`, so the junction-centred cut base is `m+L`. [Inferred: from the two
definitions] The prior reads `m+L-1` [Verified: model.py:198, r3] and stage1 counts
`end = st + L - 1` [Verified: stage1_hex_baseline.py:187,313, r3]. Neither is
junction-centred. L2a already taps `WINDOW_HALF + m + L` [Verified: model.py:641-643, r3]
and the simulator already uses `ends_all = pos + Ls` [Verified: draw.py:85, r3]. **The model
moves to their convention. L2a and the simulator do not change.**
- Width is safe, but not for the reason first given. `model_input_size=1392`, `hex_idx`
  width 1387; the required width goes 1384 → 1385, headroom 2 on `hex_idx`. [Verified: EM
  measurement, gate `em_width_check_mL`, `/tmp/check_mL_width.py`] **That gate measured the
  wrong constraint, and implementation superseded it
  [Verified: COORDINATION gate `em_reviewer_found_the_old_width_guard_was_BROKEN`].** The
  binding limit is L2a's `h`, which spans `WINDOW_HALF` each side of `P`
  (`model.py:673`), so the farthest right tap sits at `WINDOW_HALF + (P-1) + L_max`, not at
  any position `hex_idx` reaches. The requirement is therefore `WINDOW_HALF >= L_max`, one
  base tighter than `hex_idx`'s own headroom would suggest. `WINDOW_HALF=181 >= L_max=180`
  still holds, and the store's `l_seq=1908` against 1904 needed still means NO store rebuild
  — hex tables only. The conclusion is unchanged; the derivation that reached it was not.
- **The OLD guard was not merely stale — it was LATENTLY BROKEN, independent of the `m+L`
  change.** `required_width = WINDOW_HALF + tile_size + L_max - 1` at `model.py:204`, and the
  constructor check `if WINDOW_HALF < L_max - 1` with its message at `model.py:292-300`,
  derive from `hex_idx`'s width, not from `h`'s. [Verified: model.py:204, 295-299, r3] Both
  move with line 198 to drop the trailing `-1`, since `required_width` must track the
  registration offset — but that is not why the constructor check was wrong. The old check
  `WINDOW_HALF < L_max - 1` would have ADMITTED `L_max = 182` at `WINDOW_HALF = 181`, whose
  farthest tap sits exactly at the boundary of `h`'s width: a guard that did not guard.
  [Verified: COORDINATION gate `em_reviewer_found_the_old_width_guard_was_BROKEN`] The
  constructor now enforces `WINDOW_HALF >= L_max` and is tested: constructing
  `CutSiteFragmentModel(L_max=WINDOW_HALF+1)` must raise (T13, §7.4). The comments at
  `model.py:180-183, 195-196` move too.
- `L_max=180` INCLUSIVE is NOT touched. L=180 is still in the domain, and that fragment
  still occupies `m..m+179`. Only the base the 3' hexamer is centred on moves, from
  `m+179` to `m+180`. [Inferred: from the occupancy definition above]

**C. Reuse, through exactly ONE tested boundary (owner reuse ruling).** Build on
`simulator/measure.py` — `propensities` (:500), `counts_from_hexamers` (:179),
`uniform_hexamer_counts` (:395), `fl_end_weight` (:312), `FragmentLengthDist` (:202) — and
not on new counting code. [Verified: measure.py, r3] The two stacks differ on two axes:
1. **Endianness.** The model and stage1 encode little-endian, `Σ base[p+i]·4^i`
   [Verified: model.py:140-144, r3]; `hexamers.py` encodes big-endian. AAAAAC is 1024 in the
   model and 1 in `hexamers.py`. [Verified: check2.log:7] This is a pure relabelling — a
   bijection on 4096 cells — so each stack is correct on its own. It breaks only at a
   crossing.
2. **RC keying.** The cut-site model keys all four tables by the FORWARD hexamer
   [Verified: model.py:186-189, r3]. The simulator indexes `start_rev` and `end_rev` through
   `rc_permutation()` [Verified: measure.py:194-198, r3]. Table names also differ:
   simulator `end_*`, npz `stop_*` [Verified: measure.py:104, model.py:401-404, r3].

Registration stops being a difference once B lands.

**Revision 4 — superseded by implementation. No `hex_adapter.py` was built, and this is not
a simplification of the plan below; the plan's premise was wrong.** The premise was that
counting, denominators, propensities and shrinkage would all run in simulator convention and
cross into the model's convention through one adapter at the `.npz` write. Implementation
found that **nothing hexamer-keyed crosses the stack boundary in the production counting
path at all**: the counter (`count_tables.py`, S3) imports only `FragmentLengthDist` and
`fl_end_weight` from `measure.py`, both length- or position-indexed, never
`counts_from_hexamers`, `propensities` or `uniform_hexamer_counts`, so the endianness and
RC-keying differences between `hexamers.py`/`measure.py` and the model never have an
occasion to apply. [Verified: COORDINATION gates `em_unblock_phase_0b`,
`em_resolve_hex_half_offset`] The counter computes hexamer indices directly in the model's
own convention throughout. An adapter would have had nothing to adapt.

**What exists instead:** the T4b probe test (§7.4), now EXHAUSTIVE over all 4096 cells
rather than a sampled probe, checks that every one of the counter's four output tables
equals the bincount of the index `compute_log_p0` actually reads, for every cell. This is a
self-consistency guard on the counter and the model's single shared convention, not a
cross-stack conformance test — there is no second convention left to be consistent with in
this code path.

**The endianness split is therefore UNRESOLVED, not solved, and remains a deferred joint
decision with the simulator EM.** `hexamers.py`/`measure.py` are still big-endian and
forward/RC-keyed differently from the model (§1.1, §3.11); that gap only mattered for a
crossing this phase turned out not to need. It still matters for any FUTURE comparison that
deliberately reads both stacks' hexamer tables side by side (e.g. a diagnostic against the
simulator's own propensities) — that comparison still needs either an adapter or a shared
encoder, neither of which exists. Do not read "no adapter was needed" as "the split is
closed." [Verified: COORDINATION gate `em_unblock_phase_0b`]

**Out of scope this phase, named follow-up:** full consolidation onto a single hexamer
encoder. It would re-key someone else's artifacts, and it is a joint call with the simulator
EM. [Verified: COORDINATION gate architecture_reflection_r2]

**D. Denominator reuse is for hygiene, NEVER for expected gain (ruling 7).** The missing
reachability ramp — what `fl_end_weight` adds to `N_end` — was measured on this store and is
NEGLIGIBLE: cut-weighted `var/2 = 0.000078` nats per stop-table read, ~200x below the
smoothing distortion. The mean log-ratio is `-0.000012`, so the ramp is almost purely a
position-independent shift, and `rate/rate.sum()` cancels shifts. Position-level variation
IS large (`w(q)` spans 0..1; 24 zero positions; 179 below half-max = 14.9% of the window),
but it does not survive renormalisation. Sample: 4,000 of 53,321 train tiles, 45,503 stop
cuts; script `/tmp/measure_ramp_cost.py`. [Verified: EM measurement, gate
em_measure_ramp_cost] Consequences:
- Round-1 F-04's "negligible" verdict was CORRECT (§1.3 F2, §3.1).
- `measure.py`'s docstring figure "up to 156x at offset 25" [Verified: measure.py:526-527,
  r3] is a genuine POSITION-level fact. It does not propagate to renormalised tables. Do not
  cite it as motivation anywhere in this programme.
- N0 adopts `N_start`/`N_end` from `uniform_hexamer_counts` because it is the correct
  denominator and it is reused code, not because it is worth nats.

**Sequencing constraint — the counter and the reader MUST move together.** If the prior
moves to `m+L` while the counter still writes `m+L-1`, tables are keyed one way and read
another. That is verbatim the bug recorded at `model.py:123-126`: "the builder registered
at cut-2 while the model read at cut+0". [Verified: model.py:123-126, r3]
`stage1_hex_baseline.py` is owned by another agent this phase. So: **the NEW counter becomes
the single table producer, and stage1's counting path is RETIRED, not edited.**
[Verified: COORDINATION "Sequencing constraint"] What "retired" requires in practice, and
**its implementation state as of Phase 0b.1 (commit `9aa8ef5`, revision 4):**
1. **DONE.** The B commit and the new counter landed together, in ONE commit (no adapter
   landed with them — §0.4 C, revision 4). No intermediate commit had the reader at `m+L`
   and only a `m+L-1` producer. Phase 0 (G0 at `bad981d`) finished BEFORE that commit.
   [Verified: COORDINATION gates `G0_phase0`, `G0b_phase_0b1`]
2. **DONE, mechanism implemented, OWNER RULING STILL PENDING (divergence 5, §9).** The new
   counter writes a registration key into the `.npz` (implemented as `stop_cut_base_offset`,
   not the proposed `stop_cut_offset = 0`). The loader (`model.py:398-404`) rejects any
   `.npz` without it — including `stage1_hex_tables_v4_start.npz` — and refuses a `bad981d`
   checkpoint via a persisted `_hex_stop_cut_base_offset` buffer and a `load_state_dict`
   pre-hook that **cannot be bypassed with `strict=False`**, because the pre-hook runs before
   the load. [Verified: COORDINATION gate `implementation_review_phase_0b`] This mechanism
   is built and tested (T13, §7.4), but — as flagged when it was proposed — it is additive
   provenance checking the owner has not explicitly ruled on. Proceeding on it was a
   deliberate EM call (gate `em_proceed_on_registration_key_without_explicit_ruling`), not an
   owner approval; it is flagged to the owner and remains open.
3. **DONE (T14, §7.4).** `tests/test_hex_registration.py` no longer imports
   `hex_window_for_tile` from the stage1 file; the two tests that did
   (`test_builder_and_model_agree_on_registration`, `test_builder_widened_window_covers_right_edge`)
   are retargeted to the new counter, deriving the far 3' cut as `tile_size - 1 + L_max`
   rather than restating stage1's offset.
4. **DONE (T12, §7.4), and proved non-tautological by MUTATION.** The cross-check tying the
   prior's 3' read to L2a's tap and to the simulator's stop hexamer is in. Both the
   implementer and the reviewer independently reverted BOTH the producer and the reader to
   `m+L-1`: 12 tests fail, including T12a/T12b and the `propensities` conformance test.
   Reverting the READER only (the keyed-one-way-read-another state this constraint exists to
   prevent) fails 6, including T4b. `model.py` was restored byte-identical afterward (`cmp`,
   then independently MD5 `d88f62c4bb88ddb476842a6b7719db14` plus `git diff --quiet`). This is
   the direct answer to the original failure mode: a suite that already "guarded"
   registration via a test pinning a convention to itself. [Verified: COORDINATION gates
   `em_note_mutation_test_proof`, `implementation_review_phase_0b`]
5. **DONE.** Relayed to session 8d207c5986
   (`/efs/analytics/nathanboley/cut_site_model_to_simulator_em_20261010.md`), including the
   corrected note that no adapter was built (§0.4 C). [Verified: COORDINATION gate
   `em_relay_phase_0b_landing`]

**Both approved changes retire `12.142214` and `11.996969`** (and their val counterparts
12.146835 and 12.002607 [Inferred: same tables, same code path]). The prior anchor needs a
RE-SCORE: rebuild the tables on CPU locally, score through the one sanctioned evaluator.
The model number needs a RETRAIN under the new prior. [Verified: COORDINATION rulings, "Both
changes retire"] `uniform` (12.674475) and `unif+p_L` (12.501596) survive: their hexamer
tables are constant, so neither registration nor smoothing reaches them. [Inferred: C2
closed form, §1.2] Until the retrain, there is no real-data capture denominator (§6.1).

**Note (revision 4):** `docs/pending/training_analysis.md` and other older docs still
describe the registration as `m+L-1`. They are STALE on that point now that ruling 6 and
commit `9aa8ef5` have landed. This document does not edit them — `training_analysis.md` is
a separate retirement decision (COORDINATION "Remaining work" item 4) — but any reader
cross-referencing them for registration should prefer this document.

---

## 1. What was verified in source, and contradictions

### 1.1 Pointer table

| Fact | Pointer |
|---|---|
| `compute_log_p0` AT `bad981d`: strand + = `start_fwd[h(m)] + stop_fwd[h(m+L-1)] + log p_L[L]`; strand − = `start_rev[h(m+L-1)] + stop_rev[h(m)] + log p_L[L]`. **After change B (§0.4) every `m+L-1` here becomes `m+L`.** | [Verified: model.py:148-229, esp. 198, 216-226, r3] |
| `h(x)` is junction-centred: a cut at base s uses the 6-mer s-3..s+2, `CUT_OFFSET_IN_KMER=3` | [Verified: model.py:119-127] |
| All four tables use the FORWARD-strand hexamer. No reverse complement in lookup or counting. | [Verified: model.py:186-226; stage1_hex_baseline.py:203-215] |
| Hence `start_fwd` and `stop_rev` are both read at the LEFT end (m). `stop_fwd` and `start_rev` are both read at the RIGHT end (m+L-1 at `bad981d`; m+L after B). | [Verified: coordinator reading model.py:186-226; r3 model.py:216-219] |
| One `p_L` (156 values) shared by both strands. No explicit log(1/2) strand term. | [Verified: model.py:216-226] |
| Normalisation: one `log_softmax` over the flattened (1024, 156, 2) domain, together with f_θ, in `domain_log_probs`. A COMMON scale on all tables is irrelevant. **A per-table scale is NOT:** scaling `start_fwd` shifts every + cell and scaling `start_rev` shifts every − cell, so unequal scales move the strand log-odds. Only shifts with a+b = c+d are a gauge (§3.1). The loader does not renormalise (Q6), so "normalise each table to sum 1" (§0.4 A) is the gauge the model actually sees. (Revision 3: the earlier "Table scale is irrelevant" contradicted §3.1 and Q6.) | [Verified: model.py:~759-773; model.py:216-226, 399-404, r3; Inferred: the strand argument] |
| `.npz` schema: `start_fwd, stop_fwd, start_rev, stop_rev` (4096,) probabilities, `p_L` (156,), plus `k, L_min, L_max`. Floored at 1e-30, then log. Buffers if `freeze_hex`, else `nn.Parameter`. | [Verified: model.py:398-422; stage1_hex_baseline.py:505-516] |
| Counting: split hardcoded `"train"`, centred crop j=0 | [Verified: stage1_hex_baseline.py:456, 140, 159] |
| One occurrence denominator for all four tables, over START positions [0,1024) only | [Verified: stage1_hex_baseline.py:160-166] |
| Numerators: start at `st`, stop at `end=st+L-1`. Stop positions run to `tile+L_max-2` through a widened window `hex_width=tile_size+L_max-1`. This counting path is RETIRED as the table producer by §0.4 (not edited). | [Verified: stage1_hex_baseline.py:150-163, 203-215; r3 :187, :313] |
| Estimator `rate=(c+1)/(occ+4096)`, `table=rate/rate.sum()`; `p_L=length_counts/sum`, unsmoothed. REPLACED by ruling 5 (§0.4 A). | [Verified: stage1_hex_baseline.py:224-229, 239] |
| Domain D = 1024 × 156 × 2 = 319,488 cells. L ∈ [25,180] inclusive. `uniform = ln 319488 = 12.674475`. | [Verified: constants.py:48-50] |
| Admission is START-based: admitted iff `min(start,end) ∈ [0,tile)` | [Verified: dataset.py:6-9; stage1_hex_baseline.py:192-195] |
| Region-weighted NLL: `nll_i = -Σ_f log p(f)/max(N_i,1)`, mean over regions, float64 accumulation | [Verified: model.py:807-810; score_fragments.py:316-345] |
| Splits: per-region permutation, seed 42. Code 2 is `"heldout_inactive"` in dataset.py. CLI `test` maps to `(2,'heldout_inactive')`. | [Verified: build_store.py:228-237; dataset.py:31; score_fragments.py:90] |
| The store holds only start, length, strand per fragment. No mapq, GC, duplicate flag or motif. | [Verified: store.py:14-15; build_store.py:289-291] |
| `totals/N` is built from endpoint-track sums, a different count from the admission rule | [Verified: build_store.py:213-222] |
| Store sequence `l_seq = 1024 + 2·(256+186) = 1908`. That is 442 bp each side of the tile at j=0. | [Verified: store.py:62-68; T6 output l_seq=1908, max index 1646] |
| Model input tensor 1392 wide, pad 184 each side, `WINDOW_HALF=181` | [Verified: measurement agent's derivation from model.py:306-313, 438-445] |
| L2a taps `m+L`. Prior, stage1 counter and `test_hex_registration` use `m+L-1`. **Resolved by ruling 6: the prior and the counter move to `m+L`** (§0.4 B). | [Verified: model.py:641-643, 198; stage1:187,313; test_hex_registration.py:108-109, r3] |
| Trainable tables are supported (`--freeze-hex` off gives `nn.Parameter`) | [Verified: model.py:406-411] |
| f_θ: 116,032 params. L1 Conv k=6 (model.py:333-371); L2a four untied banks bilinear in (L1 features, 10 length groups, `g=min((L-25)//15,9)`) at model.py:457-467,636-680; L2b interior-span-sum prefix over [m,m+L) at model.py:471-539 (reviewer round 1: the earlier pointer swapped L2a and L2b); `len_bias[L]` (156) and `w_out` zero-init at model.py:696-710 | [Verified: model.py:333-371, 457-467, 471-539, 636-680, 696-710] |
| Sim generator: strand ~ Bern(0.5); `m ~ a(h_m)·live(m)`; `L ~ e(h_{m+L})·f(L)/t(m)`. No GC, nucleosome or band term. | [Verified: background_model/simulator/draw.py:96-162] |
| **`background_model/hexamers.py` and `background_model/simulator/measure.py` are NOT index-compatible with the model.** `hexamers.py` encodes hexamers BIG-endian (`_POW=[4^5..4^0]`); the model and stage1 encode LITTLE-endian (`base[p+i]*4^i`). AAAAAC → index 1024 in the model/stage1, index 1 in `hexamers.py`. `measure.py` right-anchors the stop end at `m+L` (not the prior's `m+L-1`) and applies the rc permutation to minus-strand tables. An `.npz` built from these pieces loads without error and the model reads every cell under the wrong 6-mer. **Revision 3:** this is no longer a reason to avoid `measure.py`. The owner's reuse ruling makes it the plan, with ONE adapter as the only crossing (§0.4 C). After B, the `m+L` registration is shared and stops being a difference; endianness, RC keying and table names remain, and the adapter handles exactly those. **Revision 4: no adapter was built, because the production counting path never imports these hexamer-keyed functions** (`counts_from_hexamers`, `propensities`, `uniform_hexamer_counts`) — only length/position-indexed `FragmentLengthDist`/`fl_end_weight`. So this incompatibility never has an occasion to bite in the implemented system. It remains REAL and UNRESOLVED for any future comparison that reads both stacks' hexamer tables side by side; that comparison still needs an adapter or a shared encoder (§0.4 C). [Verified: COORDINATION gate `em_unblock_phase_0b`] | [Verified: check2.log:7; model.py:140-144; stage1_hex_baseline.py:58-65; hexamers.py:58,95; measure.py:169-170,197; r3 measure.py:104, 194-198] |
| No GC, mappability, CpG or nucleosome track. No PWM, Markov or GLM cleavage model in the cut-site path. `fragmentomics_tools/motif.py` is a JASPAR TF-PWM class. | [Verified: research agents B, C] |

### 1.2 Contradictions of the brief

**C1. Test NLL has no single sanctioned flag.** `train.py --validate-only` runs only on val.
[Verified: train.py:169-176] The measurement agent reproduced the test number 12.142214
exactly. The method: the sanctioned per-fragment TSV (`per_fragment_prior.tsv`, which has
`split` and `scored` columns), reduced by `score_fragments.region_weighted_mult_nll` with
`CutSiteFragmentDataset(split="heldout_inactive", min_N=10, train_mode=False)`.
[Verified: /tmp/cutsite_design_diag/t1_t2_t3.py, ephemeral] This reduction is seam S2
(§4.2). It must become a committed wrapper that imports the function and does not
re-implement it.

**C2. `unif+p_L` has no producing code.** [Verified: research agent A grep] A uniform-table
`.npz` (each table 1/4096) plus the real `p_L` produces it through the sanctioned path. With
uniform tables, `p(m,L,s) = p_L[L]/2048`. So `nll_i = ln 2048 − mean_f log p_L[L_f]` per
region, with `ln 2048 = 7.624619`. [Inferred: the softmax over 1024·2 positions×strands
with constant table terms] This closed form gives a free check on the reduction. It is a test
assertion, not a second evaluator.

**C3. The fragment count the prior sees.** [Verified: T1/T3; summary json]

| Population | Fragments |
|---|---|
| Stored, 1536-bp window, all splits | 1,123,884 |
| Scored (centred crop), all splits | 749,554 |
| Unscored (jitter margin) | 374,330 (= 0.3330 of stored; 512/1536 = 0.3333) |
| TRAIN stored | 899,222 |
| TRAIN centred crop (what stage1 counts) | 599,412 |
| TRAIN jitter margin (unused by the prior) | 299,810 |

The store directory name `tile1536` names the 1536 WINDOW. `hparams.tile_size` is 1024.
[Verified: summary json]

**C4. min_N=10 drop.** [Verified: T1] train 53,321 → 50,425 (5.43%); val 6,664 → 6,286
(5.67%); test 6,664 → 6,303 (5.42%). There are 43 zero-fragment tiles (32 train, 4 val,
7 test). This reconciles 66,649 store tiles with 66,606 scored regions.

### 1.3 Additional findings

- **F1. The stage1 smoothing distorts rare hexamers, and the distortion is BIAS-side, not
  variance-side. Fix APPROVED (ruling 5, §0.4 A).** `rate=(c+1)/(occ+4096)` has the form of
  a hierarchical estimator `(c+α·r̄)/(occ+α)` with α=4096 and r̄=1/4096. [Inferred: algebra
  on stage1_hex_baseline.py:224-229] Both α and r̄ come from the CATEGORY count K=4096 of a
  Laplace-smoothed share, not from anything about a per-hexamer rate — that is the category
  error. r̄=1/4096 ≈ 0.000244 sits ~22x below the pooled empirical rate of 0.0055 cuts per
  occurrence quoted below. [Inferred: 0.0055/0.000244] So stage1 shrinks every hexamer
  toward a rate it does not have, with a weight nobody chose. The mask is all ones: stage1 counts occurrences over ALL
  53,321 TRAIN tiles, 54,600,686 total occurrences in the start domain, not "53,289 tiles"
  as an earlier draft had it, and nothing about masking lowers the denominator.
  [Verified: build_store.py:92; check2.log:8-9] Mean occurrence per hexamer = 54,600,686/4096
  = 13,330 [Inferred: arithmetic], occurrence range 199-105,176 (check2.log:9-10). The pooled
  empirical rate is 0.0055 cuts per occurrence (check2.log:15-18). Reviewer round 1 measured
  the actual distortion directly, by rebuilding stage1's four tables exactly (max rel diff 0.0
  vs the production npz; check2.log:13) and computing the cut-weighted
  Var[log(occ/(occ+4096))]/2 per table: **0.0207 / 0.0197 / 0.0219 / 0.0191** nats
  (start_fwd/stop_fwd/start_rev/stop_rev). With the +1 in the numerator, the exact
  cut-weighted var/2 of log(smoothed/raw) is **0.0178 / 0.0166 / 0.0189 / 0.0160** nats; the
  per-hexamer smoothing factor ranges [0.046, 0.963]. [Verified: check2.log:15-18] Each
  fragment reads two of these four tables, so the second-order distortion cost is
  ~0.03-0.04 nats per fragment [Inferred], 2-3x the entire k/2N noise budget (§2.5). This is
  a BIAS term, not a variance term — see F-03/§3.5 for the re-ranking this implies. The
  realised NLL gain from fixing it is still [Unverified: score an N0 npz through the
  sanctioned evaluator]: the figure above assumes the unsmoothed rate is the right table, but
  raw rates add noise, and the intended N0 form (a hierarchical `(c+w·q̄)/(occ+w)` with small
  w) trades the two rather than removing smoothing outright.
- **F2. Stop-table denominator — BOTH halves now measured, both negligible.** Stop
  numerators count ends over tile positions [24, 1202] (at `m+L-1`; [25, 1203] after B
  [Inferred: +1]). Denominators count occurrences over [0, 1024).
  [Verified: stage1_hex_baseline.py:160-166 vs 203-204] (i) Domain EXTENT: reviewer round 1
  measured the occurrence ratio (stop domain / start domain) at 1.10-1.20, median 1.15
  (check2.log:21); matching the domain moves the stop tables by a cut-weighted sd of 0.016
  nats (check2.log:19-20), i.e. a cost of order var/2 ~ 1e-4 nats. (ii) Reachability RAMP,
  the `fl_end_weight` effect: the EM's critical review objected that (i) did not cover it.
  It has since been measured at cut-weighted `var/2 = 0.000078` nats per stop-table read,
  with a mean log-ratio of `-0.000012`, so renormalisation cancels it. [Verified: EM
  measurement, gate em_measure_ramp_cost; details §0.4 D] F-04's "negligible" verdict was
  therefore CORRECT; the EM's objection was right on method only. Neither half is a
  motivation for N4 (§3.1) or for N0. N0 adopts `measure.py`'s denominators for hygiene and
  correctness only (ruling 7).
- **F3. Torch version drift.** biomarker_env has torch 2.6.0+cu124. CLAUDE.md says 2.5.1.
  [Verified: env listing]
- **F4. The best checkpoint's 12.002583 is VAL, not test.** It is also selected on val, so it
  is optimistic for val. [Verified: summary json; Inferred: selection]
- **F5. The model column is an upper bound.** 11.996969 (test) came from a run that was
  still descending. The loader phase will move it. [Verified: packet; Inferred: test split,
  because 12.142214 − 11.996969 = 0.145245 matches the published 0.145] **Both numbers are
  now RETIRED as anchors (§0.4).** The prior needs a re-score and the model a retrain under
  the new prior.
- **F6. The sim right end IS misregistered against the prior — reviewer confirmed, with three
  consequences.** The sim draws and measures the right end at `m+L` (draw.py:85,100;
  measure.py:169-170); the prior reads `m+L-1` (model.py:198); L2a taps `m+L` (model.py:643);
  `tests/test_hex_registration.py` checks only the model's internal `m+L-1` convention
  (lines 108-110, 336) and nothing ties the sim to the prior. [Verified: reviewer round 1]
  Three consequences, all [Inferred]: (1) the sim's prior-oracle gap (0.064 full, 0.227 on
  10pct) contains a registration component, not purely a sequence-signal gap; (2) part of the
  model's 56.7%/48.0% sim capture can be L2a recovering that one-base registration, so the
  "correction is worth ~4x more on real than sim" contrast (previously §6.6) is CONFOUNDED
  and must not be used as evidence — the separate conclusion that real is the arbiter is
  unchanged; (3) M5's `t̂` at `m+L-1` (§3.8) is therefore only an attenuated positive control
  on sim. The sim minus strand uses the rc track at the same windows with `end_rev` on the
  left and `start_rev` on the right (draw.py:95-97); forward counting absorbs this as a
  permutation, so N2's sim meaning is unaffected and stays clean.
  **After change B (§0.4) the misregistration is removed for every FUTURE comparison:** the
  prior, L2a and the simulator all register the 3' cut at `m+L`. [Inferred: §0.4 B] The
  three consequences above still apply to every number measured at `bad981d` — the sim gaps
  0.064/0.227 and the 56.7%/48.0% captures are historical and keep the confound. Re-score the
  sim prior with the new counter before quoting any sim gap. [Unverified: re-score both sim
  stores, P0b.4]

### 1.4 Conflicts inside the packet

- T5 uses n_val = 73,399 fragments. The val scored count is 75,308. The difference is the
  population: T5 used min_N=10 regions (73,399/6,286 = 11.68 per region). The scored count
  includes regions with N<10. Test is the same pattern: 6,303 × 11.58 ≈ 72,989 vs 74,834
  scored. [Inferred: arithmetic]
- The packet asserts that "the separable prior implies ~no dependence of the hexamer on the
  length band". I disagree in part. The separable prior still implies some dependence,
  through sequence autocorrelation at spacing L and through the per-region softmax. Short
  fragments sit nearer the strongly selected left cut, so their right-end composition differs
  from that of long ones. [Inferred: structure of compute_log_p0] T5 (a)/(b) are therefore
  only *partly* clean. The decisive test is observed vs expected-under-prior (P0.5).
- The two joint 6-mer KLs sum to 0.2274. The prior's measured sequence gain is 0.359. These
  are different quantities: pooled-train KL vs region-weighted test NLL over D. Use the ratio
  of ~1.6 only as a warning that pooled KL does not convert 1:1 to NLL. [Inferred]
- `FL_BANDS` bands are half-open `[lo,hi)` (CLAUDE.md). With `L_max=180` inclusive, a strict
  half-open band 2 would exclude L=180. This design uses `b = 0 if L < 110 else 1` for
  L ∈ [25,180]. That uses the locked edges exactly and puts L=180 in band 1. T5 did the same
  (0.5315 + 0.4685 = 1). [Unverified: `grep -rn FL_BANDS $W/background_model $W/scripts` to
  confirm how other code treats L=180] FL_BANDS itself is not open for change.

### 1.5 Items to report to the stage1 owner (not edited here)

Found during round-1 review; stage1 is owned elsewhere this phase (D2) and is not touched by
this design. [Verified: reviewer round 1]

- `stage1_hex_baseline.py:469-470` prints max/min table ratios. That violates the project's
  KL-not-max/min rule.
- It points to a nonexistent `train_cut_site.py --baseline-only`.
- It drops N-containing windows at count time, while the model reads N as A at score time.
  **The magnitude is UNRECONCILED and both figures are recorded rather than one picked.**
  Round 1 measured **31** such TRAIN fragments out of 599,412 [Unverified now:
  check2.log:8, whose script is gone]; the P0b.1 counter measured **1** dropped (at L=179)
  under the same 599,412 denominator [Verified: COORDINATION gate `G0b_phase_0b1`]. The two
  may count different things — round 1 was at the `m+L-1` registration and may have gated on
  either window, while the counter drops a fragment only when a window it actually reads is
  invalid — but a one-base registration shift is a surprising explanation for 31 -> 1, so
  treat this as open. Negligible at either value (both are <0.006% of 599,412), so it does
  not gate anything; settle it by rerunning the predictor if the N-handling inconsistency is
  ever acted on. **The reused `measure.py` path has the same behaviour**: it drops
  fragments whose start OR stop window holds a non-ACGT base [Verified: measure.py:162-171,
  r3]. The inconsistency therefore survives the switch to the new counter. Record it in the
  counter's provenance sidecar; do not patch it in this phase.
- **Revision 3 — the main item for the relay:** under the sequencing constraint (§0.4) its
  counting path is RETIRED as the table producer. Its `.npz` output is rejected by the
  `m+L` loader. Two tests in `tests/test_hex_registration.py` that import its
  `hex_window_for_tile` are retargeted to the new counter. Send to session 8d207c5986.

---

## 2. Evidence on the table

### 2.1 Anchors (real, region-weighted)

**Revision 3 status.** `uniform` and `unif+p_L` stay valid anchors. Every `prior` and
`model` row below is RETIRED as an anchor by rulings 5-6 (§0.4). They are HISTORICAL:
reproducible only with `bad981d` model code plus `stage1_hex_tables_v4_start.npz`. Phase 0
reproduces them once (G0) to prove the reduction, and never compares a candidate to them.
The replacement prior anchor comes from the P0b re-score; the replacement model number
needs a retrain. [Verified: COORDINATION rulings 5-6, "Both changes retire"]

| Row | Val | Test | Tag |
|---|---|---|---|
| uniform | 12.674475 | 12.674475 | [Verified: property of D, constants.py:48-50] |
| unif+p_L (closed form) | 12.5027343162787 | 12.50159635865178 | [Verified: reviewer recomputation, check2.log:26-27 — resolves the anchor by closed form; P0.1 must still reproduce it through the npz + sanctioned-evaluator path] |
| prior (min_N=10) | 12.14683460920793 (n=6,286) | 12.14221406310407 (n=6,303) | [Verified: measurement agent, region_weighted_mult_nll; reproduced by reviewer, check1.log:15,17] |
| prior (min_N=1) | 12.135744 | 12.136553 | [Verified: same; reviewer reproduction check1.log:16,18] |
| prior (pooled-fragment) | 12.137552 | 12.132684 | [Verified: same] |
| model (sanctioned fp32 evaluator, best ckpt, val-selected) | 12.002607100258485 | 11.996968978966706 (upper bound) | [Verified: reviewer recomputation, check1.log:19-20] |
| model (trainer, bf16-mixed, same ckpt — filename value, NOT sanctioned) | 12.002583 | — | [Verified: ckpt filename; differs from the fp32 scorer value above because `train.py`'s default precision is bf16-mixed (train.py:101-102); see F-09/§7.2] |

Derived (test): length gain 12.674475 − 12.501596 = 0.172879. Sequence gain
12.501596 − 12.142214 = 0.359382. Correction gain 12.142214 − 11.996969 = 0.145245.
[Inferred: arithmetic; all three inputs are now [Verified] per the row above] The length
gain survives the change set. The sequence and correction gains are HISTORICAL (§0.4).

The same prior differs by 0.0046 between val and test. A change of population moves the
number by 0.005-0.01. [Verified: rows above] This is the yardstick for UNPAIRED comparisons
(§6.2).

Sim anchors: full prior 12.071311, oracle 12.007078, gap 0.064233; 10pct prior 12.214898,
oracle 11.987397, gap 0.227501. The model captured 56.7% and 48.0% of these gaps. On full
sim the correction is worth 0.567 × 0.064233 ≈ 0.036, against 0.145 on real.
[Verified: packet anchors; Inferred: products] [Unverified: which split the sim anchors use;
read the sim run's summary json] The sim PRIOR numbers and the gaps are also HISTORICAL:
they were scored at `m+L-1` and carry the F6 confound. The ORACLE numbers do not read the
prior, so they survive. [Inferred: the oracle is the generator's own `p`]

### 2.2 T4: single-base information vs offset (real TRAIN, N=599,412, strands pooled)

[Verified: measurement agent T4] KL in nats of the base at the cut vs background. Plug-in
bias is ~2.5e-6 per offset.

| Quantity | Left end (start) | Right end (stop = m+L-1, as MEASURED; see the anomaly note) |
|---|---|---|
| Peak | o=0: 0.0719 | o=±2: ~0.0155 |
| o=0 | 0.0719 | 0.0009 |
| Σ single-base, 6-mer window [-3,+2] | 0.1194 | 0.0557 |
| Σ single-base, [-12,+12] | 0.1346 | 0.0666 |
| Beyond the window | +0.0152 | +0.0109 |
| Joint 6-mer KL | 0.1464 | 0.0810 |
| Joint − Σ singles (within-window dependence) | 0.0270 | 0.0253 |

Joint 6-mer plug-in bias is 4095/(2N) = 0.0034. [Inferred]

**Anomaly.** Left-end information peaks at the first fragment base. Right-end information
is ~0 at the last fragment base and peaks two bases away. A physically symmetric cut would
put a strong signal at the last base too. The cause is either a real asymmetry or the
registration. The anomaly did not invalidate the old prior, because counting and lookup used
the same index. **Revision 3:** the registration is now decided (`m+L`, ruling 6), so this
anomaly is no longer evidence for or against it. Under `m+L` the offsets in the right-hand
column shift by one: the old o=+1 is the new o=0. [Inferred: offsets are relative to the
cut base] The ±2 description does not say which side, so the table cannot be re-indexed by
hand. P0.4 re-measures it per strand at `m+L`, and the result decides where a right-end
flank term must sit.

The beyond-window terms are MARGINAL information. Part of them can be regional composition
(cuts fall in AT-rich stretches), and part is already in the 6-mer. Only an obs/exp residual
under the prior (P0.5) measures the *conditional* gain. [Inferred]

### 2.3 T5: cross-fitted dependence diagnostics (fit TRAIN, eval VAL min_N=10, n=73,399)

[Verified: measurement agent T5] These are observed-fragment quantities, not region-weighted
NLL over D. Signal ≈ held-out − permutation null. The null is the cost of fitting k cells
and matches the k/2N scale.

| Test | k | Held-out | In-sample | Null | Signal estimate | Confound status |
|---|---|---|---|---|---|---|
| (a) h5' given L-band | 8192 | +0.00120 | +0.00830 | −0.00839 | ≈0.0096 | Partly clean (§1.4) |
| (b) h3' given L-band | 8192 | +0.00613 | +0.01412 | −0.01450 | ≈0.0206 | Partly clean (§1.4) |
| (c) I(left dinuc; right dinuc given band) | 512 | +0.01186 | 0.01288 | **+0.00914** | not interpretable | CONFOUNDED: the null is positive, because a band shuffle keeps unconditional left-right dependence, which the prior also implies |
| (d) L(156) given interior GC (10 bins) | 1560 | +0.02295 | — | −0.04037 | not interpretable | CONFOUNDED: interior GC of short spans is mechanically more dispersed (var ~1/L), and the prior has the same mechanics |
| (d') L(10 groups) given GC bin | 100 | +0.00538 | — | −0.00596 | not interpretable | CONFOUNDED: same |
| (e) L given tile GC (5 bins) | 780 | +0.00069 | — | −0.00202 | ≤~0.0027 | Cleanest. Tiny. REJECT. |

Signal estimates: (a) 0.00120 + 0.00839 = 0.00959; (b) 0.00613 + 0.01450 = 0.02063;
(e) 0.00069 + 0.00202 = 0.00271. [Inferred: arithmetic] An unshrunk table keeps only
0.0012 / 0.0061, because its noise cost eats the rest. [Verified: held-out column]
(a) and (b) pool strands and use two positional tables, not four. [Verified: k=8192=4096×2]

### 2.4 T6: per-start normaliser (500 TEST tiles with ≥10 fragments, first by index)

[Verified: measurement agent T6] `t̂(m) = Σ_L stop_table[h(m+L-1)]·p_L[L]`; the rev strand
follows the model's index expression. The within-tile sd of log t̂ over all m is 0.0685 fwd /
0.1166 rev. At observed starts it is 0.0572 / 0.0982. The fragment-weighted, tile-centred
mean at observed starts is +0.0031 / +0.0212. Off-sequence cells are 0%, N-contaminated
windows 0%.

Bound: if sim-like start-then-length structure held, adding −γ·log t̂(m) changes NLL by at
most ~sd²/2: 0.0572²/2 = 0.0016 (fwd) to 0.0982²/2 = 0.0048 (rev). [Inferred: second-order
expansion] The sign on real data is not guaranteed. T6 used the old stage1 tables at
`m+L-1`. After the change set, `t̂` uses the N0 tables at `m+L`, which matches the
generator's `e(h_{m+L})`. Re-measure T6 then; do not carry these sds. [Unverified: rerun T6
after P0b]

### 2.5 Estimation-noise budget

Basis: AIC asymptotics. The expected held-out excess of an unshrunk k-parameter fit is about
k/(2N) nats per fragment. [Inferred: correctly specified, asymptotic; sparse cells make it
rough; it maps to region-weighted NLL only approximately]

| Configuration | k | N | k/2N |
|---|---|---|---|
| Current prior | 4·4095 + 155 = 16,535 | 599,412 | 0.01379 |
| + jitter-margin fragments (N1) | 16,535 | 899,222 | 0.00919 (saves 0.0046) |
| Strand-pooled tables (N2) | 2·4095 + 155 = 8,345 | 599,412 | 0.00696 |
| N2 + N1 | 8,345 | 899,222 | 0.00464 |
| Band deltas, unshrunk (M2) | +16,380 | 599,412 | +0.0137 |
| Flank PWM (M1): 19 offsets × 3 × 4 | +228 | 599,412 | +0.00019 |
| Interior GC × L-group (M3) | +100 | 599,412 | +0.0000834 (= 100/1,198,824 ≈ 8.3e-5; an earlier draft had +0.00004) |
| Interior GC × full L | +1,560 | 599,412 | +0.0013 |

Events per table cell: ~599,412/2 = ~300k per table / 4096 ≈ 73. Per band: ≈ 39 (short,
53.15%) and ≈ 34 (long, 46.85%). [Inferred: arithmetic]

**Implication.** The (historical, §0.4) 0.145 correction gain is about 10× the whole noise budget (0.014). Most
of it is therefore BIAS (missing structure), not variance. The strictly variance-side
candidates **N1, N2, N3** are worth at most ~0.01 nats together. [Inferred] **N0 is NOT in
this group.** Reviewer round 1 (§1.3 F1) measured F1's smoothing distortion directly at
0.016-0.019 nats per table, 2-3x the k/2N budget on its own — that is a BIAS cost, and N0 is
re-ranked to the top of the shortlist on that basis (§3.0, §3.5), then approved as the new
prior (ruling 5). Do not oversell N1/N2/N3. The k/2N rows above are unchanged by the change
set: B and the new denominators change no parameter count and no N. [Inferred]

The reviewer could not re-verify the 10-fold CV sd of the hex baseline (0.0059); it comes
from a section of `training_analysis.md:1450-1456` that was itself retracted.
[Verified: reviewer round 1, docs agent D] Treat 0.0059 as a demoted, unconfirmed figure,
not a working estimate of run-to-run noise.

### 2.6 What is known about the correction's 0.145

Nothing structural. No ablation of L2a, L2b or `len_bias` exists. No per-L or per-position
decomposition exists. The §10 numbers were retracted twice. [Verified: docs agent D] With the
tables frozen, f_θ is the only place that can recalibrate the tables or the length marginal.
Part of the 0.145 can therefore be pure calibration that N4 recovers.
[Inferred: `len_bias` and L2a structure, model.py:457-467, 636-680, 696-710] The 0.145 is
HISTORICAL (§0.4). Part of it was L2a doing work that the approved changes move into the
prior: the `m+L` read that L2a already had, and recalibration of the smoothed tables.
[Inferred: L2a taps `m+L`, model.py:641-643] Expect the retrained correction gain to be
smaller. Do not read that drop as a loss of model quality.

---

## 3. Ranked shortlist

### 3.0 Summary ranking

The score is roughly (expected gain × probability of a real gain) / build cost. "Bound" is
the best evidence-based ceiling, not a forecast. No gain is established.

| Rank | ID | Candidate | Layer / seam | Extra params | Bound or evidence | P(gain > 2SE) | Build cost |
|---|---|---|---|---|---|---|---|
| 1 | N0 | Counted tables with a corrected smoothing estimator, at `m+L` — **APPROVED (rulings 5-6); becomes the new prior anchor, not a candidate; BUILT** | N + S3 + B (no adapter, revision 4) | 0 | MEASURED distortion 0.016-0.019 nats/table, BIAS-side, 2-3x k/2N, upper-bound flavoured (§1.3 F1, §3.5) | medium-high | low-medium (lands with B, §0.4) |
| 2 | N4 | Region-weighted ML calibration of 4 tables + p_L | N (npz-only) + engine | 0 | unknown; the depth confound (§3.1). F1 is now absorbed by N0 | high | medium (engine) |
| 3 | N1 | Count the TRAIN jitter-margin fragments | N + S3 | 0 | ≤0.0046 (noise only) | medium (leakage verified safe on this store, §3.2) | low |
| 4 | M2 | L-band-conditional hexamer deltas, shrunk | M + S1 | ≤16,380 (or 252 lite) | ≤~0.030 observed-fragment, partly confounded | medium | medium |
| 5 | M1 | Flank PWM beyond the 6-mer window | M + S1 | 228 | ≤0.026 in-sample MARGINAL KL | low-medium | medium |
| 6 | M3 | Interior GC × length group | M + S1 | 100 | unknown; T5 (d) confounded | low-medium | medium |
| 7 | N2 | Strand-pooled (positional-tie) tables, DIAGNOSTIC | N + S3 | −8,190 | ≤0.0068 (noise only) | low-medium | low |
| 8 | M5 | Per-start normaliser −γ·log t̂(m) | M + S1 | 1 | ≤~0.005, sign unknown | low | low (after S1) |
| 9 | N3 | 6-mer backoff toward a per-offset PWM product | N + S3 | 0 (+1 hyper) | a part of ≤0.0138 | low | low |
| 10 | M4 | Left×right dinucleotide residual interaction | M + S1 | 512 | T5 (c) not interpretable | low | medium |
| — | M6 | k=8 context with backoff to k=6 | M + S1 | large | deferred until M1 passes | — | high |

**Reviewer round 1, F-03: N0 moved from rank 5 to rank 1.** The original ranking called N0
"variance-side" and subsumed by N4; that was wrong — F1's distortion is a measured BIAS term
(§1.3 F1), not variance, and the reviewer's own P0.2 predictor recomputation (check2.log)
already gives numbers, not just a placeholder. **Revision 3:** the owner then approved N0
(ruling 5) together with the `m+L` fix (ruling 6). N0 is no longer scored as a Phase-1
candidate. It is built in Phase 0b (§5) as part of the change set, and its score becomes the
new prior anchor. Ranks 2-10 are the candidates, all measured against that anchor. N4 keeps
its own motivation (the depth confound, §3.1) and runs on top of N0's tables, not instead
of them. **Stacked gains are not additive and remain unmeasured** — do not read the per-row
bounds in this table as summable.

### 3.1 N4 — region-weighted maximum-likelihood calibration of the existing prior

- **Structure exploited.** The evaluated model is log-linear with a per-region softmax. The
  counted rates are not its MLE, for two reasons (reviewer round 1, F-04: the earlier draft's
  reason (ii), the F2 stop-domain mismatch, is dropped here — measured at sd 0.016, ~1e-4
  nats, negligible; it is a correctness fix bundled into N0, §3.5, not a motivation for N4).
  Revision 3: the reachability ramp, the other half of the denominator question, is also
  measured negligible (0.000078 nats per stop-table read; §0.4 D). F-04's verdict stands.
  (i) F1 smoothing distortion (§1.3 F1; subsumed into N0's tables, which N4 should initialise
  from). (ii) A depth confound: under the model, the expected cuts at hexamer h are
  ≈ θ_h·Σ_r N_r·occ_{r,h}/Z_r. The counted estimator divides by Σ_r occ_{r,h}. These differ
  when h concentrates in regions with high N_r/Z_r (deep regions relative to their sequence
  propensity). [Inferred: algebra on compute_log_p0 and stage1:224-229] Also, `p_L` is the
  raw observed marginal. The model-implied length marginal is `p_L[L]·Σ_m a·e(L)`, which
  varies with L through sequence autocorrelation. [Inferred]
- **Cost.** 16,535 params, the same as now. No evaluator change: the output is a conforming
  `.npz`.
- **Estimation (TRAIN only) — reviewer round 1, F-02: rewritten.** The update
  `log θ_T[h] += η·(log(O+α) − log(E+α))` has fixed point O=E exactly, for any α>0: α only
  damps the step size, it is NOT a regulariser, and the earlier draft's claim that "α guards
  against chasing rare hexamers" is false. The fixed point of that update IS the unregularised
  MLE. [Verified: algebra on the update rule] Each cell reads exactly one start table, one
  stop table and `p_L` (model.py:216-226): 3 features per cell, so damped-Jacobi/GIS is
  guaranteed to converge only for **η ≤ 1/3**; the earlier η≈0.5 can overshoot. The model is
  invariant only to joint shifts (a,b,c,d,e) of (start_fwd, stop_fwd, start_rev, stop_rev,
  log p_L) with the constraint a+b = c+d (strand log-odds is NOT a free gauge); re-centring
  each log table independently after every sweep, as the earlier draft specified, changes the
  strand log-odds every sweep and is wrong. The loader does not renormalise
  (model.py:401-404), so this gauge error lands directly in the model, not in a step the
  loader would undo. Maximise Σ_r (1/N_r)·Σ_f log p(f) over the TRAIN min_N=10 population;
  the objective is **concave in log θ**. Use a real penalty instead of α-as-regulariser: a
  ridge `λ·‖log θ − log θ0‖²` with θ0 = the N0 tables, or Dirichlet pseudo-observations on the
  O side only. Optimise with L-BFGS (scipy 1.15.2 is installed) on the exact gradient O−E
  (plus the penalty gradient), or with GIS at η ≤ 1/3 if a closed-form multiplicative update
  is preferred. Fix the gauge with ONE common constant (or constants preserving a+b = c+d),
  applied once at the very end, never per sweep. Choose λ by train-CV, not by sweep count on
  val. Initialise from the N0 tables (not the raw stage1 tables).
- **Expected gain.** Unknown. [Unverified: Phase 1 N4 row] The P0.5 predictor is the
  region-weighted KL(O‖E) per table under the N0 tables at `m+L` (the new anchor; the old
  stage1 tables are retired, §0.4). That approximates the in-sample gain of one exact
  penalised step. Subtract the permutation null.
  Sim positive control: counted ≠ MLE also holds on sim, so N4 must close part of the sim
  prior-oracle gap. Use the gap RE-SCORED after the change set, not the historical 0.064 /
  0.227, which carry the F6 registration confound (§1.3 F6).
- **Overfit risk and detection.** k is unchanged, but the MLE can chase rare hexamers more
  than the smoothed counts do; the ridge/Dirichlet penalty above, not α, is what guards
  against that. The per-strand, region-weighted fit can chase the noise of low-count regions.
  Detect with (1) the held-out-fold NLL per sweep inside train-CV (early stop; val is
  confirmation only, R2-02 — revision 3 fixes this bullet, which still said "val NLL per
  sweep"), (2) the train-CV optimism gap, (3) a learning curve at 25/50/100% of TRAIN
  regions, (4) a comparison against the fragment-weighted fit.
- **Failure modes.** (a) Gain ≈ 0 because the counted rates are already ≈ MLE. P0.5 shows
  KL(O‖E) ≈ null and Phase 1 shows Δ within 2SE. (b) Gain on val that does not hold on test:
  over-tuned sweeps or λ. The test-touch ledger and the train-CV gap reveal it. (c) The
  engine drifts from the evaluator. The engine identity test (§7.3 T5) catches that.
  (d) ~~The schema cannot carry the fwd/rev relative scale if the loader renormalises each
  table.~~ Moot: the loader does not renormalise (Q6). [Verified: model.py:399-404, r3] The
  schema CAN carry the fwd/rev scale, so the end-of-fit gauge constant must preserve
  a+b = c+d, or the fit's strand log-odds is lost at write time. **Do NOT re-normalise N4's
  output to sum 1 per table** to match N0's convention: per-table normalisation applies
  shifts that need not satisfy a+b = c+d, so it would undo the fitted strand log-odds. The
  size of that error is bounded by the strand marginal, minus-strand fraction 0.4992.
  [Verified: T1; Inferred: the gauge argument]
- **Seam.** Layer N. It needs the expected-count engine (§4.4) and S2.

### 3.2 N1 — count the TRAIN jitter-margin fragments

- **Structure.** None new. It is +50% data: 899,222/599,412 = 1.50. [Inferred]
- **Cost.** 0 params. It needs the S3 counter, because stage1 hardcodes j=0.
- **Estimation.** Count fragments admitted over the 1536-bp start window, and count hexamer
  occurrences over the SAME start window. Context fits: the farthest start is
  1024+256−1 = 1279. Under `m+L` its 3' cut base is 1279+180 = 1459 and its hexamer needs
  1459+2 = 1461, inside the sequence span [−442, 1466) in tile coordinates. (At `m+L-1` it
  was 1458 and 1460.) [Inferred: from l_seq=1908 and §0.4 B] The denominators come from
  `uniform_hexamer_counts` over the same 1536-bp window, so `fl_end_weight` uses
  `region_len=1536` here, not 1024. [Inferred: measure.py:312-333, r3]
- **Expected gain.** ≤0.0046 if the noise budget holds. [Inferred: §2.5] P0.3 measures the
  real noise component.
- **Overfit risk.** None added. **Leakage — reviewer round 1, F-14: VERIFIED SAFE on this
  store.** The reviewer checked every form of overlap (train window × val/test crop, crop ×
  crop, window × window, sequence-span × crop) and found ZERO in all cases; the minimum
  tile-start gap is 1536 (windows abut exactly, never overlap), with 6,036 abutting pairs
  (check1.log:9-13). `build_store.py` has no merge or overlap logic and splits are a
  per-region permutation (build_store.py:228-237), so this is a structural property of the
  store, not a coincidence of this build. N1 is therefore leakage-safe on `quiet_tile1536_v3`.
  Keep a guard for OTHER stores: replace the `--exclude-cross-split-overlap` filter (§4.3,
  §7.3) with an **assertion** that the cross-split overlap count is 0 and abort if it is not,
  rather than silently dropping fragments — on this store the assertion simply always passes.
  Note for interpretation elsewhere: abutting tiles correlate spatially, which makes
  region-level bootstrap and region-level CV slightly anti-conservative (§6.2).
- **Failure modes.** (a) Margin composition differs from the core: compare core-only vs
  margin-only tables (held-out log-likelihood of each on the other), and check that the
  learning-curve slope predicts the gain. (b) The gain is far below 0.0046 because the
  estimator is not noise-limited: P0.3 reveals it first.

### 3.3 M2 — length-band-conditional hexamer deltas (shrunk)

- **Structure.** A cut-hexamer preference that differs between `[25,110)` and `[110,180]`.
  Bands: `b = 0 if L<110 else 1`. The edges are exactly the locked FL_BANDS. The network's L2a
  is a learned low-rank hexamer × length-group interaction, the closest structural match to a
  known network component. [Verified: model.py:457-467, 636-680 — reviewer round 1,
  F-15: the earlier pointer (471-539) is L2b's interior-span-sum prefix, not L2a]
- **Cost.** M2-full: δ_{T,b}[h] with a sum-to-zero over bands, so 4·4095 = 16,380 params.
  M2-lite: deltas on the 3-mer centred on the cut (bases s-1..s+1), 4 tables × 63 = 252
  params. [Inferred]
- **Estimation.** MAP in the region-weighted objective with a ridge penalty λ‖δ‖², fitted by
  the engine with offset log p0 (or the N4 tables). Choose λ on 5-fold train-CV. Bracket the
  curve with λ=∞ (equal to the base prior) and λ=0 (raw bands), and show the whole curve.
- **Expected gain.** ≤ about 0.030 in observed-fragment units (0.0096 left + 0.0206 right
  from T5 (a)/(b), strands pooled). [Inferred: §2.3] Part of it is mechanical (§1.4). The
  exchange rate to region-weighted NLL is unknown.
- **Overfit risk.** High if unshrunk: k/2N = 0.0137 exceeds the net held-out 0.0073. Detect
  with the λ curve, the train-CV gap, and a placebo that shuffles band labels among TRAIN
  fragments, refits and evaluates on val. The expected placebo result is ≤0 (≈ −k/2N at λ=0).
- **Failure modes.** (a) The residual against expected-under-prior is ~0, which means T5
  (a)/(b) were autocorrelation mechanics. P0.5 band residual map minus its null reveals it.
  (b) The gain sits only in the right end and moves when the registration shifts: P0.4
  shift table. (c) Sim NEGATIVE control (the generator has no band term) shows a gain: that
  is a leak or a bug.
- **Seam.** S1. N4 calibration should run jointly with δ when stacked.

### 3.4 M1 — flank PWM beyond the 6-mer window

- **Structure.** Additive per-offset 1-mer log-ratios at −12..−4 and +3..+12 around each cut,
  both ends, all four tables.
- **Cost.** 19 × 3 × 4 = 228 params. k/2N = 0.00019. [Inferred]
- **Estimation.** Fit by the engine on the region-weighted objective, with offset log p0. Do
  not use plug-in log-ratios, because flank bases correlate with the 6-mer and with regional
  composition.
- **Expected gain.** ≤ about 0.026 (0.0152 + 0.0109 marginal, in-sample pooled KL). The
  conditional gain is smaller and unknown. [Inferred: §2.2]
- **Overfit risk.** Negligible on param count. The naive-Bayes product can over-count
  dependent offsets. The fitted (not plug-in) form corrects that.
- **Failure modes.** (a) The conditional residual under the prior is ~0. P0.5 flank map shows
  it before S1 exists. (b) The gain vanishes under region-wise normalisation (the KL exchange
  rate): only the NLL table shows it. (c) A large right-end coefficient at the first flank
  offset on either side would signal a residual one-base registration error. Registration
  is now fixed at `m+L` (ruling 6), so treat such a coefficient as a bug report against
  change B, not as flank signal. Report the right-end innermost-offset coefficients
  separately. On sim, BOTH flanks are NEGATIVE controls once B lands, because the generator
  is 6-mer-only at `m` and `m+L` (draw.py:85-104). [Inferred; Verified: draw.py, r3]
- **Edge rule.** In the evaluator tensor (tile coordinates [−184, 1208)), the farthest 3'
  cut sits at 1023+180 = 1203 under `m+L` and has only 4 bases beyond it (5 at `m+L-1`).
  [Inferred: §1.1 tensor arithmetic] A term whose context runs off the tensor contributes 0.
  For ±12 at the 3' cut, that affects stop positions 1196..1203, under 1% of end positions.
  [Inferred] Report the exact affected-cell fraction.

### 3.5 N0 — counted tables with a corrected smoothing estimator

**Reviewer round 1, F-03: re-ranked to #1 (§3.0). N0 is BIAS-side, not variance-side.** The
earlier draft capped N0's expected gain at the whole variance budget (≤~0.01 nats, §2.5) and
ranked it below N4. The reviewer instead rebuilt stage1's four tables exactly (max rel diff
0.0 vs the production npz; check2.log:13) and measured the smoothing distortion directly: it
costs 0.016-0.019 nats per table (§1.3 F1), 2-3x the entire k/2N budget on its own — a bias
term, not noise.

**Revision 3: APPROVED (ruling 5), built at `m+L` (ruling 6), on `measure.py` (reuse
ruling).** N0's score becomes the new prior anchor (§0.4).

- **Structure.** Removes the F1 distortion. Pipeline, built directly in the model's OWN
  hexamer convention throughout — **revision 4: no adapter.** The round-3 plan below assumed
  the pipeline would run in simulator convention and cross through one adapter at the end
  (§0.4 C, old text); implementation found nothing hexamer-keyed needs to cross at all, so
  steps 1-4 below read as "in the model's convention" wherever they said "simulator
  convention, pending the adapter":
  1. `C(h)`: `counts_from_hexamers` on the TRAIN centred-crop fragments, with stop hexamers
     at `m+L`. [Verified: measure.py:179-199, r3]
  2. `N(h)`: `uniform_hexamer_counts` over the same crop, `FragmentLengthDist` built from the
     same TRAIN fragments. `N_start` serves `start_fwd`/`end_rev`; the ramp-weighted `N_end`
     serves `end_fwd`/`start_rev`. [Verified: measure.py:395-497, 500-552, r3] This
     replaces stage1's single start-domain denominator. It is adopted for hygiene and
     correctness only — measured value ~8e-5 nats, NOT a source of expected gain (ruling 7,
     §0.4 D). It also subsumes the F2 domain-extent fix (§1.3).
  3. Clean rate `r = C/N` from `propensities` (no smoothing, divide-by-zero guard only).
  4. Shrinkage `(C + w·q)/(N + w)` with `q` on the cuts-per-occurrence scale (the table's
     pooled empirical rate, ~0.0055; see F-07/§3.9 for the same units fix applied to N3), and
     `w` on a grid chosen by 5-fold train-CV. Limits: `w=0` gives the raw rate from step 3;
     `w→∞` gives `q`.
  5. Normalise each table to sum 1 (the loader will not, Q6), then the `.npz` write with the
     registration key and provenance sidecar. No adapter step.
  `p_L` stays the raw TRAIN centred-crop length marginal, so it must equal the v4 npz's
  `p_L` (gate G0b). [Inferred: neither ruling touches `p_L`] **Measured (revision 4,
  Phase 0b.1): 599,412 fragments admitted, 599,411 counted, 1 dropped at L=179** — not the
  ~31 this document predicted from the N-window-fragment question (Q24). `p_L` over all
  admitted fragments equals the v4 npz's `p_L` bit-for-bit (`array_equal=True`); `p_L` from
  the counted population differs from it by at most 1.658e-06. `unif+p_L` on test from the
  new counter's `p_L`, closed form, is **12.501596321**, \|Δ\| 3.2e-7 against the pre-change
  value, over 6,303 regions / 73,017 fragments — consistent with "neither ruling touches
  `p_L`." [Verified: COORDINATION gate `G0b_phase_0b1`] **Counting population note:** the
  counter runs at `min_N=0` (every tile of the split — 53,321 TRAIN tiles on the real store),
  which is the counting population and is distinct from the evaluator's `min_N=10`. Do not
  conflate the two when reading §2.5's event-per-cell arithmetic. [Verified: COORDINATION
  "What was done"] Measured runtime: the counter runs in ~78 s on real TRAIN, CPU.
  [Verified: COORDINATION "What was done"]
- **Decomposition rows (diagnostic, all at `m+L`, all through the new counter).**
  `N0_ref`: the stage1 formula `(C+1)/(N+4096)` (`w=4096`, `q=1/4096`). `N0_raw`: `w=0`.
  `N0`: `w*` from train-CV — the new anchor. Paired against the OLD prior TSV on the same
  regions, `old prior → N0_ref` is the effect of B (plus the ~1e-4 denominator hygiene), and
  `N0_ref → N0` is the effect of A. [Inferred] Neither row needs a `m+L-1` artefact, so the
  sequencing constraint holds.
- **Why fitting `w` is safe here (EM critical review, item 2).** Candidates are not
  separable. Fitting `w` against a wrong denominator, or a wrong registration, lets `w`
  absorb that error, and the result looks like a tuned success. Under A+B+C the estimator,
  the denominator and the registration land TOGETHER (§0.4), so `w` is only ever fitted
  against the final `N(h)` at `m+L`. It never sees stage1's `occ` or `m+L-1`. That is the
  reason the order is safe. The same argument fixes the order downstream: N4 initialises
  from N0 and never from stage1's tables. [Inferred]
- **Cost.** 0 params. S3 counter (built, no adapter — revision 4) and change B, both landed
  at `9aa8ef5`.
- **Expected gain.** The P0.2 predictor is already computed, not just proposed
  (check2.log:15-18): cut-weighted var/2 of log(occ/(occ+4096)) = 0.0207 / 0.0197 / 0.0219 /
  0.0191 nats per table (start_fwd/stop_fwd/start_rev/stop_rev); with the +1 numerator,
  var/2 of log(smoothed/raw) = 0.0178 / 0.0166 / 0.0189 / 0.0160; smoothing factor range
  [0.046, 0.963]; occurrence min 199, median 12,405, max 105,176 (check2.log:15-18). Each
  fragment reads two tables, so the second-order distortion cost is ~0.03-0.04 nats per
  fragment [Inferred]. **Caveat:** this figure assumes the unsmoothed rate is the right
  table; raw rates add noise of their own, so the realised NLL gain from switching to the
  hierarchical `(c+w·q_h)/(occ+w)` form can be smaller than the bias estimate above — that
  form is designed to trade the two, not just remove smoothing. The real NLL gain is still
  [Unverified: score an N0 npz through the sanctioned evaluator]. Phase 0 still re-implements
  the P0.2 predictor as a committed script (§5), even though the number above already exists
  as a one-off recomputation. This ~0.03-0.04 lies INSIDE the §0.2 stacked range of
  0.02-0.08, not on top of it (§0.2).
- **Overfit risk.** Smaller `w` means more noise. The train-CV curve over `w` detects it.
- **Failure modes.** The gain is ~0 because rare hexamers carry little cut mass. P0.2
  predictor ≈ 0 and Δ within 2SE reveal it. That outcome is still useful, because it
  attributes the N4 gain to the depth confound instead of to F1. The change stays approved
  either way: it fixes a category error, so a ~0 gain does not reopen ruling 5.

### 3.6 M3 — interior composition × length group

- **Structure.** Factor c[GC bin, g] on the exact span [m, m+L), with 10 FIXED-WIDTH GC bins
  (fixed width, to avoid the equal-frequency mechanics of T5 (d)) and the L2a groups
  `g = min((L−25)//15, 9)`. The owner has confirmed non-GC interior effects in real data
  (the reason L2b stays). Extend to CpG or dinucleotide content only if GC passes.
- **Cost.** 100 params. Prefix sums over sequence inside the S1 term.
- **Estimation.** Engine, log-linear with offset log p0. Judge against EXPECTED counts under
  the prior, never against independence.
- **Expected gain.** Unknown. T5 (d) is confounded and gives no number. [Verified: §2.3]
- **Failure modes.** The residual under the prior is ~0, so T5 (d) was purely mechanical.
  P0.5 GC×g map shows it. A gain on sim (no GC term in the generator) means a bug or a leak.

### 3.7 N2 — strand-pooled positional-tie tables (DIAGNOSTIC; owner decision D3)

- **Structure.** `start_fwd ≡ stop_rev` (both read at m) and `stop_fwd ≡ start_rev` (both read
  at m+L after B; m+L-1 at `bad981d`). **Revision 4:** the round-3 text placed the tie
  "after the adapter, because the simulator rc-permutes the `_rev` tables" — moot, since no
  adapter exists and N2's tables, like N0's, are counted directly in the model's own
  convention; write the tie there, with no RC un-permutation step. This is a POSITIONAL tie, not reverse-complement tying. It still reduces four
  tables to two, so it touches the "four untied tables" decision
  (cut_site_fragment_model.md:143-151). It needs owner sign-off before any non-diagnostic
  use. [Verified: §1.1 lookup]
- **Cost.** It saves 8,190 params. Noise saving ≤0.0068 at N=599,412. [Inferred: §2.5]
- **Failure modes.** Strand carries a real library-prep asymmetry (an R1 vs R2 end). P0.4
  shows a pooled-vs-separate held-out log-likelihood < 0, and Phase 1 shows a negative
  Δ on val. Sim tells whether pooling is valid for the sim only.
  [Unverified: the simulator's minus-strand storage convention, §8 Q8]

### 3.8 M5 — per-start normaliser

- **Structure.** −γ·log t̂(m), with γ ∈ {0, 0.5, 1} chosen on train-CV (revision 3: was "on
  val", which contradicted §6.2). The sim generator HAS this structure (`/t(m)`, with
  `t_s = Σ_L e(h_{m+L})·f(L)`, draw.py:100-101), so it is the cleanest sim POSITIVE control.
  After B, `t̂` reads the stop table at `m+L`, the generator's own registration, so the
  F6 attenuation no longer applies to new runs. [Verified: draw.py:100-101, r3; Inferred]
  Ruling 3: measured additively only; the production prior is untouched; on sim it is an
  oracle verification, never a prior.
- **Cost.** 1 param. **Expected gain.** ≤ ~0.0016-0.0048, sign unknown on real. [Inferred:
  §2.4, measured at `m+L-1`; re-measure after P0b]
- **Failure modes.** Negative on real (start-then-length does not hold): the γ grid
  includes 0, and the train-CV sign decides; val confirms. Its value is interpretive.

### 3.9 N3 — 6-mer backoff toward a per-offset product

- **Structure.** `(c_h + w·q_h)/(occ_h + w)`, with `q_h` the product of per-offset
  single-base rates (naive-Bayes PWM) on the cuts-per-occurrence scale, and w chosen on
  train-CV. Limits: `w=0` gives the raw rate; `w→∞` gives the backoff rate `q_h`.
  [Verified: reviewer round 1, F-07 — the earlier draft's `(c_h + w·occ_h·q_h)/(occ_h + w)`
  has the wrong units, tending to a raw COUNT rather than a rate as w→∞]
- **Expected gain.** A part of the ≤0.0138 noise budget. The within-window dependence
  (0.027 / 0.025) is real, so heavy backoff costs bias. [Inferred: §2.2]
- **Failure.** The CV picks w ≈ 0, so there is no gain. That is cheap to learn.

### 3.10 M4 — left×right dinucleotide residual

16 × 16 × 2 bands = 512 cells. T5 (c) is not interpretable (positive null). Build it only if
the P0.5 obs/exp map under the prior exceeds its null by the G3 threshold.

### 3.11 Rejected, with reasons

- **Tile GC → p(L).** Signal ≤ ~0.0027. [Verified: T5 (e)]
- **Band-conditioned p_L alone.** Length is already separate. N4 checks a band- or
  strand-conditional `p_L` for free, as a calibration residual.
- **Anything that consumes the simulator's generating a, e, f or hexamer counts.** Forbidden
  by constraint 5. Sim candidates are counted on the sim store's TRAIN split. Revision 3
  draws the line for the reuse ruling: reusing `measure.py`'s FUNCTIONS is required; feeding
  them the generator's own TABLES or its `FragmentLengthDist` is forbidden. Every
  `FragmentLengthDist` in this programme is built from the TRAIN fragments of the store
  being counted.
- **`biomarker.flgc.GCFlDistModel`.** Calibrated on spike-ins covering ~24-75 bp, not 25-180.
  [Verified: research agent, /home/nathanboley/src/biomarker/flgc/model.py]
- **The attic hexamer pipeline artifact.** Retired (decision 189). Only the shrinkage idea
  is reused (in N0/N3/M2), in its per-hexamer rate form `(C+w·q)/(N+w)`, not as a
  multinomial share — the multinomial form is the category error of ruling 5.
- **`fragmentomics_tools/bias_correction/`.** Superseded v1.

---

## 4. Seams and risks

### 4.1 Ownership boundary

`background_model/cut_site_model/dataset.py` and
`scripts/cut_site_model/stage1_hex_baseline.py` are owned by another agent this phase. This
design plans NO edits to them. **Reviewer round 1, F-15: resolved the home module question.**
`hex_window_for_tile` is DEFINED in `stage1_hex_baseline.py` (line 70) itself, not imported;
`jitter_matrix` and `calculate_start_and_stop_from_jitter` come FROM
`background_model_core` (stage1_hex_baseline.py:28). New code must import the jitter
helpers from `background_model_core` directly, and must NOT import `hex_window_for_tile`
from the stage1 file — that file is under refactor, and importing from it couples new code
to a file nobody here owns. Revision 3 replaced the old F-01 rule ("derive hexamer windows
via the model's own encoder") with: new code derives them via `hexamers.py`/`measure.py` and
crosses into the model's convention only through one adapter (§0.4 C). **Revision 4: that
plan was superseded by what was actually built — new code (`count_tables.py`) derives
hexamer windows directly via the model's own encoder after all, drawing only the
length/position-indexed `FragmentLengthDist`/`fl_end_weight` from `measure.py`. No adapter
exists.** New modules still IMPORT `CutSiteFragmentDataset`. Risk: the refactor renames or
changes a signature.
Mitigation: an import-smoke test of every consumed name in the new test file, and a note to
the coordinator naming the consumed symbols.

**What the change set DOES edit (all approved, none owned elsewhere):** `model.py` lines
180-204 and 292-300 plus the loader at 398-404 (change B and its guard, §0.4);
`tests/test_hex_registration.py` (convention moves to `m+L`, two tests retargeted off the
stage1 file, the new cross-check T12). It does NOT edit `dataset.py` or
`stage1_hex_baseline.py`; stage1's counting path is retired by no longer being the
producer, and the loader guard enforces that. [Verified: COORDINATION sequencing
constraint; line ranges r3]

### 4.2 S2 — reduction wrapper (no evaluator change)

`scripts/cut_site_model/baselines/reduce_split_nll.py` (TO BE WRITTEN) reads a sanctioned
per-fragment TSV. It builds `CutSiteFragmentDataset(split=..., min_N=10, train_mode=False)`
for `val` and `heldout_inactive`, and calls `score_fragments.region_weighted_mult_nll`. It
prints region-weighted NLL per split and writes JSON with provenance. With D4 approved, an
additive default-off `return_terms=True` returns per-region terms. That is about 4 lines in
`score_fragments.py`, which this phase's owner does not hold. Asserts: wrapper mean ==
function mean to 1e-12; uniform `.npz` gives ln 319488 on every region; at `bad981d` code
only, the old v4 `.npz` reproduces the RETIRED 12.146835 (val) and 12.142214 (test). That
last assert proves the reduction, not an anchor. It cannot run after change B, because the
`m+L` loader rejects the v4 `.npz` (§0.4). [Inferred]

### 4.3 S3 — counter

**BUILT (revision 4): `scripts/cut_site_model/baselines/count_tables.py`, landed at
`9aa8ef5` as part of the Phase 0b.1 change set.** It was needed because stage1 hardcodes
`split="train"` and j=0, and — after §0.4 — because it is the SINGLE table producer.
Stage1's counting path is retired. Its validity condition: `C` and `N` cover the same
candidate set (measure.py:413-418 states the same rule). It emits the fixed `.npz` schema
plus the registration key (§0.4; implemented as `stop_cut_base_offset`) and a provenance
sidecar: store path, split, config_hash, git sha, estimator, `w`, crop mode, region fraction
and seed, n_fragments, n dropped non-ACGT windows, date, **plus (added for the mask
limitation below) `mask_honoured_bases`, `cut_base_window`, `mask_unhonoured_bases`,
`mask_all_true`**. The stage1 `.npz` carried no provenance; a previous manifest named a
commit that did not produce its outputs — this sidecar is the fix for that. Counts
accumulate with `np.bincount(..., weights=float64)` or int64. Never `np.add.at` into
float32.

**Known, recorded limitation: mask handling past `l_target` (divergence 3, revision 4).**
The counter honours `tiles/mask` at the cut base, as stage1 did. But a right cut reaches up
to `L_max` bases past the tile, while `tiles/mask` spans only the `l_target` window — so the
last `cut_base_window - mask_honoured_bases` positions of the right-cut domain are counted
UNMASKED. This is harmless on the real store, whose mask is 100% True
(`mask_all_true` in the sidecar), and latent on any future store with a real mask. It is not
fixed in this phase; it is recorded in the artifact's provenance sidecar (the fields above)
so a consumer reading only the counts can see where the mask stops, rather than assuming it
covers the whole domain. [Verified: COORDINATION gate `em_apply_phase_0b_review_findings`,
divergence 3]

**Reuse plan (revision 3, owner reuse ruling; supersedes the round-1 F-01 text) —
revision 4 corrects what was actually built.** The counter is a thin driver over
`measure.py` (§0.4 C, §3.5 steps 1-5). Fragment admission comes from
`CutSiteFragmentDataset(split="train", train_mode=False)`, so the counted population cannot
drift from the evaluator's — counted at **`min_N=0`, every tile of the split** (53,321
TRAIN tiles on the real store), a different population from the evaluator's own `min_N=10`
(§3.5). Hexamer indices are computed directly in the model's OWN convention throughout.
**No adapter was built, and none was needed:** the only `measure.py` imports in this path
are `FragmentLengthDist` and `fl_end_weight`, both length- or position-indexed, never the
hexamer-keyed `counts_from_hexamers` / `propensities` / `uniform_hexamer_counts` in a form
that would require translating between conventions. [Verified: COORDINATION gate
`em_unblock_phase_0b`] This also makes the earlier "open feasibility point" about
`uniform_hexamer_counts`'s FASTA/`pysam` dependency (Q15, Q20) MOOT for the production path
— a store-backed `N_start`/`N_end` loop was written instead (§0.4 C; gate
`em_settle_Q20_reuse_feasibility`). **Every crop mode (centred, window/N1, pooled/N2) still
needs its own one-hot probe test against `compute_log_p0`** (the identity-table trick from
`tests/test_hex_registration.py`, now T4b, §7.4): build a one-hot table that is 1 at exactly
one known hexamer/offset and 0 elsewhere, run it through the counter and the model's lookup,
and assert they agree, for both strands and both ends — this is now EXHAUSTIVE over all 4096
cells for the centred crop, not a sampled probe. This was a Phase-0b gate (§5, G0b) and
PASSED: no `.npz` from this counter was scored — the new anchor included — until the probe
passed for the crop mode used. Window mode (N1) and pooled mode (N2) are not yet built and
still need their own probe test when they are.
F-12 separately notes that `hex_window_for_tile` fixes `lo = tile_start - 3`
(stage1_hex_baseline.py:89-96) and cannot index negative tile coordinates, so N1's
jitter-margin starts (window mode) need NEW registration code, not a reuse of the centred-
crop path that T4 covers — margin starts must be indexed in window coordinates, and that
path gets its own probe test too.

### 4.4 Expected-count engine (fitter utility, never an evaluator)

`scripts/cut_site_model/baselines/expected_counts.py` (TO BE WRITTEN). For each region it
computes the domain log-probs with the model's OWN `compute_log_p0` + `log_softmax` in
float64, then scatters cell probabilities onto hexamer (or feature) indices with float64
`bincount`. It never reports NLL. All reported NLL comes from `score_fragments` + S2.
- Anti-drift: a unit test asserts engine log-probs == `domain_log_probs` (zero correction)
  to **1e-5 max-abs**, or compares with `model.double()` on both sides, on a fixture.
  [Verified: reviewer round 1, F-13 — the float32 scorer path already shows
  max|logsumexp|=8.09e-7 (prior summary json), and per-cell float32 rounding at
  |log p|~12.7 is ~7.6e-7 [Inferred], so a 1e-6 max-abs gate between the float64 engine and
  the float32 evaluator will flake] [Unverified: does `compute_log_p0` accept float64
  inputs?]
- Cost: 319,488 cells × ~50k TRAIN regions ≈ 1.6e10 cell evaluations per pass. The packet
  estimates 15-30 min per pass single-core. [Unverified: time 200 regions first] With
  separable terms, the marginal over L for each m is a 1-D correlation of size P × n_L
  (~160k per strand). That is a fallback if the reference path is too slow, but only behind
  the same identity test. [Inferred]
- Run it on CPU locally. GPU is for train/score only.

### 4.5 S1 — prior-term seam (needs owner approval D1; a design question with its own risks)

New `background_model/cut_site_model/prior_terms.py`. Each `PriorTerm` exposes
`log_term(hex_idx, seq_onehot, ...) -> (B, P, n_L, 2)`. Terms load from a bundle `.npz`
(`--prior-terms`) and are ADDED to `compute_log_p0`'s output inside `domain_log_probs`. The
default is empty, which keeps the current path bit-identical. Plumbing: ~15 lines in
`model.py`, one flag in `train.py` and in `score_fragments.py`. `dataset.py` is untouched;
terms compute from the sequence already in the batch. [Unverified: batch keys that carry
sequence, via `grep -n "seq" $W/background_model/cut_site_model/model.py`]

Risks and controls:
1. **A call site misses the term.** train, validate and score must all route through
   `domain_log_probs` (model.py:759-767, score_fragments.py:234). [Verified: research agent A]
   [Unverified: `grep -rn compute_log_p0 $W/background_model $W/scripts $W/tests` shows no
   other caller]
2. **Silent change of the default path.** Test: an empty bundle gives log-probs bit-exact
   to the current prior (post-change-set: N0 at `m+L`) on a fixture.
3. **Registration.** Every term gets an identity-probe test in the style of
   `tests/test_hex_registration.py`: a one-hot table at a known k-mer or offset, and an
   assertion of where the term fires, with the 3' cut at `m+L`. No pinned NLL number.
4. **Checkpoint compatibility.** The new hparam defaults to `None`. Checkpoints trained
   AFTER the change set load and score unchanged (test). Checkpoints from `bad981d` or
   earlier carry `m+L-1` tables in their buffers and must be REFUSED, not scored (§0.4
   item 2). Revision 3: the earlier "old checkpoints load and score unchanged" is wrong after
   change B.
5. **Edge of tensor.** Terms contribute 0 where context runs past the 1392-wide tensor. Each
   term reports its affected-cell fraction.
6. **Precision.** Fitters run in float64. The forward runs in float32. Report max |Δ|
   between float32 and float64 evaluation of the same candidate on 200 regions.
7. **Scope creep.** S1 is additive. It is not a licence to change `compute_log_p0`'s
   semantics or the frozen core.

---

## 5. Phased measurement plan, gates and stop rules

Cheapest decisive experiment first. Read TEST once per final candidate set, and log each
read in the test-touch ledger (§6.4). T4, T5 and T6's numbers (§2.2-2.4) came from the
measurement agent and were NOT re-verified by the reviewer, whose own scripts are gone; keep
them tagged as measurement-agent values and make re-running them the first Phase-0 items.

**Revision 3 restructure.** The approved change set (§0.4) splits Phase 0 in three. Phase 0
runs at `bad981d` model code, because the old prior cannot be scored after change B.
Phase 0b lands the change set as ONE commit and builds the new anchor. Phase 0c runs every
diagnostic that reads a prior, under the NEW anchor. Steps keep their old IDs so that
earlier cross-references still resolve.

### Phase 0 — at `bad981d` code, no candidate fit — **COMPLETE, gate G0 PASSED (commit `2a775ce`)**

| Step | What | Output | Cost |
|---|---|---|---|
| P0.0 | Shadow assertion; suite baseline BEFORE (both `test/` and `tests/`), measured, not quoted. **Measured: 2 failed / 1071 passed / 3 skipped (1076), 145.69 s** [Verified: COORDINATION "What was done"] | log | minutes to 1 h |
| P0.1 | Write S2 + a ~20-line npz emitter (uniform; unif+p_L from the existing HEX `p_L`). Score uniform, unif+p_L, and the OLD prior on val+test (real), plus uniform + unif+p_L on both sim stores. The old-prior per-fragment TSV (`/efs/analytics/nathanboley/background_model/cut_site_runs/baselines_phase0_20261010/real/per_fragment_old_prior.tsv`) is **IRREPLACEABLE, not merely "kept"**: it is the sole input to the P0b.3 paired comparison, and it cannot be regenerated once `9aa8ef5` lands, because the `m+L` loader refuses the `bad981d`-era tables that produced it. The sim-counted prior moves to P0b.4 (it needs the new counter). **DONE: all 8 real checks passed (gate G0 below).** | **Deliverable D0: real NLL table (§6.5 rows 1-3); old prior row is HISTORICAL** | 3 real + 4 sim scorings |
| P0.4 | T4 per strand, both ends, offsets −12..+12, with offsets reported relative to BOTH `m+L-1` and `m+L`; strand-tie test (pooled vs separate held-out LL, val); sim run of the same code as a positive control (the sim right-end peak must sit where `e(h_{m+L})` puts it). Reads no prior, so it is registration-neutral. [Status: not confirmed as part of the G0 gate's 8 checks; verify before relying on it.] | tables | CPU |

### Phase 0b — the approved change set (A + B + C), ONE commit — **Phase 0b.1 COMPLETE, gate G0b PASSED (commit `9aa8ef5`). Phase 0b.2 onward NOT STARTED.**

| Step | What | Output | Cost |
|---|---|---|---|
| P0b.1 | **DONE.** T4b exhaustive probe test, in place of an adapter (§0.4 C, revision 4 — no `hex_adapter.py` was built); new counter `count_tables.py` on `measure.py` (§4.3); change B in `model.py` (line 198, guards at 204 and 292-300 corrected per the WINDOW_HALF derivation fix, comments, loader registration check `stop_cut_base_offset`); `tests/test_hex_registration.py` moved to `m+L`, two stage1-importing tests retargeted (T14), T12 and T13 added. **Suite baseline AFTER measured: 2 failed / 1111 passed / 3 skipped (1116), up from 1100 at `9fe0d31`; +16 reconciles as +2 T12, +2 net CLI, +12 new test file.** [Verified: COORDINATION gate `G0b_phase_0b1`] | one commit; suite log | engineering |
| P0b.2 | Count N0 on real TRAIN over the `w` grid, choosing `w*` by 5-fold train-CV. Fold scores go through `score_fragments` + S2 restricted to the held-out fold's regions — the one sanctioned evaluator, never a fold-local NLL. [Unverified: `score_fragments.py --splits train` is accepted; `grep -n "choices" $W/scripts/cut_site_model/score_fragments.py`] Emit `N0_ref` (stage1 formula), `N0_raw` (`w=0`) and `N0` (`w*`). | three `.npz` + sidecars | CPU + 5×grid fold scorings |
| P0b.3 | Score `N0` through the sanctioned evaluator on val+test: **this is the new prior anchor**. Score `N0_ref` and `N0_raw` as decomposition rows, paired against the P0.1 old-prior TSV. Ledger every test read. | new `prior` row; decomposition | 3 real scorings |
| P0b.4 | Count and score `N0` and `N0_ref` on both sim stores with the same code (sim TRAIN only; constraint 5). `N0` replaces the historical sim prior numbers and gaps; `N0_ref` separates the effect of B from that of A on sim. | new sim prior rows and gaps | 4 sim scorings |

### Phase 0c — diagnostics under the NEW anchor

| Step | What | Output | Cost |
|---|---|---|---|
| P0.2 | Smoothing predictor as a committed script, on the new counter's `C` and `N`: cut-weighted var/2 of log(stage1-smoothed / raw) per table; table-level held-out LL of the cut-hexamer distribution under {stage1 formula, raw, `w*`}. Confirms check2.log:15-18 (§1.3 F1, §3.5) under the new denominators and `m+L`; it does not decide whether to build N0 (ruling 5 did). | table | CPU, one pass |
| P0.3 | Learning curve of the N0 estimator at `w*`: 25/50/100% of TRAIN regions (fixed seed), scored on val | NLL(f); fit a + b/f | 3 counts + 3 val scorings |
| P0.5 | Build and unit-test the engine. Residual maps under the NEW anchor prior on TRAIN, each with a 20-permutation placebo null: per-table KL(O‖E) (N4 predictor), band × hexamer (M2), flank offsets (M1), GC × g (M3), left × right dinucleotide (M4), log t̂ (M5). Re-run T6 here. | per-feature (in-sample − null) | CPU, ~1-2 passes |

The learning-curve prediction from §2.5: noise ≈ 0.055 / 0.028 / 0.014 at 25/50/100%, a
25→100% drop of ~0.041. [Inferred: 0.0138 × 4, × 2, × 1] Smoothing damps it. The measured b
is the empirical noise at full N.

### Gates

The thresholds are proposals for the owner. 0.002 is about the run-to-run noise of TRAINED
runs and sits below every bound in §2.

- **G0 (after P0.1, at `bad981d`) — PASSED, all 8 real checks (commit `2a775ce`).** Uniform
  gives ln 319488 on every region (|Δ| < 1e-5, float32 forward). The old prior reproduces the
  RETIRED 12.146835 / 12.142214 (|Δ| < 1e-5) — a check of the reduction, not an anchor.
  unif+p_L agrees with the closed form of C2 to 1e-5. [Verified: reviewer round 1, F-13 — a
  1e-6 gate between a float64 computation and the float32 forward will flake; see §4.4]
  Measured: uniform val/test 12.674474716 (\|d\| 2.77e-7); old prior val 12.146834609
  (d −3.9e-7), test 12.142214063 (d +6.3e-8); unif+p_L val 12.502734890 vs closed form
  12.502734316, test 12.501596932 vs 12.501596359 (d +5.74e-7 for both); sim full and sim
  10pct uniform and unif+p_L also passed. Suite BEFORE 2F/1071P/3S, AFTER 2F/1079P/3S (+8
  new). [Verified: COORDINATION gate `G0_phase0`]
- **G0b (after P0b.1-P0b.3) — the change-set gate. PASSED, all 6 items (commit `9aa8ef5`).**
  ALL of: (1) T4b probe passes for the centred crop, both strands, both ends, EXHAUSTIVE
  over all 4096 cells x 4 tables — **revision 4: this is a self-consistency probe of the
  counter against `compute_log_p0`, not an adapter probe; no adapter was built (§0.4 C)**;
  (2) T12 ties the prior's 3' read to L2a's tap and to the simulator's stop hexamer; (3) T13:
  the `m+L` loader rejects `stage1_hex_tables_v4_start.npz` and a `bad981d` checkpoint;
  (4) the new counter's `p_L` vs the v4 npz's — **measured: 599,412 admitted, 599,411
  counted, 1 dropped (at L=179), NOT the ~31 earlier predicted; `p_L` over all admitted
  fragments equals the v4 npz's bit-for-bit (`array_equal=True`); `p_L` from the counted
  population differs by <= 1.658e-06; unif+p_L on test from it, closed form, is
  12.501596321, \|Δ\| 3.2e-7 over 6,303 regions / 73,017 fragments** [Verified: COORDINATION
  gate `G0b_phase_0b1`]; (5) `start_fwd` AND `stop_rev` (both, not `start_fwd` alone — the T4
  correction, divergence 1) reproduce stage1 at max relative difference 0.0 on real data;
  (6) suite AFTER = suite BEFORE plus the new tests — **measured: BEFORE 2F/1095P/3S (1100)
  at `9fe0d31`, AFTER 2F/1111P/3S (1116); +16 reconciles exactly as +2 T12, +2 net CLI, +12
  new test file.** [Verified: COORDINATION gate `G0b_phase_0b1`] If ANY had failed: stop,
  revert the whole commit, never a part, because a part is exactly the keyed-one-way-
  read-another state (§0.4). No candidate has been scored yet — Phase 0b.2 onward is NOT
  STARTED.
- **G1 (after P0.2, P0.3).** If the measured noise b < 0.002: kill N1, N2, N3 as builds;
  keep N2 only as a strand diagnostic. N0 is not gated here: ruling 5 approved it, and the
  reviewer's P0.2 recomputation (0.016-0.019, §1.3 F1) sits well above 0.002 anyway. P0.2
  confirms; it does not decide.
- **G2 (after P0.4).** If pooled-vs-separate held-out LL < 0 beyond its SE: drop N2.
- **G3 (after P0.5).** Build an M candidate only if (in-sample residual − null mean) >
  0.002 AND the null sits at or below 0. If no M candidate passes: do not request S1 (D1
  lapses). If N4's KL(O‖E) − null < 0.002: demote N4 below N1.
- **Registration gate — superseded by ruling 6.** The earlier rule ("do not change `m+L-1`
  in this phase") is withdrawn; the registration is `m+L`, enforced by T12/T13. P0.4's shift
  table is now a sanity check: if it puts the right-end peak somewhere that makes `m+L`
  look wrong, report it to the owner as new information. It does not reopen ruling 6 on its
  own. Every right-end term uses `m+L` plus a probe test.

### Phase 1 — Layer N through the evaluator (needs S2, S3 (both built), engine, G0b passed (done))

N0 is already built and scored in Phase 0b; it is the anchor, not a Phase-1 row. Then
individually: N1, N2 (diagnostic), N3, N4, each on top of N0's estimator and denominators.
Then stacked: N0→N1→N4 (reviewer round 1, F-03: N0 first, since it fixes the bias N4's
"initialise from the counted tables" step would otherwise inherit — the earlier draft's
N1→N0→N4 order is corrected). **Why this order is safe (EM critical review, item 2):**
candidates are not separable, and a hyperparameter fitted against a wrong denominator or
registration absorbs the error and looks like a tuned success. N0's `w` is fitted only
after A+B+C land together (§3.5), and N4's `λ` only after N0 exists, so no hyperparameter
here is ever fitted against a component that a later step replaces. Each candidate gets
one real `.npz` and the same estimator applied to sim-full and sim-10pct TRAIN.
Hyperparameters (the N3 backoff weight `w`, N4's ridge `λ`, sweep count) are chosen on
**train-CV only**; val is confirmation only (§6.2, F-11). Read test once for the final set.
**Deliverable D1:** §6.5 rows filled for the Layer-N candidates, Δ against the NEW anchor.

### Phase 2 — S1 + Layer-M candidates that passed G3

Order: M2-lite → M2-full → M1 → M3 → M5 → M4. Each is fitted jointly with N4 calibration
when stacked. Gains are measured stacked, never summed.

### Phase 3 — owner decision gate

Present D1/D2 tables. The owner decides which candidate, if any, is proposed as a further
alternative production prior beyond N0. The neural-arm retrain under the new prior is
needed for any new model number (§0.4). It is measured only after the loader phase,
because the model column moves. Ruling 4 (no more-epochs re-run of the OLD real model)
still stands; a retrain under the new prior is a different run.

---

## 6. Evaluation protocol

### 6.1 Capture against the PRIOR, never against uniform

- "Prior" here means the NEW anchor: N0 at `m+L`, scored in P0b.3. [Inferred: §0.4]
- Real: `capture_vs_prior = (prior − candidate)/(prior − model)`. **No valid denominator
  exists until the retrain under the new prior.** The old one, 12.142214 − 11.996969 =
  0.145245 (test), is RETIRED with both endpoints (§0.4); dividing a Δ against the new anchor
  by it would mix two priors. Until the retrain, report Δtest vs the new prior ± 2SE as the
  primary metric and leave the capture column empty. The old caveat still applies to the
  new denominator once it exists: a model run still descending gives an UPPER bound on model
  NLL, so the capture is an optimistic upper bound. [Inferred]
- Sim: `(prior − candidate)/(prior − oracle)`, using the prior re-scored in P0b.4. The
  historical denominators 0.064233 (full) and 0.227501 (10pct) carry the F6 registration
  confound and are not used.
- Report length and sequence separately, using the unif+p_L row.

### 6.2 Statistics

- Counted priors are deterministic. There is no run-to-run noise, so a third-decimal
  difference between two counted priors on the SAME regions is real if it exceeds its
  sampling SE. [Inferred] The ~0.002 run-to-run caveat applies to TRAINED runs only.
- Two noise sources remain. **Evaluation:** a paired bootstrap over test (and val) regions,
  B=2000, fixed seed; report Δ ± 2SE. This needs `return_terms` (D4). **Estimation:** 5-fold
  train-CV by region, refit per fold; report the spread and the optimism gap.
- The 0.0046 val-test gap of the unchanged prior is the yardstick for UNPAIRED comparisons
  (different splits or populations). Paired Δ uses the bootstrap SE, which is expected to be
  smaller, but only moderately: the reviewer measured the test per-region prior NLL SE at
  0.00457 (unpaired, sd 0.363, n=6,303) vs the paired prior-minus-model SE at 0.00243
  (sd 0.193, n=6,303) — pairing helps only ~2x for that contrast, not an order of magnitude.
  [Verified: reviewer round 1, check1.log:21-22]
- **Hyperparameter selection is NOT circular, by construction, as of this revision
  (reviewer round 1, F-11).** The earlier draft chose hyperparameters (α/w/λ, sweeps) on val
  and then accepted a candidate if it had "the same sign on val" — that is circular, because
  val is used for both selection and confirmation. All hyperparameters are now chosen on
  train-CV only (§5 Phase 1); val is confirmation only, never selection. ~10 candidates plus
  stacks still reach test with no multiplicity control otherwise: **pre-register ONE primary
  candidate per phase for the test read**, or apply a Holm correction across candidates that
  are read on test. Quote the measured SEs above rather than assuming pairing solves the
  multiplicity problem.
- Spatially abutting tiles (6,036 abutting pairs, minimum tile-start gap 1536; check1.log:
  9-13) make region bootstrap and region-level CV slightly anti-conservative — note this
  when interpreting bootstrap/CV intervals, it is not fixed by any change here.
- Acceptance: Δ > 2SE on test, AND the same sign and similar size on val, AND on train-CV.

### 6.3 Overfitting detection (every candidate)

1. Train-vs-held-out gap.
2. Learning curve: fit on 25/50/100% of TRAIN regions, evaluate on val. The slope gives the
   noise component. The intercept extrapolates the N→∞ ceiling.
3. Placebo: shuffle the conditioning label (band, GC bin, left-right pairing) among TRAIN
   fragments, refit, evaluate on val. The expected result is ≈ −k/2N and must be ≤ 0.
4. Report capture_vs_prior with its bootstrap interval, never as a point.

### 6.4 Ledgers

- **Test-touch ledger** (`docs/pending/cut_site_baselines_test_ledger.md`, TO BE WRITTEN):
  date, candidate, npz sha256, store config_hash, git sha, the test number as printed.
  Bind every hash to output on screen.
- **Provenance:** every `.npz` has a sidecar (§4.3). Every NLL JSON records the npz sha256.

### 6.5 Results table template (fill it in; do not edit the known cells)

| Row | Val | Test | Δtest vs prior ± 2SE | capture_vs_prior (test) | Extra params | sim-full Δ | sim-10pct Δ | sim capture |
|---|---|---|---|---|---|---|---|---|
| uniform | 12.674475 | 12.674475 | | — | — | | | |
| unif+p_L (closed form) | 12.502734 | 12.501596 | | — | 155 | | | |
| HISTORICAL: old prior (stage1 v4, `m+L-1`) — RETIRED anchor | 12.146835 | 12.142214 | (P0b.3: old → N0) | — | 16,535 | | | |
| decomposition: `N0_ref` (stage1 formula, `m+L`) | | | | — | 16,535 | | | |
| decomposition: `N0_raw` (`w=0`, `m+L`) | | | | — | 16,535 | | | |
| **prior = N0 (`w*`, `m+L`) — NEW anchor** | | | 0 | 0 | 16,535 | 0 | 0 | 0 |
| N1 + margin fragments | | | | | 0 | | | |
| N2 strand-pooled (diag) | | | | | −8,190 | | | |
| N3 PWM backoff | | | | | 0 | | | |
| N4 ML calibration | | | | | 0 | | | |
| N0→N1→N4 stacked | | | | | 0 | | | |
| + M2-lite / M2-full | | | | | 252 / 16,380 | (neg. ctrl) | (neg. ctrl) | |
| + M1 flank | | | | | 228 | (left: neg. ctrl) | | |
| + M3 GC×g | | | | | 100 | (neg. ctrl) | | |
| + M5 normaliser | | | | | 1 | (pos. ctrl) | (pos. ctrl) | |
| HISTORICAL: model on old prior (sanctioned fp32 evaluator) — RETIRED | 12.002607 (ckpt-selected) | 11.996969 (upper bound) | — | — | 116,032 | | | 0.567 / 0.480 (confounded, F6) |
| model retrained on the new prior | | | | 1 by definition | 116,032 | | | |
| oracle (sim only) | — | — | — | — | — | (P0b.4) | (P0b.4) | 1 |

**Revision 3.** The Δ column is now measured against the NEW anchor (N0 at `m+L`). The
uniform and unif+p_L Δ cells were +0.532261 / +0.359382 against the retired 12.142214 and
are blanked until P0b.3 fills the anchor. The historical rows keep their values for the
record, are never a Δ reference, and are reproducible only at `bad981d` code. The sim Δ
column and the oracle row wait for P0b.4, because the historical sim prior and gaps carry
the F6 confound. Stacked gains (N0→N1→N4, and every "+" row) are NOT additive and are
UNMEASURED until each stacked row is actually scored — do not sum the individual rows'
bounds to predict a stacked row. The unif+p_L values are the [Verified] 12.502734/12.501596
(closed form, check2.log:26-27). The historical model row uses the sanctioned fp32 scorer
value 12.002607 (not the bf16 trainer filename value 12.002583; see F-09, §2.1).

### 6.6 Sim controls

| Candidate | Sim role | Expected on sim |
|---|---|---|
| N4 | POSITIVE (counted ≠ MLE holds on sim too) | Closes part of the prior-oracle gap |
| M5 | POSITIVE (generator has `/t(m)`) | γ=1 > 0 |
| M2, M3 | NEGATIVE (no band or GC term in the generator) | ≈ −k/2N unshrunk, ≈ 0 shrunk. A gain is a leak or a bug. |
| M1 left end | NEGATIVE (6-mer only) | ≈ 0 |
| M1 right end | NEGATIVE after change B (the generator is 6-mer-only at `m+L`); was "not a control" while F6 was open | ≈ 0. A gain at the innermost right-end offset means change B is wrong — a bug, not signal |
| N2 | Depends on the simulator's minus-strand convention | Tells validity on sim only |
| ~~Right-end positive control~~ | **Retired by change B.** It existed only because the prior's schema was fixed at `m+L-1`. After B the prior itself reads `m+L`, so the P0b.4 sim prior IS the registration-matched control. | The P0b.4 sim prior-oracle gap should be smaller than the historical 0.064 / 0.227 if F6's registration component was real. Change A also moves it, so score `N0_ref` on sim too before attributing the drop to B. [Inferred] |

**Reviewer round 1, F-06: the sim right end is misregistered against the prior (§1.3 F6),
with three consequences for this table.** (1) the sim's prior-oracle gap (0.064 full, 0.227
10pct) contains a registration component, not purely a sequence-signal gap; (2) part of the
model's 56.7%/48.0% sim capture can be L2a recovering that one-base registration rather than
real sequence signal; (3) M5's sim positive control is only attenuated. **The "correction is
worth ~4x more on real than sim" contrast is CONFOUNDED by this and must NOT be used as
evidence** — the earlier draft's closing sentence to that effect is removed. Real remains the
arbiter regardless; do not accept or reject a candidate on sim alone. **Revision 3:** all
three consequences describe `bad981d`-era numbers. Change B removes the cause for every
number produced after P0b; the historical numbers keep the confound.

The oracle row may use `draw.py::oracle_nll` (reuse ruling), but only for the sim oracle.
It computes `-mean(log p)` pooled over fragments, in float64 [Verified: draw.py:181-186,
r3], whereas every other row here is REGION-weighted. Using it for any candidate row would
create a second evaluator with a different weighting. Confirm that the published oracle
anchors were computed in the same weighting as the prior rows before comparing them.
[Unverified: read the sim run's oracle computation and its summary JSON]

---

## 7. Commands, new scripts, tests

### 7.1 Environment (every session)

```bash
W=/home/nathanboley/src/fragmentomics_tools/.claude/worktrees/cut-site-model
PYENV=/home/nathanboley/miniconda3/envs/biomarker_env/bin
STORE=/efs/analytics/nathanboley/background_model/cut_site_stores/quiet_tile1536_v3.zarr
# HEX is the RETIRED m+L-1 table set. Use it ONLY at bad981d code (Phase 0, G0). After the
# change set the m+L loader rejects it (§0.4); pass the new counter's N0 .npz instead.
HEX=/efs/analytics/nathanboley/background_model/cut_site_runs/stage1_hex_tables_v4_start.npz

# Shadow check: the repo root's background_model/ shadows the worktree even with PYTHONPATH set.
cd /tmp && PATH=$PYENV:$PATH PYTHONPATH=$W python -c "import background_model,sys; p=background_model.__file__; print(p); assert p.startswith('$W'), p"

# Suite baseline BEFORE and AFTER. Measure it; do not quote. Exit 137 = HUNG: py-spy dump, do not re-run.
PATH=$PYENV:$PATH make -C $W test
# make test's default PYTEST_ARGS already collects tests/ (Makefile:65), so a separate
# `PYTEST_ARGS="tests/ -q"` run is redundant and was removed here (reviewer round 1, F-16).
# If needed: TEST_TIMEOUT=7200. Never bare pytest.
```

Read `metrics.csv` by COLUMN NAME. Never pipe `git commit` or `batch-run submit` through
head/tail. Store builds and counting run on CPU locally. GPU is for train/score only.

### 7.2 Sanctioned scoring (EXISTING)

```bash
# Smoke first; measure seconds per region before the full run.
# Reviewer round 1, F-08: --limit-regions together with --reproduce-val-mult-nll trips
# p.error at score_fragments.py:408-409 AFTER scoring and writing the TSV (exit code 2), so
# the smoke command below OMITS --reproduce-val-mult-nll. Use the full run (next block) to
# exercise that flag.
cd /tmp && PATH=$PYENV:$PATH PYTHONPATH=$W python -u $W/scripts/cut_site_model/score_fragments.py \
  --store $STORE --hex-tables <candidate.npz> --zero-correction \
  --output <out>/per_fragment_<name>.tsv --splits val test --device cpu --workers 0 \
  --limit-regions 200

# Full (val+test ≈ 13k tiles). The summary JSON is written beside the TSV.
cd /tmp && PATH=$PYENV:$PATH PYTHONPATH=$W python -u $W/scripts/cut_site_model/score_fragments.py \
  --store $STORE --hex-tables <candidate.npz> --zero-correction \
  --output <out>/per_fragment_<name>.tsv --splits val test --device cpu --workers 0 \
  --reproduce-val-mult-nll

# VAL-only cross-check via the trainer. Reviewer round 1, F-09: train.py defaults to
# --precision bf16-mixed (train.py:101-102), which perturbs the NLL — the best checkpoint's
# bf16 trainer value is 12.002583 (ckpt filename) vs the fp32 scorer's 12.002607100258485 on
# the SAME checkpoint (summary json). Always pass --precision 32 here; §2.1/§6.5 cite
# 12.002607 as the sanctioned VAL model value, not the filename's bf16 value.
cd /tmp && PATH=$PYENV:$PATH PYTHONPATH=$W python -u $W/scripts/cut_site_model/train.py \
  --store $STORE --run-dir <out> --tile-size 1024 --L-min 25 --L-max 180 \
  --hex-tables <npz> --freeze-hex --min-N 10 --batch-size 32 --validate-only --zero-correction \
  --precision 32
```

Prefer `score_fragments` for prior-only numbers; use the trainer cross-check only to confirm
the two paths agree.

### 7.3 New scripts

Every new script asserts the shadow check at start, accumulates in float64, and writes a
provenance sidecar. **Revision 4: `reduce_split_nll.py` (S2) and `count_tables.py` (S3) are
BUILT, landed in Phase 0 / 0b.1 respectively. `hex_adapter.py` was never built — removed,
§0.4 C. The remaining rows are still TO BE WRITTEN (not started; Phase 0b.2 onward).**

| Path | Purpose | Inputs → outputs |
|---|---|---|
| `scripts/cut_site_model/baselines/reduce_split_nll.py` | S2 — **BUILT (Phase 0)** | `--per-fragment-tsv --store --splits val test --min-N 10 [--paired-with TSV --bootstrap 2000 --seed 0]` → JSON: NLL per split, n_regions, Δ ± 2SE |
| `scripts/cut_site_model/baselines/emit_reference_npz.py` | uniform / unif+p_L npz | `--p-L-from <npz> --kind {uniform,unif_pL}` → conforming npz. `<npz>` is `$HEX` in Phase 0 and the N0 `.npz` after the change set. Post-change outputs carry the registration key, or the `m+L` loader rejects them; the uniform tables are registration-free, so the key only satisfies the guard. [Inferred] |
| `scripts/cut_site_model/baselines/count_tables.py` | S3 counter — **BUILT (Phase 0b.1, commit `9aa8ef5`).** The SINGLE table producer after the change set (N0 incl. `N0_ref`/`N0_raw`; N1/N2/N3 crop modes not yet built) | `--store --split train --crop {centred,window} --estimator {stage1_ref,shrink,backoff} --weight --pool-strands --region-fraction --region-seed --assert-no-cross-split-overlap` → npz (with registration key `stop_cut_base_offset`) + provenance (incl. the mask-limitation fields, §4.3). Thin driver over `measure.py` for `fl_end_weight`/`FragmentLengthDist` only; hexamer indices computed directly in the model's own convention — **no adapter** (revision 4, §0.4 C; supersedes the revision-3 "`hex_adapter.py` applied last" text). `--assert-no-cross-split-overlap` raises if overlap is found rather than silently filtering (F-14) — on `quiet_tile1536_v3.zarr` it always passes (check1.log:9-13), but it is kept as a guard for other stores. Counts TRAIN at `min_N=0` (every tile, 53,321 on the real store), ~78 s CPU. Window mode (N1) needs its own registration path for negative tile coordinates (F-12); it is not the centred-crop path T4 covers, and is not yet built. |
| `scripts/cut_site_model/baselines/phase0_diag.py` | P0.2, P0.4 | subcommands `smoothing`, `offsets --by-strand --shifts -2..2`, `strand-tie` → TSV/JSON |
| `scripts/cut_site_model/baselines/expected_counts.py` | engine + P0.5 residual maps | `residuals --hex-tables --features table,band,flank,gc_g,lr_dinuc,tlog --split train --null-permutations 20` → JSON |
| `scripts/cut_site_model/baselines/fit_ipf.py` | N4 (+ joint M terms later) | `--init npz(=N0 tables) --store --split train --optimizer {lbfgs,gis} --lambda --gis-eta 0.33 --max-sweeps 8 --weighting {region,fragment} --region-subsample 10000` → one npz per sweep/iterate. Reviewer round 1, F-02: `--eta`/`--alpha` damped-multiplicative spec is replaced — the fixed point of `log θ += η(log(O+α)-log(E+α))` is the unregularised MLE for any α, so α is not a regulariser; use a ridge `--lambda` toward the N0 tables (or Dirichlet pseudo-counts on O only), optimise with L-BFGS on the exact gradient O−E+penalty, or GIS capped at `η≤1/3` (3 features per cell). Fix the gauge with ONE common constant preserving `a+b=c+d` applied once at the very end, never per sweep. Choose BOTH the sweep count and `λ` by train-CV; val is confirmation only, never selection (reviewer round 2, R2-02 — the sweep count is itself a hyperparameter, so selecting it on val is the circularity F-11 removed). The fitter never reports NLL. |
| `background_model/cut_site_model/prior_terms.py` | S1 (after D1) | bundle npz → additive (B,P,n_L,2) terms |

Example (after the scripts exist):

```bash
cd /tmp && PATH=$PYENV:$PATH PYTHONPATH=$W python -u $W/scripts/cut_site_model/baselines/reduce_split_nll.py \
  --per-fragment-tsv <out>/per_fragment_prior.tsv --store $STORE --splits val test --min-N 10
```

### 7.4 Tests (new file(s) under `tests/`, background_model suite)

| ID | Test |
|---|---|
| T1 | All-uniform tables → every region's nll = ln 319488 (tol 1e-5, float32 forward) |
| T2 | unif+p_L → per-region `ln 2048 − mean log p_L[L_f]` (tol 1e-6) |
| T3 | S2 wrapper mean == `region_weighted_mult_nll` mean (1e-12). With D4: Σ terms / n == mean. |
| T4 | **Restated (revision 3); CORRECTED and MEASURED (P0b.1 implementation, revision 4).** The design named `start_fwd` as the one table whose registration (`m`) and denominator (start domain = `N_start`) the change set leaves unchanged. That was INCOMPLETE — a real design error: `stop_rev` is also unchanged, because it too is read at the left end `m` with the `N_start` denominator; only the right-cut tables (`stop_fwd`, `start_rev`) move. The test asserts both. **Measured on real data: `start_fwd` and `stop_rev` both reproduce stage1 at max relative difference 0.0; `stop_fwd` and `start_rev` differ by 11.67x and 37.47x respectively, by design.** [Verified: COORDINATION gate `G0b_phase_0b1`, divergence 1] Non-ACGT windows are excluded from the comparison. |
| T4b | **Phase-0b gate (F-01/F-12), BUILT AND PASSED (revision 4) — role corrected.** Revision 3 described this as a probe of the counter PLUS `hex_adapter.py` against `compute_log_p0`. No adapter was built (§0.4 C): nothing hexamer-keyed crosses stacks in the production path, so there is no second convention for this test to cross-check against. What is built and guards the counter instead: a one-hot probe of the counter against `compute_log_p0`, for the centred crop, both strands, both ends, **EXHAUSTIVE over all 4096 cells x 4 tables** (not sampled) — every table equals the bincount of the index `compute_log_p0` actually reads. [Verified: COORDINATION gate `G0b_phase_0b1`] Window/N1 and pooled/N2 crop modes are not yet built and still need their own probe test when they are. No `.npz` from the counter was scored before this probe passed. |
| T5 | Engine log-probs == `domain_log_probs` (zero correction) to **1e-5 max-abs**, or compared via `model.double()` on both sides (F-13: 1e-6 will flake against the float32 forward). |
| T6 | IPF fixed point: tables set to the MLE of a synthetic fixture do not move (|Δ log θ| < 1e-8) |
| T7 | **Restated (F-02).** "Penalised stationarity", not "O~E": after the fit, `‖O − E − penalty_gradient‖ < tol` on TRAIN, float64. (The earlier T7 asked for O≈E, which is only true at λ=0/unpenalised — the fixed point with a ridge or Dirichlet penalty is NOT O=E.) |
| T8 | S1: empty bundle is bit-exact with the current (N0, `m+L`) prior; a post-change-set checkpoint loads. (Revision 3: a `bad981d` checkpoint must be REFUSED — T13.) |
| T9 | S1 per term: identity-probe registration (as `tests/test_hex_registration.py`), right end at `m+L` (revision 3; was `m+L-1`) |
| T10 | Import-smoke of every consumed symbol from dataset.py / `background_model_core` (jitter helpers) — NOT from the stage1 file for `hex_window_for_tile` (F-15; ownership seam) |
| T11 | **Restated (F-14).** Margin counting ASSERTS zero cross-split overlap and raises if any is found (a synthetic adjacent-tile fixture exercises the raise path); it does not silently filter. On the real store the assertion is confirmed to always pass (check1.log:9-13). |
| T12 | **New (revision 3, sequencing constraint), in `tests/test_hex_registration.py`. BUILT, PASSED, and PROVED non-tautological by MUTATION (revision 4).** A CROSS-check, not a self-check: (a) T12a ties the hex_idx position `compute_log_p0` gathers for the 3' cut to the `h` position L2a taps (`WINDOW_HALF + m + L`, model.py:641-643), by SWAPPING individual bases and observing the dependency, not by restating the formula; (b) T12b ties it to the simulator's `cut_site_hexamers`, joined by k-mer STRING (`model_index_of_sim_index()`), so it is convention-independent — no adapter needed for this check either, since it compares via the string, not the index. The old probes at `:108-109`, `:252-254` and `:336` moved to `m+L` (these are the ones that used to pin the convention to itself). **Non-tautology was proved by MUTATION, independently by the implementer and the reviewer:** reverting BOTH the producer and the reader to `m+L-1` fails 12 tests (incl. T12a/T12b and the `propensities` conformance test); reverting the READER only fails 6 (incl. T4b). [Verified: COORDINATION gates `em_note_mutation_test_proof`, `implementation_review_phase_0b`] |
| T13 | **New (revision 3). BUILT (revision 4) — implemented as a key named `stop_cut_base_offset`, not the proposed `stop_cut_offset`.** The `m+L` loader rejects an `.npz` without the key (fixture: a copy of the v4 schema) and a checkpoint whose buffers carry `m+L-1` tables, via a persisted `_hex_stop_cut_base_offset` buffer plus a `load_state_dict` pre-hook that **cannot be bypassed with `strict=False`** (the pre-hook runs before the load). Also: `required_width` at model.py:204 and the constructor check at :295 fire for a geometry one base too narrow under the CORRECTED `WINDOW_HALF >= L_max` derivation (§0.4 B, revision 4) — construct with `L_max = WINDOW_HALF + 1` and the constructor must raise. This is a stronger claim than "the `-1` was removed": the OLD guard (`WINDOW_HALF < L_max - 1`) was latently broken independent of the `m+L` change, admitting `L_max=182` at the boundary of `h`'s width; the test proves the NEW guard (`WINDOW_HALF >= L_max`) catches what the old one missed. [Verified: COORDINATION gates `implementation_review_phase_0b`, `em_reviewer_found_the_old_width_guard_was_BROKEN`] **This mechanism is implemented and tested but NOT owner-ruled (divergence 5, §9):** the registration key and checkpoint refusal were the design author's proposal for enforcing the sequencing constraint; the owner has not explicitly approved the mechanism itself, only the underlying `m+L` registration (ruling 6). |
| T14 | **Retarget (revision 3).** `test_builder_and_model_agree_on_registration` (`:116-140`) and `test_builder_widened_window_covers_right_edge` (`:269-294`) import `hex_window_for_tile` from the retired stage1 file, and the second hardcodes `L_max - 1` (`:278`, `:288`). Retarget both to the new counter (far 3' cut at `tile_size − 1 + L_max`). Do not edit the stage1 file. |

---

## 8. Open questions and unverified items

| # | Question | Command or file that settles it |
|---|---|---|
| Q1 | Sim hex-table `.npz` path | Read the `hex_tables` field of `<sim run dir>/per_fragment_prior.tsv.summary.json` |
| Q2 | Other callers of `compute_log_p0` | `grep -rn compute_log_p0 $W/background_model $W/scripts $W/tests` |
| Q3 | Does the cut-site dataset use `/endpoints/*`? | `grep -n endpoints $W/background_model/cut_site_model/dataset.py` |
| Q4 | `train.py` precision default | **RESOLVED (reviewer round 1):** default is `--precision bf16-mixed` (train.py:101-102); always pass `--precision 32` for prior cross-checks (F-09, §7.2) |
| Q5 | Home module of stage1 helpers | `grep -n "^from\|^import" $W/scripts/cut_site_model/stage1_hex_baseline.py` |
| Q6 | ~~Does the loader renormalise tables (fwd/rev scale)?~~ **RESOLVED: no.** `model.py:399-404` takes `np.log(np.maximum(data[key], 1e-30))` with no normalisation step, so a table-scale change lands in the model rather than being absorbed. This is why N4's gauge must preserve `a+b=c+d` (§3.1). | [Verified: model.py:399-404, reviewer round 2 R2-03] |
| Q7 | ~~Sim right-end index convention vs model (F6)~~ **RESOLVED (revision 3):** the simulator uses `ends_all = pos + Ls`, i.e. `m+L` [Verified: draw.py:83-85, r3]; the prior moves to it under ruling 6. Endianness and RC keying remain — **revision 4: NOT handled by an adapter, because none was built.** Nothing hexamer-keyed crosses the production path, so the split is simply unused there; it stays open and deferred as a joint call with the simulator EM (§0.4 C). | [Verified: draw.py:83-85, measure.py:179-199, r3; COORDINATION gate `em_unblock_phase_0b`] |
| Q8 | Sim minus-strand storage convention (N2 control) | Read the strand branch of `draw.py` and the simulator writer |
| Q9 | Cross-split tile adjacency (N1 leakage) | **RESOLVED (reviewer round 1):** ZERO overlaps found, every form checked (window×crop, crop×crop, window×window, seq-span×crop); min tile-start gap 1536, 6,036 abutting pairs (check1.log:9-13). Leakage-safe on this store (F-14); keep an assertion for other stores. |
| Q10 | Does `compute_log_p0` accept float64? | Call it on a 1-region fixture with float64 tables |
| Q11 | Per-region nll sd (bootstrap SE scale) | **PARTIALLY RESOLVED:** reviewer measured the TEST unpaired per-region prior NLL SE at 0.00457 (sd 0.363, n=6,303) and the paired prior-minus-model SE at 0.00243 (sd 0.193, n=6,303) (check1.log:21-22). Still need the equivalent VAL numbers and the `return_terms` path (D4) for arbitrary candidates. |
| Q12 | Reproduce unif+p_L 12.501596 | **RESOLVED by closed form:** reviewer recomputed 12.5027343162787 (val) / 12.50159635865178 (test) directly (check2.log:26-27), matching the published test figure. P0.1 must still reproduce it through the npz + sanctioned-evaluator path, which is a different check (exercises S2 and the npz schema, not just the formula). |
| Q13 | Smoothing distortion NLL cost | Bias bound measured (check2.log:15-18); realised NLL gain from P0b.3 (`N0_ref → N0`) |
| Q14 | Right-end per-offset table by strand | P0.4 |
| Q15 | Is pysam installed? **Now load-bearing:** `uniform_hexamer_counts` imports it inside each worker (measure.py:343) | `$PYENV/python -c "import pysam"` |
| Q16 | How other code treats L=180 in FL_BANDS | `grep -rn FL_BANDS $W/background_model $W/scripts` |
| Q17 | Which split the sim anchors use | Read the sim run summary JSONs |
| Q18 | Engine time per region | Time 200 regions |
| Q19 | Library transfer: tables from RD-56413-Lib1 can encode library-specific bias. One library per store (COORDINATION item 6). | Untestable this phase. A second-library store would settle it. |
| Q20 | ~~Can `uniform_hexamer_counts` run on the store's TRAIN crop?~~ **RESOLVED: NO — and do not try.** `_uniform_block` opens `pysam.FastaFile` and fetches per region; `uniform_hexamer_counts` takes `rdf` + `fasta_path`. Four reasons not to bridge it: (1) the model reads `tiles/seq`, so counting denominators from a fresh FASTA fetch risks counting one sequence and applying rates to another — invisible, because a wrong denominator changes values not shapes; (2) it assumes `[gstart, gstop)` + flanks, while our domain is a centred 1024 crop inside `l_seq=1908` with reach `L_max` past it, so it needs synthesised coordinate triples nothing checks; (3) it gates on `valid` only, while our counting must honour `tiles/mask` — different positions, different denominator; (4) it adds a reference-FASTA artifact and `pysam` to a step that needs neither, for a ramp worth ~8e-5 nats. **Write the `N_start`/`N_end` loop store-backed** (~15-20 lines). Reuse the PURE pieces: `fl_end_weight` (`:312-333`, closed form) and `propensities` (`:500-560`) as a fixture cross-check. **This also makes Q15 (pysam) moot for the production path.** | [Verified: measure.py:336-392, :395-403; EM gate `em_settle_Q20_reuse_feasibility`] |
| Q21 | ~~Do `bad981d` checkpoints carry the hex tables in their state dict?~~ **RESOLVED: YES.** `register_buffer` for all four tables under `--freeze-hex`, and always for `_log_pL`; PyTorch persists buffers in `state_dict`. So every `bad981d` checkpoint carries `m+L-1` tables, and loading one under `m+L` code would gather old tables at new positions while loading SILENTLY, since buffer shapes stay `(4096,)`. The retrain is mandatory and the registration guard MUST cover checkpoints, not only the npz. | [Verified: model.py:406-408, :412-415; EM gate `em_verify_checkpoint_tables_baked_in`; reviewer round 3 R3-03] |
| Q22 | Were the published sim oracle anchors computed region-weighted, like the prior rows, or with `oracle_nll`'s pooled mean? (§6.6) | Read the sim run's oracle computation and summary JSON |
| Q23 | ~~Does `score_fragments.py` accept `--splits train`?~~ **RESOLVED: YES.** `SPLITS` includes `"train": (0, "train")` and the flag uses `choices=list(SPLITS)`, so P0b.2's train-CV can score folds through the sanctioned path. | [Verified: score_fragments.py:90, :367; reviewer round 3 R3-01] |
| Q24 | ~~Does stage1's `p_L` exclude the N-window fragments it drops from the tables?~~ **RESOLVED: NO, it includes them.** `length_counts[L - L_min] += 1` runs BEFORE the hex-validity check, so `p_L` counts every admitted fragment while the tables drop invalid windows. G0b item 4 must therefore expect the new counter's `p_L` to differ from the v4 npz by exactly the N-window fragments, not to match bit-for-bit. | [Verified: stage1_hex_baseline.py:197-198 vs :203-215; reviewer round 3 R3-02] |

**Scope limits — corrected (reviewer round 1, F-10).** `min_N` counts ALL stored fragments in
the 1536-bp window (dataset.py:118-127), NOT centred-crop fragments; the earlier draft's
"regions with ≥10 centred-crop fragments" is wrong. Results here hold for regions with ≥10
stored (window) fragments, which excludes 5.4-5.7% of regions (§0.1 item 4; [Verified: T1]).
At `min_N=1` (all fragments admitted regardless of count), 6 val and 2 test regions still
have zero SCORED (centred-crop) fragments (check1.log:16,18) — so even the min_N=1 rows in
§2.1/§6.5 are not "every region scored", and N4's TRAIN min_N=10 population and N1's margin
population (§3.1, §3.2) should both be read in these same window-vs-crop terms, not
crop-only.

---

## 9. Self-grade: **B+** (revision 3; round-2 reviewer gave B+ and said A- once R2-01..03 were applied)

I do not claim A-. Revision 3 is a large, UNREVIEWED edit: a new §0.4, a three-part Phase 0,
a new gate G0b, four new tests (T12-T14, T4b extended) and five new open questions. Round 1
showed that bulk unreviewed edits to this document are where new errors enter. A- should
come from a round-3 reviewer, not from me.

What revision 3 earns:
- The four owner decisions are reflected as decided, not as open items. The sequencing
  constraint sits at the top (§0.4), with a mechanism that enforces it without editing the
  stage1 file: one commit, a loader registration key, and a checkpoint refusal.
- The two carried EM findings are answered in place: the 0.02-0.08 / 0.03-0.04 relation is
  stated (§0.2, §3.5), and the separability argument is the stated reason the order is safe
  (§3.5, §5 Phase 1).
- The ramp measurement is applied in the direction it points: F-04 is confirmed, the 156x
  figure is fenced off, and the denominator reuse carries no gain claim anywhere.
- Revision 3 read the source directly for every new pointer and found five residual
  inconsistencies that rounds 1-2 left: §1.1 "Table scale is irrelevant" (contradicted Q6
  and §3.1); §3.1 "val NLL per sweep" and §3.8 "γ chosen on val" (contradicted R2-02/F-11);
  S1 risk 4 "old checkpoints score unchanged" (false after B); and two registration tests
  that import the stage1 file being retired.

What holds it below A-:
- Phase 0b depends on five unverified mechanics: Q20 (FASTA vs store sequence for
  `uniform_hexamer_counts`), Q15 (`pysam`), Q21 (checkpoint buffers), Q23 (`--splits train`)
  and Q24 (stage1's `p_L` population). Any one can force a design change in §4.3 or G0b.
- The adapter's exact permutation (digit reversal plus rc un-permutation) is derived, not
  tested. The probe T4b, not this document, is the authority.
- No gain is measured. The new anchor itself is still a blank row.
- The loader registration key and the checkpoint refusal are my proposals for enforcing the
  sequencing constraint. The owner has not ruled on them.

### Round-1 self-grade (historical): B-

The reviewer found the index-convention incompatibility of the sanctioned "reusable pieces"
(F-01), a mathematically wrong claim about N4's regulariser (F-02), a mis-classification of
N0 as variance-side when it is measured bias (F-03), and a half-dozen smaller pointer/number
errors (F-05 through F-16) — none of which the original draft's self-review caught. That is
the honest basis for B-, not B.

What earns it:
- Every number now traces to either the original packet, this review's own recomputation
  (check1.log/check2.log), or carries its arithmetic. The brief contradictions and the
  stale-doc finding are surfaced at the top.
- The plan keeps the single sanctioned evaluator. New code sits behind named seams, and no
  owned file is edited.
- Several challenges from the original draft held up under review: the N1 leakage question
  (now answered: zero overlap, F-14), the sim right-end misregistration (F6, now shown to
  have three concrete downstream consequences, F-06), partial confounding of T5 (a)/(b), and
  stage1's smoothing read as a hierarchical prior.

What holds it below B+:
- Round 1 found a real correctness bug in the design, not just missing verification: the
  "reusable pieces" (`hexamers.py`, `measure.py`) use the OPPOSITE hexamer index convention
  from the model (F-01), which would have let a plausible-looking `.npz` silently score every
  cell under the wrong hexamer. That is exactly the kind of error this document's own
  tag-everything discipline exists to catch, and it slipped through round 0.
- The N4 estimation spec (§3.1) had a genuine math error (α is not a regulariser; the
  per-table re-centring breaks the model's actual gauge freedom) that would have produced a
  wrong fit if implemented as originally written (F-02).
- Some load-bearing mechanics are STILL unverified after round 1: loader renormalisation
  (Q6, since RESOLVED in round 2), other `compute_log_p0` callers (Q2), float64 support in `compute_log_p0` (Q10),
  engine runtime (Q18), and the VAL-split bootstrap SE (Q11, test side only resolved).
- No candidate gain is measured, even after round 1. The bounds come from observed-fragment
  diagnostics with an unknown exchange rate to region-weighted NLL. The 0.02-0.08 expectation
  is a judgement, not a measurement, and stacked gains (N0→N1→N4, the M-layer "+" rows) are
  explicitly unmeasured and non-additive.
- The G-gate thresholds (0.002) are still proposals, not derived limits, though G0/G1 are now
  internally consistent with the measured float32 precision floor (F-13) and the measured N0
  distortion (F-03).

### Review log

| Round | Reviewer grade | Finding IDs | Disposition |
|---|---|---|---|
| 1 | B- | F-01 | Applied: index-convention mismatch flagged in §1.1/§4.3; S3 required to derive indices from the model's own encoder or a validated permutation adapter; one-hot probe test per crop mode made a Phase-0 gate (T4b). |
| 1 | B- | F-02 | Applied: §3.1 estimation rewritten — ridge/Dirichlet penalty replaces α-as-regulariser, L-BFGS or GIS at η≤1/3, gauge fixed once at the end with a+b=c+d preserved, λ chosen by train-CV; T7 restated as penalised stationarity. |
| 1 | B- | F-03 | Applied: N0 re-ranked to #1 in §3.0, re-classified BIAS-side in §2.5/§1.3 F1, moved to the first Phase-1 item in §5; measured numbers from check2.log inserted throughout; headline/ranking/phase-plan made mutually consistent. |
| 1 | B- | F-04 | Applied: F2 dropped as an N4 motivation in §3.1; kept as a one-line correctness note bundled into N0 (§3.5) with the measured negligible cost. |
| 1 | B- | F-05 | Applied: 3' `m+L`/`m+L-1` decision removed from the closed-decisions list in §0.3; marked open, not touched this phase. |
| 1 | B- | F-06 | Applied: three consequences of the sim registration mismatch added to §1.3 F6 and §6.6; the "~4x more on real" claim removed as confounded; a sim-only right-end positive-control diagnostic added to §6.6. |
| 1 | B- | F-07 | Applied: N3 formula corrected to `(c_h+w·q_h)/(occ_h+w)` with stated limits (§3.9); N0's analogous estimator in §3.5 written in the same corrected units. |
| 1 | B- | F-08 | Applied: `--reproduce-val-mult-nll` removed from the §7.2 smoke command, with the p.error/exit-2 interaction explained inline. |
| 1 | B- | F-09 | Applied: `--precision 32` added to the §7.2 trainer command; 12.002607 cited as the sanctioned VAL model value in §2.1/§6.5, with the bf16 filename value (12.002583) kept as a separate, explicitly non-sanctioned row. |
| 1 | B- | F-10 | Applied: §8 scope-limit sentence corrected from "centred-crop fragments" to "stored window fragments"; min_N=1 zero-scored-region counts added. |
| 1 | B- | F-11 | Applied: §6.2 rewritten — hyperparameters chosen on train-CV only, val as confirmation only; measured paired/unpaired SEs inserted; pre-registration/Holm-correction requirement added for the multiplicity problem; abutting-tile anti-conservatism note added. |
| 1 | B- | F-12 | Applied: combined with F-01 in §4.3 — window-mode (N1) registration flagged as needing new code, not a reuse of the centred-crop path; T4b probe test covers it explicitly. |
| 1 | B- | F-13 | Applied: G0 and T5 tolerances changed from 1e-6 to 1e-5 max-abs (or `model.double()` comparison), with the measured float32 precision floor cited. |
| 1 | B- | F-14 | Applied: N1 marked leakage-safe on this store (zero overlap measured, check1.log:9-13); the filter flag replaced with an assertion in §4.3/§7.3/T11; abutting-tile spatial-correlation note added to §6.2. |
| 1 | B- | F-15 | Applied: all pointer/number corrections made — L2a/L2b model.py pointers swapped back (§1.1, §2.6, §3.3), the retracted 10-fold CV sd demoted (§2.5), tile_size's location corrected, the all-ones mask and 53,321/54,600,686 figures corrected (§1.3 F1), M3's k/2N corrected to 8.3e-5 (§2.5), `hex_window_for_tile`'s/jitter helpers' home modules corrected (§4.1), T10 corrected to import jitter helpers from `background_model_core` not the stage1 file. |
| 1 | B- | F-16 | Applied: a "report to the stage1 owner" list added (§1.5); the redundant second `make test` line removed from §7.1. |

No finding was rejected. Every finding's disposition above is "applied" or "applied as part
of a combined fix" (F-04 and F-12, each folded into a neighbouring finding's edit).

## Review Notes (2026-10-10, round 2)

**Verdict**: APPROVED WITH CONDITIONS
**Grade**: B+

### Findings

| # | Severity | Category | Finding |
|---|----------|----------|---------|
| R2-01 | Medium | Consistency | §0.2 headline omits N0 and calls N4 the lead candidate, but §3.0 ranking puts N0 at rank 1. The fix pass updated §3.0, §3.5, §5 but missed §0.2. |
| R2-02 | Medium | Consistency | §7.3 fit_ipf.py says "Choose the sweep/λ by scoring on val" but §6.2 says "val is confirmation only, never selection." Contradicts F-11's fix. |
| R2-03 | Low | Consistency | Q6 marked open in §8 but answered in §3.1: loader does NOT renormalize (model.py:399-404). Close as RESOLVED. |
| R2-04 | Info | Accuracy | All 16 round-1 findings verified as correctly applied. |
| R2-05 | Info | Accuracy | 30+ source-file line-number ranges verified against the bad981d worktree. |
| R2-06 | Info | Accuracy | Endianness split, sim registration at m+L, and N4 gauge constraint a+b=c+d all confirmed correct. |
| R2-07 | Info | Design | All binding constraints respected. No open owner decision silently assumed. |
| R2-08 | Info | Accuracy | Measurement-agent numbers properly attributed, not promoted to reviewer-verified. |

### Conditions

1. Update §0.2 to name N0 as rank-1 candidate ahead of N4.
2. Change fit_ipf.py spec to select sweep count by train-CV, matching §6.2.
3. Close Q6 as RESOLVED.

### Risks

1. Implementer reads §0.2 headline, builds N4 first, initializes from stage1's biased tables instead of N0's corrected ones.
2. fit_ipf.py spec says to select sweeps on val while §6.2 forbids it — implementer picks one and the other is violated.
3. Five Q-items (Q2, Q3, Q10, Q18, Q19) remain genuinely open and gate Phase-1 work.

### Key Tradeoffs

- N0 before N4 (not simultaneous): N0's corrected tables become N4's initialization, so the ordering matters. The alternative (running N4 on raw stage1 tables) inherits the F1 smoothing distortion. Chosen direction is correct.
- Ridge penalty vs Dirichlet pseudo-counts for N4: both offered, neither mandated. The implementer picks based on scipy convenience. Either is sound because the objective is concave.
- Sweep count on train-CV vs val: train-CV chosen (after this review's R2-02 fix) to keep val as a clean confirmation split. The cost is slightly more computation (5-fold CV per candidate vs one val pass).

## Revision log

### Revision 3 (2026-10-10) — owner rulings 5-7, the reuse ruling, two carried EM findings

Targeted edits only. No research was redone and no section was rewritten wholesale. The
reviser read the source pointers listed in the header; every new pointer is tagged `r3`.
The reviser ran no code. No number below was taken from anywhere except COORDINATION, the
brief, this document's earlier rounds, or the source read this session.

| Change | What was edited |
|---|---|
| A — estimator fix APPROVED (ruling 5) | New §0.4 A states the category error and the replacement (clean `C/N` from `propensities`, `(C+w·q)/(N+w)` with `w` on train-CV, normalise to sum 1, loader does not renormalise). §1.3 F1 marked approved, with the category-error reading added. §3.5 restructured as a five-step pipeline on `measure.py`, plus decomposition rows `N0_ref`/`N0_raw`. §0.2, §3.0 and §5 re-cast N0 as the approved new prior anchor, not a candidate. D2 closed. G1 no longer gates N0. |
| B — 3' registration fixed to `m+L` (ruling 6) | Moved out of "open" (§0.3) into §0.4 B, with the junction derivation, the width check (no store rebuild), the two stale `-1` guards (model.py:204, :295), and `L_max=180` INCLUSIVE unchanged. Every `m+L-1` reference updated or labelled `bad981d`-era: §1.1 rows, F2, F6, T4 anomaly (§2.2), T6 (§2.4), N1 context arithmetic (§3.2), M1 failure mode and edge rule (§3.4), N2 (§3.7), M5 (§3.8), registration gate (§5), §6.6 controls, T9. The "right-end positive control" row in §6.6 is retired, because the prior now reads `m+L` itself. Anchors: `12.142214` and `11.996969` (and val 12.146835 / 12.002607) marked RETIRED in §0.1, §2.1, F5, §4.2, G0, §6.1 and §6.5; re-score for the prior, retrain for the model. |
| Sequencing constraint | Stated prominently in §0.4: the counter and reader move in ONE commit; the new counter is the single producer; stage1's counting path is retired, not edited. Added enforcement: a registration key in the `.npz` and a loader/checkpoint refusal (T13, a proposal). The `test_hex_registration.py` cross-check to L2a AND the simulator added as T12; two tests that import the stage1 file retargeted as T14. Phase 0 split into 0 (at `bad981d`), 0b (the change set) and 0c (diagnostics under the new anchor); new gate G0b. §1.5 relay item to session 8d207c5986. |
| C — reuse through ONE adapter | §0.4 C names the reused `measure.py` functions, the two remaining differences (endianness, RC keying; plus table names), and `hex_adapter.py` as the only crossing, guarded by the one-hot probe T4b. Single-encoder consolidation recorded as a named follow-up, out of scope. §1.1 incompatibility row, §4.1, §4.3 and §7.3 rewritten from "do not reuse" to "reuse through the adapter". §3.11 fences reuse of FUNCTIONS (required) from reuse of the generator's TABLES or f(L) (forbidden). §6.6 restricts `draw.py::oracle_nll` to the sim oracle row, because it is fragment-pooled, not region-weighted. |
| D — ramp measured NEGLIGIBLE; ruling 7 | §0.4 D carries the measurement [Verified: EM measurement, gate em_measure_ramp_cost]. F2 (§1.3) now covers both halves (domain extent and ramp) and states that F-04's verdict was CORRECT. The 156x figure is labelled position-level and barred as motivation. §3.1 and §3.5 state that the denominator reuse is for hygiene and correctness only, never justified by expected gain. |
| EM finding — §0.2 vs F-03 | §0.2 now states that N0's ~0.03-0.04 sits INSIDE the 0.02-0.08 stacked range (largest single component; the range floor is below N0's bound because the bound is upper-bound flavoured). Neither figure is a share of the 0.145 any more. The same sentence is cross-referenced from §3.5. |
| EM finding — candidates not separable | §3.5 ("Why fitting `w` is safe here") and §5 Phase 1 state that a hyperparameter fitted against a wrong denominator or registration absorbs the error, and that A+B+C landing together is the reason the order is safe. N4 initialises only from N0. |
| Residual inconsistencies found and fixed while re-checking | §1.1 "Table scale is irrelevant" → per-table scale moves strand log-odds; §3.1 overfit detection "val NLL per sweep" → held-out fold inside train-CV; §3.1 failure (d) closed against Q6, plus "do not re-normalise N4 output per table"; §3.8 M5 γ "chosen on val" → train-CV; §4.5 S1 risk 4 "old checkpoints score unchanged" → refused; §3.0 N1 "needs a leakage check" → verified safe; §7.1 `$HEX` labelled retired. Q7 resolved; Q13 and Q15 updated; Q20-Q24 added. Self-grade re-stated as B+ with justification (§9). |

Not applied: nothing from the brief was skipped. One item is applied as a PROPOSAL rather
than a decision: the loader registration key and checkpoint refusal (§0.4 item 2, T13).
They enforce the sequencing constraint, but the owner has not ruled on the mechanism.

### Revision 4 (2026-10-10) — reconciled against Phase 0 and Phase 0b.1 as implemented

Targeted edits only, against `COORDINATION.cut-site-model.md`'s "Design-Implementation
Divergences" section and its gate log (commit `9aa8ef5`+, Phase 0b.1 reviewed A-). No
section was rewritten, no conclusion, candidate ranking, or gate threshold changed. Every
new number is tagged `[Verified: COORDINATION gate <name>]`, not asserted from this
session's own reading.

| Divergence | What was edited |
|---|---|
| 1 — T4 was INCOMPLETE (design error) | T4 (§7.4) and G0b gate item 5 (§5) corrected: `stop_rev` is unchanged by changes A+B, not just `start_fwd` — both are read at the left end `m` with the `N_start` denominator. Measured: both at max relative difference 0.0 vs stage1; `stop_fwd`/`start_rev` differ by 11.67x/37.47x by design. |
| 2 — WINDOW_HALF derivation was wrong; old guard LATENTLY BROKEN | §0.4 B rewritten: the binding limit is L2a's `h` (`WINDOW_HALF` each side of `P`, farthest tap `WINDOW_HALF+(P-1)+L_max`), giving `WINDOW_HALF >= L_max`, not "drop the `-1` from `L_max-1`". The old guard `WINDOW_HALF < L_max-1` would have admitted `L_max=182` at the boundary of `h`'s width — LATENTLY BROKEN, not merely stale. T13 (§7.4) updated to describe the stronger claim and the `CutSiteFragmentModel(L_max=WINDOW_HALF+1)` raise test. |
| 3 — Mask handling past `l_target` | New paragraph in §4.3 documents the limitation as known and recorded (harmless on the real store, mask 100% True) rather than silent: the provenance sidecar's `mask_honoured_bases`, `cut_base_window`, `mask_unhonoured_bases`, `mask_all_true` fields are added to the S3 spec. |
| 4 — No `hex_adapter.py` built | §0.4 C, §3.5, §3.7 (N2), §4.1, §4.3, §7.3, T4b, T12, Q7, and the §1.1 incompatibility pointer all rewritten: no adapter exists; nothing hexamer-keyed crosses the simulator/model boundary in the production counting path (only `FragmentLengthDist`/`fl_end_weight` are imported, both length/position-indexed); the counter computes directly in the model's own convention. T4b is now an exhaustive (4096-cell) self-consistency probe, not a cross-stack conformance test. The endianness/RC-keying split is marked UNRESOLVED and deferred as a joint call with the simulator EM, not solved. |
| 5 — T13 guard not owner-ruled | §0.4 sequencing-constraint item 2 and T13 (§7.4) updated with the implemented mechanism's real name (`stop_cut_base_offset` key, `_hex_stop_cut_base_offset` buffer, a `load_state_dict` pre-hook that cannot be bypassed with `strict=False`) and reaffirmed as flagged-but-not-owner-ruled, not merely "proposed" (it is built and tested). |
| Measured facts folded in | `p_L` 599,412 admitted / 599,411 counted / 1 dropped (L=179), bit-for-bit match to v4 over admitted fragments (§0.4 C via §3.5; G0b gate item 4); `unif+p_L` test 12.501596321, \|Δ\| 3.2e-7 (§3.5, G0b gate item 4); suite 2F/1111P/3S (1116) at `9aa8ef5`, up from 1100 at `9fe0d31` (§5 P0b.1 row, G0b gate item 6; also Phase 0's own 2F/1071P/3S→2F/1079P/3S in the G0 gate); counter runtime ~78 s CPU and `min_N=0` counting population, distinct from the evaluator's `min_N=10` (§3.5, §4.3); non-tautology proved by mutation (12 tests fail reverting producer+reader, 6 fail reverting the reader only) folded into T12 (§7.4) and the sequencing-constraint item 4 (§0.4). |
| Also fixed | §0.4 sequencing-constraint items 1-5 and §9/T13 updated from future tense to DONE/implemented, since Phase 0 and Phase 0b.1 landed at `9aa8ef5`. New "Implementation status" block at the top names Phase 0 and 0b.1 COMPLETE (gates G0, G0b PASSED) and Phase 0b.2 onward NOT STARTED. The P0.1 "old-prior TSV" note is upgraded from "kept" to IRREPLACEABLE, with its path and the reason (the `m+L` loader now refuses the tables that produced it). A one-line note added after §0.4's retirement paragraph flagging `docs/pending/training_analysis.md` and older docs as stale on `m+L-1`, not edited here. |

Not applied: no edit to this document's conclusions, candidate ranking (§3.0), or gate
thresholds (§5). `docs/pending/training_analysis.md` was flagged stale, not edited (out of
scope, a separate retirement decision). Could not reconcile: none — all five divergences and
every "also fix" item in the brief had a clear landing spot in the existing structure.

## Review Notes (2026-10-10, round 3)

**Verdict**: APPROVED
**Grade**: A-

### Findings

| # | Severity | Category | Finding |
|---|----------|----------|---------|
| R3-01 | Low | Accuracy | Q23 is answerable: YES. `score_fragments.py:90` defines `SPLITS` with `"train"` as a key; `:367` uses `choices=list(SPLITS)`. Close as RESOLVED. |
| R3-02 | Low | Accuracy | Q24 is answerable: NO, stage1's `p_L` does NOT exclude N-window fragments. `stage1_hex_baseline.py:197-198` increments `length_counts` before the hex validity check at `:203-215`. The G0b gate at line 1087-1090 correctly anticipates this. Close with this answer. |
| R3-03 | Low | Accuracy | Q21 was already confirmed by EM gate `em_verify_checkpoint_tables_baked_in`. The doc lists it as `[Unverified]` — conservative, not wrong. Close as VERIFIED. |
| R3-04 | Info | Accuracy | All r3-tagged pointers verified against source (25+ checked). All match. |
| R3-05 | Info | Consistency | Every appearance of retired anchors `12.142214` and `11.996969` is correctly tagged RETIRED or HISTORICAL. No live reference found. |
| R3-06 | Info | Consistency | The `m+L` derivation is correct. `rfa.stops_0` is "0 based exclusive" (`fragment_array.py:246`), confirming `measure.py:170` uses `m+L`. Width arithmetic and `L_max=180` INCLUSIVE are correct. |
| R3-07 | Info | Consistency | The `0.02-0.08` / `0.03-0.04` reconciliation is consistent across §0.2, §1.3, §3.5, and the revision log. |
| R3-08 | Info | Consistency | Val/train-CV discipline is maintained throughout. All pre-R3 contradictions were fixed. |
| R3-09 | Info | Design | No binding constraint violated. |
| R3-10 | Info | Accuracy | The N0_ref decomposition (`w=4096, q=1/4096`) correctly reproduces stage1's formula. |

### What was verified vs could not be verified

**Verified this session:**
- 25+ r3-tagged source pointers against `bad981d` code in the worktree
- Every appearance of retired anchors (systematic grep)
- Every `m+L-1` reference for correct contextualisation (systematic grep)
- Val/train-CV consistency across all sections (systematic grep)
- `0.02-0.08` / `0.03-0.04` / `156x` / `0.145` consistency (systematic grep)
- Q23 (YES), Q24 (NO), Q21 (already confirmed by EM gate)
- `rfa.stops_0` convention (`fragment_array.py:246`: 0-based exclusive = `m+L`)
- The `m+L` derivation's correctness
- No binding constraint violations
- N0_ref algebra

**Could not verify (no code execution):**
- Q15: whether `pysam` is installed in `biomarker_env`
- Q20: whether `uniform_hexamer_counts` can be driven from store tile coordinates + FASTA
- The adapter's digit-reversal + rc un-permutation produces correct output (T4b is the authority)
- Any NLL number (no scoring runs)

### Risks

1. Q15 (pysam) or Q20 (FASTA-vs-store) fails during Phase 0b, forcing a fallback to store-backed hexamer counting. The fallback is documented (§4.3 lines 961-963) but adds scope.
2. The adapter permutation is derived but untested. An error silently scrambles all four tables. T4b is the sole guard and must pass before any `.npz` is scored.
3. The registration key and checkpoint refusal are proposals without an owner ruling on the mechanism. An implementer might defer them, leaving the sequencing constraint unenforced.

### Conditions

1. Close Q23 as RESOLVED (YES).
2. Close Q24 with: NO, stage1's `p_L` does not exclude N-window fragments. The new counter should match this or document the ~31-fragment difference in the provenance sidecar.
3. Close Q21 as VERIFIED (EM gate `em_verify_checkpoint_tables_baked_in`).

### Key Tradeoffs

- **Adapter vs single-encoder consolidation:** the adapter is the correct immediate step. Full consolidation would re-key artifacts owned by the simulator EM and is properly scoped as a follow-up.
- **Registration key as a proposal vs an owner-ruled mechanism:** the constraint is owner-ruled (ruling 6 + the sequencing requirement). The enforcement mechanism is the author's proposal. It is sound and the alternative (no enforcement) is unsafe.
- **N0 normalise-to-sum-1 vs N4 preserve-gauge:** correctly handled as a deliberate difference — N0 produces probabilities, N4 refines them under a gauge constraint (a+b=c+d) that per-table normalisation would violate.

### Disagreement with the author's self-grade

The author graded B+ and said A- should come from a reviewer. The stated reasons for withholding A- are weaker than presented: three of the five "unverified" questions are answerable from source (Q23, Q24, Q21), leaving only Q15 and Q20 genuinely open, both with documented fallbacks. The adapter permutation is correctly derived and guarded by T4b. The registration key proposal is the only sound mechanism for the sequencing constraint. No contradiction was found in the revision-3 edits despite a systematic search across all flagged patterns. The document is ready for implementation.
