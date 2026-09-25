# Splitting computation from reporting in the oracle scripts

Status: DRAFT, awaiting design review. Written by the EM directly after two
design-agent delegations (`8da6e730-653`, `dc302d9d-764`) died to API rate
limits inside 50s with zero output — the fifth and sixth occurrences of that
pattern on this project. The pre-registered stop rule (retry once, then write
it myself) fired.

## 1. The problem, stated as what actually happened

Three scripts compute anchor values for the v3_A simulation study and publish
them as JSON on `/efs`:

| script | lines | `main()` | runtime | artifact |
|---|---|---|---|---|
| `scripts/nb_oracle_v2.py` | 837 | 173–833 (661 lines) | ~18 min | `oracle_nb_v2.json` |
| `scripts/nb_oracle_perhex.py` | 859 | 277–855 (579 lines) | ~3.1 h | `oracle_nb_perhex.json` |
| `scripts/nb_oracle_sweep.py` | 236 | 46–236 | minutes | stdout only |

*(Corrected after review. My original `main()` ranges ran past the
`if __name__` guard at v2:836 / perhex:858. The reviewer's file totals were
each one higher than `wc -l` reports; its `main()` boundaries were right and
mine were not. Both re-verified here.)*

Those artifacts are the *denominators* for every model-quality percentage
quoted in `docs/pending/training_analysis.md` and now on two Confluence pages.

### 1.1 A 664-line `main()` that computes and renders in one pass

`nb_oracle_v2.py` does everything inside one function: caches val pairs, runs
the sweep, measures determinism, selects two anchors, verifies alignment,
scores four models, evaluates five gates, prints a table, and assembles the
published JSON *including all of its explanatory prose*.

Because the prose is built in the same pass as the numbers, **fixing a
sentence costs 18 minutes.** That is not a theoretical objection. It has
produced, measurably:

- Three separate in-place hand-edits of published artifacts to dodge the cost
  (the `%`-bias anchor correction, the `_not_an_anchor` text alignment, and a
  near-miss today). Each left the committed script able to emit something
  different from what the artifact actually said.
- One of those drifts was caught by implementation review; another was caught
  only by reading the two files side by side. Neither was caught by a test.
- Today a `_note` inside the published JSON was found asserting that the
  analysis window "excludes the rising tail above r~3000 because it is genuine
  signal" — a claim the script's own sweep data refutes (r=3506.3 scores
  4.026652, *lower* than 4.027178 at r=2458.8, which is inside the window).
  Correcting that one sentence cost a full 18-minute re-run.

The deeper point: the review that examined this code verified that the
statistics matched their names, and did not ask whether the *prose* was true.
Prose that ships inside a data artifact is not covered by any test we have.

### 1.2 Scoring logic exists in three copies

| script | construct | closure over |
|---|---|---|
| `nb_oracle_v2.py:260` | `eval_loss_at_log_r` | `oracle_data` |
| `nb_oracle_perhex.py:391` | `eval_scalar_loss` | `val_oracle_data` |
| `nb_oracle_sweep.py:148` | inline loop | `pair_logits` |

All three build `torch.full((1, C, L), log_r)` and call the same frozen-core
`MaskedNegativeBinomialOffsetNLLLoss` with the same configuration.

These have already diverged in a way that mattered. `nb_oracle_perhex.py`
re-fit its own scalar anchor using scipy alone — the method that
`nb_oracle_v2.py` had superseded with min-over-union — landing at 4.026408539
against v2's published 4.026351354, a **5.72e-5** disagreement. Every delta and
percentage in that file was computed against the wrong denominator.
Partitioning work by file prevented a *write* collision but not this *method*
divergence.

**Two corrections to an earlier draft of this section, both from review:**

- The method divergence alone is worth **0.04pp** (the scalar-anchor block read
  61.35% before the fix, 61.31% after). The draft attributed the larger
  61.68% → 61.31% swing to it, but 61.68% comes from the *per-hexamer anchor*
  block, which uses a different denominator **by design**. Both numbers were
  genuinely in the file and "same model, same checkpoint, different numbers"
  remains true — but the cause is mostly anchor choice, not method divergence,
  and conflating them overstates this pain.
- The draft called the script's guard `assert abs(scalar - REF) < 0.001`
  "three orders of magnitude too loose". **Measured: 1e-3 / 5.72e-5 = 17.5x,
  which is 1.24 orders of magnitude.** The guard is still far too loose — it
  passed straight through the real divergence, which is the point — but the
  quantitative claim was wrong and is corrected here.

### 1.3 A third form of the same disease, found while researching this design

`nb_oracle_perhex.py:61-67` hardcodes **six floats transcribed by hand** from
v2's published JSON, plus a seventh derived from two of them:

```python
REF_SCALAR_ORACLE = 4.026351353675127
REF_UNIFORM       = 4.115219691395760
REF_TRAINED_KEN   = 4.060737457573413
REF_TRAINED_HYBRID= 4.067427050471306
REF_UNTRAINED_KEN = 4.117498033046722
REF_UNTRAINED_HYBRID = 4.125973814576864
REF_GAP = REF_UNIFORM - REF_SCALAR_ORACLE
```

This is a manual copy of another artifact's contents into source. It is
currently correct — I checked all six against the live JSON by parsing them
out of the source and comparing with `==`, and all six match bitwise, with
`REF_GAP` deriving exactly to the published `gap_uniform_minus_oracle` — but
nothing enforces that. If v2's numbers move, perhex keeps using stale ones and its
internal cross-check (`assert abs(scalar - REF) < 0.001`) is **~17x too loose**
(1e-3 against a 5.72e-5 divergence, i.e. 1.24 orders of magnitude — an earlier
draft said three) to notice: it passed happily through the disagreement
described in §1.2.

**The three pains are one pain.** Numbers computed in one place are re-entered
by hand somewhere else, and nothing binds the copy to the original.

## 2. Goal

1. Correcting an explanatory note is cheap and *cannot* desync from the
   artifact it describes.
2. Scoring logic has exactly one home.
3. Derived artifacts read their inputs rather than embedding transcribed
   copies.
4. Every currently published number is bitwise unchanged.

## 3. Proposed structure

Split each script into three layers.

```
  compute  ──►  raw artifact  ──►  render  ──►  published artifact
 (minutes)      (on /efs)        (seconds)      (on /efs, cited)
```

**Compute** does the expensive work and emits *only numbers* — no prose, no
formatting, no interpretation. For `nb_oracle_v2` that is: the sweep curve,
the determinism probe results, the two selected anchors with their sources,
the alignment measurements, and the four model scores.

**Raw artifact** (`oracle_nb_v2.raw.json`) persists exactly that. It is the
unit of expensive work.

**Render** reads the raw artifact and produces the published JSON: the derived
statistics (spans, percentages, gate verdicts) and all explanatory prose. It
must be pure — same raw in, same published out — and take seconds.

### 3.1 Why this addresses each pain

- §1.1: a note fix re-runs `render` only. **Measured after Phase 2: 19.9s against compute's 701.2s** — seconds, not minutes, though not the "2 seconds" an earlier draft of this section claimed. The
  incentive to hand-edit an artifact disappears, and with it the drift class.
- §1.2: `compute` is where the shared scoring helper lands (§4), so the three
  copies collapse to one.
- §1.3: perhex's `render` reads `oracle_nb_v2.json` (or its raw) for the
  anchor instead of embedding seven transcribed floats.

### 3.2 What I am *not* proposing

Not a framework, not a plugin system, not a general-purpose report engine.
Two functions and a file per script. The scripts are research code whose
output is cited in documents; the goal is that a correction is cheap and a
copy cannot go stale, nothing more.

### 3.3 The raw artifact schema (added after review)

The design review traced all 661 lines of `nb_oracle_v2.main()` and established
the thing I could not: **every variable crossing the compute→render boundary is
a JSON-serialisable scalar, string, or small dict.** No torch tensors, models,
datasets, closures or FASTA handles are required in the render half; `pct_bias`
needs only `uniform_loss` and `gap`, both scalars. Phase 2 is therefore **low
risk, not medium** — my original estimate was pessimistic.

`oracle_nb_v2.raw.json`:

```
_schema_version   int          bump on ANY field add/remove/rename
_computed_utc     str
_runtime_s        float
store/sim_dir/fasta/design_doc          str    provenance
n_val_pairs, tile_size, l_target, crop, gc_mode
sweep_curve       [{log_r, r, loss} x 27]
determinism       {r, eval1, eval2, bitwise_equal} x 4
oracle            {loss, log_r, r, source}
uniform           {loss, log_r, r, source}
alignment         {best_shift, best_r, r_at_shift_0, r_at_shift_minus128,
                   ratio_0_vs_minus128, n_tiles}
models            {trained_ken, trained_hybrid, untrained_ken,
                   untrained_hybrid} -> {nb_loss, checkpoint?,
                   nb_val_loss_from_training?, seed?}
loss_config       {class_name, max_dispersion_ratio, clamp_margin,
                   dispersion_window_size, matches_training_config}
```

Everything else in the published JSON is **derived** and belongs to render:
`gap_uniform_minus_oracle`, `pct_bias_captured`, `noise_floor` (span,
max-adjacent-delta, plateau range/interval), `profiled_nuisance_r`
(`not_identified`, plateau interval, the reference markers), the `verification`
block, `supersedes`, and every `_note`.

**Gate placement — decided (review finding #6):** verification gates go in
**render**. They depend only on scalars already in the raw artifact, and the
whole point of the split is that a wrong gate — or wrong gate *prose* — is
correctable without recompute. This project has already shipped one gate that
could never pass and one note that was false; both would have been cheap to fix
under this placement. Render is therefore "derive + format + judge", not pure
formatting, and that is deliberate.

**Staleness — decided (review finding #5):** render reads `_schema_version` and
compares against a constant in the render module. On mismatch or missing file it
**raises and exits non-zero**, naming the expected and found versions and the
command to regenerate. It must **never** silently recompute: a surprise
18-minute run inside what should be a 2-second render is exactly the behaviour
that gets worked around by hand-editing, which is the disease being cured.

## 4. The frozen-core question — DECIDED 2026-09-25

> **OWNER RULING: `scripts/_oracle_scoring.py`.** The recommendation below was
> accepted. The helper does NOT go in the frozen core, and this refactor
> therefore requires no frozen-core approval. The 2026-09-24 approval that
> said "helper in the FROZEN core" is superseded for this widened scope.
> A reviewer should still challenge the *reasoning* — but the decision is
> made, not open.

The shared scoring helper is a candidate for `background_model_core.py`.

**The case for:** it consumes `MaskedNegativeBinomialOffsetNLLLoss` and
`_prepare_mask`, both of which already live there; CLAUDE.md directs callers
to "consume its primitives rather than reimplementing them," and three
reimplementations is precisely the failure that instruction exists to prevent.

**The case against:** CLAUDE.md declares the module a frozen statistical
specification whose docstring is authoritative. Adding a convenience wrapper
widens its surface. A helper that merely arranges tensors and calls an
existing loss is *plumbing*, and plumbing is explicitly listed as fine without
sign-off — but it would live inside the file that is not.

**Recommendation:** put it in a new `scripts/_oracle_scoring.py`, not the
frozen core. It is consumed only by scripts, it is not part of the statistical
specification, and keeping it out means this refactor needs no frozen-core
approval at all. If it later earns a place in the core, moving it is a
separate, smaller decision.

**This is flagged rather than assumed.** The 2026-09-24 approval covered the
*narrower* extraction as "helper in the FROZEN core". That approval predates
this widened scope and should be re-confirmed or redirected.

## 5. Acceptance test

The refactor is a no-op on every published number. The test is not
"spot-check the headline value" — that method has missed regressions here that
leaf-diffing caught.

**Method.** For each artifact: keep the pre-refactor file, regenerate, flatten
both JSONs to `path -> scalar` leaf maps, and diff. Required outcome:

- **Zero** differing leaves except `created_utc` and `runtime_s`.
- `oracle_nb_nll` bitwise `4.026351353675127` — compared with `==`, not
  `round()` or `isclose`.
- `uniform_nb_nll` and `gap_uniform_minus_oracle` bitwise equal.
- `pct_bias_captured` unchanged at KEN 61.31 / Hybrid 53.78 in **both**
  artifacts.

Any movement is **a finding to report, not something to adjust until it
matches.** A pure refactor that moves a number means the two code paths were
never equivalent, which is information about the old code, not a problem with
the new.

This exact method has now caught or confirmed every change in this work
stream: it proved the `not_identified` dedup was pure (only timestamps moved),
proved today's note fixes were pure, and proved the perhex text alignment
touched zero numeric leaves.

**Second sub-test, added after review (finding #9).** The leaf-diff proves
numeric fidelity but cannot prove the raw artifact is *complete* — render might
reproduce the published JSON only because it quietly re-reads something else
(the run `summary.json` files, the store, the FASTA). So:

> **Render-from-raw-alone.** Run render with nothing available but the raw
> artifact, and diff against the published JSON. If render needs any other
> input, the raw schema is incomplete and that is a finding, not something to
> paper over by letting render open the extra file.

Without this, §3.3's schema is an assertion rather than a tested property.

## 6. Open decisions

**6.1 Is the raw artifact committed, cached, or throwaway?**
Recommend: **cached on `/efs`, never committed.** It is large and derived;
CLAUDE.md already bars committing stores and pileup arrays. Render must fail
loudly if it is absent or if its schema version does not match — never fall
back to recomputing silently, because a silent 18-minute recompute inside what
should be a 2-second render is exactly the surprise that gets worked around.
The mechanism is now specified in §3.3 rather than left as a principle.

**6.2 Do the existing artifacts get regenerated?**
`oracle_nb_v2.json` costs 18 minutes: regenerate, no question.
`oracle_nb_perhex.json` cost a **3.1-hour** fit (11101.9s) and has been
hand-edited twice.

**First, what that 3.1 hours actually buys.** Measured from the artifact:

| block | cost | status |
|---|---|---|
| `with_scalar_anchor` → KEN 61.31 / Hybrid 53.78 | milliseconds (arithmetic on the `REF_*` constants) | **`_canonical`** — the only quotable numbers |
| `perhex_oos`, `perhex_insample`, `fitted_r_summary` | **the 3.1 hours** | evidence behind the verdict |
| `with_oos_perhex_anchor` → 61.68 / 54.11 | derived from that fit | stamped `_not_an_anchor: DO NOT QUOTE` |

So every number anyone cites from this file is cheap; the expensive half
produces only do-not-quote figures and the evidence for `NOT_MATERIAL`. That
verdict is still load-bearing in one specific way — it is what closed open
question §7.2 and justified keeping the scalar anchor — so it is not free to
leave unverified either.

**An earlier draft of this section claimed regenerating "is the only way to
prove the refactor is clean for that file". That is false**, and the option it
missed is the one now chosen.

### DECIDED 2026-09-25 (owner): function-level bitwise verification

The load-bearing computation lives in three **pure** module-level functions,
verified by AST inspection:

| function | args | module globals used |
|---|---|---|
| `score_val_perhex` | 5 explicit | **none** |
| `nb_nll_positions` | 3 explicit | **none** |
| `fit_perhex_grid` | 4 explicit | 4 grid constants (`LOG_R_LO/HI`, `NHEX`, `N_GRID`) |

So the expensive computation is verifiable **without running the expensive
pipeline**. The acceptance test for Phase 3 is:

1. Call all three on a **fixed small input** (a few hundred pairs, or
   synthetic, pinned in the test) before and after the refactor and compare
   outputs **bitwise**.
2. Verify the render half in full with the §5 leaf-diff — this covers the
   entire canonical/quotable block.
3. `git diff` the extracted compute code to show it moved without changing.

Minutes, not hours, and it removes the need for the cost argument the review
rightly attacked: the objection becomes moot rather than excusing a gap.

**Residual risk, stated plainly:** the Phase C/D **orchestration** inside
`main()` — how data flows between phases — is still not verified end to end. A
bug there could survive this test. That is a narrower and more honest thing to
accept knowingly than "the compute half is unverified". Stamp the artifact with
a field recording exactly this: function-level verified, orchestration not.

**6.3 Migration for existing readers.**
Both artifacts are cited in `training_analysis.md` and on Confluence pages
4964155436 and 4963532846. The published JSON *schema* must not change in this
refactor — same keys, same nesting. Schema changes, if wanted, are a separate
change with its own review, so that a schema break is never confounded with a
refactor that was supposed to move nothing.

## 7. Phasing

| phase | scope | risk |
|---|---|---|
| 1 | Extract `scripts/_oracle_scoring.py`; point all three scripts at it. No structural change. | low — bitwise test covers it |
| 2 | Split `nb_oracle_v2.py` into compute/render with a raw artifact. | **low** (was "medium") — review traced the 661-line `main()` and the boundary is clean; schema now fixed in §3.3 |
| 3 | Same split for `nb_oracle_perhex.py`; delete the seven `REF_*` constants in favour of reading v2's artifact. | medium — see 6.2 |
| 4 | `nb_oracle_sweep.py` (stdout only, no artifact). | low |

Each phase: implement → review → fix → bitwise acceptance test → commit.
Phase 1 is independently valuable and could ship alone.

## 8. Self-assessment

**Grade: B.**

Honest about what it is. The problem statement is strong because it is built
entirely from things that actually happened, with the evidence attached, and
§1.3 is a genuine finding surfaced while researching rather than a restatement
of the brief. The acceptance test is concrete and has a track record.

What held it below an A — **and what review resolved (2026-09-25, grade A-):**

- ~~**I have not read all 664 lines of `nb_oracle_v2.main()`.**~~ **RESOLVED.**
  The reviewer traced all 661 lines and found the boundary clean: every
  crossing variable is a JSON-serialisable scalar or small dict. My "medium
  risk" was pessimistic; Phase 2 is low risk and the schema is now pinned in
  §3.3. This was the single largest uncertainty and it resolved in the
  favourable direction.
- **Two of my factual claims were WRONG and are corrected in §1.2:** the guard
  is ~17x too loose, not three orders of magnitude; and the 61.68% figure comes
  from the per-hexamer anchor, so attributing it to method divergence
  overstated that pain (the real figure is 0.04pp). Both were caught by review,
  not by me — which is the same failure mode this project keeps hitting, now
  committed by the author of the document warning about it.
- ~~**No measurement of the claimed win.**~~ **RESOLVED.** Measured after
  Phase 2: render is **19.9s** against compute's 701.2s. The premise holds,
  but my "2-second" figure was optimistic by 10x and is corrected in §3.1.
- **6.2 is a judgement call I have made on the owner's behalf** — grandfathering
  a 3.1-hour artifact is a real trade and I recommend it partly because the run
  is expensive, which is not a correctness argument.
- I wrote this after two failed delegations, so it has had no independent
  research pass. That is exactly the condition under which the last
  un-reviewed NB oracle shipped wrong, which is why §9 of `nb_oracle.md`
  exists.

**This design must not be implemented without a design review at ≥ A-.**

## Review Notes (2026-09-25)

**Verdict**: APPROVED WITH CONDITIONS
**Grade**: A-

### Method

Full source verification against `nb_oracle_v2.py` (838 lines),
`nb_oracle_perhex.py` (860 lines), `nb_oracle_sweep.py` (237 lines), both
published JSON artifacts on `/efs`, and the live test suite (447 passed, 102
warnings). All seven REF_* constants verified bitwise against
`oracle_nb_v2.json`. The 661-line `main()` in `nb_oracle_v2.py` was traced
end-to-end to map the compute/render boundary.

### Key finding: the boundary is clean

Every variable that crosses the compute→render boundary is a JSON-serializable
scalar, string, or small dict (~20 fields + the 27-entry sweep curve + nested
metadata). No torch tensors, model objects, datasets, closures, or file
handles are needed in the render half. `pct_bias` uses only `uniform_loss` and
`gap` (both scalars). The design's self-assessed "medium risk" for Phase 2 is
conservative — the boundary is straightforward.

### Conditions for approval

1. **Correct "three orders of magnitude" (§1.2).** Actual ratio: threshold
   (1e-3) / divergence (5.72e-5) = 17.5x ≈ 1.24 OoM, not three. The guard IS
   too loose, but the quantitative claim is wrong.
2. **Add a raw artifact schema** — even a sketch listing the fields the raw
   artifact will contain. This reduces Phase 2 risk and prevents the
   implementer from making ad hoc boundary decisions.
3. **Define the staleness detection mechanism** referenced in §6.1. What field
   carries the schema version? What value? What does render do on mismatch?
4. **State explicitly in §6.2** that grandfathering means the perhex
   compute→raw split is NOT verified end-to-end. The economic argument is
   valid; the lost guarantee must be visible.

### Risks

1. Without the schema specification, the implementer must re-trace the
   boundary. (Mitigated: this review documents that the boundary is clean and
   identifies the ~20 fields.)
2. Grandfathered perhex compute→raw split could contain a bug the skipped
   acceptance test would have caught.
3. Schema staleness detection is undefined; a render failure on a stale raw
   artifact could produce a confusing error rather than a helpful one.

### Accuracy corrections

- §1.2 "reading KEN at 61.68%" conflates per-hexamer-anchor divergence with
  scalar method divergence. The scalar-only divergence is 0.04pp (61.35% →
  61.31%); the 61.68% figure comes from the per-hexamer anchor section, which
  uses a different denominator by design.
- Table in §1: v2.py main() is 173–833 (661 lines, not 173–837 / 664 lines);
  perhex.py main() ends at 855 (not 849); sweep.py has 237 lines (not 236).
  Minor; doc acknowledges line numbers rot.

### Key tradeoffs

- **Scoring helper in `scripts/_oracle_scoring.py` vs frozen core**: the
  design's recommendation is correct. The helper is plumbing, not statistical
  specification, and keeping it out avoids frozen-core approval overhead for a
  refactor that should need none.
- **Grandfathering perhex**: defensible for a NOT_MATERIAL negative result, but
  the implementer should understand it is a cost/correctness tradeoff, not
  full verification.
- **Verification gates**: the design should decide whether gates go in compute
  (raw artifact records outcomes) or render (gate logic correctable without
  recompute). Either works; the choice should be deliberate.
