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
| `scripts/nb_oracle_v2.py` | 837 | 173–837 | ~18 min | `oracle_nb_v2.json` |
| `scripts/nb_oracle_perhex.py` | 859 | 277–849 | ~3.1 h | `oracle_nb_perhex.json` |
| `scripts/nb_oracle_sweep.py` | 236 | 46–236 | minutes | stdout only |

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
against v2's published 4.026351354. Every delta and percentage in that file
was computed against the wrong denominator, reading KEN at 61.68% where the
canonical artifact said 61.31%. Same model, same checkpoint, two numbers.
Partitioning work by file prevented a *write* collision but not this *method*
divergence.

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
internal cross-check (`assert abs(scalar - REF) < 0.001`) is three orders of
magnitude too loose to notice: it passed happily through the 5.7e-5
disagreement described in §1.2.

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

- §1.1: a note fix re-runs `render` only. Seconds, not 18 minutes. The
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

## 6. Open decisions

**6.1 Is the raw artifact committed, cached, or throwaway?**
Recommend: **cached on `/efs`, never committed.** It is large and derived;
CLAUDE.md already bars committing stores and pileup arrays. Render must fail
loudly if it is absent or if its schema version does not match — never fall
back to recomputing silently, because a silent 18-minute recompute inside what
should be a 2-second render is exactly the surprise that gets worked around.

**6.2 Do the existing artifacts get regenerated?**
`oracle_nb_v2.json` costs 18 minutes: regenerate, no question.
`oracle_nb_perhex.json` cost a **3.1-hour** fit and has been hand-edited twice.
Regenerating it is the only way to prove the refactor is clean for that file,
but it is expensive and its verdict (NOT_MATERIAL, a negative result) is not
load-bearing. Options: (a) regenerate once and accept the cost; (b) split it
so the cheap `render` half is verified and the expensive `compute` half is
grandfathered with its provenance recorded. **Recommend (b)**, and say so in
the artifact.

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
| 2 | Split `nb_oracle_v2.py` into compute/render with a raw artifact. | medium — the 664-line `main()` is the hard part |
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

What holds it below an A:

- **I have not read all 664 lines of `nb_oracle_v2.main()`.** I know its phase
  structure from its output and from targeted greps. The compute/render seam
  in Phase 2 is where the real difficulty lives, and I am proposing the split
  without having traced every variable across that boundary. A reviewer should
  push hard there; my estimate that it is "medium risk" is not well founded.
- **No measurement of the claimed win.** I assert render is seconds. That is
  inference from what it does, not timed.
- **6.2 is a judgement call I have made on the owner's behalf** — grandfathering
  a 3.1-hour artifact is a real trade and I recommend it partly because the run
  is expensive, which is not a correctness argument.
- I wrote this after two failed delegations, so it has had no independent
  research pass. That is exactly the condition under which the last
  un-reviewed NB oracle shipped wrong, which is why §9 of `nb_oracle.md`
  exists.

**This design must not be implemented without a design review at ≥ A-.**
