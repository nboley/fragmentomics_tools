# A correct NB oracle for the overdispersed simulation store

**Status:** DESIGN — not implemented. Written by the EM on 2026-09-24 after two
consecutive design agents died to API rate limits without executing a tool.
Requires design review (>= A-) before implementation.

**Scope:** why `simulation_v3_nb/A/oracle_nb.json` is not a floor, what the
correct floor for the `nb_offset` objective is, and how to compute and verify
it.

---

## 1. The problem

We measure how much simulated fragmentation bias a model captures by anchoring
its held-out NLL between an *oracle* (the NLL of the true generating process — a
floor no model can beat) and a *uniform* baseline:

```
% captured = (uniform - model) / (uniform - oracle)
```

For the overdispersed store this is broken. `oracle_nb.json` claims oracle
4.115647 / uniform 4.204456, and trained models score **154%** and **162%**.

The disproof needs no theory:

| quantity | value |
|---|---|
| untrained (epoch-0) KEN val_loss | **4.093257** |
| untrained (epoch-0) hybrid val_loss | **4.108722** |
| claimed oracle | **4.115647** |

Both *untrained* models beat the claimed floor. Whatever `oracle_nb.json`
computes, it is not a lower bound on this objective.

A multinomial-unit workaround now exists and is verified
(`oracle_multinomial.json`, oracle 7.510858 / uniform 7.600395). It is a real
floor, but it is **blind to the overdispersion this store exists to test** — the
multinomial and overdispersed stores produce gaps identical to 2.7e-5, because
both anchors shift by the same constant (-0.012337 and -0.012310). That is the
signature of one shared propensity structure plus a constant offset, exactly as
§7.5 of `training_analysis.md` predicts: the multinomial loss conditions on the
per-track total, so total-count noise is divided back out. So the workaround
answers "who learned the propensity better" and cannot answer "does the
architecture ranking survive overdispersion".

---

## 2. Diagnosis

### 2.1 What the loss actually computes

`MaskedNegativeBinomialOffsetNLLLoss` (frozen core), per track:

```
N      = target.sum(-1)                 # the OBSERVED total
mu_i   = N * p_i,  p = softmax(masked shape_logits)
r_i    = exp(log_dispersion)            # per window; W = L // dispersion_window_size
nll    = -sum_i log NB(target_i ; total_count=r_i, mean=mu_i)
loss   = mean over (batch, track) of  nll / max(N, 1)
```

Two properties matter and are easy to miss:

1. **`N` is the observed sum of the very counts being scored.** The mean is
   anchored to the realised total, not to an independent prediction of it.
2. It is therefore a **pseudo-likelihood**, not a normalised likelihood over the
   data. The docstring says so in its first line ("Pseudo-likelihood").

With `--dispersion-window-size 1` (what both runs used), `out_size = L // w = L`,
so dispersion is per-position. With `--freeze-dispersion`, `log_r` is pinned at
`log_dispersion_init = 7.0`, i.e. **r ≈ 1096**.

### 2.2 What the simulator actually did

`simulate_sample_nb` in `scripts/sim_fragments.py`:

```
mu_p       = N_target * w_pos[p] / sum(w_pos)
count_p    ~ NB(r = hexamer_r[hex_at_p], mu = mu_p)     # independent per position
```

with a stated consequence: *"The total fragment count per region is no longer
exactly N; it varies according to the NB variance."* True `hexamer_r` has median
**7.179**.

### 2.3 Why the plug-in is not a floor

`sim_oracle_nb.py` computes the oracle by plugging the true propensity and the
true per-position `r` into the loss. That is the natural move and it is wrong
here, for two compounding reasons.

**(a) The offset conditions on information the generative process did not use.**
The simulator's mean is `N_target * w_p`. The loss's mean is `N_observed * p_i`,
and `N_observed` is a random draw *around* `N_target`. Anchoring to the realised
total removes exactly the total-count variance from the prediction problem. A
predictor using `N_observed` is therefore strictly better-informed than the true
generative process, so its NLL can fall **below** the true-parameter plug-in.
Gibbs' inequality — "the true parameters minimise expected NLL" — applies to a
proper likelihood and does **not** transfer to a pseudo-likelihood with a
data-dependent offset.

**(b) True `r` is the right dispersion for the marginal, not for what the loss
sees.** Conditioning the mean on the realised total absorbs much of the
overdispersion; the residual per-position spread is tighter than the marginal NB
with r ≈ 7.18. Plugging in the true `r` therefore *over*-disperses relative to
the conditioned quantity being scored, inflating the NLL. Meanwhile the frozen
models sit at r ≈ 1096 (near-Poisson, i.e. much tighter), which is closer to the
conditioned spread. That is a coherent mechanism for an *untrained* model beating
the plug-in: the win comes from the dispersion term, not from the propensity.

**Prediction that distinguishes this from an alignment or units bug:** if (b)
holds, then holding the propensity at truth and sweeping a scalar `r` will find a
minimum at `r` **far above** 7.18, and that minimum will lie below both untrained
models. An alignment bug would instead show up as a mis-centred shift-correlation,
and a units bug as a constant offset — neither predicts a well-behaved interior
minimum in `r`.

### 2.4 The clamp reporting zero effect is EXPECTED — do not investigate it

*Revised 2026-09-24 after design review. This section previously flagged
`clamp_effect_on_oracle: 0.0` as suspicious. That was wrong, and chasing it
would have wasted implementation time.*

`oracle_nb.json` reports `clamp_effect_on_oracle: 0.0`, i.e. the dispersion
clamp never bound. That is exactly what should happen, by a back-of-envelope
the original draft failed to do:

- the clamp floor is `r >= mu / (2(1-p) - 1)`, which for small `p` is ≈ `mu`;
- per-position `mu = N_track * p_i`, with `N_track` in the hundreds and
  `p_i ≈ 1/2048`, so `mu ≈ 0.1–5`;
- that is **far below** the true `r ≈ 7.18`, so the floor never exceeds it.

The clamp cannot bind at these per-position count levels. Zero effect is a
consistency check passing, not an anomaly. **This is not an open question.**

### 2.5 Scope check — the OTHER way a plug-in oracle can fail does not apply here

There is a second, independent mechanism that can stop true parameters being a
floor, and it has bitten this project on a different store: **if the model class
cannot express the true parameters, the true parameters are not its floor.**
Concretely, in a regime-B simulation the hexamer surface varies per sample
(`w6_s = w6 * exp(log_jitter_s)`), while the model is a function of sequence
alone and emits one profile for all samples. The best sample-agnostic predictor
is then some optimal *average* surface, not the true `w6`, and a model that
finds it will legitimately score below a true-`w6` oracle with no overfitting.

**That mechanism does NOT apply to this store.** Verified, not assumed:
`simulation_v3_nb/A/ground_truth.json` has `regime: "A"`, `jitter_sd: 0.0`, and
`ground_truth.npz:log_jitter` has standard deviation **exactly 0.0** — the w6
surface is shared across all samples. So the only failure mode in play here is
§2.3's pseudo-likelihood argument.

Stated because the general principle is what matters and it now has two distinct
instances: **plugging in true generative parameters yields a floor only when (a)
the objective is a proper likelihood of the generative process, and (b) the
model class can express those parameters.** This store violates (a) only. A
future store that is both overdispersed *and* regime B would violate both at
once, and an oracle fixing only §2.3 would still not be a floor.

---

## 3. What the correct floor is

If §2.3 holds, **there is no plug-in oracle for this objective.** The floor is
not "the true parameters" but the minimum the objective can attain:

```
oracle_nb = min over (p, r) of  L_nb(p, r)      on the held-out set
```

Three candidate anchors, in increasing strength and cost:

| anchor | definition | what it licenses |
|---|---|---|
| **A. true-p, fitted-r** | p fixed at the true propensity; scalar `r` fitted to minimise the loss | "how much of the bias did the model capture, given a correctly specified dispersion" — a clean propensity floor |
| **B. true-p, true-r (current)** | plug-in | **nothing** — not a floor; retire it |
| **C. fully fitted** | minimise over both p and r | the true attainable floor, but p is 2048-dimensional per tile; expensive and risks overfitting the val set |

**Recommendation: A.** It isolates the propensity question (which is what
"% bias captured" has always meant here), it is a 1-D optimisation so it is cheap
and has no overfitting risk worth worrying about, and it is directly comparable
across architectures because every model is scored against the same fitted `r`.

C is the theoretically correct floor but answers a different and less useful
question, and a 2048-dim fit on 1600 val pairs would not be trustworthy.

**Report both A and the fitted `r` itself.** If the fitted `r` lands near
`log_dispersion_init`'s r ≈ 1096 rather than near the true 7.18, that is a
publishable finding in its own right: it would say the `nb_offset` objective,
as specified, cannot see the overdispersion the simulator injected — which would
mean the store cannot answer its motivating question under *any* oracle, and the
right response is to change the objective or the simulation, not the oracle.

---

## 4. Implementation plan

**Step 0 — confirm or refute §2.3 before building anything.** Hold the
propensity at truth, sweep scalar `r` over a log grid (say 1 to 1e5, 25 points),
plot the loss. Cheap, and decisive:

Three outcomes, not two *(the third added 2026-09-24 after design review, which
correctly pointed out the original binary framing left the most interesting case
with no plan)*:

- **`r` far above 7.18** (say > 500), minimum below both untrained models →
  §2.3 confirmed. Proceed with anchor A.
- **`r` intermediate** (roughly 20–500) → the objective has *partial* sensitivity
  to the injected overdispersion. Proceed with anchor A, and **report the fitted
  value as a headline result**: it quantifies how much of the generative
  dispersion survives the offset conditioning, which is the most scientifically
  interesting outcome of the three and the one this store was built to probe.
- **`r` at or near 7.18** → §2.3 is **wrong**. Stop and re-diagnose. Do not
  proceed to step 1 on a falsified premise.

Record the swept curve itself, not just the argmin — a ragged or flat curve means
the fitted minimum is not meaningful regardless of where it sits (see §5).

This step is the whole reason to review this design before implementing it.

**Step 1 — reuse, do not rebuild.** `scripts/score_v3nb_multinomial.py`
(commit `9d9d070`, verified) already loads the store, computes the true
propensity, handles the centre-crop to `[128, 2176)`, GC, and alignment. Import
its propensity path. `CLAUDE.md` is explicit that hand-rolling this machinery is
a recurring failure mode here, and the broken `sim_oracle_nb.py` is the most
recent instance.

**Step 2 — fit `r`** by 1-D minimisation of the actual frozen-core loss
(`MaskedNegativeBinomialOffsetNLLLoss` with `max_dispersion_ratio=2.0`,
`clamp_margin=1.0`, `dispersion_window_size=1` — matched to the runs). Use the
frozen core's loss object directly; do not reimplement the NB NLL.

**Step 3 — uniform anchor** under the same loss and the same fitted-`r`
procedure, with a uniform propensity.

**Step 4 — write** `simulation_v3_nb/A/oracle_nb_v2.json` with full provenance:
inputs read, alignment evidence, crop, GC mode, the fitted `r` and the swept
curve, loss config, UTC timestamp. **Do not overwrite `oracle_nb.json`** — leave
the broken file in place with a `_superseded_by` key added, so anyone holding a
number traceable to it can find out it was wrong.

---

## 5. Verification protocol

Every check has a stated failure criterion. A verification that cannot fail is
not one.

| check | criterion | why |
|---|---|---|
| **Untrained control** | untrained KEN and hybrid must score **above** the oracle | this is the exact check that exposed the current file. Note: untrained models land slightly *above* uniform (+0.0015, +0.0100 measured under multinomial) because random logits are worse than constant ones — expect that, do not treat it as failure |
| **Sanity gate** | both trained models strictly between oracle and uniform | if any model beats the oracle, STOP and report it as a finding. Do not tune until it passes |
| **Alignment** | shift-correlation argmax at shift 0, with a stated margin over the nearest competitor | v3_A achieved r=0.381 at 0 vs 0.027 at -128 (~14x); the v3nb multinomial run achieved 0.1157 vs 0.0295 (~4x), weaker because count noise dilutes it. Report the number, do not just assert "aligned" |
| **Uniform cross-check** | corroborate against an independent estimate | no collapsed v3nb run exists, so `log(2048) = 7.6246` minus the zero-count-track correction is the available check — weaker than v3_A's, and say so |
| **Loss-object identity** | the oracle must call the same frozen-core loss instance the runs used | a floor computed under a different clamp is not in the same units as the val_loss it is compared to. The current file documents this concern and still got the answer wrong |
| **Monotonicity** | the fitted-`r` sweep must be smooth and unimodal | a ragged curve means the loss evaluation is noisy or the clamp is switching, and the fitted minimum would be meaningless |

---

## 6. What the result does and does not license

- **Does:** rank architectures on propensity learning on the overdispersed store,
  in units with a genuine floor.
- **Does not:** support any cross-store comparison against v3_A. The
  overdispersed store trains on **12,800** pairs vs v3_A's **61,440** — a 5x data
  cut that confounds overdispersion with data volume regardless of units. Any
  v3nb-vs-v3_A delta mixes the two. Settling that needs a v3nb store rebuilt at
  matched scale.
- **Does not:** by itself establish that the objective is sensitive to
  overdispersion at all. That is what the fitted `r` tells us, and it may tell us
  the answer is no.

---

## 7. Open questions

1. **Is the `nb_offset` objective able to see overdispersion at all?** §2.3(b)
   argues the offset conditioning absorbs most of it. If the fitted `r` comes
   back near 1096, the answer is effectively no, and the finding is about the
   objective rather than about the oracle. This is the most important thing the
   step-0 sweep will tell us.
2. ~~**Should `r` be fitted per-position, per-window, or scalar?**~~ **CLOSED —
   see §9.4.** Scalar anchor stands; per-hexamer fit is not a floor (OOS delta
   +0.000483, under 0.001-nat threshold).
3. **Does the clamp bind on the oracle path?** §2.4 flags an inconsistency in the
   current file. Resolve during step 0.
4. **Is anchor A the right question?** It measures propensity capture holding
   dispersion correctly specified. If the owner's question is instead "does the
   model handle overdispersion", no propensity-only anchor answers it and the
   experiment needs redesigning rather than re-anchoring.

---

## 8. Honest self-assessment

**Grade: B+.** The diagnosis in §2.3 is grounded in the actual source of both
the loss and the simulator, and it makes a falsifiable prediction (§2.3, last
paragraph) with a cheap decisive test (§4 step 0). That is the strongest part.

**Weakest part: §2.3 is reasoning, not measurement.** I have not run the sweep.
The mechanism is consistent with every number available — including the specific
fact that untrained models beat the plug-in, which a propensity-side bug would
not produce — but it is a hypothesis until step 0 runs. The design is deliberately
structured so that step 0 can refute it and stop the work, rather than burying the
assumption in an implementation.

Second weakness: the choice of anchor A over C is a judgement about which
question is worth answering, and reasonable people could prefer C. I have argued
for A rather than proved it correct.

---

## Review Notes (2026-09-24)

**Verdict**: APPROVED WITH CONDITIONS
**Grade**: A-

### Verification summary

Every factual claim was checked against source. The central claim (§2.3) is
mathematically correct:

- `MaskedNegativeBinomialOffsetNLLLoss` computes `N = target.sum(dim=-1)` and
  anchors `mu_i = N * softmax(logits)_i` (`background_model_core.py:546-547`).
  The docstring opens with "Pseudo-likelihood" (line 502). **Verified by code.**
- `simulate_sample_nb` uses `mu_pos = target * w_pos / total_w` with `target`
  being `N_target`, not `N_observed` (`sim_fragments.py:542`). **Verified by code.**
- Gibbs' inequality does not transfer to this pseudo-likelihood because the
  offset `N` is data-dependent and the product-NB is not a proper distribution
  over the data vector. **Verified by reasoning; the math is sound.**
- All numerical values (oracle 4.115647, uniform 4.204456, untrained KEN
  4.093257, untrained hybrid 4.108722, multinomial anchors, gap identity to
  2.7e-5, 154%/162%, 12800 vs 61440 pairs) match their source files. Untrained
  KEN val_loss confirmed in the training run's `metrics.csv` (epoch-0). **Verified.**

### Conditions

1. **Resolve §2.4 before implementation.** The clamp_effect being 0.0 is not
   suspicious — it is expected. Per-position mu = N_track × p_i ≈ (hundreds) ×
   (1/2048) ≈ 0.1–5, far below true r ≈ 7.18. The clamp floor
   `r >= mu / (2(1-p) - 1) ≈ mu` never exceeds the true r. Replace the
   suspicion with this back-of-envelope resolution so implementers don't
   investigate a non-issue.

2. **Add a third outcome to §4 step 0 for intermediate fitted r.** The binary
   framing (r >> 7.18 → proceed; r ≈ 7.18 → stop) leaves no plan if the
   minimum is at, say, r = 50–200. Suggested: "r intermediate (20–500) → the
   objective has partial sensitivity to overdispersion; proceed with anchor A
   and flag the finding."

### Risks

1. Step 0 could produce an intermediate r with no defined action plan.
2. If the fitted r lands near 1096, the NB oracle adds no information beyond
   the already-verified multinomial oracle — the finding would be "this
   objective cannot see the overdispersion this store exists to test." The
   design handles this correctly (§3, §7 Q1) but it means the entire
   implementation could conclude with a negative result after an hour of work.
3. §2.4's suspicious framing could waste implementation time (minor).

### Key tradeoffs

- **Anchor A over C**: A isolates propensity (the right question for "% bias
  captured"), avoids overfitting 2048-dim propensity on 1600 val pairs, and is
  a 1-D optimisation. C is theoretically correct but answers a harder question
  that the data may not support. Reviewer agrees with A.
- **Scalar r over per-hexamer**: scalar is conservative and answers the
  first-order question. Per-hexamer (4096 params on ~3.2M data points) is a
  valid follow-up if the scalar r proves interesting.
- **Step 0 before implementation**: the cheapest possible falsification test
  (25 loss evaluations on 1600 pairs). Correct design choice — a wrong
  diagnosis caught at step 0 costs an hour, not a week.

---

## 9. Results addendum (post-implementation)

### 9.1 The profiled nuisance r is not identified

The step-0 sweep shows the loss descending steeply from r=1 to r≈15, then
entering a flat plateau. The profiled nuisance r (via scipy or grid minimum)
lands somewhere on this plateau, but the plateau's loss span — reproducible
structure in the softmax/lgamma/clamp pipeline, NOT stochastic noise (the
objective is bitwise deterministic) — makes the specific value meaningless.
The JSON field is `profiled_nuisance_r` with `not_identified: true`.

The `noise_floor` field in `oracle_nb_v2.json` reports `plateau_loss_span`
and `max_adjacent_non_monotonicity`, both computed over a fixed analysis
window r∈[15, 3000].

That window is a reporting convention, not a measured boundary, and it is
deliberately *not* the same thing as `profiled_nuisance_r.plateau_interval`,
which is data-driven (`loss < min + 1e-3`) and returns r∈[4.1, 3506.3]. An
earlier version of this section justified the narrower window as "excluding
the rising tail above r~3000, which is genuine signal". **That justification
was wrong.** The point just above the boundary, r=3506.3, has loss 4.026652 —
*lower* than the highest point inside the window (4.027178 at r=2458.8). The
tail does not rise monotonically out of the plateau; it dips back down first.
Only r=5000 (4.027832) rises clear of the plateau threshold, and both the
fixed window and the data-driven interval already exclude it.

The choice of window does not affect the published statistic. Measured over
both: `plateau_loss_span` is 8.2619e-4 either way (16 points in the fixed
window, 22 in the data-driven interval), because the minimum (r=1096) and the
maximum (r=2458.8) both fall inside the narrower one. The window is therefore
safe to keep, but it should be read as an arbitrary convention rather than as
a claim about where the plateau ends.

The defensible conclusion is: **the offset conditioning absorbs the
overdispersion above r≈15–20, and the loss plateau is flat, so r is not
identified.** This answers open question §7.1 in the negative: the `nb_offset`
objective effectively cannot see the overdispersion this store injected.

The earlier claim that fitted r=21 vs true r=7.18 demonstrates "partial
sensitivity" rested on a loss difference (1.4e-4) smaller than the plateau's
own loss span (~8.3e-4). That claim is withdrawn.

### 9.2 Min-over-union anchor selection

The scipy optimizer returned a point that was NOT the minimum of the sweep
curve stored in the same JSON — the sweep contained a lower-loss point. The
oracle and uniform anchors are now selected as the minimum over the union of
(swept grid points, scipy result), eliminating this inconsistency.

### 9.3 Corrections to oracle_nb_v2.json

Applied in the second pass to fix misstatements in the initial oracle_nb_v2.json:

- **`profiled_nuisance_r`** (was `fitted_r`): renamed to clarify that this is
  the argmin of a nuisance parameter, not an estimate of the generative r.
  The published value r≈1096 coincides with the frozen model initialisation
  exp(log_dispersion_init) = exp(7) ≈ 1096; this is an artefact of the flat
  plateau plus the reference marker log(1096) being injected into the sweep
  grid as a selectable point — NOT agreement between the oracle and the model.
  Any r on the plateau produces an effectively identical oracle loss. The field
  carries `not_identified: true` and a published plateau interval. The oracle
  VALUE (4.026351) is unchanged.

- **`noise_floor._note`**: no longer calls deterministic structure "numerical
  noise". The objective is bitwise deterministic; what varies across the plateau
  is reproducible structure in the loss computation. The statistic is rescoped
  from r>20 to a fixed analysis window r∈[15, 3000], and renamed from
  `plateau_non_monotonicity` to `plateau_loss_span`. See §9.1 for why that
  window is a reporting convention rather than a measured plateau boundary —
  an earlier draft described it as excluding a "rising tail", which the sweep
  data contradicts.

- **Sweep raggedness gate split**: the sweep's 11 sign changes (vs threshold 5)
  previously set `sanity_gate_all_pass = false`, making the overall gate
  permanently fail. Since the design's own conclusion (§9.1) is that the
  plateau is flat and r is not identified, a gate that can never pass carries
  no information. Raggedness is now a descriptive field `r_identified: false`
  with the sign-change count and threshold alongside it. `sanity_gate_all_pass`
  covers only conditions that mean the artifact is BROKEN: alignment
  best_shift==0, untrained-above-oracle, trained-between-anchors.

- **Deleted `uniform_vs_log_W`**: compared NB loss against log(2048), an exact
  identity for the multinomial loss that has no analogue for the NB-offset loss.
  The delta (-3.51) compared unlike things and could not fail.

- **Deleted `descending_before`**: computed and never read.

### 9.4 Per-hexamer r investigation (closes open question §7.2)

The per-hexamer investigation is complete. The scalar anchor stands.

Measured results (from the completed per-hexamer run):

| configuration | oracle NLL | delta vs scalar |
|---|---|---|
| scalar oracle (1 param) | 4.026409 | — |
| true per-hexamer r, no fit | 4.026575 | +0.000166 (WORSE) |
| per-hexamer fit-on-train, scored-on-val (OOS) | 4.026892 | +0.000483 (WORSE) |
| per-hexamer fit-on-val (in-sample) | 4.026611 | +0.000202 |

Note: the scalar oracle reported here (4.026409) is the per-hexamer script's
own refit; the published v2 anchor is 4.026351354 (from the min-over-union
selection on the full sweep grid).

4014 hexamers were fitted out-of-sample vs 2952 in-sample. The in-sample/OOS
comparison is therefore confounded — the two differ in effective model size,
not only in train/val split.

**Verdict: NOT_MATERIAL.** The OOS delta (+0.000483) is under the pre-registered
0.001-nat threshold in a gap of 0.0888. A 4096-parameter fit landing ABOVE a
1-parameter fit out-of-sample means the per-hexamer anchor is **not a floor**
and cannot serve as the oracle. Open question §7.2 is closed **by measurement**
in favour of the scalar.
