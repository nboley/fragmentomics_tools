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

### 2.4 Secondary corroboration

`oracle_nb.json` reports `clamp_effect_on_oracle: 0.0` — the dispersion clamp
never bound. With true r ≈ 7.18 the clamp floor `r >= mu / (2(1-p) - 1)` should
bind at every position where `mu` exceeds ~7. That it reported exactly zero
effect suggests either the clamp was not applied on the oracle path or `mu` is
far smaller than assumed. Worth resolving during implementation; it is a
consistency check on the whole computation, not a separate bug report.

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

- Interior minimum far above 7.18, below both untrained models → §2.3 confirmed,
  proceed with anchor A.
- Minimum at or near 7.18 → §2.3 is **wrong**; stop and re-diagnose. Do not
  proceed to step 1 on a falsified premise.

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
2. **Should `r` be fitted per-position, per-window, or scalar?** Scalar is
   proposed for tractability. A per-hexamer fit would be closer to the generative
   structure but reintroduces overfitting risk on 1600 val pairs.
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
