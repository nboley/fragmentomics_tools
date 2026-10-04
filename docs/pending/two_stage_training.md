# Two-Stage Cut-Site Fragment Model Training

## Motivation

The current cut-site model predicts fragment probabilities from scratch via a
neural network outputting logits over the full (position × length × strand)
domain. Training curves show the model converges almost immediately and barely
improves — suggesting it struggles to discover the positional structure from
gradient signal alone.

The core insight: most of the fragment probability structure is already
observable in the empirical cut-site frequencies and can be computed
analytically. The neural network should learn *corrections* to this empirical
baseline, not the entire distribution from scratch.

## Stage 1: Analytical Baseline (no gradient descent)

### Step 1a — Empirical cut-site probabilities

Compute per-position cut probabilities from observed fragment endpoints,
pooled across all training regions:

    p_start(hexamer) = cuts_starting_at_hexamer / occurrences_of_hexamer
    p_stop(hexamer)  = cuts_ending_at_hexamer   / occurrences_of_hexamer

This is the MLE of a linear Conv1d(4, 1, kernel_size=6) on one-hot sequence —
solvable analytically by counting. Four tables total: start_fwd, stop_fwd,
start_rev, stop_rev.

**Two model variants:**
- **Conv baseline**: a linear Conv1d whose weights ARE the hex table values
  (closed-form solution, no training needed)
- **K-mer embedding (KEN-style)**: embedding table indexed by hexamer,
  initialized to the empirical log-probabilities

### Step 1b — Marginal fragment length distribution

Compute the empirical length distribution pooled over all training fragments:

    p(L) = count(fragments with length L) / total_fragments

for L in [L_min, L_max].

### Step 1c — Factored baseline probability

For each fragment (centre m, length L, strand s):

    log p₀(m, L, s) = log p_start(start_pos | seq, s)
                     + log p_stop(stop_pos | seq, s)
                     + log p(L)
                     - log Z

where Z normalizes over all valid (m, L, s) in the tile. This factored model
assumes start and stop positions are conditionally independent given the
sequence context — the same factorization the old 12-track model uses.

## Stage 2: Neural Correction (trained)

The model output is an additive log-correction to the baseline:

    log p(m, L, s) = log p₀(m, L, s) + f_θ(seq, m, L, s)

followed by log_softmax over the valid domain to re-normalize. The correction
f_θ starts at zero (uniform correction = no change to baseline), so:

- At initialization, the model reproduces the analytical baseline exactly
- Training only needs to learn deviations: correlations between start/stop
  positions, sequence effects beyond the hexamer window, region-specific
  structure
- The learning problem is centered and much easier

## Properties

- **Sim data sanity check**: on simulated data generated from hexamer tables,
  the Stage 1 baseline IS the ground truth. Stage 2 corrections should
  converge to zero — a perfect test that the pipeline is correct.

- **Interpretability**: the baseline NLL gives an analytical lower bound for
  the factored model. Any improvement from Stage 2 measures the value of
  modeling start/stop correlations and longer-range sequence effects.

- **Warm start**: the model starts at ~12.3 nats (the old model's fragment
  NLL) rather than near-uniform (~log(P × n_L × 2) ≈ 14+). This puts
  gradients in a useful regime from step 1.

## Implementation Outline

1. Compute hex tables from training data (or load from simulator manifest
   for sim data)
2. Compute marginal p(L) from training fragments
3. Materialize log p₀ per tile on-the-fly from stored sequence (cheap —
   just hexamer lookups + length table + log Z)
4. Model forward pass produces f_θ(seq) with same shape as logits
5. Loss = log_softmax(log_p0 + f_theta) evaluated at observed fragments
6. Both conv and embedding variants share the same correction architecture;
   they differ only in how p₀ is parameterized

## Open Questions

- Should the hex tables be frozen after Stage 1, or remain trainable
  (jointly optimized with the correction network)?
- Optimal kernel size for the conv variant — k=6 matches the simulator,
  but real biology may have longer-range effects
- Whether to include the NB count loss in Stage 2 or keep it separate
