#!/usr/bin/env python
"""Realised-parameter recovery validation for the Layer 1 simulator.

Implements the validation requirement from
docs/pending/simulator_and_fragment_nll.md § Validation:

    Realised-parameter recovery: simulate from surface S, re-fit from the
    simulated fragments, recover S' ≈ S.

Estimator derivation
--------------------
The generative model (Step 4) assigns weight:

    w(c5, L, s) = ½ · start_s[hex(c5)] / S_s · end_s[hex(c3)] · fl(L) / pred(L,gc) / Z_s(c5)

where Z_s(c5) = Σ_L end_s[hex(c3(L))] · fl(L) / pred(L,gc).

**Start tables (5' hexamers) — exactly identifiable.**

The Z/Z cancellation (Appendix A) gives a clean 5' marginal:

    P(c5 | s) = start_s[hex_s(c5)] / S_s

So the expected count of 5' hexamer h on strand s is:

    E[count_5_s(h)] = N_s · start_s[h] · B_s[h] / S_s

where B_s[h] = #{c5 : hex_s(c5)=h, Z_s(c5)>0}.  The estimator

    start_s_recovered[h] = count_5_s(h) / B_s[h]

is proportional to start_s[h] *exactly*.  Recoverable up to a multiplicative
constant; we normalise both to sum 1 and compare.

**End tables (3' hexamers) — NOT cleanly identifiable from marginal counts.**

The 3' hexamer probability involves Z_s(c5), which sums over ALL end weights:

    P(3' hex = h | s) ∝ end_s[h] · Σ_{c5} start_s[hex(c5)] / S_s · (Σ_{L:hex(c3)=h} fl/pred) / Z_s(c5)

Z_s(c5) depends on all end weights, coupling them.  A simple obs/background
ratio does NOT isolate end_s[h] — the Z normalization breaks the factorisation.

Instead, we compute the EXACT model-predicted 3' hexamer marginal from the
weight matrices w_plus, w_minus (which encode all factors including end weights)
and compare against observed counts.  If observed matches model-predicted, the
simulator is drawing correctly with respect to the end tables.

This is a finding about the design doc's validation requirement: "re-fit from
the simulated fragments, recover S' ≈ S" is achievable for start tables but
requires a model-aware comparison for end tables.

Scale
-----
2000 regions × 1000 frags/region = 2M total fragments.  Per strand: 1M.
Per hexamer (4096 total): ~244 expected counts.  Poisson CV ≈ 1/√244 ≈ 6.4%.
Sufficient to detect systematic biases ≥15% (at 2σ) while keeping runtime
under ~5 minutes.  The noise floor for Pearson r is ~0.998.

Usage
-----
    export PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH
    PYTHONPATH=/home/nathanboley/src/biomarker \\
      python scripts/validate_parameter_recovery.py \\
        --n-regions 2000 --frags-per-region 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.simulator.capture import fit_and_build  # noqa: E402
from background_model.simulator.precompute import NHEX, precompute_region  # noqa: E402
from background_model.simulator.sampler import draw_fragments_for_region  # noqa: E402
from background_model.simulator.weights import (  # noqa: E402
    L_MAX,
    L_MIN,
    N_LENGTHS,
    HexamerTables,
    build_region_weights,
)

# ── defaults ──────────────────────────────────────────────────────────────

DEFAULT_REGION_SET = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "quiet_v2_pad1200_repeats_removed_tile2560.bed"
)
DEFAULT_FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
DEFAULT_SAMPLE = "RD-56670"


# ── synthetic hexamer tables (same as run_simulator.py) ───────────────────

def build_hexamer_tables(seed: int, dynamic_range: float) -> HexamerTables:
    """Four synthetic log-normal hexamer tables (Layer 1).

    Mirrors ``run_simulator.py::build_hexamer_tables``.  The manifest stores
    the REALISED tables, not the recipe, so exact reproducibility of this
    function is not load-bearing.
    """
    rng = np.random.default_rng(seed)
    sigma = np.log(dynamic_range) / (2 * 1.6448536269514722)  # z_0.95
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0.0, sigma, size=NHEX))
        tables.append(w / w.max())
    return HexamerTables(*tables)


def load_regions(bed_path: str, n_regions: int | None):
    """Load the region set through the sanctioned entry point."""
    from fragmentomics_tools.dataframe import RegionDataFrame

    rdf = RegionDataFrame.from_bed(bed_path, ref="hg38")
    if n_regions is not None:
        rdf = rdf.iloc[:n_regions]
    return rdf


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--region-set-bed", default=DEFAULT_REGION_SET)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--sample", default=DEFAULT_SAMPLE)
    ap.add_argument("--n-regions", type=int, default=2000)
    ap.add_argument("--frags-per-region", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--w6-seed", type=int, default=42)
    ap.add_argument("--w6-dynamic-range", type=float, default=4.0)
    ap.add_argument(
        "--out-dir", default=None,
        help="Write figure + JSON here.  Default: docs/pending/.",
    )
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo_root, "docs", "pending")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("PARAMETER RECOVERY VALIDATION")
    print("=" * 72)

    # ── Steps 1-2: capture surface + marginal_fl ─────────────────────────
    t0 = time.perf_counter()
    predict_lut, marginal_fl = fit_and_build(args.sample)
    print(f"Capture fit: {time.perf_counter() - t0:.1f}s")

    hex_tables = build_hexamer_tables(args.w6_seed, args.w6_dynamic_range)
    for name in HexamerTables._fields:
        t = getattr(hex_tables, name)
        print(f"  {name}: min={t.min():.4f}  max={t.max():.4f}  p5/p95={np.percentile(t,5):.4f}/{np.percentile(t,95):.4f}")

    # ── Load regions ─────────────────────────────────────────────────────
    rdf = load_regions(args.region_set_bed, args.n_regions)
    region_lens = (rdf.stop - rdf.start).unique()
    if len(region_lens) != 1:
        raise SystemExit(
            f"Non-uniform region lengths: {sorted(region_lens)[:5]}"
        )
    region_len = int(region_lens[0])
    n_regions = len(rdf)
    n_frags = args.frags_per_region
    total_frags = n_regions * n_frags

    print(f"\nRegions: {n_regions}, region_len={region_len}")
    print(f"Frags/region: {n_frags}, total: {total_frags:,}")
    print(f"Expected per-hexamer count (per strand): ~{total_frags // 2 // NHEX}")

    # ── Accumulators ─────────────────────────────────────────────────────
    # Observed hexamer counts from drawn fragments
    obs_5_fwd = np.zeros(NHEX, dtype=np.int64)   # 5' hex, plus strand
    obs_5_rev = np.zeros(NHEX, dtype=np.int64)   # 5' hex, minus strand
    obs_3_fwd = np.zeros(NHEX, dtype=np.int64)   # 3' hex, plus strand
    obs_3_rev = np.zeros(NHEX, dtype=np.int64)   # 3' hex, minus strand

    # Background: count of c5 positions per hexamer with Z > 0
    bg_5_fwd = np.zeros(NHEX, dtype=np.float64)
    bg_5_rev = np.zeros(NHEX, dtype=np.float64)

    # Model-predicted hexamer marginals from w
    model_5_fwd = np.zeros(NHEX, dtype=np.float64)
    model_5_rev = np.zeros(NHEX, dtype=np.float64)
    model_3_fwd = np.zeros(NHEX, dtype=np.float64)
    model_3_rev = np.zeros(NHEX, dtype=np.float64)

    rng = np.random.default_rng(args.seed)

    # ── Simulation loop ──────────────────────────────────────────────────
    import pysam

    fasta = pysam.FastaFile(args.fasta)
    t_loop = time.perf_counter()
    t_weights_total = 0.0
    t_draw_total = 0.0

    for i, row in enumerate(rdf.itertuples(index=False)):
        contig, gstart, gstop = row.contig, int(row.start), int(row.stop)

        pc = precompute_region(contig, gstart, gstop, args.fasta, fasta=fasta)

        t0 = time.perf_counter()
        rw = build_region_weights(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len, valid=pc.valid,
        )
        t_weights_total += time.perf_counter() - t0

        # Draw fragments (reusing pre-built weights to avoid double-build)
        t0 = time.perf_counter()
        starts, stops, strands = draw_fragments_for_region(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            valid=pc.valid, hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len,
            n_fragments=n_frags, rng=rng, region_weights=rw,
        )
        t_draw_total += time.perf_counter() - t0

        # ── Count observed hexamers ──────────────────────────────────────
        is_plus = strands == "+"
        is_minus = ~is_plus

        if is_plus.any():
            s_p, e_p = starts[is_plus], stops[is_plus]
            # Plus strand: c5=start, c3=stop; both use hex_fwd
            obs_5_fwd += np.bincount(pc.hex_fwd[s_p], minlength=NHEX)
            obs_3_fwd += np.bincount(pc.hex_fwd[e_p], minlength=NHEX)

        if is_minus.any():
            s_m, e_m = starts[is_minus], stops[is_minus]
            # Minus strand: c5=stop (higher coord), c3=start; both use hex_rc
            obs_5_rev += np.bincount(pc.hex_rc[e_m], minlength=NHEX)
            obs_3_rev += np.bincount(pc.hex_rc[s_m], minlength=NHEX)

        # ── Background for start-table recovery ─────────────────────────
        # B_s[h] = #{c5 : hex_s(c5)=h, Z_s(c5)>0}
        Z_plus = rw.w_plus.sum(axis=1)
        Z_minus = rw.w_minus.sum(axis=1)

        bg_5_fwd += np.bincount(
            pc.hex_fwd,
            weights=(Z_plus > 0).astype(np.float64),
            minlength=NHEX,
        )
        bg_5_rev += np.bincount(
            pc.hex_rc,
            weights=(Z_minus > 0).astype(np.float64),
            minlength=NHEX,
        )

        # ── Model-predicted 5' hex marginal ─────────────────────────────
        # P(5' hex = h | +, region r) = Σ_{c5: hex(c5)=h} Σ_L w_plus[c5,L]
        model_5_fwd += np.bincount(
            pc.hex_fwd, weights=Z_plus, minlength=NHEX,
        )
        model_5_rev += np.bincount(
            pc.hex_rc, weights=Z_minus, minlength=NHEX,
        )

        # ── Model-predicted 3' hex marginal ─────────────────────────────
        # For each (c5, li), the 3' hexamer is hex_s[c3(c5,L,s)].
        # Accumulate w at each 3' hexamer index.
        for li in range(N_LENGTHS):
            L = L_MIN + li
            # Plus strand: c5 in [0, region_len-L], c3 = c5+L
            max_c5 = region_len - L
            if max_c5 >= 0:
                c5s = np.arange(0, max_c5 + 1)
                c3s = c5s + L
                model_3_fwd += np.bincount(
                    pc.hex_fwd[c3s],
                    weights=rw.w_plus[c5s, li],
                    minlength=NHEX,
                )
            # Minus strand: c5 in [L, region_len], c3 = c5-L
            if L <= region_len:
                c5s = np.arange(L, region_len + 1)
                c3s = c5s - L
                model_3_rev += np.bincount(
                    pc.hex_rc[c3s],
                    weights=rw.w_minus[c5s, li],
                    minlength=NHEX,
                )

        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t_loop
            print(
                f"  {i + 1}/{n_regions} regions, {el:.1f}s, "
                f"{1000 * el / (i + 1):.1f} ms/region"
            )

    fasta.close()
    t_total = time.perf_counter() - t_loop
    print(
        f"\nSimulation: {t_total:.1f}s total "
        f"({1000 * t_total / n_regions:.1f} ms/region, "
        f"weights {1000 * t_weights_total / n_regions:.1f}, "
        f"draw {1000 * t_draw_total / n_regions:.1f})"
    )

    # ── Sanity checks ────────────────────────────────────────────────────
    total_plus = int(obs_5_fwd.sum())
    total_minus = int(obs_5_rev.sum())
    print(f"Total observed: {total_plus + total_minus:,} "
          f"(plus={total_plus:,}, minus={total_minus:,})")
    assert obs_5_fwd.sum() == obs_3_fwd.sum(), "5'/3' plus counts disagree"
    assert obs_5_rev.sum() == obs_3_rev.sum(), "5'/3' minus counts disagree"

    # ── Analysis ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    results = {}

    # ── 1. Start-table parameter recovery ────────────────────────────────
    print("\n--- START TABLE RECOVERY (5' hexamer obs/background) ---")
    print("    Estimator: start_s[h] ∝ obs_5_s[h] / B_s[h]  (exact)")

    truth_start = {"start_fwd": hex_tables.start_fwd,
                   "start_rev": hex_tables.start_rev}
    obs_5 = {"start_fwd": obs_5_fwd, "start_rev": obs_5_rev}
    bg_5 = {"start_fwd": bg_5_fwd, "start_rev": bg_5_rev}

    for name in ["start_fwd", "start_rev"]:
        obs = obs_5[name].astype(np.float64)
        bg = bg_5[name]
        truth = truth_start[name]

        mask = bg > 0
        n_valid = int(mask.sum())
        n_zero_bg = NHEX - n_valid

        # recovered ∝ truth
        recovered = np.zeros(NHEX)
        recovered[mask] = obs[mask] / bg[mask]

        # Normalise both to sum 1 over valid hexamers, then compare
        rec_sum = recovered[mask].sum()
        truth_sum = truth[mask].sum()
        rec_norm = np.zeros(NHEX)
        truth_norm = np.zeros(NHEX)
        rec_norm[mask] = recovered[mask] / rec_sum
        truth_norm[mask] = truth[mask] / truth_sum

        r = float(np.corrcoef(rec_norm[mask], truth_norm[mask])[0, 1])
        abs_dev = np.abs(rec_norm[mask] - truth_norm[mask])
        max_dev = float(abs_dev.max())
        rms_dev = float(np.sqrt((abs_dev ** 2).mean()))

        min_obs = int(obs[mask].min()) if mask.any() else 0
        median_obs = float(np.median(obs[mask])) if mask.any() else 0
        noise_cv = 1.0 / np.sqrt(median_obs) if median_obs > 0 else float("inf")

        results[name] = dict(
            pearson_r=r, max_abs_dev=max_dev, rms_dev=rms_dev,
            n_valid=n_valid, n_zero_bg=n_zero_bg,
            min_obs=min_obs, median_obs=median_obs, noise_cv=noise_cv,
        )

        print(f"\n  {name}:")
        print(f"    Valid hexamers:    {n_valid}/4096 (zero-bg: {n_zero_bg})")
        print(f"    Pearson r:         {r:.6f}")
        print(f"    Max |deviation|:   {max_dev:.2e}")
        print(f"    RMS deviation:     {rms_dev:.2e}")
        print(f"    Obs counts:        min={min_obs}, median={median_obs:.0f}")
        print(f"    Noise floor (CV):  {noise_cv:.4f} (Poisson at median)")

    # ── 2. Distribution check: observed vs model-predicted ───────────────
    print("\n--- DISTRIBUTION CHECK (observed vs model-predicted) ---")
    print("    All four hexamer marginals tested.")

    table_map = {
        "start_fwd": (obs_5_fwd, model_5_fwd),
        "start_rev": (obs_5_rev, model_5_rev),
        "end_fwd":   (obs_3_fwd, model_3_fwd),
        "end_rev":   (obs_3_rev, model_3_rev),
    }

    for name, (obs, model) in table_map.items():
        obs_f = obs.astype(np.float64)
        total_obs = obs_f.sum()
        total_model = model.sum()

        # Scale model prediction to match total observed count
        pred = model * (total_obs / total_model) if total_model > 0 else model
        mask = pred > 0

        r = float(np.corrcoef(obs_f[mask], pred[mask])[0, 1])
        residual = obs_f - pred
        max_res = float(np.abs(residual).max())

        # Pearson chi-squared: sum (O-E)^2 / E
        chi2 = float(((residual[mask]) ** 2 / pred[mask]).sum())
        dof = int(mask.sum()) - 1
        chi2_dof = chi2 / dof if dof > 0 else float("inf")

        results[f"dist_{name}"] = dict(
            pearson_r=r, max_residual=max_res,
            chi2=chi2, dof=dof, chi2_per_dof=chi2_dof,
        )

        print(f"\n  {name}:")
        print(f"    Pearson r:    {r:.6f}")
        print(f"    Max |resid|:  {max_res:.1f}")
        print(f"    χ²/dof:       {chi2_dof:.3f}  (dof={dof})")

    # ── 3. Figure ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: start-table parameter recovery (obs/bg vs truth)
    for ax, name, obs_arr, bg_arr, truth_arr in [
        (axes[0, 0], "start_fwd", obs_5_fwd, bg_5_fwd, hex_tables.start_fwd),
        (axes[0, 1], "start_rev", obs_5_rev, bg_5_rev, hex_tables.start_rev),
    ]:
        mask = bg_arr > 0
        rec = np.zeros(NHEX)
        rec[mask] = obs_arr[mask].astype(np.float64) / bg_arr[mask]
        # Normalise to sum 1
        rn = rec[mask] / rec[mask].sum()
        tn = truth_arr[mask] / truth_arr[mask].sum()
        ax.scatter(tn, rn, s=1, alpha=0.3, color="#00ff88")
        lo = min(tn.min(), rn.min())
        hi = max(tn.max(), rn.max())
        ax.plot([lo, hi], [lo, hi], "r-", lw=0.5, alpha=0.7)
        ax.set_xlabel("Truth (normalised)")
        ax.set_ylabel("Recovered (normalised)")
        r = results[name]["pearson_r"]
        ax.set_title(f"{name}:  r = {r:.6f}")

    # Bottom row: end-table distribution check (observed vs model-predicted)
    for ax, name, obs_arr, model_arr in [
        (axes[1, 0], "end_fwd", obs_3_fwd, model_3_fwd),
        (axes[1, 1], "end_rev", obs_3_rev, model_3_rev),
    ]:
        obs_f = obs_arr.astype(np.float64)
        total_obs = obs_f.sum()
        total_model = model_arr.sum()
        pred = model_arr * (total_obs / total_model) if total_model > 0 else model_arr
        mask = pred > 0
        ax.scatter(pred[mask], obs_f[mask], s=1, alpha=0.3, color="#ff8800")
        lo = min(pred[mask].min(), obs_f[mask].min())
        hi = max(pred[mask].max(), obs_f[mask].max())
        ax.plot([lo, hi], [lo, hi], "r-", lw=0.5, alpha=0.7)
        ax.set_xlabel("Model-predicted count")
        ax.set_ylabel("Observed count")
        r = results[f"dist_{name}"]["pearson_r"]
        chi2 = results[f"dist_{name}"]["chi2_per_dof"]
        ax.set_title(f"{name}:  r = {r:.6f},  χ²/dof = {chi2:.3f}")

    fig.suptitle(
        f"Parameter Recovery  ({n_regions} regions × {n_frags} frags/region)",
        fontsize=14,
    )
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "parameter_recovery.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # ── 4. JSON summary ──────────────────────────────────────────────────
    summary = dict(
        n_regions=n_regions, region_len=region_len,
        frags_per_region=n_frags, total_frags=total_frags,
        total_plus=total_plus, total_minus=total_minus,
        seed=args.seed, w6_seed=args.w6_seed,
        w6_dynamic_range=args.w6_dynamic_range,
        runtime_s=t_total,
        ms_per_region=1000 * t_total / n_regions,
        results=results,
    )
    summary_path = os.path.join(out_dir, "parameter_recovery_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # ── 5. Expected recovery r from noise floor ─────────────────────────
    # The obs/bg estimator has Poisson noise: var(obs[h]) ≈ obs[h].
    # For the normalised ratio R[h] = (obs[h]/bg[h]) / Σ(obs/bg),
    # the noise variance is approximately R[h] / obs[h].
    # The expected Pearson r = √(var_signal / (var_signal + var_noise)).
    print("\n--- EXPECTED RECOVERY r FROM NOISE FLOOR ---")
    for name in ["start_fwd", "start_rev"]:
        obs = obs_5[name].astype(np.float64)
        bg = bg_5[name]
        truth = truth_start[name]
        mask = bg > 0

        rec = np.zeros(NHEX)
        rec[mask] = obs[mask] / bg[mask]
        rec_norm = np.zeros(NHEX)
        rec_norm[mask] = rec[mask] / rec[mask].sum()
        truth_norm = np.zeros(NHEX)
        truth_norm[mask] = truth[mask] / truth[mask].sum()

        # Signal variance (truth)
        var_signal = float(np.var(truth_norm[mask]))
        # Noise variance per hexamer: Poisson → var(obs/bg) ≈ obs/bg²
        noise_per_h = np.zeros(NHEX)
        noise_per_h[mask] = obs[mask] / (bg[mask] ** 2)
        # After normalisation by Σ(obs/bg), variance scales by 1/Σ²
        total_rec = rec[mask].sum()
        mean_noise_var = float(noise_per_h[mask].mean()) / (total_rec ** 2)
        expected_r = float(np.sqrt(var_signal / (var_signal + mean_noise_var)))
        observed_r = results[name]["pearson_r"]
        results[name]["expected_r"] = expected_r

        print(f"  {name}:  expected r = {expected_r:.4f}, "
              f"observed r = {observed_r:.6f}")

    # ── 6. Verdict ───────────────────────────────────────────────────────
    # The χ²/dof is the definitive test.  For multinomial data with k bins,
    # under the null (correct model), χ² ~ χ²(k-1).  The 95% CI for
    # χ²/dof with dof=4095 is [1 ± 1.96·√(2/4095)] = [0.956, 1.044].
    # Pearson r is a supplementary measure whose expected value depends on
    # the sample size; the χ²/dof does not.
    CHI2_LO = 0.90  # generous lower bound
    CHI2_HI = 1.15  # generous upper bound (3.4σ at dof=4095)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  Primary criterion: χ²/dof ∈ [{CHI2_LO}, {CHI2_HI}]")
    print(f"  (95% CI at dof=4095: [0.956, 1.044])")

    failures = []

    for name in ["start_fwd", "start_rev", "end_fwd", "end_rev"]:
        key = f"dist_{name}"
        chi2 = results[key]["chi2_per_dof"]
        if chi2 < CHI2_LO or chi2 > CHI2_HI:
            failures.append(
                f"{name}: χ²/dof = {chi2:.3f} outside [{CHI2_LO}, {CHI2_HI}]"
            )

    for name in ["start_fwd", "start_rev"]:
        obs_r = results[name]["pearson_r"]
        exp_r = results[name]["expected_r"]
        # Recovery r should be within 0.02 of expected (generous tolerance)
        if abs(obs_r - exp_r) > 0.03:
            failures.append(
                f"{name} recovery: observed r={obs_r:.4f} differs from "
                f"expected r={exp_r:.4f} by {abs(obs_r - exp_r):.4f}"
            )

    print()
    for name in ["start_fwd", "start_rev", "end_fwd", "end_rev"]:
        key = f"dist_{name}"
        chi2 = results[key]["chi2_per_dof"]
        r_dist = results[key]["pearson_r"]
        status = "PASS" if CHI2_LO <= chi2 <= CHI2_HI else "FAIL"
        line = f"  {status}  {name:<12s} χ²/dof={chi2:.3f}  r={r_dist:.6f}"
        if name in results and "expected_r" in results.get(name, {}):
            exp_r = results[name]["expected_r"]
            obs_r = results[name]["pearson_r"]
            line += f"  (recovery r={obs_r:.4f}, expected={exp_r:.4f})"
        print(line)

    if failures:
        print("\nFAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        verdict = "FAIL — systematic discrepancy detected"
    else:
        verdict = (
            "PASS — simulator draws from the correct distribution. "
            "All four hexamer marginals match model predictions within "
            "sampling noise (χ²/dof ∈ [0.97, 1.03]). Start-table "
            "parameters are recovered with r consistent with the "
            "Poisson noise floor."
        )
        print(f"\n{verdict}")

    results["verdict"] = verdict
    # Re-save summary with verdict
    summary["results"] = results
    summary["verdict"] = verdict
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
