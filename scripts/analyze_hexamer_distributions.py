#!/usr/bin/env python
"""Compare real vs synthetic cut-site hexamer enrichment distributions.

Loads the 5 IBD sample Parquets, computes per-hexamer enrichment
(observed / background), and compares against synthetic tables from build_w6.

Figures go to docs/pending/cut_site_hexamer_plots/.
Stats are printed to stdout for the analysis doc.

Env: /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
Run: PYTHONPATH=. python scripts/analyze_hexamer_distributions.py
"""
from __future__ import annotations

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── matplotlib setup (dark background per standing instruction) ──────────
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hex")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from scipy.stats import pearsonr, spearmanr

# ── paths ────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = "/efs/analytics/nathanboley/background_model/cut_site_hexamers"
PLOT_DIR = os.path.join(REPO, "docs", "pending", "cut_site_hexamer_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Bright palette for dark backgrounds (avoid muddy tab10 blues/browns)
COLORS = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe",
          "#fd79a8", "#74b9ff", "#55efc4", "#ffeaa7", "#dfe6e9"]
TABLE_COLORS = {"start_fwd": "#00d4aa", "start_rev": "#ff6b6b",
                "end_fwd": "#74b9ff", "end_rev": "#ffe66d"}
SAMPLE_COLORS = COLORS[:5]

TABLES = ["start_fwd", "start_rev", "end_fwd", "end_rev"]


# ── build_w6: copied from sim_fragments.py (DO NOT MODIFY) ──────────────
def build_w6(seed: int, dynamic_range: float) -> np.ndarray:
    """4096 synthetic log-normal hexamer weights, normalised to max 1."""
    rng = np.random.default_rng(seed)
    sigma = np.log(dynamic_range) / (2 * 1.6448536269514722)
    w = np.exp(rng.normal(0.0, sigma, size=4096))
    return w / w.max()


# ── load real data ───────────────────────────────────────────────────────
def load_all_samples():
    """Load all 5 Parquet files, return dict of sample_name -> DataFrame."""
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    samples = {}
    for f in files:
        name = os.path.basename(f).replace(".cut_site_hexamers.parquet", "")
        df = pq.read_table(f).to_pandas()
        samples[name] = df
    return samples


def compute_enrichment_band_collapsed(df):
    """Collapse across FL bands by summing observed & background, then divide.

    Enrichment = sum(observed across bands) / sum(background across bands)
    per (hexamer, table). This is a count-weighted average — bands with more
    fragments contribute proportionally. Normalised so each table sums to 1
    (i.e., enrichment is relative to the within-table mean).

    What this collapse hides: any FL-dependent variation in hexamer preference.
    If a hexamer is strongly preferred at short lengths but not long ones,
    the marginal enrichment averages that out, weighted by how many fragments
    fell in each band. The synthetic tables have NO band structure, so this
    is the only like-for-like comparison possible.
    """
    grouped = df.groupby(["hexamer", "table"]).agg(
        obs=("observed", "sum"), bg=("background", "sum")
    ).reset_index()
    grouped["enrichment"] = grouped["obs"] / grouped["bg"]
    # Normalise within each table so enrichments are relative (sum to 1)
    for tbl in TABLES:
        mask = grouped["table"] == tbl
        total = grouped.loc[mask, "enrichment"].sum()
        grouped.loc[mask, "enrichment"] /= (total / len(grouped.loc[mask]))
    return grouped


def compute_enrichment_per_band(df, table_name):
    """Per-band enrichment for one table, returns (n_bands, 4096) array."""
    sub = df[df["table"] == table_name].copy()
    bands = sorted(sub[["band_lo", "band_hi"]].drop_duplicates().values.tolist())
    result = np.full((len(bands), 4096), np.nan)
    for i, (lo, hi) in enumerate(bands):
        bsub = sub[(sub["band_lo"] == lo) & (sub["band_hi"] == hi)]
        obs = bsub["observed"].values.astype(np.float64)
        bg = bsub["background"].values.astype(np.float64)
        enr = obs / np.where(bg > 0, bg, np.nan)
        # Normalise to mean 1
        enr_mean = np.nanmean(enr)
        if enr_mean > 0:
            enr /= enr_mean
        result[i] = enr
    return result, bands


# ── figure functions ─────────────────────────────────────────────────────

def fig1_spread_comparison(real_enrichments, syn_tables):
    """Log-scale distribution of enrichment: real (4 tables, 5 samples) vs synthetic."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    stats = {}
    for i, tbl in enumerate(TABLES):
        ax = axes[i]
        # Real: overlay 5 samples
        for j, (sname, enr_df) in enumerate(real_enrichments.items()):
            vals = enr_df[enr_df["table"] == tbl]["enrichment"].values
            vals = vals[np.isfinite(vals) & (vals > 0)]
            log_vals = np.log10(vals)
            ax.hist(log_vals, bins=80, alpha=0.35, color=SAMPLE_COLORS[j],
                    label=sname if i == 0 else None, density=True)
            if j == 0:
                stats[tbl] = {"real_min": float(vals.min()),
                              "real_max": float(vals.max()),
                              "real_p5": float(np.percentile(vals, 5)),
                              "real_p95": float(np.percentile(vals, 95)),
                              "real_dynamic_range": float(vals.max() / vals.min()),
                              "real_p95_p5": float(np.percentile(vals, 95) / np.percentile(vals, 5))}

        # Synthetic
        syn = syn_tables[tbl]
        syn_log = np.log10(syn[syn > 0])
        ax.hist(syn_log, bins=80, alpha=0.7, color="white", histtype="step",
                linewidth=2, label="synthetic" if i == 0 else None, density=True)
        stats[tbl]["syn_min"] = float(syn.min())
        stats[tbl]["syn_max"] = float(syn.max())
        stats[tbl]["syn_p5"] = float(np.percentile(syn, 5))
        stats[tbl]["syn_p95"] = float(np.percentile(syn, 95))
        stats[tbl]["syn_dynamic_range"] = float(syn.max() / syn.min())
        stats[tbl]["syn_p95_p5"] = float(np.percentile(syn, 95) / np.percentile(syn, 5))

        ax.set_xlabel("log10(enrichment)", fontsize=11)
        ax.set_ylabel("density", fontsize=11)
        ax.set_title(tbl, fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Enrichment spread: real (5 samples) vs synthetic (build_w6, DR=4.0)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig1_spread_comparison.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return stats


def fig2_qq_lognormal(real_enrichments, syn_tables):
    """QQ plots: real enrichments vs theoretical log-normal (which the synthetic IS)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, tbl in enumerate(TABLES):
        ax = axes[i]
        # Use first sample as representative
        first_sample = list(real_enrichments.keys())[0]
        enr_df = real_enrichments[first_sample]
        real_vals = enr_df[enr_df["table"] == tbl]["enrichment"].values
        real_vals = np.sort(real_vals[np.isfinite(real_vals) & (real_vals > 0)])

        # Theoretical quantiles from a log-normal with same mean/sd of log
        log_real = np.log(real_vals)
        mu, sigma = log_real.mean(), log_real.std()
        theoretical = np.sort(np.exp(np.random.default_rng(42).normal(mu, sigma, len(real_vals))))

        ax.scatter(np.log10(theoretical), np.log10(real_vals), s=2, alpha=0.3,
                   color=TABLE_COLORS[tbl])
        # Reference line
        lo = min(np.log10(theoretical).min(), np.log10(real_vals).min())
        hi = max(np.log10(theoretical).max(), np.log10(real_vals).max())
        ax.plot([lo, hi], [lo, hi], "--", color="#ff6348", linewidth=1.5)

        ax.set_xlabel("log10(theoretical log-normal)", fontsize=10)
        ax.set_ylabel("log10(real enrichment)", fontsize=10)
        ax.set_title(tbl, fontsize=13, fontweight="bold")

    fig.suptitle("QQ: real enrichment vs fitted log-normal (1 sample)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig2_qq_lognormal.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def fig3_shape_comparison(real_enrichments, syn_tables):
    """Overlay log-enrichment histograms + KDE for shape comparison.

    Real = mean across 5 samples (to reduce noise). Show skewness and kurtosis.
    """
    from scipy.stats import skew, kurtosis

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    shape_stats = {}
    for i, tbl in enumerate(TABLES):
        ax = axes[i]

        # Mean enrichment across all 5 samples
        all_enr = []
        for sname, enr_df in real_enrichments.items():
            vals = enr_df[enr_df["table"] == tbl]["enrichment"].values
            all_enr.append(vals)
        all_enr = np.array(all_enr)  # (5, 4096)
        mean_enr = np.nanmean(all_enr, axis=0)
        valid = np.isfinite(mean_enr) & (mean_enr > 0)
        log_real = np.log(mean_enr[valid])

        syn = syn_tables[tbl]
        log_syn = np.log(syn[syn > 0])

        ax.hist(log_real, bins=80, alpha=0.6, color=TABLE_COLORS[tbl],
                density=True, label="real (5-sample mean)")
        ax.hist(log_syn, bins=80, alpha=0.5, color="white", histtype="step",
                linewidth=2, density=True, label="synthetic")

        real_skew = skew(log_real)
        real_kurt = kurtosis(log_real)
        syn_skew = skew(log_syn)
        syn_kurt = kurtosis(log_syn)

        ax.text(0.02, 0.98,
                f"Real:  skew={real_skew:.3f}  excess_kurt={real_kurt:.3f}\n"
                f"Synth: skew={syn_skew:.3f}  excess_kurt={syn_kurt:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

        shape_stats[tbl] = {
            "real_skew": float(real_skew), "real_excess_kurtosis": float(real_kurt),
            "syn_skew": float(syn_skew), "syn_excess_kurtosis": float(syn_kurt),
            "real_log_std": float(log_real.std()),
            "syn_log_std": float(log_syn.std()),
        }

        ax.set_xlabel("log(enrichment)", fontsize=11)
        ax.set_ylabel("density", fontsize=11)
        ax.set_title(tbl, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Shape: log-enrichment distributions (skew, kurtosis)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig3_shape_comparison.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return shape_stats


def fig4_correlation_structure(real_enrichments):
    """Correlation matrix across the 4 tables — within-end vs across-end.

    Uses the first sample as the primary, overlays all 5 as violin/box for
    the key correlations.
    """
    # Build per-hexamer enrichment matrix: (4096, 4) per sample
    sample_names = list(real_enrichments.keys())
    n_samples = len(sample_names)

    # Collect per-table enrichment vectors for each sample
    all_corr_matrices = []
    pair_corrs = {pair: [] for pair in [
        ("start_fwd", "start_rev"), ("end_fwd", "end_rev"),  # within-end
        ("start_fwd", "end_fwd"), ("start_rev", "end_rev"),  # across-end, same strand
        ("start_fwd", "end_rev"), ("start_rev", "end_fwd"),  # across-end, cross strand
    ]}

    for sname in sample_names:
        enr_df = real_enrichments[sname]
        vecs = {}
        for tbl in TABLES:
            sub = enr_df[enr_df["table"] == tbl].sort_values("hexamer")
            vecs[tbl] = sub["enrichment"].values
        # Correlation matrix
        mat = np.full((4, 4), np.nan)
        for a_i, a_tbl in enumerate(TABLES):
            for b_i, b_tbl in enumerate(TABLES):
                va = vecs[a_tbl]; vb = vecs[b_tbl]
                ok = np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
                if ok.sum() > 10:
                    mat[a_i, b_i] = pearsonr(np.log(va[ok]), np.log(vb[ok]))[0]
        all_corr_matrices.append(mat)

        for pair in pair_corrs:
            va = vecs[pair[0]]; vb = vecs[pair[1]]
            ok = np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
            if ok.sum() > 10:
                pair_corrs[pair].append(pearsonr(np.log(va[ok]), np.log(vb[ok]))[0])

    mean_corr = np.nanmean(all_corr_matrices, axis=0)

    # Figure: heatmap + grouped bar chart of correlation pairs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Heatmap
    im = ax1.imshow(mean_corr, cmap="coolwarm", vmin=0, vmax=1)
    ax1.set_xticks(range(4)); ax1.set_xticklabels(TABLES, rotation=45, ha="right", fontsize=10)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(TABLES, fontsize=10)
    for a_i in range(4):
        for b_i in range(4):
            ax1.text(b_i, a_i, f"{mean_corr[a_i, b_i]:.3f}",
                     ha="center", va="center", fontsize=10, fontweight="bold",
                     color="black" if mean_corr[a_i, b_i] > 0.5 else "white")
    fig.colorbar(im, ax=ax1, shrink=0.8)
    ax1.set_title("Pearson(log-enrichment)\n5-sample mean", fontsize=12, fontweight="bold")

    # Grouped bar: within-end vs across-end
    within_pairs = [("start_fwd", "start_rev"), ("end_fwd", "end_rev")]
    across_same = [("start_fwd", "end_fwd"), ("start_rev", "end_rev")]
    across_cross = [("start_fwd", "end_rev"), ("start_rev", "end_fwd")]

    categories = []
    means = []
    stds = []
    colors_bar = []
    for pairs, label, col in [
        (within_pairs, "within-end", "#00d4aa"),
        (across_same, "across-end\n(same strand)", "#74b9ff"),
        (across_cross, "across-end\n(cross strand)", "#ffe66d"),
    ]:
        for pair in pairs:
            vals = pair_corrs[pair]
            categories.append(f"{pair[0]}\nvs\n{pair[1]}")
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors_bar.append(col)

    x = np.arange(len(categories))
    bars = ax2.bar(x, means, yerr=stds, color=colors_bar, edgecolor="white",
                   linewidth=0.5, capsize=3, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylabel("Pearson r (log-enrichment)", fontsize=11)
    ax2.set_title("Correlation structure across tables\n(mean +/- SD over 5 samples)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    # Add value labels
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{m:.3f}", ha="center", fontsize=9, fontweight="bold", color="white")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#00d4aa", label="within-end"),
                       Patch(facecolor="#74b9ff", label="across-end (same strand)"),
                       Patch(facecolor="#ffe66d", label="across-end (cross strand)")]
    ax2.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.suptitle("Table correlation structure — justifying (or not) untied tables",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig4_correlation_structure.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return pair_corrs, mean_corr


def fig5_sample_variability(real_enrichments):
    """Cross-sample consistency: pairwise correlation of enrichments within each table."""
    sample_names = list(real_enrichments.keys())
    n = len(sample_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    variability_stats = {}
    for ti, tbl in enumerate(TABLES):
        ax = axes[ti]
        # Extract enrichment vectors
        vecs = []
        for sname in sample_names:
            enr_df = real_enrichments[sname]
            sub = enr_df[enr_df["table"] == tbl].sort_values("hexamer")
            vecs.append(sub["enrichment"].values)
        vecs = np.array(vecs)  # (5, 4096)

        # Pairwise correlations
        corrs = []
        for i in range(n):
            for j in range(i+1, n):
                va = vecs[i]; vb = vecs[j]
                ok = np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
                if ok.sum() > 10:
                    corrs.append(pearsonr(np.log(va[ok]), np.log(vb[ok]))[0])

        variability_stats[tbl] = {
            "pairwise_r_mean": float(np.mean(corrs)),
            "pairwise_r_min": float(np.min(corrs)),
            "pairwise_r_max": float(np.max(corrs)),
        }

        # Scatter: sample 0 vs sample 1 as representative
        va = vecs[0]; vb = vecs[1]
        ok = np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
        ax.scatter(np.log10(va[ok]), np.log10(vb[ok]), s=2, alpha=0.2,
                   color=TABLE_COLORS[tbl])
        lo = min(np.log10(va[ok]).min(), np.log10(vb[ok]).min())
        hi = max(np.log10(va[ok]).max(), np.log10(vb[ok]).max())
        ax.plot([lo, hi], [lo, hi], "--", color="#ff6348", linewidth=1)

        r_val = pearsonr(np.log(va[ok]), np.log(vb[ok]))[0]
        ax.text(0.02, 0.98,
                f"r = {r_val:.4f}\nAll pairs: {np.mean(corrs):.4f} [{np.min(corrs):.4f}, {np.max(corrs):.4f}]",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

        ax.set_xlabel(f"log10(enrichment) — {sample_names[0]}", fontsize=10)
        ax.set_ylabel(f"log10(enrichment) — {sample_names[1]}", fontsize=10)
        ax.set_title(tbl, fontsize=13, fontweight="bold")

    fig.suptitle("Sample-to-sample stability (5 IBD samples)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig5_sample_variability.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return variability_stats


def fig6_start_vs_end_asymmetry(real_enrichments):
    """Direct scatter of start vs end enrichments (both fwd), showing asymmetry."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use mean across samples
    mean_enr = {}
    for tbl in TABLES:
        all_vals = []
        for sname, enr_df in real_enrichments.items():
            sub = enr_df[enr_df["table"] == tbl].sort_values("hexamer")
            all_vals.append(sub["enrichment"].values)
        mean_enr[tbl] = np.nanmean(all_vals, axis=0)

    for ax, (t1, t2), title in [
        (axes[0], ("start_fwd", "end_fwd"), "fwd strand: start vs end"),
        (axes[1], ("start_rev", "end_rev"), "rev strand: start vs end"),
    ]:
        v1 = mean_enr[t1]; v2 = mean_enr[t2]
        ok = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
        ax.scatter(np.log10(v1[ok]), np.log10(v2[ok]), s=3, alpha=0.2,
                   color="#a29bfe")
        lo = min(np.log10(v1[ok]).min(), np.log10(v2[ok]).min())
        hi = max(np.log10(v1[ok]).max(), np.log10(v2[ok]).max())
        ax.plot([lo, hi], [lo, hi], "--", color="#ff6348", linewidth=1)

        r_val = pearsonr(np.log(v1[ok]), np.log(v2[ok]))[0]
        ratio = np.std(np.log(v1[ok])) / np.std(np.log(v2[ok]))
        ax.text(0.02, 0.98,
                f"r = {r_val:.4f}\nlog-SD ratio (start/end) = {ratio:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        ax.set_xlabel(f"log10(enrichment) — {t1}", fontsize=11)
        ax.set_ylabel(f"log10(enrichment) — {t2}", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

    fig.suptitle("Start vs end enrichment asymmetry (5-sample mean)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig6_start_end_asymmetry.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def fig7_band_stability(samples, sample_name):
    """Enrichment stability across FL bands for one sample, one table."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    df = samples[sample_name]
    for ti, tbl in enumerate(TABLES):
        ax = axes[ti]
        enr_bands, bands = compute_enrichment_per_band(df, tbl)
        # Pick 4 representative bands
        band_indices = [0, 5, 10, 15]  # [25-35], [75-85], [125-135], [175-181]
        for bi in band_indices:
            if bi >= len(bands):
                continue
            lo, hi = bands[bi]
            vals = enr_bands[bi]
            valid = np.isfinite(vals) & (vals > 0)
            log_vals = np.log(vals[valid])
            ax.hist(log_vals, bins=60, alpha=0.4, density=True,
                    label=f"[{lo},{hi})")

        ax.set_xlabel("log(enrichment)", fontsize=10)
        ax.set_ylabel("density", fontsize=10)
        ax.set_title(tbl, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, title="FL band")

    fig.suptitle(f"FL-band stability of enrichment ({sample_name})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "fig7_band_stability.png"), dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading real data...", flush=True)
    samples = load_all_samples()
    print(f"  Loaded {len(samples)} samples: {list(samples.keys())}", flush=True)

    # Compute band-collapsed enrichment per sample
    print("Computing enrichments...", flush=True)
    real_enrichments = {}
    for sname, df in samples.items():
        real_enrichments[sname] = compute_enrichment_band_collapsed(df)

    # Generate synthetic tables (4 independent draws, matching the sim's use
    # of build_w6 -- the sim uses ONE table, but here we generate 4 to show
    # what independent log-normal draws look like)
    print("Generating synthetic tables...", flush=True)
    syn_tables = {}
    for i, tbl in enumerate(TABLES):
        # Different seed per table to make them independent
        syn_tables[tbl] = build_w6(seed=1337 + i, dynamic_range=4.0)

    # ── Figures ──────────────────────────────────────────────────────────
    print("Generating figures...", flush=True)

    print("  Fig 1: Spread comparison", flush=True)
    spread_stats = fig1_spread_comparison(real_enrichments, syn_tables)

    print("  Fig 2: QQ log-normal", flush=True)
    fig2_qq_lognormal(real_enrichments, syn_tables)

    print("  Fig 3: Shape comparison", flush=True)
    shape_stats = fig3_shape_comparison(real_enrichments, syn_tables)

    print("  Fig 4: Correlation structure", flush=True)
    pair_corrs, mean_corr = fig4_correlation_structure(real_enrichments)

    print("  Fig 5: Sample variability", flush=True)
    var_stats = fig5_sample_variability(real_enrichments)

    print("  Fig 6: Start vs end asymmetry", flush=True)
    fig6_start_vs_end_asymmetry(real_enrichments)

    print("  Fig 7: Band stability", flush=True)
    first_sample = list(samples.keys())[0]
    fig7_band_stability(samples, first_sample)

    # ── Print stats report ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("MEASURED STATISTICS")
    print("="*80)

    print("\n--- SPREAD (Dynamic Range) per table ---")
    print(f"{'Table':<12} {'Real min/max':>16} {'Real DR':>10} {'Real p95/p5':>12} "
          f"{'Syn min/max':>16} {'Syn DR':>10} {'Syn p95/p5':>12}")
    for tbl in TABLES:
        s = spread_stats[tbl]
        print(f"{tbl:<12} {s['real_min']:.4f}/{s['real_max']:.4f}  "
              f"{s['real_dynamic_range']:>10.1f}x  {s['real_p95_p5']:>10.2f}x  "
              f"{s['syn_min']:.4f}/{s['syn_max']:.4f}  "
              f"{s['syn_dynamic_range']:>10.1f}x  {s['syn_p95_p5']:>10.2f}x")

    print("\n--- SHAPE (log-enrichment) ---")
    print(f"{'Table':<12} {'Real skew':>10} {'Real kurt':>10} {'Real log-SD':>12} "
          f"{'Syn skew':>10} {'Syn kurt':>10} {'Syn log-SD':>12}")
    for tbl in TABLES:
        s = shape_stats[tbl]
        print(f"{tbl:<12} {s['real_skew']:>10.3f} {s['real_excess_kurtosis']:>10.3f} "
              f"{s['real_log_std']:>12.4f}  "
              f"{s['syn_skew']:>10.3f} {s['syn_excess_kurtosis']:>10.3f} "
              f"{s['syn_log_std']:>12.4f}")

    print("\n--- CORRELATION STRUCTURE (Pearson of log-enrichment) ---")
    print("Category                          Pair                        Mean    SD")
    within = [("start_fwd", "start_rev"), ("end_fwd", "end_rev")]
    across_same = [("start_fwd", "end_fwd"), ("start_rev", "end_rev")]
    across_cross = [("start_fwd", "end_rev"), ("start_rev", "end_fwd")]
    for label, pairs in [("Within-end", within),
                         ("Across-end (same strand)", across_same),
                         ("Across-end (cross strand)", across_cross)]:
        for pair in pairs:
            vals = pair_corrs[pair]
            print(f"  {label:<32} {pair[0]:>10} vs {pair[1]:<10} "
                  f"{np.mean(vals):.4f}  {np.std(vals):.4f}")

    print("\n--- SAMPLE-TO-SAMPLE VARIABILITY ---")
    print(f"{'Table':<12} {'Mean r':>8} {'Min r':>8} {'Max r':>8}")
    for tbl in TABLES:
        s = var_stats[tbl]
        print(f"{tbl:<12} {s['pairwise_r_mean']:>8.4f} {s['pairwise_r_min']:>8.4f} "
              f"{s['pairwise_r_max']:>8.4f}")

    # Also compute the all-sample-mean enrichment stats per table for the doc
    print("\n--- ALL-SAMPLE-MEAN ENRICHMENT STATS ---")
    for tbl in TABLES:
        all_vals = []
        for sname, enr_df in real_enrichments.items():
            sub = enr_df[enr_df["table"] == tbl].sort_values("hexamer")
            all_vals.append(sub["enrichment"].values)
        mean_enr = np.nanmean(all_vals, axis=0)
        valid = np.isfinite(mean_enr) & (mean_enr > 0)
        v = mean_enr[valid]
        print(f"{tbl:<12} min={v.min():.4f}  max={v.max():.4f}  "
              f"DR={v.max()/v.min():.1f}x  p5={np.percentile(v,5):.4f}  "
              f"p95={np.percentile(v,95):.4f}  p95/p5={np.percentile(v,95)/np.percentile(v,5):.2f}x")

    print(f"\nPlots saved to: {PLOT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
