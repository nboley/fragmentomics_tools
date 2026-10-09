"""Correlation structure of per-sample cut-site hexamer tables.

Answers the question the disattenuation gate raises but does not resolve: do the
samples that deviate from the pool deviate TOGETHER (a real subgroup) or
independently (individual/technical)?

Panels, per table:
  A  sample x sample correlation of RAW log-enrichment  -- what the samples are
  B  the same on the shrunk POSTERIOR weights           -- what the prior makes them
  C  table x table correlation on the pooled tables

Raw and posterior are shown side by side deliberately: the posteriors are all
pulled toward one pool, so their mutual correlations are inflated BY
CONSTRUCTION. The difference between the panels is the attenuation itself.

Scratch/exploratory (leading underscore, matching the repo convention).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from build_hexamer_prior import load_all_samples_ordered, load_artifact  # noqa: E402

D = "/efs/analytics/nathanboley/background_model/cut_site_hexamers"
ARTIFACT = f"{D}/hexamer_prior_92samples.json"
SHEET = f"{D}/cohort92_resolved_paths.tsv"
OUTDIR = os.path.join(_REPO, "docs", "pending", "hexamer_disattenuation_plots")

TABLES = ["start_fwd", "end_fwd", "start_rev", "end_rev"]
# min r_disattenuated across all four tables < 0.80 -> blocked by the proposed gate
BLOCKED = ["RD-56171-Lib1", "RD-56818-Lib1", "RD-57090-Lib1", "RD-56676-Lib1"]


def log_enrich(counts: np.ndarray, bg_prop: np.ndarray) -> np.ndarray:
    """log(observed proportion / background proportion), float64 throughout."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum(dtype=np.float64)
    out = np.full(c.shape, np.nan, dtype=np.float64)
    nz = (c > 0) & (bg_prop > 0)
    out[nz] = np.log(c[nz] / n) - np.log(bg_prop[nz])
    return out


def corr_on_joint_support(mat: np.ndarray) -> np.ndarray:
    """Pairwise Pearson r, each pair restricted to hexamers finite in BOTH."""
    n = mat.shape[0]
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(mat[i]) & np.isfinite(mat[j])
            if m.sum() < 10:
                out[i, j] = out[j, i] = np.nan
                continue
            a, b = mat[i][m], mat[j][m]
            r = np.corrcoef(a, b)[0, 1]
            out[i, j] = out[j, i] = r
    return out


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    art = load_artifact(ARTIFACT)
    names = list(art["provenance"]["sample_names"])
    obs, _bg_accum, _metas, hex_order_loaded = load_all_samples_ordered(D, names)
    # Raw counts and posterior weights are indexed by position, so a divergent
    # hexamer order would misalign the two panels SILENTLY. Fail instead.
    if list(hex_order_loaded) != list(art["hex_order"]):
        raise AssertionError(
            "loader hexamer order != artifact hex_order; raw/posterior would misalign"
        )
    print(f"[corr] loaded {len(names)} samples, hex_order verified", flush=True)

    plt.style.use("dark_background")

    for tn in TABLES:
        bg = np.asarray(art["background"][tn], dtype=np.float64)
        bg_prop = bg / bg.sum(dtype=np.float64)

        raw = np.vstack([log_enrich(obs[s][tn], bg_prop) for s in names])
        post = np.vstack(
            [np.log(np.asarray(art["posteriors"][s][tn], dtype=np.float64)) for s in names]
        )

        c_raw = corr_on_joint_support(raw)
        c_post = corr_on_joint_support(post)

        # cluster on the RAW structure; apply the same order to both panels so the
        # comparison is like-for-like
        d = 1.0 - np.nan_to_num(c_raw, nan=0.0)
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2.0
        order = leaves_list(linkage(squareform(d, checks=False), method="average"))

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for ax, c, title in (
            (axes[0], c_raw, f"RAW log-enrichment — {tn}"),
            (axes[1], c_post, f"SHRUNK posterior — {tn}"),
        ):
            im = ax.imshow(
                c[np.ix_(order, order)], cmap="magma", vmin=0.5, vmax=1.0, aspect="equal"
            )
            ax.set_title(title, color="white", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
            # mark the gate-blocked samples
            for b in BLOCKED:
                if b in names:
                    pos = int(np.where(order == names.index(b))[0][0])
                    ax.axhline(pos, color="#00e5ff", lw=0.8, alpha=0.9)
                    ax.axvline(pos, color="#00e5ff", lw=0.8, alpha=0.9)

        off = ~np.eye(len(names), dtype=bool)
        fig.suptitle(
            f"Sample x sample hexamer correlation, {tn}   "
            f"(mean off-diag raw {np.nanmean(c_raw[off]):.3f} -> "
            f"posterior {np.nanmean(c_post[off]):.3f});  "
            "cyan lines = samples the proposed gate blocks (min r_disatt < 0.80)",
            color="white",
            fontsize=11,
        )
        fig.tight_layout()
        p = os.path.join(OUTDIR, f"fig5_corr_structure_{tn}.png")
        fig.savefig(p, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(
            f"[corr] {tn}: mean off-diag raw {np.nanmean(c_raw[off]):.4f} "
            f"posterior {np.nanmean(c_post[off]):.4f} -> {p}",
            flush=True,
        )

    # table x table, on the pooled tables
    pooled = np.vstack(
        [
            log_enrich(
                np.asarray(art["prior"][tn]["pooled_obs"], dtype=np.float64),
                np.asarray(art["background"][tn], dtype=np.float64)
                / np.asarray(art["background"][tn], dtype=np.float64).sum(dtype=np.float64),
            )
            for tn in TABLES
        ]
    )
    ct = corr_on_joint_support(pooled)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(ct, cmap="magma", vmin=-1, vmax=1)
    ax.set_xticks(range(4), TABLES, rotation=45, ha="right", color="white")
    ax.set_yticks(range(4), TABLES, color="white")
    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, f"{ct[i, j]:.3f}", ha="center", va="center",
                color="#00e5ff" if abs(ct[i, j]) < 0.6 else "black", fontsize=10,
            )
    ax.set_title("Pooled table x table correlation\n(log-enrichment over 4096 hexamers)",
                 color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "fig6_table_x_table_corr.png")
    fig.savefig(p, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[corr] table x table -> {p}", flush=True)
    print(ct.round(4))


if __name__ == "__main__":
    main()
