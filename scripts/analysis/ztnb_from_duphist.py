"""Fit a ZTNB GCFlDistModel from a precomputed PE duplicate histogram.

Input: /efs/analytics/nathanboley/ctcf_fa_cache/duphist_merged/<sample>__duphist_wg.tsv.gz
with columns (length, gc, multiplicity, molecule_keys).

Why this source rather than a BAM: the retained per-result BAMs are SINGLE-END,
where the observable is read length (capped by sequencing chemistry, ~64bp here)
rather than fragment length. These histograms are paired-end, and the
mononucleosome mode at 166bp confirms it, so the length axis is comparable with
the spike panel's true oligo lengths.

Bins are centred on the spike grid points so the two surfaces land on the SAME
grid. The library's default ZTNB bins are much coarser than the spike panel
(8 length bins vs 9 spike lengths), which makes a cell-by-cell comparison invent
distinctions ZTNB cannot express.
"""

import numpy as np
import pandas as pd

from flgc.model import GCFlDistModel

DUPHIST_DIR = "/efs/analytics/nathanboley/ctcf_fa_cache/duphist_merged"

SPIKE_LENGTHS = [24, 32, 42, 52, 75, 100, 125, 150, 175]
SPIKE_GCS = [30, 40, 50, 60, 70]

# bins centred on the spike grid points
LENGTH_BINS = [(20, 28), (29, 37), (38, 47), (48, 63), (64, 87),
               (88, 112), (113, 137), (138, 162), (163, 200)]
GC_BINS = [(25, 35), (36, 45), (46, 55), (56, 65), (66, 75)]

MAX_SANE_LENGTH = 1000  # above this is artifact (1.5% of mass, up to a 65535 sentinel)


def load_duphist(sample):
    path = f"{DUPHIST_DIR}/{sample}__duphist_wg.tsv.gz"
    df = pd.read_csv(path, sep="\t")
    return df[(df.length >= 1) & (df.length <= MAX_SANE_LENGTH)]


def build_cell_map(df):
    """(length, gc) -> (k_vals, obs, kmax, seen_unique), the shape fit() wants."""
    cell_map = {}
    for (length, gc), g in df.groupby(["length", "gc"], sort=False):
        k = g.multiplicity.to_numpy()
        obs = g.molecule_keys.to_numpy()
        cell_map[(int(length), float(gc))] = (k, obs, int(k.max()), int(obs.sum()))
    return cell_map


def fit_ztnb(sample, min_cell_size=200):
    df = load_duphist(sample)
    cell_map = build_cell_map(df)
    model = GCFlDistModel()
    model.fit(cell_map, length_bins=LENGTH_BINS, gc_bins=GC_BINS,
              min_cell_size=min_cell_size)
    diag = {
        "sample": sample,
        "molecules": int(df.molecule_keys.sum()),
        "reads": int((df.molecule_keys * df.multiplicity).sum()),
        "cells_fitted": len(model._records),
        "cells_possible": len(LENGTH_BINS) * len(GC_BINS),
    }
    diag["duplication"] = diag["reads"] / diag["molecules"]
    return model, diag


def p_seen_on_spike_grid(model):
    """ZTNB p_seen at each spike grid point. NaN where unfitted (NOT zero)."""
    from flgc.model import _bin_index

    grid = np.full((len(SPIKE_LENGTHS), len(SPIKE_GCS)), np.nan)
    for a, L in enumerate(SPIKE_LENGTHS):
        for b, G in enumerate(SPIKE_GCS):
            i = _bin_index(L, model.length_bins)
            j = _bin_index(G, model.gc_bins)
            if i is None or j is None:
                continue
            p = model.p_seen[i, j]
            if np.isfinite(p):
                grid[a, b] = float(p)
    return grid


if __name__ == "__main__":
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "RD-56670"
    model, diag = fit_ztnb(sample)
    print(f"{diag['sample']}: {diag['molecules']:,} molecules, "
          f"{diag['reads']:,} reads, duplication={diag['duplication']:.4f}")
    print(f"cells fitted: {diag['cells_fitted']} / {diag['cells_possible']}")

    grid = p_seen_on_spike_grid(model)
    out = pd.DataFrame(grid, index=SPIKE_LENGTHS, columns=SPIKE_GCS)
    print("\nZTNB p_seen on the spike grid:")
    print(out.round(3).to_string())
    n_clamped = int(np.nansum(grid < 1.0 / 3.0))
    print(f"\ncells below 1/MAX_WEIGHT (would clamp): {n_clamped} / "
          f"{int(np.isfinite(grid).sum())} fitted")
