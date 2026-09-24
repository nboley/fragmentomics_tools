"""Per-sample (length, GC) recovery shape from spikes, for comparison with ZTNB.

Method (docs/pending/absolute_capture_probability.md sections 10-12):

    d            = SPANK reads / SPANK deduped      per sample, per SPANK length
    molecules    = GC-dSpark reads / d              transfer d to the non-UMI panel
    p_seen_x_vol = molecules / MPM_input            MPM = molarity * N_A * 1e-6
    p_seen_ratio = p_seen_x_vol / (value at ref cell)

WHAT THE GRID QUANTITY IS. Writing V for the effective volume sampled:

    molecules_observed(L,G) = P(seen|L,G) * MPM(L,G) * V
    p_seen_x_vol(L,G)       = molecules / MPM = P(seen|L,G) * V      [units of uL]
    p_seen_ratio(L,G)       = P(seen|L,G) / P(seen|ref)              [dimensionless]

So the grid is a RATIO OF PROBABILITIES, not a probability. V cancels, which is
why this is computable without knowing the volume and equally why the absolute
scale is out of reach: P(seen) = molecules / (MPM * V) is one equation in the two
unknowns P and V. Do not read `p_seen_ratio` as P(seen).

For correcting production's MPM the ratio is in fact the quantity wanted:

    MPM_pathogen = (EDR / ddspank) * MPM_spank = C_path * (P_path / P_spank)

is correct only if the pathogen recovers like a 52bp/50%GC oligo. P(L,G)/P(spank)
is exactly the factor that relaxes that.

d is PER SAMPLE. The spread in deduped SPANK counts across the cohort is pooling
differences between samples, not noise, so a cohort-level d would be wrong for
every individual sample.

CONFOUNDING (section 12). The panel carries ONE oligo per (length, GC) cell, so
sequence-specific recovery cannot be separated from the cell effect. The surface
is reproducible (cross-sample Spearman 0.82-0.95) but is not a pure GC/length
bias: it alternates between adjacent GC cells in a way a smooth bias cannot.
"""

import numpy as np
import pandas as pd

from flgc.spike_data import load_spikes_df, load_spikes_for_result

AVOGADRO_PER_UL = 6.02214076e17  # molecules per uL at 1 M

SPANK_KEYS = {  # SPANK length -> (reads key, deduped key)
    52: ("SPANK-52C", "deduped SPANK-52C"),
    75: ("SPANK-75B", "deduped SPANK-75B"),
}

REF_LENGTH, REF_GC = 52, 50.0  # the SPANK calibration point


def duplication_rates(counts):
    """reads / deduped for each SPANK. Returns {length: d}."""
    out = {}
    for length, (rk, dk) in SPANK_KEYS.items():
        reads, dedup = counts.get(rk, 0), counts.get(dk, 0)
        if reads > 0 and dedup > 0:
            out[length] = reads / dedup
    return out


def spike_surface(result_id, profile="SNMv4C", env="prod"):
    """Per-cell p_seen_ratio for one sample. Returns (DataFrame, diagnostics)."""
    counts = load_spikes_for_result(result_id, env=env)
    d_by_len = duplication_rates(counts)
    if not d_by_len:
        raise ValueError(f"{result_id}: no usable SPANK counts")

    # Length-flatness holds to ~1.7% across sampled results, so one mean d is
    # used; the per-length values are returned so the assumption stays visible.
    d = float(np.mean(list(d_by_len.values())))

    model = load_spikes_df(result_id, profile, env=env)
    model = model[model.use_with_model].copy()

    # undo the library's molarity scaling to recover the raw count
    model["raw_reads"] = model.molarity_scaled_count / model.molarity_scale_factor - 1.0

    # ds templates contribute two strands, so 2x the molecules their listed
    # molarity implies (owner-confirmed, tested)
    strand_factor = np.where(model.ss_or_ds.astype(str).str.strip() == "ds", 2.0, 1.0)
    model["mpm_input"] = model.molarity * AVOGADRO_PER_UL * strand_factor

    model["molecules"] = model.raw_reads / d
    model["p_seen_x_vol"] = model.molecules / model.mpm_input  # = P(seen) * V

    ref = model.iloc[
        (
            (model.length - REF_LENGTH).abs()
            + (model.model_gc_perc - REF_GC).abs() / 10.0
        ).argmin()
    ]
    model["p_seen_ratio"] = model.p_seen_x_vol / ref.p_seen_x_vol

    diag = {
        "result_id": result_id,
        "d_by_length": d_by_len,
        "d": d,
        "d_ratio": d_by_len[52] / d_by_len[75] if len(d_by_len) == 2 else np.nan,
        "ref_cell": (int(ref.length), float(ref.model_gc_perc)),
        "n_cells": len(model),
    }
    cols = [
        "name", "length", "model_gc_perc", "ss_or_ds", "molarity",
        "raw_reads", "mpm_input", "molecules", "p_seen_x_vol", "p_seen_ratio",
    ]
    return model[cols].sort_values(["length", "model_gc_perc"]), diag


def ztnb_ratio_grid(ztnb_model, lengths, gcs, ref_length=REF_LENGTH, ref_gc=REF_GC):
    """ZTNB p_seen normalised at the same reference cell, ON THE SPIKE GRID.

    NaN marks cells where ZTNB has NO FIT -- these are not zeros. `predict()`
    would return MAX_WEIGHT there, which is a default rather than an estimate,
    so they are excluded instead of being silently compared.
    """
    from flgc.model import _bin_index

    def p_at(length, gc):
        i = _bin_index(length, ztnb_model.length_bins)
        j = _bin_index(gc, ztnb_model.gc_bins)
        if i is None or j is None:
            return np.nan
        p = ztnb_model.p_seen[i, j]
        return float(p) if np.isfinite(p) else np.nan

    ref = p_at(ref_length, ref_gc)
    grid = np.full((len(lengths), len(gcs)), np.nan)
    for a, L in enumerate(lengths):
        for b, G in enumerate(gcs):
            grid[a, b] = p_at(L, G)
    if not (np.isfinite(ref) and ref > 0):
        return grid, np.nan
    return grid / ref, ref


if __name__ == "__main__":
    import sys

    rids = [int(x) for x in sys.argv[1:]] or [200000]
    for rid in rids:
        surf, diag = spike_surface(rid)
        print(f"\n=== result {rid} ===")
        print(
            f"d(52)={diag['d_by_length'].get(52, float('nan')):.4f}  "
            f"d(75)={diag['d_by_length'].get(75, float('nan')):.4f}  "
            f"ratio={diag['d_ratio']:.4f}  using d={diag['d']:.4f}"
        )
        print(f"reference cell: {diag['ref_cell']}   cells: {diag['n_cells']}")
        piv = surf.pivot_table(
            index="length", columns="model_gc_perc", values="p_seen_ratio"
        )
        print("\np_seen_ratio = P(seen|L,G) / P(seen|ref)   [NOT a probability]")
        print(piv.round(3).to_string())
