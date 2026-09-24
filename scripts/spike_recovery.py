"""Per-sample (length, GC) recovery from spikes, for comparison against ZTNB.

Method (docs/pending/absolute_capture_probability.md sections 10-11):

  d            = SPANK reads / SPANK deduped        per sample, per SPANK length
  molecules    = GC-dSpark reads / d                 transfer d to the non-UMI panel
  recovery     = molecules / MPM_input               MPM = molarity * N_A * 1e-6

`recovery` carries an unresolved constant factor (the effective volume sampled),
so it is NOT an absolute P(seen). It is reported normalised to the cell nearest
the SPANK calibration point, which makes its SHAPE across (length, GC) directly
comparable to a ZTNB p_seen surface normalised the same way.

d is computed PER SAMPLE. The 191x spread in deduped SPANK counts across the
cohort is pooling differences between samples, not noise, so a cohort-level d
would be wrong for every individual sample.
"""

import numpy as np
import pandas as pd

from flgc.spike_data import load_spikes_df, load_spikes_for_result

AVOGADRO_PER_UL = 6.02214076e17  # molecules per uL at 1 M

SPANK_KEYS = {  # length -> (reads key, deduped key)
    52: ("SPANK-52C", "deduped SPANK-52C"),
    75: ("SPANK-75B", "deduped SPANK-75B"),
}


def duplication_rates(counts):
    """reads / deduped for each SPANK, per sample. Returns {length: d}."""
    out = {}
    for length, (rk, dk) in SPANK_KEYS.items():
        reads, dedup = counts.get(rk, 0), counts.get(dk, 0)
        if reads > 0 and dedup > 0:
            out[length] = reads / dedup
    return out


def recovery_surface(result_id, profile="SNMv4C", env="prod"):
    """Per-cell recovery for one sample. Returns (DataFrame, diagnostics)."""
    counts = load_spikes_for_result(result_id, env=env)
    d_by_len = duplication_rates(counts)
    if not d_by_len:
        raise ValueError(f"{result_id}: no usable SPANK counts")

    # Length-flatness held to ~1.7% across sampled results, so a single mean d
    # is used. The per-length values are returned so the assumption stays visible.
    d = float(np.mean(list(d_by_len.values())))

    df = load_spikes_df(result_id, profile, env=env).copy()
    model = df[df.use_with_model].copy()

    # undo the library's molarity scaling to recover the raw count
    model["raw_reads"] = model.molarity_scaled_count / model.molarity_scale_factor - 1.0

    # ds templates contribute two strands, hence 2x the molecules their listed
    # molarity implies (owner-confirmed, tested)
    strand_factor = np.where(model.ss_or_ds.astype(str).str.strip() == "ds", 2.0, 1.0)
    model["mpm_input"] = model.molarity * AVOGADRO_PER_UL * strand_factor

    model["molecules"] = model.raw_reads / d
    model["recovery"] = model.molecules / model.mpm_input

    # normalise at the cell closest to the SPANK calibration point (52bp, 50% GC)
    ref = model.iloc[
        ((model.length - 52).abs() + (model.model_gc_perc - 50).abs() / 10.0).argmin()
    ]
    model["recovery_norm"] = model.recovery / ref.recovery

    diag = {
        "result_id": result_id,
        "d_by_length": d_by_len,
        "d": d,
        "d_ratio": (
            d_by_len[52] / d_by_len[75] if len(d_by_len) == 2 else np.nan
        ),
        "ref_cell": (int(ref.length), float(ref.model_gc_perc)),
        "n_cells": len(model),
    }
    cols = [
        "name", "length", "model_gc_perc", "ss_or_ds", "molarity",
        "raw_reads", "mpm_input", "molecules", "recovery", "recovery_norm",
    ]
    return model[cols].sort_values(["length", "model_gc_perc"]), diag


if __name__ == "__main__":
    import sys

    rids = [int(x) for x in sys.argv[1:]] or [200000, 200001, 200002]
    for rid in rids:
        try:
            surf, diag = recovery_surface(rid)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"{rid}: {type(exc).__name__}: {str(exc)[:70]}")
            continue

        print(f"\n=== result {rid} ===")
        print(
            f"d(52)={diag['d_by_length'].get(52, float('nan')):.4f}  "
            f"d(75)={diag['d_by_length'].get(75, float('nan')):.4f}  "
            f"ratio={diag['d_ratio']:.4f}  using d={diag['d']:.4f}"
        )
        print(f"reference cell: {diag['ref_cell']}   cells: {diag['n_cells']}")
        piv = surf.pivot_table(
            index="length", columns="model_gc_perc", values="recovery_norm"
        )
        print("\nrecovery, normalised to the reference cell:")
        print(piv.round(3).to_string())
