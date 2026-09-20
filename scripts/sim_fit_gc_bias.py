"""Fit the empirical GC x fragment-length bias surface from SPARK spike-in oligos.

This produces the GROUND-TRUTH GC bias used by the fragment simulator.

Where the numbers come from
---------------------------
Karius spikes synthetic "SPARK" oligos of known length, GC% and molarity into
libraries.  Their recovered counts measure how efficiently a fragment of a given
(length, GC) survives library prep + sequencing.  Per-oligo counts live in each
result's pipeline metrics under names like ``SPARK-v4-024-30-ss``.

The production model (``karius_biomarker.flgc.GCFlDistModel``) stores
``correction_factor = log(molarity_scaled_count / grand_mean)`` and applies
``weight = exp(-cf)`` -- i.e. it DIVIDES the bias out.

>>> IMPORTANT SIGN CONVENTION <<<
For SIMULATION we need the bias itself, not the correction:

    bias(L, GC) = exp(+cf) = 1 / weight

Multiplying acceptance by the production ``predict()`` output would *anti-bias*
the simulation -- exactly backwards.  This script therefore emits ``bias``.

Caveats carried into the output
-------------------------------
* All model oligos share molarity (5e-13), so the molarity scaling cancels in
  the grand-mean ratio; we use raw counts + 1 pseudocount, matching the
  production ``molarity_scaled_count`` up to that common factor.
* The LENGTH axis is contaminated by single-stranded oligo degradation (longer
  ss oligos are recovered less because they degrade, not purely because of
  library bias), and only lengths 24-75bp have GC coverage.  Real cfDNA runs to
  167bp+.  The simulator therefore uses the GC axis of this surface and takes
  the fragment-length distribution empirically from real data instead.
* GC is computed over the WHOLE FRAGMENT, in percent, matching production.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd

AWS = "/home/nathanboley/miniconda3/envs/biomarker_env/bin/aws"
SPIKE_DIR = ("/efs/analytics/nathanboley/mtdna_chimerism/transplant_poc/"
             "full_cohort_results/spikein")
PAT = re.compile(r"^SPARK-v4-(\d+)-(\d+)-(ss|ds)$")
# production clamps the correction weight to [0.10, 3.0]; the matching bias clamp
BIAS_LO, BIAS_HI = 1.0 / 3.0, 10.0


def result_ids(limit):
    ids = []
    for fn in sorted(os.listdir(SPIKE_DIR)):
        if not fn.endswith(".spikein.json"):
            continue
        try:
            d = json.load(open(os.path.join(SPIKE_DIR, fn)))
        except Exception:
            continue
        if d.get("status") == "ok" and d.get("result_id"):
            ids.append(int(d["result_id"]))
        if len(ids) >= limit:
            break
    return ids


def spark_counts(result_id, tmpdir):
    """Per-oligo ss SPARK counts for one result, or None if unavailable."""
    dst = os.path.join(tmpdir, f"m{result_id}.json")
    uri = f"s3://results.prod.kariusdx.com/{result_id}/analyze-alignments/metrics.json"
    r = subprocess.run([AWS, "s3", "cp", uri, dst, "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        return None
    try:
        d = json.load(open(dst))
    except Exception:
        return None
    rows = []
    for m in d:
        mm = PAT.match(str(m.get("name", "")))
        if mm and m.get("type") == "count" and mm.group(3) == "ss":
            rows.append(dict(length=int(mm.group(1)), gc=int(mm.group(2)),
                             count=float(m["value"])))
    os.remove(dst)
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--out", default="/efs/analytics/nathanboley/background_model/"
                                     "simulation/gc_bias_grid.json")
    args = ap.parse_args()

    ids = result_ids(args.n_samples)
    print(f"candidate results: {len(ids)}")

    cfs, used = [], []
    with tempfile.TemporaryDirectory() as td:
        for rid in ids:
            df = spark_counts(rid, td)
            if df is None or df.empty:
                continue
            piv = df.pivot_table(index="length", columns="gc", values="count")
            # production restricts the interpolation grid to length <= 75 and
            # drops GC outside [25,75] (too few/unreliable oligos there)
            piv = piv.loc[piv.index <= 75, [c for c in piv.columns if 25 <= c <= 75]]
            if piv.isna().all().all():
                continue
            piv = piv.fillna(piv.mean().mean()) + 1.0        # pseudocount
            cf = np.log(piv / piv.values.mean())             # vs GRAND mean
            cfs.append(cf)
            used.append(rid)
            print(f"  {rid}: ok ({len(used)})")

    if not cfs:
        raise SystemExit("no SPARK grids recovered")

    # average the log-surface across samples -> geometric mean of the bias
    cf_mean = sum(cfs) / len(cfs)
    cf_sd = (sum((c - cf_mean) ** 2 for c in cfs) / max(len(cfs) - 1, 1)) ** 0.5

    bias = np.exp(cf_mean).clip(BIAS_LO, BIAS_HI)

    # --- the 2-D surface the simulator uses -----------------------------------
    # GC bias INTERACTS with fragment length: the fitted GC slope (70% vs 30%)
    # runs -0.425 (24bp, GC-disfavouring) -> +1.406 (75bp, strongly GC-favouring),
    # every contrast significant at >2 SEM.  Collapsing to a GC marginal would
    # average a sign flip away, so we keep the full grid.
    #
    # We row-centre (divide each length row by its own mean) to strip the LENGTH
    # MAIN EFFECT, which is contaminated by single-stranded oligo degradation --
    # longer ss SPARKs are recovered less because they degrade, not because of
    # library bias.  The simulator draws lengths from the empirical real-data
    # distribution instead, so re-imposing a length marginal here would also
    # double-count.  What survives is purely the GC x length INTERACTION.
    bias2d = bias.div(bias.mean(axis=1), axis=0)

    gc_curve = np.exp(cf_mean.mean(axis=0))
    gc_curve = gc_curve / gc_curve.mean()

    out = dict(
        n_samples_used=len(used), result_ids=used,
        lengths=[int(x) for x in bias.index],
        gc_percents=[int(x) for x in bias.columns],
        bias_grid=bias.values.tolist(),
        bias_grid_2d_row_centred=bias2d.values.tolist(),
        cf_sd_grid=cf_sd.values.tolist(),
        cf_sem_grid=(cf_sd / np.sqrt(len(cfs))).values.tolist(),
        gc_curve={int(g): float(v) for g, v in gc_curve.items()},
        gc_slope_by_length={int(l): float(bias2d.loc[l].iloc[-1] / bias2d.loc[l].iloc[0])
                            for l in bias2d.index},
        sign_convention="bias = exp(+cf) = 1/production_weight",
        surface_for_simulation="bias_grid_2d_row_centred",
        extrapolation_policy=(
            "HOLD at the grid edges: L<24 uses the 24bp row, L>75 uses the 75bp "
            "row, GC<30 uses 30%, GC>70 uses 70%. The grid has no coverage above "
            "75bp (real cfDNA reaches 167bp+), and the GC slope is still rising "
            "there, so extrapolating the trend would invent a very strong bias "
            "that the data cannot support. Holding is conservative and explicit."
        ),
        clamp=[BIAS_LO, BIAS_HI],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\nfitted from {len(used)} samples -> {args.out}")
    print("\n2-D bias surface used by the simulator (row-centred; rows=length, cols=GC%):")
    print("        " + "".join(f"{g:>8d}" for g in out["gc_percents"]))
    for i, l in enumerate(out["lengths"]):
        print(f"{l:>5d}bp " + "".join(f"{v:8.3f}" for v in out["bias_grid_2d_row_centred"][i]))
    print("\nGC slope (70%/30%) by length -- the interaction:")
    for l, v in sorted(out["gc_slope_by_length"].items()):
        print(f"  {l:>3d}bp : {v:.3f}")
    print(f"\nbias grid range: {bias.values.min():.3f} .. {bias.values.max():.3f}")
    print(f"median across-sample sd of log-bias: {np.median(cf_sd.values):.3f}")


if __name__ == "__main__":
    main()
