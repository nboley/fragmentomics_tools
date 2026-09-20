"""Build the CONTROL ("not accessible in blood") CTCF motif site set.

Phase 4 control experiment.  This is the discriminating negative control for the
main CTCF meta-profile run (scripts/ctcf_pileup_build_sites.py): the main run uses
CTCF motifs that ARE bound/accessible in blood; this run uses motifs that are real
CTCF motifs in OTHER tissues but have NO blood occupancy.  cfDNA is blood-derived,
so at these sites there should be NO footprint in the raw data, while the SEQUENCE
(and hence the model's predicted bias) is unchanged.  If the correction is real
sequence-bias removal it should leave these flat; if it is an artifact it will
manufacture a spurious footprint peak.

Site-set cascade (counts reported at every step, like sites_report.json):
  A. START from the NON-blood / NON-hematopoietic cell types only.
  B. dedupe across those cell types on (contig, start, stop, strand); keep max score.
  C. REMOVE motifs present in ANY hematopoietic CTCF set:
       - exact (contig,start,stop,strand) matches (reported), AND
       - anything within +-200 bp of a hematopoietic site (superset of exact).
     The +-200bp overlap is done with the library (RegionDataFrame.drop_overlapping_regions
     against a +-200bp-widened hematopoietic BED), NOT hand-rolled.
  D. REMOVE motifs overlapping blood DHS (the accessibility filter) via
     RegionDataFrame.from_bed(...) + drop_overlapping_regions (CLAUDE.md sanctioned).
Then the SAME remaining filters as the main run:
  1. standard contigs (chr1-22, chrX)
  2. drop sites whose +-AGG_HALF window overlaps the blacklist
  3. drop sites too near contig edges (full TILE + model margin must fit)
  4. keep strong motifs (top-quartile tf_top_score), cap to N_MAX strongest.

Interval overlap goes through the library (pybedtools -> bedtools binary).  bedtools
ships in the biomarker_env conda env's bin/; ensure that bin is on PATH before running
(e.g. `export PATH=.../envs/biomarker_env/bin:$PATH`).  If bedtools is unavailable the
RegionDataFrame ops will raise -- there is deliberately NO hand-rolled overlap fallback
for the hematopoietic/DHS filters, so a silent divergence cannot creep in.

Emits sites.npz (same schema as the main builder) + sites_report.json.  CPU-only, cheap.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/nathanboley/src/fragmentomics_tools")
from fragmentomics_tools.contig import CONTIG_LENGTHS  # noqa: E402
from fragmentomics_tools.dataframe import RegionDataFrame  # noqa: E402

TILE = 16_384
MODEL_MARGIN = 124  # (calc_input_region_size(16384) - 16384)//2 for default model
AGG_HALF = 1_000  # +-1000 bp aggregation window around motif center

CTCF_DIR = "/home/nathanboley/src/biomarker-projects/shared_data/ctcf/cell_type_merged"
# NON-blood, NON-hematopoietic cell types -> the control motif source
NONBLOOD_CELL_TYPES = {
    "dorsolateral_prefrontal_cortex": "CTCF.dorsolateral_prefrontal_cortex.tsv",
    "mucosa_of_descending_colon": "CTCF.mucosa_of_descending_colon.tsv",
    "sigmoid_colon": "CTCF.sigmoid_colon.tsv",
    "transverse_colon": "CTCF.transverse_colon.tsv",
    "stomach": "CTCF.stomach.tsv",
    "neural_cell": "CTCF.neural_cell.tsv",
    "neural_progenitor_cell": "CTCF.neural_progenitor_cell.tsv",
    "neural_crest_cell": "CTCF.neural_crest_cell.tsv",
    "osteoblast": "CTCF.osteoblast.tsv",
}
# hematopoietic sets to EXCLUDE against (any blood occupancy disqualifies a motif)
HEMATOPOIETIC_CELL_TYPES = {
    "B_cell": "CTCF.B_cell.tsv",
    "CD14-positive_monocyte": "CTCF.CD14-positive_monocyte.tsv",
    "CD8-positive_alpha-beta_T_cell": "CTCF.CD8-positive,_alpha-beta_T_cell.tsv",
    "natural_killer_cell": "CTCF.natural_killer_cell.tsv",
    "neutrophil": "CTCF.neutrophil.tsv",
}
HEMATOPOIETIC_PAD = 200  # drop anything within +-200bp of a hematopoietic site

BLACKLIST_BED = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "exclusion_4_blacklist_encode_v2.bed"
)
BLOOD_DHS_BED = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "exclusion_1_dhs_blood_pad500.bed"
)
OUT_DIR = "/efs/analytics/nathanboley/background_model/ctcf_pileup_noblood"

SCORE_QUANTILE = 0.75
N_MAX = 5000  # cap after top-quartile (deterministic strongest-N)
STD_CONTIGS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def load_blacklist():
    bl = pd.read_csv(
        BLACKLIST_BED, sep="\t", header=None, names=["contig", "start", "stop"]
    )
    by_contig = {}
    for contig, grp in bl.groupby("contig"):
        starts = grp["start"].to_numpy()
        stops = grp["stop"].to_numpy()
        order = np.argsort(starts)
        by_contig[contig] = (starts[order], stops[order])
    return by_contig


def overlaps_blacklist(bl_by_contig, contig, lo, hi):
    """True if [lo, hi) overlaps any blacklist interval on contig."""
    if contig not in bl_by_contig:
        return False
    starts, stops = bl_by_contig[contig]
    idx = np.searchsorted(starts, hi)  # intervals with start < hi are [:idx]
    if idx == 0:
        return False
    return bool(np.any(stops[:idx] > lo))


def _write_bed(df, cols=("contig", "start", "stop", "strand")):
    """Write df[cols] to a temp headerless BED; return the path (caller unlinks)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
    df[list(cols)].to_csv(tmp.name, sep="\t", index=False, header=False)
    tmp.close()
    return tmp.name


def drop_overlapping_bed(sites_df, other_df):
    """Return the rows of sites_df (0..N-1 RangeIndex) NOT overlapping other_df,
    using the library (RegionDataFrame.from_bed + drop_overlapping_regions)."""
    sites_df = sites_df.reset_index(drop=True)
    sp = _write_bed(sites_df)
    op = _write_bed(other_df)
    try:
        sites_rdf = RegionDataFrame.from_bed(sp, ref="hg38")
        other_rdf = RegionDataFrame.from_bed(op, ref="hg38")
        kept = sites_rdf.drop_overlapping_regions(other_rdf)
        keep_idx = sorted(int(i) for i in kept.index)
    finally:
        os.unlink(sp)
        os.unlink(op)
    return sites_df.loc[keep_idx].reset_index(drop=True)


def load_ctcf(cell_types):
    frames = {}
    for ct, fname in cell_types.items():
        df = pd.read_csv(f"{CTCF_DIR}/{fname}", sep="\t")
        df = df[["contig", "start", "stop", "strand", "tf_top_score"]].copy()
        frames[ct] = df
    return frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {"cell_type_rows": {}, "hematopoietic_rows": {}, "steps": []}

    # ---- A. start from non-blood cell types -----------------------------
    nb_frames = load_ctcf(NONBLOOD_CELL_TYPES)
    for ct, df in nb_frames.items():
        report["cell_type_rows"][ct] = int(len(df))
    allsites = pd.concat(list(nb_frames.values()), ignore_index=True)
    report["steps"].append(("raw_concat_nonblood", int(len(allsites))))

    # ---- B. dedupe on (contig,start,stop,strand); keep max score --------
    allsites = (
        allsites.sort_values("tf_top_score", ascending=False)
        .drop_duplicates(subset=["contig", "start", "stop", "strand"], keep="first")
        .reset_index(drop=True)
    )
    report["steps"].append(("dedupe_coord_strand", int(len(allsites))))

    # ---- C. remove motifs present in ANY hematopoietic set --------------
    hema_frames = load_ctcf(HEMATOPOIETIC_CELL_TYPES)
    for ct, df in hema_frames.items():
        report["hematopoietic_rows"][ct] = int(len(df))
    hema = pd.concat(list(hema_frames.values()), ignore_index=True)
    hema = hema.drop_duplicates(
        subset=["contig", "start", "stop", "strand"]
    ).reset_index(drop=True)

    # C.1 report exact (contig,start,stop,strand) matches (informational)
    exact = allsites.merge(
        hema[["contig", "start", "stop", "strand"]].drop_duplicates(),
        on=["contig", "start", "stop", "strand"], how="inner",
    )
    report["hematopoietic_exact_matches"] = int(len(exact))

    # C.2 drop anything within +-HEMATOPOIETIC_PAD bp of a hematopoietic site.
    # Widen the hematopoietic intervals by +-PAD (clip start at 0), then use the
    # library overlap.  This +-PAD widening is a superset of the exact matches.
    hema_pad = hema.copy()
    hema_pad["start"] = (hema_pad["start"] - HEMATOPOIETIC_PAD).clip(lower=0)
    hema_pad["stop"] = hema_pad["stop"] + HEMATOPOIETIC_PAD
    allsites = drop_overlapping_bed(allsites, hema_pad)
    report["steps"].append(
        (f"drop_hematopoietic_pad{HEMATOPOIETIC_PAD}", int(len(allsites)))
    )

    # ---- D. remove motifs overlapping blood DHS (accessibility filter) ---
    dhs = pd.read_csv(
        BLOOD_DHS_BED, sep="\t", header=None,
        names=["contig", "start", "stop"],
        usecols=[0, 1, 2],
    )
    dhs["strand"] = "."
    allsites = drop_overlapping_bed(allsites, dhs)
    report["steps"].append(("drop_blood_dhs", int(len(allsites))))

    # ---- SAME remaining filters as the main run -------------------------
    # 1. standard contigs
    allsites = allsites[allsites["contig"].isin(STD_CONTIGS)].reset_index(drop=True)
    report["steps"].append(("std_contigs", int(len(allsites))))

    # motif center (center = start + (stop-start)//2)
    allsites["center"] = (
        allsites["start"] + (allsites["stop"] - allsites["start"]) // 2
    ).astype(int)

    hg = CONTIG_LENGTHS["hg38"]

    # 2. blacklist overlap of the +-AGG_HALF window
    bl_by_contig = load_blacklist()
    keep_bl = np.array(
        [
            not overlaps_blacklist(
                bl_by_contig, r.contig, r.center - AGG_HALF, r.center + AGG_HALF + 1
            )
            for r in allsites.itertuples()
        ]
    )
    allsites = allsites[keep_bl].reset_index(drop=True)
    report["steps"].append(("blacklist_window_clear", int(len(allsites))))

    # 3. contig-edge: full TILE window + model margin must fit in-contig
    half = TILE // 2
    pad = half + MODEL_MARGIN
    keep_edge = np.array(
        [
            (r.center - pad >= 0) and (r.center + pad <= hg[r.contig])
            for r in allsites.itertuples()
        ]
    )
    allsites = allsites[keep_edge].reset_index(drop=True)
    report["steps"].append(("contig_edge_clear", int(len(allsites))))

    # 4. strong motifs: top-quartile tf_top_score
    thresh = float(np.quantile(allsites["tf_top_score"], SCORE_QUANTILE))
    allsites = allsites[allsites["tf_top_score"] >= thresh].reset_index(drop=True)
    report["score_quantile"] = SCORE_QUANTILE
    report["score_threshold"] = thresh
    report["steps"].append(("top_quartile_score", int(len(allsites))))

    # optional deterministic cap to N_MAX strongest
    if len(allsites) > N_MAX:
        allsites = (
            allsites.sort_values("tf_top_score", ascending=False)
            .head(N_MAX)
            .reset_index(drop=True)
        )
        report["steps"].append((f"cap_top_{N_MAX}", int(len(allsites))))

    n_plus = int((allsites["strand"] == "+").sum())
    n_minus = int((allsites["strand"] == "-").sum())
    report["n_sites"] = int(len(allsites))
    report["n_plus"] = n_plus
    report["n_minus"] = n_minus

    np.savez(
        f"{OUT_DIR}/sites.npz",
        contig=allsites["contig"].to_numpy().astype("U6"),
        center=allsites["center"].to_numpy().astype(np.int64),
        strand=allsites["strand"].to_numpy().astype("U1"),
        score=allsites["tf_top_score"].to_numpy().astype(np.float64),
        TILE=np.int64(TILE),
        MODEL_MARGIN=np.int64(MODEL_MARGIN),
        AGG_HALF=np.int64(AGG_HALF),
    )
    with open(f"{OUT_DIR}/sites_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
