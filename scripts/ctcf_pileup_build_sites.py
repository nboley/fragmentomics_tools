"""Build the CTCF motif-aligned pileup site set (Phase 4 flagship, step A).

Reads the blood/hematopoietic CTCF motif TSVs, dedupes across cell types on
(contig, start, stop, strand), then filters:
  1. keep standard contigs (chr1-22, chrX) present in hg38 CONTIG_LENGTHS
  2. drop sites whose +-AGG_HALF aggregation window overlaps the blacklist
  3. drop sites too near contig edges (full TILE + model margin must fit)
  4. keep strong motifs (top-quartile tf_top_score), optional cap to N_MAX

Emits sites.npz (contig, center, strand, score) + sites_report.json with the
kept/dropped counts at each step.  CPU-only, cheap; run in the sandbox.
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/nathanboley/src/fragmentomics_tools")
from fragmentomics_tools.contig import CONTIG_LENGTHS  # noqa: E402
from fragmentomics_tools.dataframe import RegionDataFrame  # noqa: E402

TILE = 16_384
MODEL_MARGIN = 124  # (calc_input_region_size(16384) - 16384)//2 for default model
AGG_HALF = 1_000  # +-1000 bp aggregation window around motif center

CTCF_DIR = "/home/nathanboley/src/biomarker-projects/shared_data/ctcf/cell_type_merged"
CELL_TYPES = {
    "B_cell": "CTCF.B_cell.tsv",
    "CD14-positive_monocyte": "CTCF.CD14-positive_monocyte.tsv",
    "CD8-positive_alpha-beta_T_cell": "CTCF.CD8-positive,_alpha-beta_T_cell.tsv",
}
BLACKLIST_BED = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "exclusion_4_blacklist_encode_v2.bed"
)
OUT_DIR = "/efs/analytics/nathanboley/background_model/ctcf_pileup"

SCORE_QUANTILE = 0.75
N_MAX = 5000  # cap after top-quartile (deterministic strongest-N)
STD_CONTIGS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def drop_blacklisted_windows(allsites, half):
    """Drop sites whose +-half aggregation window touches the blacklist.

    Uses the library (RegionDataFrame.from_bed + drop_overlapping_regions) per
    CLAUDE.md rather than a hand-rolled interval test, so this script and
    ctcf_pileup_build_sites_noblood.py share ONE blacklist implementation.  Two
    implementations of the same filter can drift apart at boundaries (half-open
    vs inclusive, padding), and the correction path masks positions using this
    same BED -- a disagreement would silently leave masked positions inside
    "blacklist-clear" sites.

    Requires the bedtools binary on PATH (ships in the biomarker_env conda
    env's bin/).  There is deliberately NO hand-rolled fallback.
    """
    windows = pd.DataFrame(
        {
            "contig": allsites["contig"].to_numpy(),
            "start": (allsites["center"] - half).to_numpy(),
            "stop": (allsites["center"] + half + 1).to_numpy(),
        }
    )
    win_rdf = RegionDataFrame(windows, ref="hg38")
    bl_rdf = RegionDataFrame.from_bed(BLACKLIST_BED, ref="hg38")
    kept = win_rdf.drop_overlapping_regions(bl_rdf)
    return allsites.loc[allsites.index.isin(kept.index)].reset_index(drop=True)


def main():
    report = {"cell_type_rows": {}, "steps": []}

    frames = []
    for ct, fname in CELL_TYPES.items():
        df = pd.read_csv(f"{CTCF_DIR}/{fname}", sep="\t")
        df = df[["contig", "start", "stop", "strand", "tf_top_score"]].copy()
        report["cell_type_rows"][ct] = int(len(df))
        frames.append(df)
    allsites = pd.concat(frames, ignore_index=True)
    report["steps"].append(("raw_concat", int(len(allsites))))

    # dedupe on (contig, start, stop, strand); keep max score across cell types
    allsites = (
        allsites.sort_values("tf_top_score", ascending=False)
        .drop_duplicates(subset=["contig", "start", "stop", "strand"], keep="first")
        .reset_index(drop=True)
    )
    report["steps"].append(("dedupe_coord_strand", int(len(allsites))))

    # 1. standard contigs
    allsites = allsites[allsites["contig"].isin(STD_CONTIGS)].reset_index(drop=True)
    report["steps"].append(("std_contigs", int(len(allsites))))

    # motif center (17bp motif: center = start + (stop-start)//2)
    allsites["center"] = (
        allsites["start"] + (allsites["stop"] - allsites["start"]) // 2
    ).astype(int)

    hg = CONTIG_LENGTHS["hg38"]

    # 2. blacklist overlap of the +-AGG_HALF window (library, see docstring)
    allsites = drop_blacklisted_windows(allsites, AGG_HALF)
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
