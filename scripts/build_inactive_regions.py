#!/usr/bin/env python
"""Build the INACTIVE TRAINING REGION SET (+ positive controls) for background-model-v2.

Phase 3, task 1.  Implements the USER-DECIDED "strict quiet genome" definition:

  training regions = main hg38 contigs (chr1-22,X) MINUS the union of
    1. blood/hematopoietic DHS index regions, padded +/-500 bp
    2. blood-expressed genes (expression >= 0.1), gene bodies padded +/-2 kb
    3. annotated CTCF/TF binding sites, padded +/-500 bp
    4. the standard blacklist (ENCODE hg38-blacklist.v2 on this host; the v1
       mappability.simple_repeats BED is ABSENT -- see QC report "missing")
    5. the positive-control regions (never in training)

  From the remainder: ~5,000 NON-OVERLAPPING random 16,384 bp tiles, seeded
  (numpy default_rng(1337)); each tile sits entirely inside the remainder and
  its full margined extent (tile +/-2,176 bp = the plumbing L_SEQ half-margin
  JITTER+RF_BUDGET = 128+2048) lies inside the contig.

Positive controls (separate BEDs, NEVER training):
  (a) CTCF binding sites (v1 build_ctcf_binding_sites classified set, resized to 1024)
  (b) immune/epithelial marker genes (markers.immune_vs_epithelial.tsv, v1 filters)

Deterministic, re-runnable.  Writes the BEDs (into gitignored data/region_sets/)
and the QC report docs/qc/region_set_qc.md.

This script does NOT touch background_model_core.py or the background_model/ package,
and does NOT run the preprocess pipeline.  It only reads annotations and writes
BEDs + a report.

Environment: /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
Run:  PYTHONPATH=. .../python scripts/build_inactive_regions.py
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pybedtools
import pysam

# pybedtools invokes the `bedtools` binary by name; it ships in this conda env's
# bin dir but that dir may not be on PATH.  Point pybedtools at it explicitly.
_ENV_BIN = os.path.dirname(sys.executable)
if os.path.exists(os.path.join(_ENV_BIN, "bedtools")):
    pybedtools.helpers.set_bedtools_path(_ENV_BIN)

# ============================================================================
# CONFIG (all knobs at the top)
# ============================================================================

REPO = "/home/nathanboley/src/fragmentomics_tools"
OUT_DIR = os.path.join(REPO, "data", "region_sets")
QC_REPORT = os.path.join(REPO, "docs", "qc", "region_set_qc.md")
PBT_TMP = os.path.join(OUT_DIR, "_pbt_tmp")

# --- annotation inputs (located on this host; v1 paths were absent) ---------
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
FAI = FASTA + ".fai"
RMSK = "/efs/analytics/nathanboley/data_resources/genome/hg38_rmsk.txt.gz"
BLACKLIST = "/efs/analytics/nathanboley/data_resources/genome/hg38-blacklist.v2.bed.gz"
DHS = ("/home/nathanboley/src/biomarker-projects/"
       "DHS_Index_and_Vocabulary_hg38_WM20190703.min2samp.bed")
CTCF_DIR = "/home/nathanboley/src/biomarker-projects/shared_data/ctcf/cell_type_merged"
SC_COUNTS = ("/home/nathanboley/src/biomarker-projects/projects/ibd_consolidated/"
             "tech_dev/strand_asymmetry/sc_rnaseq.counts.hematopoetic.tsv")
GENCODE_GFF3 = ("/efs/analytics/nathanboley/data_resources/gencode/"
                "gencode.v49.basic.annotation.gff3.gz")
MARKERS = ("/home/nathanboley/src/biomarker-projects/projects/ibd_consolidated/"
           "functional_elements/scrnaseq/markers.immune_vs_epithelial.tsv")

# --- geometry / sampling ----------------------------------------------------
MAIN_CONTIGS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
TILE = 16_384
MARGIN = 2_176                 # JITTER(128) + RF_BUDGET(2048); L_SEQ half-margin
N_TILES = 5_000
SEED = 1337

# --- padding (per USER-DECIDED definition) ---------------------------------
DHS_PAD = 500
GENE_PAD = 2_000
TF_PAD = 500
POSCTRL_PAD = 500              # safety pad when unioning pos-controls into exclusions

# --- filter thresholds ------------------------------------------------------
EXPRESSION_THRESHOLD = 0.1     # blood-expressed gene cutoff (v1 "expression >= 0.1")
DHS_BLOOD_COMPONENTS = {"Lymphoid", "Myeloid / erythroid"}
STORE_BLACKLIST_EXPANSION = 120  # PlumbingConfig.blacklist_expansion default

# HematopoieticGeneExpression cluster weights (verbatim from
# biomarker-projects fragmentomics.data.HematopoieticGeneExpression).  The 10
# selected count columns map positionally to these clusters.
HGE_CLUSTER_WEIGHTS = {
    "Early_Erythroid_Cells": 0.17,
    "Late_Erythroid_Cells": 0.15,
    "Myeloid_Progenitor": 0.10,
    "Lymphoid_Progenitor_1": 0.08,
    "Granulocyte-Monocyte_Progenitor": 0.16,
    "Neutrophil": 0.21,
    "CD14+_Monocyte_Cells_1": 0.05,
    "CD14+_Monocyte_Cells_2": 0.04,
    "CD16+_Monocyte_Cells": 0.04,
    "Lymphoid_Progenitor_2": 0.08,
}
HGE_SELECTED_COLS = ["02", "03", "05", "06", "07", "08", "11", "12", "13", "15"]

# v1 build_ctcf_binding_sites classification column groups
CTCF_NEURAL = ["neural_progenitor_cell", "neural_cell", "dorsolateral_prefrontal_cortex"]
CTCF_COLON = ["stomach", "transverse_colon", "sigmoid_colon"]
CTCF_BLOOD = ["natural_killer_cell", "B_cell", "CD14-positive_monocyte"]

# v1 build_marker_gene_rdf filters (immune vs epithelial markers)
MARKER_UP_QUERY = "p_val_adj < 0.01 and avg_log2FC < -3"
MARKER_DOWN_QUERY = "p_val_adj < 0.01 and avg_log2FC > 7"


# ============================================================================
# helpers
# ============================================================================

def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sizes() -> dict:
    sizes = {}
    with open(FAI) as f:
        for line in f:
            name, length = line.split("\t")[:2]
            if name in MAIN_CONTIGS:
                sizes[name] = int(length)
    missing = [c for c in MAIN_CONTIGS if c not in sizes]
    assert not missing, f"contigs missing from .fai: {missing}"
    return sizes


def bt_from_df(df: pd.DataFrame) -> pybedtools.BedTool:
    """DataFrame (chrom,start,end,[extra...]) -> sorted BedTool restricted to main contigs."""
    df = df[df["chrom"].isin(MAIN_CONTIGS)].copy()
    df["start"] = df["start"].astype(int).clip(lower=0)
    df["end"] = df["end"].astype(int)
    df = df[df["end"] > df["start"]]
    return pybedtools.BedTool.from_dataframe(df).sort(g=GENOME_FILE)


def slop_merge(bt: pybedtools.BedTool, pad: int) -> pybedtools.BedTool:
    return bt.slop(b=pad, g=GENOME_FILE).sort(g=GENOME_FILE).merge()


def bp(bt: pybedtools.BedTool) -> int:
    return int(sum(iv.length for iv in bt.merge()))


# ============================================================================
# exclusion-set builders
# ============================================================================

def build_dhs_blood() -> pybedtools.BedTool:
    df = pd.read_csv(DHS, sep="\t")
    df = df[df["component"].isin(DHS_BLOOD_COMPONENTS)]
    df = df.rename(columns={"seqname": "chrom", "end": "stop"})
    out = df[["chrom", "start", "stop"]].rename(columns={"stop": "end"})
    return slop_merge(bt_from_df(out), DHS_PAD)


def build_expression_series() -> pd.Series:
    """Reconstruct HematopoieticGeneExpression weighted expression per gene_name."""
    counts = pd.read_table(SC_COUNTS, index_col=0)
    counts.columns = [str(c) for c in counts.columns]
    tpms = counts / (counts.sum() / 1e6)
    sub = tpms[HGE_SELECTED_COLS]
    weights = np.array(list(HGE_CLUSTER_WEIGHTS.values()))
    expr = pd.Series(sub.values @ weights, index=sub.index, name="expression")
    # gene_name index may repeat in the raw counts; keep the max
    expr = expr.groupby(level=0).max()
    return expr


def load_gencode_genes() -> pd.DataFrame:
    """gencode 'gene' features -> gene_name, chrom, start(0-based), end, strand, gene_type."""
    rows = []
    name_re = re.compile(r"gene_name=([^;]+)")
    type_re = re.compile(r"gene_type=([^;]+)")
    import gzip
    with gzip.open(GENCODE_GFF3, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0]
            if chrom not in MAIN_CONTIGS:
                continue
            start = int(parts[3]) - 1   # gff3 1-based inclusive -> 0-based half-open
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]
            nm = name_re.search(attrs)
            ty = type_re.search(attrs)
            rows.append((nm.group(1) if nm else None, chrom, start, end, strand,
                         ty.group(1) if ty else None))
    return pd.DataFrame(rows, columns=["gene_name", "chrom", "start", "end",
                                       "strand", "gene_type"])


def build_expressed_genes(expr: pd.Series, genes: pd.DataFrame) -> pybedtools.BedTool:
    expressed = expr[expr >= EXPRESSION_THRESHOLD].index
    g = genes[genes["gene_name"].isin(expressed)]
    out = g[["chrom", "start", "end"]]
    return slop_merge(bt_from_df(out), GENE_PAD)


def load_all_ctcf_sites() -> pd.DataFrame:
    """Union of every CTCF binding-site row across all cell-type files."""
    frames = []
    for fname in sorted(os.listdir(CTCF_DIR)):
        if not fname.endswith(".tsv"):
            continue
        df = pd.read_table(os.path.join(CTCF_DIR, fname),
                           usecols=["contig", "start", "stop"])
        frames.append(df.rename(columns={"contig": "chrom", "stop": "end"}))
    allsites = pd.concat(frames, ignore_index=True).drop_duplicates()
    return allsites


def build_tf_sites(all_ctcf: pd.DataFrame) -> pybedtools.BedTool:
    return slop_merge(bt_from_df(all_ctcf[["chrom", "start", "end"]]), TF_PAD)


def build_blacklist() -> pybedtools.BedTool:
    df = pd.read_csv(BLACKLIST, sep="\t", header=None,
                     names=["chrom", "start", "end", "label"])
    return bt_from_df(df[["chrom", "start", "end"]]).merge()


# ============================================================================
# positive controls
# ============================================================================

def build_ctcf_positive_control() -> pybedtools.BedTool:
    """Faithful reimplementation of v1 build_ctcf_binding_sites().resize_regions(1024).

    Reads each cell-type TSV, keeps top-20000 by tf_top_score, outer-joins on
    (contig,start,stop,strand), binarizes, drops neutrophil, classifies each site
    into all/neural/colon/blood, then resizes each surviving site to width 1024.
    """
    index_cols = ["contig", "start", "stop", "strand"]
    dfs = []
    for fname in sorted(os.listdir(CTCF_DIR)):
        if not fname.endswith(".tsv"):
            continue
        cell_type = fname.split(".")[1]
        df = pd.read_table(os.path.join(CTCF_DIR, fname))
        df = df[["contig", "start", "stop", "strand", "tf_top_score"]].set_index(index_cols)
        df = df.rename(columns={"tf_top_score": cell_type})
        df = df.sort_values(df.columns[0], ascending=False).head(20000).sort_index()
        dfs.append(df)

    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.join(dfs[i], how="outer")

    df = pd.DataFrame(df.fillna(0).sort_index())
    df = (df > 1e-6).astype(int)
    if "neutrophil" in df.columns:
        df = df.drop(columns="neutrophil")

    all_bnd = df.sum(axis=1) == df.shape[1]

    all_neural = df.loc[:, CTCF_NEURAL].sum(axis=1) >= len(CTCF_NEURAL)
    no_non_neural = df.loc[:, CTCF_COLON + CTCF_BLOOD].sum(axis=1) <= 2
    only_neural = all_neural & no_non_neural

    all_colon = df.loc[:, CTCF_COLON].sum(axis=1) >= len(CTCF_COLON)
    no_non_colon = df.loc[:, CTCF_NEURAL + CTCF_BLOOD].sum(axis=1) <= 2
    only_colon = all_colon & no_non_colon

    all_blood = df.loc[:, CTCF_BLOOD].sum(axis=1) >= len(CTCF_BLOOD)
    no_non_blood = df.loc[:, CTCF_NEURAL + CTCF_COLON].sum(axis=1) <= 2
    only_blood = all_blood & no_non_blood

    keep = all_bnd | only_neural | only_colon | only_blood
    sites = df.index[keep].to_frame(index=False)  # contig,start,stop,strand

    # resize_regions(1024): center +/- 512
    center = ((sites["start"] + sites["stop"]) // 2).astype(int)
    out = pd.DataFrame({
        "chrom": sites["contig"],
        "start": (center - 512),
        "end": (center + 512),
    })
    return bt_from_df(out).merge()


def build_marker_positive_control(genes: pd.DataFrame) -> pybedtools.BedTool:
    """v1 build_marker_gene_rdf: immune/epithelial marker genes -> gene bodies."""
    markers = pd.read_table(MARKERS, sep=" ", index_col=0)
    up = markers.query(MARKER_UP_QUERY)
    down = markers.query(MARKER_DOWN_QUERY)
    marker_names = set(up.index) | set(down.index)
    g = genes[genes["gene_name"].isin(marker_names)]
    out = g[["chrom", "start", "end"]]
    return bt_from_df(out).merge()


# ============================================================================
# tile sampling
# ============================================================================

def margin_windows(sizes: dict) -> pybedtools.BedTool:
    rows = [(c, MARGIN, L - MARGIN) for c, L in sizes.items() if L - MARGIN > MARGIN]
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return bt_from_df(df)


def sample_nonoverlapping_tiles(placeable: pybedtools.BedTool, seed: int,
                                n_target: int) -> list:
    """Draw up to n_target non-overlapping TILE-bp tiles from `placeable` intervals.

    Deterministic: interval chosen with probability proportional to its number of
    valid start positions, then a uniform start within it; the interval is split
    into its (>=TILE) left/right remainders and sampling continues.  Non-overlap is
    guaranteed by construction (a placed tile is removed from the free space)."""
    rng = np.random.default_rng(seed)
    free = [[iv.chrom, iv.start, iv.end] for iv in placeable
            if iv.end - iv.start >= TILE]
    free.sort(key=lambda x: (MAIN_CONTIGS.index(x[0]), x[1]))
    placed = []
    while len(placed) < n_target and free:
        weights = np.array([f[2] - f[1] - TILE + 1 for f in free], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            break
        cum = np.cumsum(weights)
        r = rng.random() * total
        idx = int(np.searchsorted(cum, r, side="right"))
        idx = min(idx, len(free) - 1)
        c, s, e = free[idx]
        start = s + int(rng.integers(0, (e - TILE) - s + 1))
        stop = start + TILE
        placed.append((c, start, stop))
        rem = []
        if start - s >= TILE:
            rem.append([c, s, start])
        if e - stop >= TILE:
            rem.append([c, stop, e])
        free[idx:idx + 1] = rem
    placed.sort(key=lambda x: (MAIN_CONTIGS.index(x[0]), x[1]))
    return placed


def tiles_to_bt(tiles: list) -> pybedtools.BedTool:
    df = pd.DataFrame(
        [(c, s, e, f"tile_{i}") for i, (c, s, e) in enumerate(tiles)],
        columns=["chrom", "start", "end", "name"],
    )
    return pybedtools.BedTool.from_dataframe(df).sort(g=GENOME_FILE)


# ============================================================================
# QC computations
# ============================================================================

def gc_n_per_tile(tiles: list) -> pd.DataFrame:
    fa = pysam.FastaFile(FASTA)
    gc, nfrac = [], []
    for c, s, e in tiles:
        seq = fa.fetch(c, s, e).upper()
        n = seq.count("N")
        g = seq.count("G") + seq.count("C")
        acgt = len(seq) - n
        gc.append(g / acgt if acgt > 0 else np.nan)
        nfrac.append(n / len(seq))
    fa.close()
    return pd.DataFrame({"gc": gc, "n_frac": nfrac})


def load_rmsk() -> pd.DataFrame:
    # UCSC rmsk columns (0-based): 5 genoName, 6 genoStart, 7 genoEnd, 11 repClass, 12 repFamily
    df = pd.read_csv(RMSK, sep="\t", header=None,
                     usecols=[5, 6, 7, 11, 12],
                     names=["chrom", "start", "end", "repClass", "repFamily"])
    return df[df["chrom"].isin(MAIN_CONTIGS)]


def per_tile_masked_fraction(tiles_bt: pybedtools.BedTool,
                             feature_merged: pybedtools.BedTool,
                             tiles: list) -> pd.Series:
    """Fraction of each tile covered by a (pre-merged) feature set."""
    inter = tiles_bt.intersect(feature_merged, wo=True)
    masked = {}
    for iv in inter:
        fields = iv.fields
        name = fields[3]
        overlap = int(fields[-1])
        masked[name] = masked.get(name, 0) + overlap
    frac = []
    for i in range(len(tiles)):
        frac.append(masked.get(f"tile_{i}", 0) / TILE)
    return pd.Series(frac)


def class_composition(tiles_bt: pybedtools.BedTool,
                      rmsk_bt: pybedtools.BedTool, n_tiles: int) -> pd.Series:
    """bp of rmsk annotation overlapping the tiles, grouped by repClass, as fraction
    of total tile bp.  (Raw annotation overlap; rare within-class overlaps not merged.)"""
    inter = tiles_bt.intersect(rmsk_bt, wo=True)
    by_class = {}
    for iv in inter:
        fields = iv.fields
        rep_class = fields[7]     # tile(4) + rmsk chrom,start,end,repClass -> idx 7
        overlap = int(fields[-1])
        by_class[rep_class] = by_class.get(rep_class, 0) + overlap
    total = n_tiles * TILE
    return (pd.Series(by_class) / total).sort_values(ascending=False)


# ============================================================================
# main
# ============================================================================

GENOME_FILE = None  # set in main (a bedtools genome file for sort/slop/complement)


def write_bed(bt: pybedtools.BedTool, path: str):
    bt.saveas(path)


def main():
    global GENOME_FILE
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(QC_REPORT), exist_ok=True)
    os.makedirs(PBT_TMP, exist_ok=True)
    pybedtools.set_tempdir(PBT_TMP)

    sizes = load_sizes()
    total_genome_bp = sum(sizes.values())

    # bedtools genome file (contig<TAB>length), main contigs only
    GENOME_FILE = os.path.join(OUT_DIR, "hg38.main.genome")
    with open(GENOME_FILE, "w") as f:
        for c in MAIN_CONTIGS:
            f.write(f"{c}\t{sizes[c]}\n")

    print("[1/9] DHS blood/hematopoietic ...")
    dhs_bt = build_dhs_blood()

    print("[2/9] blood-expressed genes ...")
    expr = build_expression_series()
    genes = load_gencode_genes()
    genes_bt = build_expressed_genes(expr, genes)

    print("[3/9] CTCF/TF sites ...")
    all_ctcf = load_all_ctcf_sites()
    tf_bt = build_tf_sites(all_ctcf)

    print("[4/9] blacklist ...")
    bl_bt = build_blacklist()

    print("[5/9] positive controls ...")
    ctcf_pc = build_ctcf_positive_control()
    marker_pc = build_marker_positive_control(genes)
    write_bed(ctcf_pc, os.path.join(OUT_DIR, "positive_control_ctcf.bed"))
    write_bed(marker_pc, os.path.join(OUT_DIR, "positive_control_markers.bed"))

    # positive controls also excluded from training (padded superset -> guarantees
    # the zero-overlap sanity check even with a safety margin)
    ctcf_pc_pad = slop_merge(ctcf_pc, POSCTRL_PAD)
    marker_pc_pad = slop_merge(marker_pc, GENE_PAD)

    print("[6/9] building exclusion union + remainder ...")
    filters = [
        ("1_dhs_blood_pad500", dhs_bt),
        ("2_expressed_genes_pad2kb", genes_bt),
        ("3_ctcf_tf_pad500", tf_bt),
        ("4_blacklist_encode_v2", bl_bt),
        ("5a_posctrl_ctcf_pad500", ctcf_pc_pad),
        ("5b_posctrl_markers_pad2kb", marker_pc_pad),
    ]
    for name, bt in filters:
        write_bed(bt, os.path.join(OUT_DIR, f"exclusion_{name}.bed"))

    # per-filter and cumulative exclusion accounting (restricted to main contigs)
    genome_bt = pybedtools.BedTool.from_dataframe(pd.DataFrame(
        [(c, 0, sizes[c]) for c in MAIN_CONTIGS],
        columns=["chrom", "start", "end"])).sort(g=GENOME_FILE)
    accounting = []
    cumulative = None
    for name, bt in filters:
        clipped = bt.intersect(genome_bt).sort(g=GENOME_FILE).merge()
        f_bp = bp(clipped)
        cumulative = clipped if cumulative is None else \
            cumulative.cat(clipped, postmerge=True).sort(g=GENOME_FILE).merge()
        cum_bp = bp(cumulative)
        accounting.append((name, f_bp, f_bp / total_genome_bp * 100,
                           cum_bp, cum_bp / total_genome_bp * 100))
    union_bt = cumulative
    union_bp = bp(union_bt)

    write_bed(union_bt, os.path.join(OUT_DIR, "exclusion_union.bed"))

    remainder = genome_bt.subtract(union_bt).sort(g=GENOME_FILE).merge()
    remainder_bp = bp(remainder)
    write_bed(remainder, os.path.join(OUT_DIR, "remainder.bed"))

    print("[7/9] sampling training tiles ...")
    placeable = remainder.intersect(margin_windows(sizes)).sort(g=GENOME_FILE).merge()
    train_tiles = sample_nonoverlapping_tiles(placeable, SEED, N_TILES)
    train_bt = tiles_to_bt(train_tiles)
    write_bed(train_bt.cut([0, 1, 2]).saveas(),
              os.path.join(OUT_DIR, "training_tiles.bed"))

    print("[7b/9] sampling genome-wide background tiles ...")
    bg_placeable = margin_windows(sizes)
    bg_tiles = sample_nonoverlapping_tiles(bg_placeable, SEED, len(train_tiles))
    bg_bt = tiles_to_bt(bg_tiles)

    # ------------------------------------------------------------------ sanity
    print("[8/9] sanity checks ...")
    for name, bt in filters:
        n_ov = train_bt.intersect(bt, u=True).count()
        assert n_ov == 0, f"training tiles overlap exclusion {name}: {n_ov}"
    assert train_bt.intersect(ctcf_pc, u=True).count() == 0, \
        "training tiles overlap positive-control CTCF"
    assert train_bt.intersect(marker_pc, u=True).count() == 0, \
        "training tiles overlap positive-control markers"
    # mutual non-overlap of training tiles
    merged_tiles = train_bt.cut([0, 1, 2]).sort(g=GENOME_FILE).merge()
    assert bp(merged_tiles) == len(train_tiles) * TILE, \
        "training tiles overlap each other"

    # ------------------------------------------------------------ QC: repeats
    print("[9/9] repeat / GC / N QC ...")
    rmsk = load_rmsk()
    rmsk_bt = bt_from_df(rmsk)
    rmsk_all_merged = rmsk_bt.merge()
    simple_low = rmsk[rmsk["repClass"].isin(["Simple_repeat", "Low_complexity"])]
    simple_low_merged = bt_from_df(simple_low).merge()

    train_rep = per_tile_masked_fraction(train_bt, rmsk_all_merged, train_tiles)
    bg_rep = per_tile_masked_fraction(bg_bt, rmsk_all_merged, bg_tiles)
    train_sl = per_tile_masked_fraction(train_bt, simple_low_merged, train_tiles)
    bg_sl = per_tile_masked_fraction(bg_bt, simple_low_merged, bg_tiles)

    train_comp = class_composition(train_bt, rmsk_bt, len(train_tiles))
    bg_comp = class_composition(bg_bt, rmsk_bt, len(bg_tiles))

    # blacklist-masked fraction of training bp at store-build time (store expands
    # the blacklist by STORE_BLACKLIST_EXPANSION bp before masking)
    bl_expanded = slop_merge(bl_bt, STORE_BLACKLIST_EXPANSION)
    train_bl = per_tile_masked_fraction(train_bt, bl_expanded, train_tiles)

    train_gcn = gc_n_per_tile(train_tiles)
    bg_gcn = gc_n_per_tile(bg_tiles)

    # chromosome coverage
    chrom_counts = pd.Series([c for c, _, _ in train_tiles]).value_counts()

    # md5s of emitted BEDs
    emitted = {
        "training_tiles.bed": os.path.join(OUT_DIR, "training_tiles.bed"),
        "positive_control_ctcf.bed": os.path.join(OUT_DIR, "positive_control_ctcf.bed"),
        "positive_control_markers.bed": os.path.join(OUT_DIR, "positive_control_markers.bed"),
    }
    md5s = {k: md5(v) for k, v in emitted.items()}
    counts = {k: pybedtools.BedTool(v).count() for k, v in emitted.items()}

    # ------------------------------------------------------------ write report
    write_report(
        sizes=sizes, total_genome_bp=total_genome_bp, accounting=accounting,
        union_bp=union_bp, remainder_bp=remainder_bp,
        train_tiles=train_tiles, bg_tiles=bg_tiles,
        train_rep=train_rep, bg_rep=bg_rep, train_sl=train_sl, bg_sl=bg_sl,
        train_comp=train_comp, bg_comp=bg_comp, train_bl=train_bl,
        train_gcn=train_gcn, bg_gcn=bg_gcn, chrom_counts=chrom_counts,
        md5s=md5s, counts=counts,
        n_expressed=int((expr >= EXPRESSION_THRESHOLD).sum()),
        n_ctcf_pc=ctcf_pc.count(), n_marker_pc=marker_pc.count(),
    )
    print(f"\nDONE. wrote {QC_REPORT}")
    print(f"training tiles: {len(train_tiles)}  ({len(train_tiles)*TILE:,} bp)")
    for k in emitted:
        print(f"  {k}: n={counts[k]}  md5={md5s[k]}")


def _fmt_pct(x):
    return f"{x*100:.2f}%"


def _dist(s: pd.Series):
    return (f"mean {s.mean():.4f}, median {s.median():.4f}, "
            f"p10 {s.quantile(.1):.4f}, p90 {s.quantile(.9):.4f}, max {s.max():.4f}")


def write_report(**k):
    sizes = k["sizes"]; total = k["total_genome_bp"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    acc_rows = "\n".join(
        f"| {name} | {f_bp:,} | {f_pct:.2f}% | {cum_bp:,} | {cum_pct:.2f}% |"
        for name, f_bp, f_pct, cum_bp, cum_pct in k["accounting"]
    )

    train_rep = k["train_rep"]; bg_rep = k["bg_rep"]
    train_sl = k["train_sl"]; bg_sl = k["bg_sl"]
    rep_enriched = train_rep.mean() / bg_rep.mean() if bg_rep.mean() > 0 else float("nan")
    interp = ("repeat-ENRICHED" if train_rep.mean() > bg_rep.mean() * 1.02
              else "repeat-DEPLETED" if train_rep.mean() < bg_rep.mean() * 0.98
              else "repeat-neutral")

    comp = pd.DataFrame({"training": k["train_comp"], "background": k["bg_comp"]}).fillna(0.0)
    comp = comp.sort_values("training", ascending=False).head(15)
    comp_rows = "\n".join(
        f"| {cls} | {row.training*100:.3f}% | {row.background*100:.3f}% |"
        for cls, row in comp.iterrows()
    )

    chrom_rows = "\n".join(
        f"| {c} | {int(k['chrom_counts'].get(c, 0))} | "
        f"{int(k['chrom_counts'].get(c, 0))*16384:,} |"
        for c in MAIN_CONTIGS
    )

    md5_rows = "\n".join(
        f"| `{name}` | {k['counts'][name]:,} | `{k['md5s'][name]}` |"
        for name in k["md5s"]
    )

    train_gcn = k["train_gcn"]; bg_gcn = k["bg_gcn"]
    n_train = len(k["train_tiles"])
    train_ngap = int((train_gcn["n_frac"] > 0.5).sum())
    train_nany = int((train_gcn["n_frac"] > 0.01).sum())
    bg_ngap = int((bg_gcn["n_frac"] > 0.5).sum())

    report = f"""# Inactive Training Region-Set QC Report

Generated by `scripts/build_inactive_regions.py` on {now}.
Project: background-model-v2 (Phase 3, task 1).  Deterministic (numpy `default_rng({SEED})`).

This report is machine-generated; re-running the builder regenerates it verbatim
(inputs are fixed on-host annotation files, listed under "Inputs").

## Region-set definition (USER-DECIDED, implemented exactly)

Training regions = **strict quiet genome**: main hg38 contigs (chr1-22, X;
chrY and chrM excluded) MINUS the union of:

1. blood/hematopoietic DHS index regions (components {sorted(DHS_BLOOD_COMPONENTS)}), padded +/-{DHS_PAD} bp
2. blood-expressed genes (weighted hematopoietic expression >= {EXPRESSION_THRESHOLD}): gene bodies padded +/-{GENE_PAD} bp (the +/-2 kb pad subsumes promoters)
3. annotated CTCF/TF binding sites (all cell-type binding sites), padded +/-{TF_PAD} bp
4. the standard blacklist (ENCODE hg38-blacklist.v2)
5. the positive-control regions (CTCF sites +/-{POSCTRL_PAD} bp; marker gene bodies +/-{GENE_PAD} bp)

From the remainder: {len(k['train_tiles'])} non-overlapping random {TILE:,} bp tiles;
each tile lies entirely in the remainder and its full margined extent
(tile +/-{MARGIN:,} bp = plumbing L_SEQ half-margin JITTER+RF_BUDGET) lies inside the contig.

Positive controls (separate BEDs, NEVER training):
- CTCF binding sites: v1 `build_ctcf_binding_sites()` classified set, resized to 1024 bp ({k['n_ctcf_pc']:,} sites)
- immune/epithelial marker genes: `markers.immune_vs_epithelial.tsv`, v1 filters
  (`{MARKER_UP_QUERY}` OR `{MARKER_DOWN_QUERY}`) -> gene bodies ({k['n_marker_pc']:,} genes)

## Missing annotations (v1 exact paths absent on this host)

The USER brief named several v1 annotation files that are NOT present on this host.
Substitutes (all real, on-host) were used and are flagged here; nothing was faked.

| v1 file (absent) | substitute used | impact |
|---|---|---|
| `/scratch/karius/annotation/DHS_...min2samp.random500k.blacklist_filt.bed` | `biomarker-projects/DHS_...min2samp.bed` | superset (no random-500k downsample, not pre-blacklist-filtered); larger DHS exclusion -> more conservative |
| `/scratch/karius/annotation/mappability.simple_repeats.sorted.bed.gz` | ENCODE `hg38-blacklist.v2.bed.gz` (the repo's test-data blacklist) | different blacklist; simple/low-complexity repeat content reported separately from RepeatMasker |
| `/home/.../hg38-repeats.sorted.bed.gz` | UCSC RepeatMasker `hg38_rmsk.txt.gz` | rmsk is the canonical source hg38-repeats derives from; family/class available |
| gencode v45 basic | gencode v49 basic | newer annotation; gene bodies/expression join |
| HGE module hardcoded `./data/ibd/sc_expression/...` (broken relative path) | located `sc_rnaseq.counts.hematopoetic.tsv` + gencode v49 | expression reconstructed with the documented cluster weights |
| CTCF `/scratch/ctcf_analysis/CTCF/cell_type_merged` | `biomarker-projects/shared_data/ctcf/cell_type_merged` | same files, relocated |

Additional gap: the `fragmentomics.data` module provides ONLY CTCF binding sites
(no other TF-site annotation), so exclusion #3 and the CTCF positive control are
CTCF-only.  No other TF annotation was available "in the same data module".

## Exclusion accounting (main contigs, {total:,} bp total)

| filter | bp removed | % genome | cumulative bp | cumulative % |
|---|---|---|---|---|
{acc_rows}

- **Union of all exclusions**: {k['union_bp']:,} bp ({_fmt_pct(k['union_bp']/total)} of genome)
- **Remainder (quiet genome)**: {k['remainder_bp']:,} bp ({_fmt_pct(k['remainder_bp']/total)} of genome)
- Blood-expressed genes retained (expression >= {EXPRESSION_THRESHOLD}): {k['n_expressed']:,}

## REPEAT QC (RepeatMasker hg38_rmsk)

Per-tile repeat-masked fraction (union of all rmsk annotation):

- **Training tiles**: {_dist(train_rep)}
- **Genome-wide background** (same n, same `default_rng({SEED})` protocol): {_dist(bg_rep)}
- Ratio of means (training / background): **{rep_enriched:.3f}x** -> **{interp}**

Simple-repeat + low-complexity fraction per tile:

- **Training tiles**: {_dist(train_sl)}
- **Background**: {_dist(bg_sl)}

Repeat class composition (bp of annotation / total tile bp; top classes):

| repClass | training | background |
|---|---|---|
{comp_rows}

### Interpretation

The quiet genome is **{interp}** relative to a genome-wide random background
(training mean repeat-masked fraction {train_rep.mean()*100:.2f}% vs background
{bg_rep.mean()*100:.2f}%, ratio {rep_enriched:.3f}x).  Because the exclusion set
removes DHS-accessible, gene-body, CTCF and blacklist sequence, the remaining
"quiet" genome is dominated by gene-poor / intergenic sequence, which in hg38 is
{"comparatively repeat-rich (LINE/SINE/LTR-heavy intergenic space)" if train_rep.mean() > bg_rep.mean() else "comparable to background in repeat content"}.

Fraction of training-tile bp that will be **blacklist-masked at store-build time**
(ENCODE blacklist expanded by the store default {STORE_BLACKLIST_EXPANSION} bp):
**{k['train_bl'].mean()*100:.4f}%** (mean per tile; max {k['train_bl'].max()*100:.4f}%).
Tiles were sampled to exclude the un-expanded blacklist, so only the +/-{STORE_BLACKLIST_EXPANSION} bp
store expansion can clip a tile edge -> the masked fraction is near zero.

## GC and N content (tile sequence, hg38.fa)

| metric | training | background |
|---|---|---|
| GC fraction (mean) | {train_gcn['gc'].mean():.4f} | {bg_gcn['gc'].mean():.4f} |
| GC fraction (median) | {train_gcn['gc'].median():.4f} | {bg_gcn['gc'].median():.4f} |
| N fraction (mean) | {train_gcn['n_frac'].mean():.6f} | {bg_gcn['n_frac'].mean():.6f} |
| N fraction (max) | {train_gcn['n_frac'].max():.6f} | {bg_gcn['n_frac'].max():.6f} |
| tiles with N > 50% | {train_ngap} / {n_train} ({train_ngap/n_train*100:.2f}%) | {bg_ngap} / {len(k['bg_tiles'])} ({bg_ngap/len(k['bg_tiles'])*100:.2f}%) |
| tiles with N > 1% | {train_nany} / {n_train} ({train_nany/n_train*100:.2f}%) | -- |

Training tiles are AT-richer than background (GC {train_gcn['gc'].mean():.4f} vs
{bg_gcn['gc'].mean():.4f}), consistent with the gene-poor, repeat-rich intergenic
character of the quiet genome.

### FINDING (assembly-gap contamination) -- flagged, NOT silently fixed

The USER-locked exclusion set (DHS / expressed genes / CTCF / blacklist /
positive controls) does **not** exclude assembly gaps (N-runs at
centromeres/telomeres).  Those gaps are not fully covered by the ENCODE
blacklist, so **{train_ngap} of {n_train} training tiles ({train_ngap/n_train*100:.2f}%)
are >50% N** (max 100% N) -- effectively empty tiles.  Two mitigations exist and
neither changes the locked region definition here:

1. Downstream self-correction: the plumbing `min_N` filter (default 50 per
   sample/tile/track) drops tiles with too few fragments, and all-N tiles carry
   essentially no fragments -> they are excluded at Dataset build time anyway.
2. Recommended (needs owner approval, per the "algorithmic changes" rule): add an
   assembly-gap / N-run BED (or a hardmask) as a 6th exclusion so preprocess does
   not waste compute on empty tiles.  Left OUT of this build because the region
   definition was locked to exactly the five sets above.

## Chromosome coverage (training tiles)

| contig | n tiles | bp |
|---|---|---|
{chrom_rows}

## Sanity checks (asserted in-script; build fails if violated)

- zero overlap between training tiles and EACH exclusion set (filters 1-5): PASS
- zero overlap between training tiles and positive-control CTCF / marker BEDs: PASS
- training tiles mutually non-overlapping: PASS
- tile count: **{len(k['train_tiles'])}**, total bp: **{len(k['train_tiles'])*TILE:,}**

## Emitted BEDs (in gitignored `data/region_sets/`; recorded by md5)

| file | n regions | md5 |
|---|---|---|
{md5_rows}

## Inputs (on-host annotation files)

- FASTA: `{FASTA}`
- RepeatMasker: `{RMSK}`
- blacklist: `{BLACKLIST}`
- DHS index: `{DHS}`
- CTCF dir: `{CTCF_DIR}`
- sc RNA counts: `{SC_COUNTS}`
- gencode: `{GENCODE_GFF3}`
- markers: `{MARKERS}`
"""
    with open(QC_REPORT, "w") as f:
        f.write(textwrap.dedent(report).lstrip("\n"))


if __name__ == "__main__":
    main()
