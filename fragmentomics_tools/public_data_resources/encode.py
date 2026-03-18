import os

import pandas as pd

from fragmentomics_tools.dataframe import RegionDataFrame

DEFAULT_CTCF_BASE_DIR = "/scratch/ctcf_analysis/CTCF/cell_type_merged"


def build_ctcf_binding_sites(base_dir=None):
    """Build a RegionDataFrame of CTCF binding sites classified by tissue group.

    Loads per-cell-type CTCF ChIP-seq peak files from ENCODE, merges them,
    binarizes by peak presence, and classifies each binding site as 'blood',
    'colon', 'neural', or 'all' based on the following rule: a site is assigned
    to a tissue group if it is bound in ALL members of the target group and in
    at most 2 members of the other groups. Sites bound in all cell types are
    labeled 'all'.

    The 14 cell types used (after dropping neutrophil) are split into:
      - neural: neural_progenitor_cell, neural_cell, dorsolateral_prefrontal_cortex
      - colon: stomach, transverse_colon, sigmoid_colon
      - blood: natural_killer_cell, B_cell, CD14-positive_monocyte

    Args:
        base_dir: Directory containing per-cell-type merged CTCF binding site
            TSV files. Each file should have columns: contig, start, stop,
            strand, tf_top_score. Defaults to DEFAULT_CTCF_BASE_DIR
            ('/scratch/ctcf_analysis/CTCF/cell_type_merged').

    Returns:
        A RegionDataFrame (ref='hg38') with columns: contig, start, stop,
        strand, cell_type. The cell_type column is one of 'all', 'neural',
        'colon', or 'blood'.
    """
    if base_dir is None:
        base_dir = DEFAULT_CTCF_BASE_DIR

    ### load and merge all of the binding sites #######################################################################
    index_cols = ["contig", "start", "stop", "strand"]
    dfs = []
    for fname in os.listdir(base_dir):
        # extract the biosample type from the filename
        cell_type = fname.split(".")[1]
        # load the data into a region dataframe
        df = pd.read_table(os.path.join(base_dir, fname))
        # subset the dataframe by the binding site locations
        df = df[["contig", "start", "stop", "strand", "tf_top_score"]].set_index(
            index_cols
        )
        # rename the score column to the biosample_type
        df = df.rename(columns={"tf_top_score": cell_type})
        df = df.sort_values(df.columns[0], ascending=False).head(20000).sort_index()
        dfs.append(df)

    # merge everything
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.join(dfs[i], how="outer")

    # binarize based upon peak presence
    df = pd.DataFrame(df.fillna(0).sort_index())
    df = (df > 1e-6).astype(int)

    # neutrophils look weird
    df = df.drop(columns="neutrophil")

    ### Build all of the masks ########################################################################################
    # we're excluding one blood and one neural so that the counts remain the same
    neural_columns = [
        "neural_progenitor_cell",
        "neural_cell",
        "dorsolateral_prefrontal_cortex",
    ]  # 'neural_crest_cell',
    colon_columns = ["stomach", "transverse_colon", "sigmoid_colon"]
    blood_columns = [
        "natural_killer_cell",
        "B_cell",
        "CD14-positive_monocyte",
    ]  # , 'CD8-positive,_alpha-beta_T_cell'
    all_columns = neural_columns + colon_columns + blood_columns

    all_bnd_mask = df.sum(axis=1) == df.shape[1]

    all_neural_bnd_mask = df.loc[:, neural_columns].sum(axis=1) >= len(neural_columns)
    no_non_neural_bnd_mask = df.loc[:, colon_columns + blood_columns].sum(axis=1) <= 2
    only_neural_mask = all_neural_bnd_mask & no_non_neural_bnd_mask

    all_colon_bnd_mask = df.loc[:, colon_columns].sum(axis=1) >= len(colon_columns)
    no_non_colon_bnd_mask = df.loc[:, neural_columns + blood_columns].sum(axis=1) <= 2
    only_colon_mask = all_colon_bnd_mask & no_non_colon_bnd_mask

    all_blood_bnd_mask = df.loc[:, blood_columns].sum(axis=1) >= len(blood_columns)
    no_non_blood_bnd_mask = df.loc[:, neural_columns + colon_columns].sum(axis=1) <= 2
    only_blood_mask = all_blood_bnd_mask & no_non_blood_bnd_mask

    assert (
        all_bnd_mask.sum()
        + only_neural_mask.sum()
        + only_colon_mask.sum()
        + only_blood_mask.sum()
        == (all_bnd_mask | only_neural_mask | only_colon_mask | only_blood_mask).sum()
    )

    # set the class column, and drop all of the binary labels
    df.loc[:, "cell_type"] = None
    df.loc[all_bnd_mask, "cell_type"] = "all"
    df.loc[only_neural_mask, "cell_type"] = "neural"
    df.loc[only_colon_mask, "cell_type"] = "colon"
    df.loc[only_blood_mask, "cell_type"] = "blood"
    df = df[["cell_type"]].dropna()

    return RegionDataFrame(df.reset_index(), ref="hg38")
