import sys

sys.path.insert(0, "/home/nboley/src/fragmentomics_tools/projects/")

import os

import pandas as pd

from torch import utils
import lightning as L

from fragmentomics_tools.bias_correction.model import BackgroundModelModule
from fragmentomics_tools.bias_correction.data import FragmentEndpointsDataset
from fragmentomics_tools.dataframe import RegionDataFrame

from ibd.data import HematopoieticGeneExpression, build_ctcf_binding_sites
from ibd.lib import load_ibd_sample_df

def build_marker_gene_rdf(genes_rdf):
    marker_genes = pd.read_table(
        "/scratch/karius/reference/markers.immune_vs_epithelial.tsv", sep=" "
    )
    up_marker_genes = marker_genes.query("p_val_adj < 0.01 and avg_log2FC < -3").assign(
        direction="up_in_colon"
    )
    down_marker_genes = marker_genes.query(
        "p_val_adj < 0.01 and avg_log2FC > 7"
    ).assign(direction="down_in_colon")
    marker_genes = pd.concat([up_marker_genes, down_marker_genes])
    return (
        genes_rdf.set_index("gene_name")
        .join(marker_genes, how="inner")
        .reset_index()
        .reset_index(drop=True)
    )  # .query("direction == 'up_in_colon' and expression < 0.2")


def build_gene_rdfs():
    hge_raw = HematopoieticGeneExpression.load().expand_regions(2048, 2048)
    test_rdf = build_marker_gene_rdf(hge_raw)
    non_test_rdf = hge_raw.query(
        "expression < 0.1 and gene_name not in @test_rdf.gene_name"
    ).sample(frac=1.0, random_state=1337)
    train_rdf = non_test_rdf.head(3800)
    val_rdf = non_test_rdf.tail(200)
    return train_rdf, val_rdf, test_rdf


def build_ctcf_rdfs():
    n_train, n_val = 100000, 5000
    dhs_rdf = RegionDataFrame(
        pd.read_csv("/scratch/karius/annotation/DHS_Index_and_Vocabulary_hg38_WM20190703.min2samp.random500k.blacklist_filt.bed", sep="\t"),
        ref='hg38'
    ).resize_regions(1024)
    dhs_rdf['strand'] = '.'
    ctcf_rdf = build_ctcf_binding_sites().resize_regions(1024)
    intersect_rdf = dhs_rdf.intersect_with_rdf(ctcf_rdf).drop_duplicates()
    train_and_val_rdf = dhs_rdf.loc[~dhs_rdf.index.isin(intersect_rdf.index), :].sample(n_train + n_val)
    train_rdf = train_and_val_rdf.head(n_train)
    val_rdf = train_and_val_rdf.tail(n_val)
    return train_rdf, val_rdf, ctcf_rdf


def train(model, sdf, train_rdf, val_rdf, max_epochs=10):
    import warnings

    warnings.filterwarnings("ignore", ".*Received a 3D input to dropout2d*")

    train_dataset = FragmentEndpointsDataset(train_rdf, sdf, model)
    val_dataset = FragmentEndpointsDataset(val_rdf, sdf, model)

    val_loader = utils.data.DataLoader(val_dataset, batch_size=32)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    trainer = L.Trainer(
        max_epochs=max_epochs
    )  # , strategy="ddp_find_unused_parameters_true" )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return model

valid_columns = [
    'strand_+__fl_40_65__coverage_first',
    'strand_+__fl_40_65__coverage_last',
    'strand_+__fl_40_65__coverage_midpoint',
    'strand_+__fl_120_175__coverage_first',
    'strand_+__fl_120_175__coverage_last',
    'strand_+__fl_120_175__coverage_midpoint',
    'strand_-__fl_40_65__coverage_first',
    'strand_-__fl_40_65__coverage_last',
    'strand_-__fl_40_65__coverage_midpoint',
    'strand_-__fl_120_175__coverage_first',
    'strand_-__fl_120_175__coverage_last',
    'strand_-__fl_120_175__coverage_midpoint'
]

def main():
    sdf = load_ibd_sample_df().sample(20)
    train_rdf, val_rdf, test_rdf = build_ctcf_rdfs()
    model = BackgroundModelModule(num_residual_layers=2, output_columns=valid_columns, loss="negative_binomial")
    model = train(model, sdf, train_rdf, val_rdf, max_epochs=3)
    model.save('/scratch/karius/bias_correction_model/merged.accessible_regions.negative_binomial.2.res')
    return
    for i, sample_id in enumerate(sdf.sample_id.tolist()):
        if os.path.exists(
            f"/scratch/karius/bias_correction_model/merged.negative_binomial.{sample_id}.2.res"
        ):
            continue
        print(i, len(sdf), sample_id)
        model = BackgroundModelModule.load(
            "/scratch/karius/bias_correction_model/merged.negative_binomial.merged.2.res"
        )
        model = train(
            model,
            sdf.query("sample_id == @sample_id"),
            train_rdf,
            val_rdf,
            max_epochs=5,
        )
        model.save(
            f"/scratch/karius/bias_correction_model/merged.negative_binomial.{sample_id}.2.res"
        )


if __name__ == "__main__":
    main()
