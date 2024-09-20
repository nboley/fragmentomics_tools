import os
import pandas

from datamanifest import DataManifest

from fragmentomics_tools.dataframe import DataFrameBase, RegionDataFrame

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data/"))


def get_data_path(key):
    return os.path.join(DATA_PATH, key)


CELL_NAME_TO_CELL_EID = {
    "IMR90_Cell_Line": "E017",
    "ES-WA7_Cell_Line": "E002",
    "H9_Cell_Line": "E008",
    "ES-I3_Cell_Line": "E001",
    "HUES6_Cell_Line": "E015",
    "HUES48_Cell_Line": "E014",
    "HUES64_Cell_Line": "E016",
    "H1_Cell_Line": "E003",
    "4star": "E024",
    "iPS-20b_Cell_Line": "E020",
    "iPS-18_Cell_Line": "E019",
    "iPS-15b_Cell_Line": "E018",
    "iPS_DF_6.9_Cell_Line": "E021",
    "iPS_DF_19.11_Cell_Line": "E022",
    "H1_Derived_Neuronal_Progenitor_Cultured_Cells": "E007",
    "H9_Derived_Neuronal_Progenitor_Cultured_Cells": "E009",
    "H9_Derived_Neuron_Cultured_Cells": "E010",
    "hESC_Derived_CD56+_Mesoderm_Cultured_Cells": "E013",
    "hESC_Derived_CD56+_Ectoderm_Cultured_Cells": "E012",
    "hESC_Derived_CD184+_Endoderm_Cultured_Cells": "E011",
    "H1_BMP4_Derived_Mesendoderm_Cultured_Cells": "E004",
    "H1_BMP4_Derived_Trophoblast_Cultured_Cells": "E005",
    "H1_Derived_Mesenchymal_Stem_Cells": "E006",
    "Peripheral_Blood_Mononuclear_Primary_Cells": "E062",
    "CD3_Primary_Cells_Peripheral_UW": "E034",
    "CD4+_CD25int_CD127+_Tmem_Primary_Cells": "E045",
    "CD3_Primary_Cells_Cord_BI": "E033",
    "CD4+_CD25+_CD127-_Treg_Primary_Cells": "E044",
    "CD4+_CD25-_Th_Primary_Cells": "E043",
    "CD4+_CD25-_CD45RA+_Naive_Primary_Cells": "E039",
    "CD4+_CD25-_IL17-_PMA-Ionomycin_stimulated_MACS_purified_Th_Primary_Cells": "E041",
    "CD4+_CD25-_IL17+_PMA-Ionomcyin_stimulated_Th17_Primary_Cells": "E042",
    "CD4+_CD25-_CD45RO+_Memory_Primary_Cells": "E040",
    "CD4_Memory_Primary_Cells": "E037",
    "CD8_Memory_Primary_Cells": "E048",
    "CD4_Naive_Primary_Cells": "E038",
    "CD8_Naive_Primary_Cells": "E047",
    "CD14_Primary_Cells": "E029",
    "CD19_Primary_Cells_Cord_BI": "E031",
    "CD34_Primary_Cells": "E035",
    "Mobilized_CD34_Primary_Cells_Male": "E051",
    "Mobilized_CD34_Primary_Cells_Female": "E050",
    "CD34_Cultured_Cells": "E036",
    "CD19_Primary_Cells_Peripheral_UW": "E032",
    "CD56_Primary_Cells": "E046",
    "CD15_Primary_Cells": "E030",
    "Bone_Marrow_Derived_Mesenchymal_Stem_Cell_Cultured_Cells": "E026",
    "Chondrocytes_from_Bone_Marrow_Derived_Mesenchymal_Stem_Cell_Cultured_Cells": "E049",
    "Adipose_Derived_Mesenchymal_Stem_Cell_Cultured_Cells": "E025",
    "Mesenchymal_Stem_Cell_Derived_Adipocyte_Cultured_Cells": "E023",
    "Muscle_Satellite_Cultured_Cells": "E052",
    "Penis_Foreskin_Fibroblast_Primary_Cells_skin01": "E055",
    "Penis_Foreskin_Fibroblast_Primary_Cells_skin02": "E056",
    "Penis_Foreskin_Melanocyte_Primary_Cells_skin01": "E059",
    "Penis_Foreskin_Melanocyte_Primary_Cells_skin03": "E061",
    "Penis_Foreskin_Keratinocyte_Primary_Cells_skin02": "E057",
    "Penis_Foreskin_Keratinocyte_Primary_Cells_skin03": "E058",
    "Breast_vHMEC": "E028",
    "Breast_Myoepithelial_Cells": "E027",
    "Neurosphere_Cultured_Cells_Ganglionic_Eminence_Derived": "E054",
    "Neurosphere_Cultured_Cells_Cortex_Derived": "E053",
    "Thymus": "E112",
    "Fetal_Thymus": "E093",
    "Brain_Hippocampus_Middle": "E071",
    "Brain_Substantia_Nigra": "E074",
    "Brain_Anterior_Caudate": "E068",
    "Brain_Cingulate_Gyrus": "E069",
    "Brain_Inferior_Temporal_Lobe": "E072",
    "Brain_Angular_Gyrus": "E067",
    "Brain_Mid_Frontal_Lobe": "E073",
    "Brain_Germinal_Matrix": "E070",
    "Fetal_Brain_Female": "E082",
    "Fetal_Brain_Male": "E081",
    "Adipose_Nuclei": "E063",
    "Psoas_Muscle": "E100",
    "Skeletal_Muscle_Female": "E108",
    "Skeletal_Muscle_Male": "E107",
    "Fetal_Muscle_Trunk": "E089",
    "Fetal_Muscle_Leg": "E090",
    "Fetal_Heart": "E083",
    "Right_Atrium": "E104",
    "Left_Ventricle": "E095",
    "Right_Ventricle": "E105",
    "Aorta": "E065",
    "Duodenum_Smooth_Muscle": "E078",
    "Colon_Smooth_Muscle": "E076",
    "Rectal_Smooth_Muscle": "E103",
    "Stomach_Smooth_Muscle": "E111",
    "Fetal_Stomach": "E092",
    "Fetal_Intestine_Small": "E085",
    "Fetal_Intestine_Large": "E084",
    "Small_Intestine": "E109",
    "Sigmoid_Colon": "E106",
    "Colonic_Mucosa": "E075",
    "Rectal_Mucosa.Donor_29": "E101",
    "Rectal_Mucosa.Donor_31": "E102",
    "Stomach_Mucosa": "E110",
    "Duodenum_Mucosa": "E077",
    "Esophagus": "E079",
    "Gastric": "E094",
    "Placenta_Amnion": "E099",
    "Fetal_Kidney": "E086",
    "Fetal_Lung": "E088",
    "Ovary": "E097",
    "Pancreatic_Islets": "E087",
    "Fetal_Adrenal_Gland": "E080",
    "Fetal_Placenta": "E091",
    "Adult_Liver": "E066",
    "Pancreas": "E098",
    "Lung": "E096",
    "Spleen": "E113",
    "A549_EtOH_0.02pct_Lung_Carcinoma": "E114",
    "Dnd41_TCell_Leukemia": "E115",
    "GM12878_Lymphoblastoid": "E116",
    "HeLa-S3_Cervical_Carcinoma": "E117",
    "HepG2_Hepatocellular_Carcinoma": "E118",
    "HMEC_Mammary_Epithelial": "E119",
    "HSMM_Skeletal_Muscle_Myoblasts": "E120",
    "HSMMtube_Skeletal_Muscle_Myotubes_Derived_from_HSMM": "E121",
    "HUVEC_Umbilical_Vein_Endothelial_Cells": "E122",
    "K562_Leukemia": "E123",
    "Monocytes-CD14+_RO01746": "E124",
    "NH-A_Astrocytes": "E125",
    "NHDF-Ad_Adult_Dermal_Fibroblasts": "E126",
    "NHEK-Epidermal_Keratinocytes": "E127",
    "NHLF_Lung_Fibroblasts": "E128",
    "Osteoblasts": "E129",
}


CELL_EID_TO_CELL_NAME = {v: k for k, v in CELL_NAME_TO_CELL_EID.items()}

"""
See
https://egg2.wustl.edu/roadmap/web_portal/meta.html
"""
# these were identified by looking at the descriptions in data/roadmap/rna/expression/EG.name.txt.gz
# and https://docs.google.com/spreadsheets/d/1yikGx4MsO9Ei36b64yOy9Vb6oPC5IBGlFbYEt-N6gOM/edit#gid=15
# 3-34,37-45,47-48,61
CELL_EID_TO_CELL_ANATOMY = {
    "E017": "LUNG",
    "E002": "ESC",
    "E008": "ESC",
    "E001": "ESC",
    "E015": "ESC",
    "E014": "ESC",
    "E016": "ESC",
    "E003": "ESC",
    "E024": "ESC",
    "E020": "IPSC",
    "E019": "IPSC",
    "E018": "IPSC",
    "E021": "IPSC",
    "E022": "IPSC",
    "E007": "ESC_DERIVED",
    "E009": "ESC_DERIVED",
    "E010": "ESC_DERIVED",
    "E013": "ESC_DERIVED",
    "E012": "ESC_DERIVED",
    "E011": "ESC_DERIVED",
    "E004": "ESC_DERIVED",
    "E005": "ESC_DERIVED",
    "E006": "ESC_DERIVED",
    "E062": "BLOOD",
    "E034": "BLOOD",
    "E045": "BLOOD",
    "E033": "BLOOD",
    "E044": "BLOOD",
    "E043": "BLOOD",
    "E039": "BLOOD",
    "E041": "BLOOD",
    "E042": "BLOOD",
    "E040": "BLOOD",
    "E037": "BLOOD",
    "E048": "BLOOD",
    "E038": "BLOOD",
    "E047": "BLOOD",
    "E029": "BLOOD",
    "E031": "BLOOD",
    "E035": "BLOOD",
    "E051": "BLOOD",
    "E050": "BLOOD",
    "E036": "BLOOD",
    "E032": "BLOOD",
    "E046": "BLOOD",
    "E030": "BLOOD",
    "E026": "STROMAL_CONNECTIVE",
    "E049": "STROMAL_CONNECTIVE",
    "E025": "FAT",
    "E023": "FAT",
    "E052": "MUSCLE",
    "E055": "SKIN",
    "E056": "SKIN",
    "E059": "SKIN",
    "E061": "SKIN",
    "E057": "SKIN",
    "E058": "SKIN",
    "E028": "BREAST",
    "E027": "BREAST",
    "E054": "BRAIN",
    "E053": "BRAIN",
    "E112": "THYMUS",
    "E093": "THYMUS",
    "E071": "BRAIN",
    "E074": "BRAIN",
    "E068": "BRAIN",
    "E069": "BRAIN",
    "E072": "BRAIN",
    "E067": "BRAIN",
    "E073": "BRAIN",
    "E070": "BRAIN",
    "E082": "BRAIN",
    "E081": "BRAIN",
    "E063": "FAT",
    "E100": "MUSCLE",
    "E108": "MUSCLE",
    "E107": "MUSCLE",
    "E089": "MUSCLE",
    "E090": "MUSCLE_LEG",
    "E083": "HEART",
    "E104": "HEART",
    "E095": "HEART",
    "E105": "HEART",
    "E065": "VASCULAR",
    "E078": "GI_DUODENUM",
    "E076": "GI_COLON",
    "E103": "GI_RECTUM",
    "E111": "GI_STOMACH",
    "E092": "GI_STOMACH",
    "E085": "GI_INTESTINE",
    "E084": "GI_INTESTINE",
    "E109": "GI_INTESTINE",
    "E106": "GI_COLON",
    "E075": "GI_COLON",
    "E101": "GI_RECTUM",
    "E102": "GI_RECTUM",
    "E110": "GI_STOMACH",
    "E077": "GI_DUODENUM",
    "E079": "GI_ESOPHAGUS",
    "E094": "GI_STOMACH",
    "E099": "PLACENTA",
    "E086": "KIDNEY",
    "E088": "LUNG",
    "E097": "OVARY",
    "E087": "PANCREAS",
    "E080": "ADRENAL",
    "E091": "PLACENTA",
    "E066": "LIVER",
    "E098": "PANCREAS",
    "E096": "LUNG",
    "E113": "SPLEEN",
    "E114": "LUNG",
    "E115": "BLOOD",
    "E116": "BLOOD",
    "E117": "CERVIX",
    "E118": "LIVER",
    "E119": "BREAST",
    "E120": "MUSCLE",
    "E121": "MUSCLE",
    "E122": "VASCULAR",
    "E123": "BLOOD",
    "E124": "BLOOD",
    "E125": "BRAIN",
    "E126": "SKIN",
    "E127": "SKIN",
    "E128": "LUNG",
    "E129": "BONE",
}

CELL_ANATOMY_TO_CELL_EID = {"all": []}
for cell_eid in CELL_EID_TO_CELL_ANATOMY.keys():
    anatomy = CELL_EID_TO_CELL_ANATOMY[cell_eid]
    if anatomy.lower() not in CELL_ANATOMY_TO_CELL_EID:
        CELL_ANATOMY_TO_CELL_EID[anatomy.lower()] = []
    CELL_ANATOMY_TO_CELL_EID[anatomy.lower()].append(cell_eid)
    CELL_ANATOMY_TO_CELL_EID["all"].append(cell_eid)

CELL_ANATOMY_TO_CELL_EID_WITH_EXPRESSION = dict(
    all=[
        "E000",
        "E003",
        "E004",
        "E005",
        "E006",
        "E007",
        "E011",
        "E012",
        "E013",
        "E016",
        "E024",
        "E027",
        "E028",
        "E037",
        "E038",
        "E047",
        "E050",
        "E053",
        "E054",
        "E055",
        "E056",
        "E057",
        "E058",
        "E059",
        "E061",
        "E062",
        "E065",
        "E066",
        "E070",
        "E071",
        "E079",
        "E082",
        "E084",
        "E085",
        "E087",
        "E094",
        "E095",
        "E096",
        "E097",
        "E098",
        "E100",
        "E104",
        "E105",
        "E106",
        "E109",
        "E112",
        "E113",
        "E114",
        "E116",
        "E117",
        "E118",
        "E119",
        "E120",
        "E122",
        "E123",
        "E127",
        "E128",
    ],
    blood=["E037", "E038", "E047", "E062"],
    breast=["E027", "E028", "E119"],
    liver=["E066"],
    brain=["E070", "E071", "E082"],
    esophagus=["E079"],
    gastric=["E094"],
    lung=["E096"],
    ovary=["E097"],
    pancreas=["E098"],
    colon=["E106"],
    thymus=["E112"],
    spleen=["E113"],
)


class GeneExpression(DataFrameBase):
    _metadata = ["sample_info"]
    _required_metadata = ["sample_info"]

    @classmethod
    def from_fname_s3_or_local(cls, expression_path, metadata_path):
        return cls(
            data=pandas.read_table(expression_path),
            sample_info=pandas.read_table(metadata_path, index_col=0),
        )

    @property
    def data_cols(self):
        return set(self.sample_info.index)

    @property
    def info_cols(self):
        return set(self.columns) - self.data_cols

    def aggregate_across_sample_types(self, func="median"):
        # aggregate the expression data over the expression columns
        aggregated_exp_data = {}
        # add the non-expression columns (e.g. ensemble_id)
        for column_name in self.info_cols:
            aggregated_exp_data[column_name] = self.loc[:, column_name]

        # add the aggregated expression columns
        aggregated_sample_info_rows = []

        for sample_type, df in self.sample_info.groupby("sample_type"):
            # Aggregate the data columns for this sample_type
            sample_exp = self.loc[:, df.index].aggregate(func, axis=1)
            aggregated_exp_data[sample_type] = sample_exp

            # Aggregate the sample_info
            # drop the columns that can't be aggregated
            df = df.drop(
                [
                    "biological_replicate_id",
                    "technical_replicate_id",
                    "source",
                    "fname",
                    "parser",
                ],
                axis="columns",
            )
            # the rest of the columns should contain identical information
            df = df.drop_duplicates()
            assert (
                len(df) == 1
            ), f"Could not aggregate sample_info for sample_type {sample_type}:\n{df}"
            aggregated_sample_info_rows.append(df.iloc[0])

        aggregated_sample_info = pandas.DataFrame(
            aggregated_sample_info_rows
        ).set_index("sample_type")

        # combine everything together, and initialize the aggregated dataframe
        return type(self)(aggregated_exp_data, sample_info=aggregated_sample_info)


class TSSs(RegionDataFrame):
    def merge_gene_expression(self, gene_expression_df: GeneExpression):
        if not isinstance(gene_expression_df, GeneExpression):
            raise TypeError("'gene_expression_df' must be of type GeneExpression.")

        rv = self.merge(gene_expression_df, on="ens_id")
        if rv.ens_id.duplicated().any():
            raise ValueError("Duplicate ensemble ids detected in merged df")

        return rv

    def filter_by_genes(self, ens_gene_ids):
        return self.loc[self["ens_id"].isin(ens_gene_ids)]


def load_gene_expression_df(ref):
    assert ref in ("hg19", "hg38")
    return GeneExpression.from_fname_s3_or_local(
        get_data_path("annotations/roadmap/gene_expression.tsv.gz"),
        get_data_path("annotations/roadmap/gene_expression_metadata.tsv"),
    )


def load_tss_s_df(ref):
    assert ref in ("hg19", "hg38")
    tss_input_path = get_data_path(f"annotations/encode/tss.all_rampage.{ref}.bed.gz")
    rv = TSSs(pandas.read_table(tss_input_path), ref=ref)
    # strip off the gene id patch id
    rv["ens_id"] = [x.split(".")[0] for x in rv["ens_id"].tolist()]
    return rv


def load_chrom_hmm_states():
    rdf = RegionDataFrame(
        pandas.read_table(
            "/scratch/nboley/test_unet_pytorch/DHS_Index_and_Vocabulary_hg38_WM20190703.min2samp.noblacklist.bed", 
            low_memory=False,
            # dtype=dict(zip(['contig', 'start', 'stop', 'identifier', 'mean_signal', 'numsamples', 'summit', 'core_start', 'core_end', 'component', 'strand'], [str, int, int, str, float, int, int, int, int, str]))
        ).rename(columns={'seqname': 'contig'}).dropna(),
        ref='hg38'
    ).convert_dtypes()
    
    blood_rdf = rdf.query("component in ['Lymphoid', 'Myeloid / erythroid']").copy().center_on_summit().resize_regions(1024)
    all_rdf = rdf.query("component in ['Tissue invariant']").copy().center_on_summit().resize_regions(1024)
    digestive_rdf = rdf.query("component in ['Digestive']").copy().center_on_summit().resize_regions(1024)
    neural_rdf = rdf.query("component in ['Neural']").copy().center_on_summit().resize_regions(1024)
    
    srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf.sample(10000), sdf.query("label == 1").sample(10)).attach_fragment_arrays(num_cores=32, min_mapq=10, fragment_array_callback=lambda fa: fa.drop_duplicate_fragments())
