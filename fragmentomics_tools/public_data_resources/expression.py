import pandas as pd
import numpy as np

from fragmentomics_tools.dataframe import RegionDataFrame
from fragmentomics_tools.public_data_resources.gencode import load_gencode_genes_rdf

class HematopoieticGeneExpression(RegionDataFrame):
    """Hematopoietic Gene Expression from SC Bone Marrow and PBMC's

    This was taken from this paper:
    https://www.nature.com/articles/s41587-019-0332-7
    https://github.com/GreenleafLab/MPAL-Single-Cell-2019
    """

    @staticmethod
    def load_cell_labels():
        _ = """
        1	HSC - Hematopoietic Stem Cells
        2	Early Eryth. - Early Erythroid Cells
        3	Late Eryth. - Late Erythroid Cells
        4	Early Basophil Cells
        5	CMP/LMPP - Common Myeloid Progenitor / lymphoid-primed multipotent progenitors
        6	CLP 1 - Common Lymphoid Progenitor
        7	GMP - Granulocyte-Monocyte progenitors
        8	GMP/Neut - Granulocyte-Monocyte progenitors / Neutrophil
        9	pDC - Plasmacytoid Dendritic Cell
        10	cDC - Classical Dendritic CElls
        11	CD14 Mono 1 - CD14+ Monocyte Cells
        12	CD14 Mono 2 - CD14+ Monocyte Cells
        13	CD16 Mono - CD16+ Monocyte Cells
        14	Unk - Unkown
        15	CLP 2 - Common Lymphoid Progenitor
        16	Pre B - Pre B-Cell Progenitor
        17	B - B Cells
        18	Plasma - Plasma Cells
        19	CD8 N - CD8+ Naïve Cells
        20	CD4 N 1 - CD4+ Naïve Cells
        21	CD4 N 2 - CD4+ Naïve Cells
        22	CD4 M - CD4+ Memory Cells
        23	CD8 EM - CD8+ Effector Memory Cells
        24	CD8 CM - CD8+ Central Memory Cells
        25	NK - Natural Killer Cells
        26	Unk - Unkown
        """
        code_to_weight = [line.strip().split("\t") for line in _.strip().splitlines()]
        code_to_weight = [(k.zfill(2), v.split(" - ")[0], v.split(" - ")[1]) for k, v in code_to_weight]
        return code_to_weight

    @staticmethod
    def cluster_names_and_weights():
        cluster_names_and_weights = """
        Early_Erythroid_Cells 0.17
        Late_Erythroid_Cells 0.15
        Myeloid_Progenitor 0.1
        Lymphoid_Progenitor_1 0.08
        Granulocyte-Monocyte_Progenitor 0.16
        Neutrophil 0.21
        CD14+_Monocyte_Cells_1 0.05
        CD14+_Monocyte_Cells_2 0.04
        CD16+_Monocyte_Cells 0.04
        Lymphoid_Progenitor_2 0.08
        """.strip().split("\n")
        return dict((x.split()[0], float(x.split()[1])) for x in cluster_names_and_weights)

    @classmethod
    def load(cls):
        grpd_counts = pd.read_table("data/expression/sc_rnaseq.counts.hematopoetic.tsv", index_col=0)
        cluster_names_and_weights = cls.cluster_names_and_weights()
        
        tpms = grpd_counts/(grpd_counts.sum()/1e6)
        tpms = tpms[['02', '03', '05', '06', '07', '08', '11', '12', '13', '15']]
        # taking advantage of dict keys remaining sorted
        tpms.columns = list(cluster_names_and_weights.keys())
        
        gencode_rdf = load_gencode_genes_rdf()
    
        # join the tpms to gencode genes
        tpms = pd.DataFrame(
            gencode_rdf.drop(columns=['tss', 'tes']).reset_index()
        ) \
        .set_index('gene_name') \
        .join(tpms) \
        .reset_index() \
        .rename(columns={'index': 'gene_name'}) \
        .set_index(['gene_name', 'gene_id', 'contig', 'strand', 'start', 'stop']) \
        .dropna()
    
        # set expression by taking a weighted sum over the relevant sub-types
        expression_df = pd.DataFrame(
            np.matmul(tpms.values, np.array(list(cluster_names_and_weights.values()))), 
            index=tpms.index, 
            columns=['expression']
        ).reset_index()
    
        return cls(expression_df, ref='hg38')