import pandas as pd

from fragmentomics_tools.dataframe import RegionDataFrame

def load_gencode_genes_rdf(ref='hg38'):
    assert ref == 'hg38'

    # load the 5' and 3' UTR elements
    rdf = pd.read_table(
        "/scratch/nboley/gencode/gencode.v39.basic.annotation.protein_coding.gff3",
        usecols=[0, 2, 3, 4, 6, 8],
        names=["contig", "element", "start", "stop", "strand", "meta"],
    ).query("contig not in ['chrM', 'chrY'] and element in ['three_prime_UTR', 'five_prime_UTR']") # 'gene', 'transcript', 'exon', 
    rdf['gene_id'] = rdf.meta.str.extract(';gene_id=(ENSG?\d+)\.\d+;')
    rdf['gene_name'] = rdf.meta.str.extract(';gene_name=(.+?);')
    rdf = rdf.drop(columns=['meta'])
    rdf = rdf.drop_duplicates()

    # aggregate into a single gene, using the most common 5' and 3' ends
    tes_s = []
    for idx, sub_df in rdf.query(f"strand == '+' and element == 'three_prime_UTR'").groupby('gene_id')['stop']:
        tes_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])
    for idx, sub_df in rdf.query(f"strand == '-' and element == 'three_prime_UTR'").groupby('gene_id')['start']:
        tes_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])

    tss_s = []
    for idx, sub_df in rdf.query(f"strand == '+' and element == 'five_prime_UTR'").groupby('gene_id')['start']:
        tss_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])
    for idx, sub_df in rdf.query(f"strand == '-' and element == 'five_prime_UTR'").groupby('gene_id')['stop']:
        tss_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])

    tss_and_tes = pd.DataFrame(tss_s, columns=['gene_id', 'tss']).set_index('gene_id').join(pd.DataFrame(tes_s, columns=['gene_id', 'tes']).set_index('gene_id')).dropna().astype(int)
    # return tss_and_tes, _rdf # .set_index('gene_id')[['contig', 'strand']]

    rdf = rdf[['gene_id', 'gene_name', 'contig', 'strand']].drop_duplicates().set_index('gene_id').join(tss_and_tes, how='inner')
    rdf['start'] = rdf[['tss', 'tes']].min(axis=1)
    rdf['stop'] = rdf[['tss', 'tes']].max(axis=1)

    return RegionDataFrame(rdf.sort_values(["contig", "start"]), ref=ref)