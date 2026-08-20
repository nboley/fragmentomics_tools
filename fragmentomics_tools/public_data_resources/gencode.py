import pandas as pd

from fragmentomics_tools.dataframe import RegionDataFrame

DEFAULT_GENCODE_V39_GFF3 = "/scratch/nboley/gencode/gencode.v39.basic.annotation.gff3.gz"
DEFAULT_GENCODE_V49_GFF3 = "/efs/analytics/nathanboley/data_resources/gencode/gencode.v49.basic.annotation.gff3.gz"


def load_gencode_genes_rdf(ref="hg38", path=None):
    assert ref == "hg38"
    if path is None:
        path = DEFAULT_GENCODE_V49_GFF3

    # load the 5' and 3' UTR elements
    rdf = pd.read_table(
        path,
        usecols=[0, 2, 3, 4, 6, 8],
        names=["contig", "element", "start", "stop", "strand", "meta"],
        comment="#",
    ).query(
        "contig not in ['chrM', 'chrY'] and element in ['three_prime_UTR', 'five_prime_UTR']"
    )  # 'gene', 'transcript', 'exon',
    rdf["gene_id"] = rdf.meta.str.extract(";gene_id=(ENSG?\d+)\.\d+;")
    rdf["gene_name"] = rdf.meta.str.extract(";gene_name=(.+?);")
    rdf = rdf.drop(columns=["meta"])
    rdf = rdf.drop_duplicates()

    # aggregate into a single gene, using the most common 5' and 3' ends
    tes_s = []
    for idx, sub_df in rdf.query(
        f"strand == '+' and element == 'three_prime_UTR'"
    ).groupby("gene_id")["stop"]:
        tes_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])
    for idx, sub_df in rdf.query(
        f"strand == '-' and element == 'three_prime_UTR'"
    ).groupby("gene_id")["start"]:
        tes_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])

    tss_s = []
    for idx, sub_df in rdf.query(
        f"strand == '+' and element == 'five_prime_UTR'"
    ).groupby("gene_id")["start"]:
        tss_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])
    for idx, sub_df in rdf.query(
        f"strand == '-' and element == 'five_prime_UTR'"
    ).groupby("gene_id")["stop"]:
        tss_s.append([idx, sub_df.agg(lambda x: pd.Series.mode(x)[0])])

    tss_and_tes = (
        pd.DataFrame(tss_s, columns=["gene_id", "tss"])
        .set_index("gene_id")
        .join(pd.DataFrame(tes_s, columns=["gene_id", "tes"]).set_index("gene_id"))
        .dropna()
        .astype(int)
    )
    # return tss_and_tes, _rdf # .set_index('gene_id')[['contig', 'strand']]

    rdf = (
        rdf[["gene_id", "gene_name", "contig", "strand"]]
        .drop_duplicates()
        .set_index("gene_id")
        .join(tss_and_tes, how="inner")
    )
    rdf["start"] = rdf[["tss", "tes"]].min(axis=1)
    rdf["stop"] = rdf[["tss", "tes"]].max(axis=1)

    return RegionDataFrame(rdf.sort_values(["contig", "start"]), ref=ref)


def build_bed_line(df):
    """Build a transcript bed line from a df contianing gtf exons and a transcript.

    This is useful for visualization.
    """
    # extract the transcript record
    transcript = df.query("element == 'transcript'")
    assert len(transcript) == 1
    transcript = transcript.iloc[0]

    # extract the exons
    exons = df.query("element == 'exon'").sort_values("start", ascending=True)
    # assert exons.start.min() == transcript.start
    # assert exons.stop.max() == transcript.stop
    blockCount = str(len(exons))
    blockSizes = ",".join([str(e.stop - e.start) for e in exons.itertuples()]) + ","
    blockStarts = (
        ",".join([str(e.start - transcript.start) for e in exons.itertuples()]) + ","
    )
    rv = [
        transcript.contig,
        str(transcript.start),
        str(transcript.stop),
        transcript.gene_name,
        "1000",
        transcript.strand,
        str(transcript.start),
        str(transcript.start),
        "0",
        blockCount,
        blockSizes,
        blockStarts,
    ]
    return "\t".join(rv)


def load_gencode_gtf(path):
    rdf = pd.read_table(
        path,
        usecols=[0, 2, 3, 4, 6, 8],
        names=["contig", "element", "start", "stop", "strand", "meta"],
        comment="#",
    ).query("contig not in ['chrM', 'chrY'] and element in ['transcript', 'exon']")
    rdf["gene_id"] = rdf.meta.str.extract('gene_id "(ENSG?\d+\.\d+)"; ')
    rdf["gene_name"] = rdf.meta.str.extract('gene_name "(.+?)"; ')
    rdf["transcript_id"] = rdf.meta.str.extract('transcript_id "(.+?)"; ')
    rdf = rdf.drop(columns=["meta"])
    rdf = rdf.drop_duplicates()


def build_transcript_bed_from_gencode_gtf(input_fname, output_fname):
    # "/scratch/karius/annotation/gencode/gencode.v45.basic.annotation.gtf.gz"
    # "/scratch/karius/annotation/gencode/gencode.v45.basic.annotation.bed"
    with open(output_fname, "w") as ofp:
        for _, sub_df in tqdm.tqdm(rdf.groupby("transcript_id")):
            ofp.write(build_bed_line(sub_df) + "\n")
    # bedtools sort -i /scratch/karius/annotation/gencode/gencode.v45.basic.annotation.bed > /scratch/karius/annotation/gencode/gencode.v45.basic.annotation.sorted.bed
    # rm /scratch/karius/annotation/gencode/gencode.v45.basic.annotation.bed
    # bgzip /scratch/karius/annotation/gencode/gencode.v45.basic.annotation.sorted.bed
    # mv /scratch/karius/annotation/gencode/gencode.v45.basic.annotation.sorted.bed.gz /scratch/karius/annotation/gencode/gencode.v45.basic.annotation.bed.gz


def load_tss_rdf(gencode_path=None):
    """Load transcription start sites from a GENCODE GFF3 annotation file.

    Parses transcript records, extracts the TSS (strand-aware), resizes to 16bp
    windows, and groups by unique genomic position, aggregating transcript IDs
    and names.

    Args:
        gencode_path: Path to a GENCODE GFF3 file. Defaults to the GENCODE v39
            basic annotation at DEFAULT_GENCODE_V39_GFF3.

    Returns:
        A RegionDataFrame with columns: contig, start, stop, strand, gene_id,
        gene_name, gene_type, element, transcript_id (comma-separated),
        transcript_name (comma-separated), n_transcripts.
    """
    if gencode_path is None:
        gencode_path = DEFAULT_GENCODE_V39_GFF3
    rdf = pd.read_table(
            gencode_path,
            usecols=[0, 2, 3, 4, 6, 8],
            comment="#",
            names=["contig", "element", "start", "stop", "strand", "meta"],
    ).query("element == 'transcript'")
    rdf["transcript_id"] = rdf.meta.str.extract(r";transcript_id=(ENST?\d+)\.\d+;")
    rdf["transcript_name"] = rdf.meta.str.extract(r";transcript_name=(.+?);")
    rdf["gene_id"] = rdf.meta.str.extract(r";gene_id=(ENSG?\d+)\.\d+;")
    rdf["gene_name"] = rdf.meta.str.extract(r";gene_name=(.+?);")
    rdf["gene_type"] = rdf.meta.str.extract(r";gene_type=(.+?);")
    rdf = rdf.drop(columns=["meta"])
    rdf = rdf.drop_duplicates()
    rdf = RegionDataFrame(rdf, ref='hg38').reset_index(drop=True)
    rdf = rdf.truncate_regions(right_amt=rdf.region_lengths-1, strand_aware=True).resize_regions(16).reset_index(drop=True)
    rdf = RegionDataFrame(rdf.df.groupby(list(set(rdf.columns) - {'transcript_id', 'transcript_name'})).apply(
        lambda x: pd.Series(dict(transcript_id=",".join(x.transcript_id), transcript_name=",".join(x.transcript_name), n_transcripts=x.shape[0]))
        , include_groups=False
    ).reset_index(), ref='hg38')
    return rdf.reorder_columns()


def load_exons_rdf(gencode_path=None):
    """Load protein-coding exon regions from a GENCODE GFF3 annotation file.

    Parses exon records and filters to protein_coding gene_type only.

    Args:
        gencode_path: Path to a GENCODE GFF3 file. Defaults to the GENCODE v39
            basic annotation at DEFAULT_GENCODE_V39_GFF3.

    Returns:
        A RegionDataFrame with columns: contig, start, stop, strand, element,
        gene_id, gene_name, gene_type.
    """
    if gencode_path is None:
        gencode_path = DEFAULT_GENCODE_V39_GFF3
    rdf = pd.read_table(
            gencode_path,
            usecols=[0, 2, 3, 4, 6, 8],
            comment="#",
            names=["contig", "element", "start", "stop", "strand", "meta"],
    ).query("element == 'exon'")
    rdf["gene_id"] = rdf.meta.str.extract(r";gene_id=(ENSG?\d+)\.\d+;")
    rdf["gene_name"] = rdf.meta.str.extract(r";gene_name=(.+?);")
    rdf["gene_type"] = rdf.meta.str.extract(r";gene_type=(.+?);")
    rdf = rdf.drop(columns=["meta"])
    rdf = rdf.query("gene_type == 'protein_coding'")
    rdf = rdf.drop_duplicates()
    rdf = RegionDataFrame(rdf, ref='hg38').reset_index(drop=True)
    return rdf
