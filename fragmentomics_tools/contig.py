import os
from collections import OrderedDict
import subprocess

import tempfile
from enum import Enum
from typing import Dict


from fragmentomics_tools.constants import get_data_path


REFERENCE_ANNOTATIONS = ["hg16", "hg17", "hg18", "hg19", "hg38"]

# mapping from reference annotation name to fasta file key
REFERENCE_FASTA_DATA_MANIFEST_KEY = {
    "hg16": "reference/GRCh34/GRCh34.primary_assembly.genome.fa.gz",
    "hg17": "reference/GRCh35/GRCh35.primary_assembly.genome.fa.gz",
    "hg18": "reference/GRCh36/GRCh36.primary_assembly.genome.fa.gz",
    "hg19": "reference/GRCh37/GRCh37.primary_assembly.genome.fa.gz",
    "hg38": "reference/GRCh38/GRCh38.p12.genome.fa.gz",
}
REFERENCE_CHROM_SIZES_MANIFEST_KEY = {
    "hg16": "reference/GRCh34/chrom.sizes",
    "hg17": "reference/GRCh35/chrom.sizes",
    "hg18": "reference/GRCh36/chrom.sizes",
    "hg19": "reference/GRCh37/chrom.sizes",
    "hg38": "reference/GRCh38/chrom.sizes",
}


def get_canonical_reference_name(ref_name):
    grc_to_canonical = {
        "GRCh34": "hg16",
        "GRCh35": "hg17",
        "GRCh36": "hg18",
        "GRCh37": "hg19",
        "GRCh38": "hg38",
    }
    if ref_name in REFERENCE_ANNOTATIONS:
        return ref_name
    elif ref_name in grc_to_canonical:
        return grc_to_canonical[ref_name]
    else:
        raise ValueError(f"invalid ref_name: {ref_name}")


class ReferenceInferenceError(ValueError):
    pass


def infer_reference_genome_from_fname(fname, expected=None, spec=None):
    """Infer reference genome from fname.

    fname   : the filename to infer the reference from
    expected: the expected reference.
              This serves as a default in case reference isn't able to be identified,
              and verifies that they match if it is able to infer one.
    spec    : mapping from references to reference strings
    """
    if spec is None:
        spec = dict(
            hg16=["hg16", "GRCh34"],
            hg17=["hg17", "GRCh35"],
            hg18=["hg18", "GRCh36"],
            hg19=["hg19", "GRCh37"],
            hg38=["hg38", "GRCh38"],
        )

    matches = dict()
    for ref_name, aliases in spec.items():
        if any(alias.lower() in fname.lower() for alias in aliases):
            matches[ref_name] = True

    if len(matches) == 0:
        if expected is not None:
            return expected
        raise ReferenceInferenceError(
            f"Can not infer reference for '{fname}'.\nHint: fname should contain one of {REFERENCE_ANNOTATIONS}"
        )
    elif len(matches) == 1:
        inferred = list(matches)[0]
        if expected is not None and expected != inferred:
            raise ReferenceInferenceError(
                f"Inferred reference '{inferred}' does not match expected '{expected}'"
            )
        return inferred
    else:
        raise ReferenceInferenceError(f"Can not infer reference for '{fname}'.\nMultiple matches found")


def infer_reference_genome_from_fnames(fnames, expected=None, spec=None):
    """Infer the reference genome from a group of filenames, raising an error if they're not all the same."""
    refs = {x: infer_reference_genome_from_fname(x, expected=expected, spec=spec) for x in fnames}
    ref = next(iter(refs.values()))
    if not all(ref == x for x in refs.values()):
        raise ValueError("Input files have different input references.\n'{}'".format(refs))
    return ref


def _get_ordered_contigs_and_lengths(ref):
    def _strip_chr_or_none(name):
        if name.startswith("chr"):
            return name[3:]
        else:
            return None

    def iter_weird_variants(name):
        for prefix in [
            "KI",  # chr16_KI270728v1_random'
            "GL",  # chr14_GL000225v1_random
            "KQ",  # KQ759759.1
        ]:
            index = name.find(prefix)
            if index >= 0:
                yield name[index : index + 10]
                yield name[index : index + 10].replace("v", ".")

        return

    assert ref in REFERENCE_ANNOTATIONS
    contigs = []
    contig_lengths = OrderedDict()
    with open(get_data_path(f"{ref}.chrom.sizes")) as fp:
        for name, length in (line.split() for line in fp):
            contigs.append(name)
            contig_lengths[name] = int(length)
            if _strip_chr_or_none(name):
                contigs.append(_strip_chr_or_none(name))
                contig_lengths[_strip_chr_or_none(name)] = int(length)
            for name_variant in iter_weird_variants(name):
                contigs.append(name_variant)
                contig_lengths[name_variant] = int(length)

    return contigs, contig_lengths


CONTIGS = {}
CONTIG_LENGTHS = {}
for ref in REFERENCE_ANNOTATIONS:
    _contigs, _lengths = _get_ordered_contigs_and_lengths(ref)
    CONTIGS[ref] = _contigs
    CONTIG_LENGTHS[ref] = _lengths

# Discussion of various contig types
# https://software.broadinstitute.org/gatk/documentation/article?id=7857
SEX_CHROMS = ["chrX", "chrY"]
AUTOSOMES = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
]
MITOCHONDRIAL_CHROMS = [
    "chrM",
]
STANDARD_CHROMS = AUTOSOMES + SEX_CHROMS


def _standardize_name(contig):
    if not contig.startswith("chr"):
        return "chr" + contig


def contig_is_autosome(contig):
    return _standardize_name(contig) in AUTOSOMES


def contig_is_assembled(contig):
    contig = _standardize_name(contig)
    return contig in AUTOSOMES or contig in SEX_CHROMS or contig in MITOCHONDRIAL_CHROMS


def contig_is_alternate(contig):
    if contig.endswith("_alt"):
        return True
    if contig[-5:-1] == "_hap":
        return True
    return False


def contig_is_unlocalized(contig):
    return contig.endswith("random")


def contig_is_unplaced(contig):
    return _standardize_name(contig.startswith("chrU"))


class ChromOrdering(Enum):
    natural = "natural"
    lexicographical = "lexicographical"


def make_chrom_sizes_file(out_fname, assembly, chroms=None):
    """
    Creates a chrom.sizes file for a given assembly with a given set of chromosomes

    :param assembly: one of REFERENCE_ANNOTATIONS
    :param chroms: list of chromosomes to use.  Defaults to STANDARD_CHROMS.
    """
    if chroms is None:
        chroms = STANDARD_CHROMS

    with open(out_fname, "w") as fp:
        for chrom in chroms:
            length = CONTIG_LENGTHS[assembly][chrom]
            fp.write(f"{chrom}\t{length}\n")


_CHROMOSOME_Q_ARM_STARTS_HG38 = None  #: cache for chromosome q_arm_starts


def get_flattened_genome_offsets(assembly):
    """
    A flattened genome is one where the chromosomes have been flattened so that chr2's first position
    is chr1.length + 1, and so on.  Mainly used for plotting.

    >>> from pprint import pprint

    Note that pprint sorts the chromosome keys before printing
    >>> pprint(get_flattened_genome_offsets('hg38'), indent=1)
    {'chr1': 0,
     'chr10': 1674883629,
     'chr11': 1808681051,
     'chr12': 1943767673,
     'chr13': 2077042982,
     'chr14': 2191407310,
     'chr15': 2298451028,
     'chr16': 2400442217,
     'chr17': 2490780562,
     'chr18': 2574038003,
     'chr19': 2654411288,
     'chr2': 248956422,
     'chr20': 2713028904,
     'chr21': 2777473071,
     'chr22': 2824183054,
     'chr3': 491149951,
     'chr4': 689445510,
     'chr5': 879660065,
     'chr6': 1061198324,
     'chr7': 1232004303,
     'chr8': 1391350276,
     'chr9': 1536488912,
     'chrX': 2875001522,
     'chrY': 3031042417}
    """
    flattened_chrom_offsets = dict()
    for i, chrom in enumerate(STANDARD_CHROMS):
        if i == 0:
            flattened_chrom_offsets[chrom] = 0
        elif i == 1:
            # offset should be chr1 length
            assert chrom == "chr2"
            flattened_chrom_offsets[chrom] = CONTIG_LENGTHS[assembly][STANDARD_CHROMS[i - 1]]
        else:
            flattened_chrom_offsets[chrom] = (
                flattened_chrom_offsets[STANDARD_CHROMS[i - 1]]
                + CONTIG_LENGTHS[assembly][STANDARD_CHROMS[i - 1]]
            )
    return flattened_chrom_offsets


def get_chromosome_q_arm_starts(assembly: str) -> Dict[str, int]:
    """
    Gets the chromosome -> long arm start positions
    >>> from pprint import pprint
    >>> pprint(get_chromosome_q_arm_starts('hg38'), indent=1)
    {'chr1': 123400000,
     'chr10': 39800000,
     'chr11': 53400000,
     'chr12': 35500000,
     'chr13': 17700000,
     'chr14': 17200000,
     'chr15': 19000000,
     'chr16': 36800000,
     'chr17': 25100000,
     'chr18': 18500000,
     'chr19': 26200000,
     'chr2': 93900000,
     'chr20': 28100000,
     'chr21': 12000000,
     'chr22': 15000000,
     'chr3': 90900000,
     'chr4': 50000000,
     'chr5': 48800000,
     'chr6': 59800000,
     'chr7': 60100000,
     'chr8': 45200000,
     'chr9': 43000000,
     'chrX': 61000000,
     'chrY': 10400000}
    """
    from fbio.formats import BedReader

    global _CHROMOSOME_Q_ARM_STARTS_HG38
    if assembly != "hg38":
        raise NotImplementedError("Implemented only for hg38")

    if _CHROMOSOME_Q_ARM_STARTS_HG38 is None:
        with load_data_manifest(DEFAULT_DATA_MANIFEST_PATH) as dm:
            fname = dm.sync_and_get("annotations/GRCh38_extras/hg38.cytobands.bed.gz").path
            df = BedReader.load_dataframe(fname)
            # get the first occurance of a cytoband that starts with a "q" for each chromosome
            long_arm_starts = dict(
                df[df["name"].str.startswith("q")].groupby("chrom").first()["start"].items()
            )
            _CHROMOSOME_Q_ARM_STARTS_HG38 = long_arm_starts

    return _CHROMOSOME_Q_ARM_STARTS_HG38
