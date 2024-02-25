import pytest

from fragmentomics_tools.contig import infer_reference_genome_from_fname, ReferenceInferenceError


def test_infer_reference_genome_from_fname():
    assert infer_reference_genome_from_fname("tmp.hg19.bed") == "hg19"
    assert infer_reference_genome_from_fname("tmp.hg38.bed") == "hg38"
    assert infer_reference_genome_from_fname("GRCh37/tmp.bed") == "hg19"
    assert infer_reference_genome_from_fname("grch37/tmp.bed") == "hg19"
    assert infer_reference_genome_from_fname("tmp.hg38.bed") == "hg38"
    assert infer_reference_genome_from_fname("GRCh38/tmp.bed") == "hg38"

    with pytest.raises(ReferenceInferenceError):
        infer_reference_genome_from_fname("tmp.hg39.bed")
    with pytest.raises(ReferenceInferenceError):
        infer_reference_genome_from_fname("hg19/tmp.hg38.bed")
    with pytest.raises(ReferenceInferenceError):
        infer_reference_genome_from_fname("GRCh37/tmp.hg38.bed")
