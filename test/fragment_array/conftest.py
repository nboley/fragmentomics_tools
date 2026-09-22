import os
import tempfile

import pytest

from fragments_h5.fragments_h5 import build_fragments_h5

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")

# The fixture BAM/FASTA cover chr6:99,110,000-99,130,000. This locus has 11
# fragments within +/-256 and 12 within +/-512, which is why tests that need an
# even fragment count (e.g. the k-way split) use the wider window.
TEST_CHROM = "chr6"
TEST_POS = 99119615


@pytest.fixture(scope="session")
def bam_path():
    return os.path.join(DATA_DIR, "small.chr6.bam")


@pytest.fixture(scope="session")
def fasta_file_path():
    return os.path.join(DATA_DIR, "GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")


@pytest.fixture(scope="session")
def small_h5_path(bam_path, fasta_file_path):
    """Build the fragments h5 once per session.

    Building scans the full chr6 reference and takes ~25s, so this is session
    scoped and shared by every test module in this directory.
    """
    with tempfile.TemporaryDirectory() as dirname:
        ofname = os.path.join(dirname, os.path.basename(bam_path) + ".frag.h5")
        build_fragments_h5(bam_path, ofname, fasta_filename=fasta_file_path)
        yield ofname
