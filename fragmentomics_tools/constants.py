import os

# from contig import CONTIG_LENGTHS, CONTIGS, STANDARD_CHROMS, REFERENCE_FASTA_DATA_MANIFEST_KEY


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # /home/user/src/fragmentomics_tools/
DATA_DIR = os.path.join(BASE_DIR, "data")  # /home/user/src/fragmentomics_tools/data


DEFAULT_REFERENCE = "hg38"


DEFAULT_VPLOT_SUMPOOL_BY = 16
# preserve power of two, when including the 0 frag.  This constant is primarily for ML models
DEFAULT_MAX_FRAG_LEN = 511
DEFAULT_MIN_MAPQ = 0


def get_data_path(p):
    assert not p.startswith("/")
    return os.path.join(DATA_DIR, p)


def get_repo_data_path(rel_path):
    repo_data_path = os.path.abspath(
        os.path.join(BASE_DIR, "./repo_data/")
    )
    return os.path.abspath(os.path.join(repo_data_path, rel_path))
