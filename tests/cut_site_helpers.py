"""Shared helpers for the cut-site simulator tests: the toy genome, h5
building, the recording rng and the draw-frame builders.

One home, so the split test files (owner decision 188) do not each carry a
copy.  Fixtures built from these are in ``tests/conftest.py``.  Moved verbatim
from ``tests/test_cut_site_simulator.py``.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pysam

import cut_site_oracle as oracle

from background_model.constants import HEX_HALF, L_MAX


TESTS_DATA = os.path.join(os.path.dirname(__file__), "data")
CHR6_FASTA = os.path.join(TESTS_DATA, "GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")
GOLDEN_H5 = os.path.join(TESTS_DATA, "golden.small.chr6.frag.h5")

TANDEM_HEX = "AACGTC"
TANDEM_RC = oracle.RC(TANDEM_HEX)
TANDEM_REPEATS = 50
DB_CORE_LEN = 4101  # de Bruijn B(4,6): 4096 + 5


def _build_toy_genome():
    """Build a synthetic contig with known properties."""
    core = oracle.de_bruijn(4, 6)
    assert len(core) == DB_CORE_LEN

    tandem = TANDEM_HEX * TANDEM_REPEATS  # 300 bp
    single_n = "A"  # pad before N
    n_block = "ACGTAC" + "N" + "ACGTAC"  # single N in context
    n_run = "N" * 10
    lowercase = core[100:200].lower()  # 100 bp soft-masked copy

    genome = core + tandem + n_block + n_run + lowercase
    filler_len = 6200 - len(genome)
    assert filler_len > 0, f"genome is {len(genome)} bp, need < 6200"
    rng = np.random.RandomState(42)
    filler = "".join("ACGT"[b] for b in rng.randint(0, 4, filler_len))
    genome = genome + filler
    return genome


def _write_fasta(genome, tmpdir, contig="chrT"):
    """Write genome to FASTA and index it. Returns path."""
    fa_path = os.path.join(tmpdir, "toy.fa")
    with open(fa_path, "w") as f:
        f.write(f">{contig}\n")
        for i in range(0, len(genome), 80):
            f.write(genome[i:i + 80] + "\n")
    pysam.faidx(fa_path)
    return fa_path


def _build_h5(bed_rows, fasta_path, tmpdir, name="test"):
    """Write BED rows, bgzip+tabix, build h5. Returns h5 path.

    ``bed_rows``: list of (contig, start, stop, strand, mapq1, mapq2).
    """
    bed_path = os.path.join(tmpdir, f"{name}.bed")
    with open(bed_path, "w") as f:
        for contig, start, stop, strand, mq1, mq2 in sorted(
            bed_rows, key=lambda r: (r[0], r[1], r[2])
        ):
            f.write(f"{contig}\t{start}\t{stop}\t\t0\t{strand}\t{mq1}\t{mq2}\n")

    gz = pysam.tabix_index(bed_path, preset="bed", force=True)
    h5_path = os.path.join(tmpdir, f"{name}.frag.h5")
    subprocess.run(
        ["build-fragments-h5", gz, h5_path, "--fasta", fasta_path, "--quiet"],
        check=True,
        capture_output=True,
    )
    return h5_path


class _RecordingRng:
    """Records the exact call sequence sample_region makes."""

    def __init__(self, n_plus_values, choice_returns, random_values):
        self._n_plus_iter = iter(n_plus_values)
        self._choice_returns = list(choice_returns)
        self._choice_idx = 0
        self._random_values = list(random_values)
        self._random_idx = 0
        self.recorded_start_weights = []
        self.recorded_binomial_calls = []

    def binomial(self, n, p):
        self.recorded_binomial_calls.append((n, p))
        return next(self._n_plus_iter)

    def choice(self, a, size=None, replace=True, p=None):
        self.recorded_start_weights.append(p.copy() if p is not None else None)
        result = self._choice_returns[self._choice_idx]
        self._choice_idx += 1
        return np.array(result)

    def random(self, shape):
        vals = self._random_values[self._random_idx]
        self._random_idx += 1
        return np.array(vals).reshape(shape)


def _draw_frame(genome, regions, *, contigs=None, region_index=None):
    """A frame ``simulate_fragments_to_bed`` accepts, built without an h5.

    The writer reads only coordinates, ``fragment_array.length`` and the
    padded ``sequence``, so empty fragment arrays suffice.  ``contig`` is a
    LABEL to the writer -- the draw reads ``sequence`` only -- which lets two
    rows carry identical sequence yet stay distinguishable in the output.
    """
    from fragmentomics_tools import RegionFragmentArray
    from fragmentomics_tools.region import Region

    return pd.DataFrame({
        "contig": contigs if contigs is not None else ["chrT"] * len(regions),
        "start": [g0 for g0, _ in regions],
        "stop": [g1 for _, g1 in regions],
        "fragment_array": [
            RegionFragmentArray([], [], Region("chrT", g0, g1), L_MAX)
            for g0, g1 in regions
        ],
        "sequence": [
            genome[g0 - HEX_HALF:g1 + L_MAX + HEX_HALF].encode()
            for g0, g1 in regions
        ],
        "region_index": (np.arange(len(regions)) if region_index is None
                         else np.asarray(region_index)),
    })


def _read_outputs(bed_path, stats):
    """``(bed_lines, sidecar_lines)`` -- the sidecar decompressed, since the
    gzip header carries an mtime."""
    import gzip
    with open(bed_path) as f:
        bed = f.read().splitlines()
    with gzip.open(stats["p_sidecar"], "rt") as f:
        side = f.read().splitlines()
    return bed, side
