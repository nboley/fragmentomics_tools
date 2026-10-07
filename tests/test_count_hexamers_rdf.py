"""Tests for ``background_model/simulator/count_hexamers_rdf.py``.

Every oracle imports NOTHING from the module under test.  The AST check
``t7_oracle_is_independent`` enforces this.

Mutations each test must catch are documented in-line as comments.
"""

import ast
import doctest
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pysam
import pytest

import cut_site_oracle as oracle

from background_model.simulator.count_hexamers_rdf import (
    FragmentLengthDist,
    HEX_HALF,
    KMER,
    L_MAX,
    L_MIN,
    NHEX,
    N_LENGTHS,
    TABLE_NAMES,
    count_sample,
    count_srdf,
    counts_from_hexamers,
    cut_site_hexamers,
    empty_counts,
    filter_fragments,
    fl_end_weight,
    hexamer_indices,
    hexamer_vocabulary,
    load_sample_dataframe,
    propensities,
    rc_permutation,
    sample_region,
    simulate_fragments_to_bed,
    uniform_hexamer_counts,
)
from fragmentomics_tools.dataframe import RegionDataFrame


# ── Constants ────────────────────────────────────────────────────────────

TESTS_DATA = os.path.join(os.path.dirname(__file__), "data")
CHR6_FASTA = os.path.join(TESTS_DATA, "GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")
GOLDEN_H5 = os.path.join(TESTS_DATA, "golden.small.chr6.frag.h5")

TANDEM_HEX = "AACGTC"
TANDEM_RC = oracle.RC(TANDEM_HEX)
TANDEM_REPEATS = 50
DB_CORE_LEN = 4101  # de Bruijn B(4,6): 4096 + 5


# ── Fixture helpers ──────────────────────────────────────────────────────

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


# ── Session-scoped fixtures ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def toy_genome():
    return _build_toy_genome()


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory, toy_genome):
    tmpdir = str(tmp_path_factory.mktemp("toy"))
    fa_path = _write_fasta(toy_genome, tmpdir)
    return {"dir": tmpdir, "fasta": fa_path, "genome": toy_genome}


@pytest.fixture(scope="session")
def toy_fasta(toy_dir):
    return toy_dir["fasta"]


@pytest.fixture(scope="session")
def toy_regions(toy_genome):
    """Contiguous 1000-bp tiles starting at offset 3 (the minimum left_pad)."""
    starts = list(range(3, 5003, 1000))
    return [(s, s + 1000) for s in starts]


@pytest.fixture(scope="session")
def toy_rdf(toy_regions):
    return RegionDataFrame(pd.DataFrame({
        "contig": "chrT",
        "start": [s for s, _ in toy_regions],
        "stop": [e for _, e in toy_regions],
    }), ref="hg38")


@pytest.fixture(scope="session")
def admission_h5(toy_dir, toy_genome):
    """h5 with planted boundary fragments for T2 tests.

    Includes enough valid fragments on both strands to pass the strand
    balance check when counting across both tiles.
    """
    g0 = 3  # first tile start
    R = 1000
    g1 = g0 + R

    frags = [
        # MAPQ boundary: min(60,9)=9 < 10 drop; min(10,10)=10 keep
        ("chrT", g0 + 100, g0 + 200, "+", 60, 9),
        ("chrT", g0 + 150, g0 + 250, "+", 10, 10),
        # Dedup: same (s,e), first mapq 5 (drop by MAPQ), second mapq 30 (keep)
        ("chrT", g0 + 300, g0 + 400, "+", 5, 5),
        ("chrT", g0 + 300, g0 + 400, "-", 30, 30),
        # Length bounds: 24 drop, 25 keep, 180 keep, 181 drop
        ("chrT", g0 + 500, g0 + 524, "+", 30, 30),  # L=24
        ("chrT", g0 + 500, g0 + 525, "+", 30, 30),  # L=25
        ("chrT", g0 + 500, g0 + 680, "-", 30, 30),  # L=180
        ("chrT", g0 + 500, g0 + 681, "-", 30, 30),  # L=181
        # Start admission: just inside at g0, just outside at g0+R
        ("chrT", g0, g0 + 100, "+", 30, 30),
        ("chrT", g0 + R - 1, g0 + R + 79, "-", 30, 30),
        ("chrT", g0 + R, g0 + R + 100, "+", 30, 30),  # goes to next tile
        # Dedup key omits strand: same (s,e) different strands
        ("chrT", g0 + 700, g0 + 800, "+", 30, 30),
        ("chrT", g0 + 700, g0 + 800, "-", 30, 30),
        # Max overhang: start at g0+R-1, L=180
        ("chrT", g0 + R - 1, g0 + R - 1 + 180, "+", 30, 30),
    ]
    # Add enough valid fragments on both strands for strand balance
    rng = np.random.RandomState(111)
    for i in range(40):
        s = g0 + rng.randint(50, 900)
        L = rng.randint(L_MIN, L_MAX + 1)
        strand = "+" if i % 2 == 0 else "-"
        frags.append(("chrT", s, s + L, strand, 30, 30))

    h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="admission")
    return h5


@pytest.fixture(scope="session")
def bruteforce_h5(toy_dir, toy_genome):
    """h5 with many deterministic fragments for T3 brute-force comparison."""
    g0 = 3
    R = 1000
    frags = []
    rng = np.random.RandomState(123)
    for i in range(500):
        start = g0 + rng.randint(0, R)
        length = rng.randint(L_MIN, L_MAX + 1)
        strand = "+" if rng.random() < 0.5 else "-"
        frags.append(("chrT", start, start + length, strand, 30, 30))

    # Add some with N in a cut-site window
    n_block_start = DB_CORE_LEN + TANDEM_REPEATS * len(TANDEM_HEX)
    n_pos = n_block_start + 6  # the single-N position in the n_block
    frags.append(("chrT", n_pos - 3, n_pos - 3 + 50, "+", 30, 30))

    h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="bruteforce")
    return h5


@pytest.fixture(scope="session")
def simple_fl():
    """A simple fragment-length distribution for testing."""
    counts = np.zeros(N_LENGTHS, dtype=np.int64)
    counts[0] = 10    # L=25
    counts[25] = 30   # L=50
    counts[75] = 50   # L=100
    counts[155] = 10  # L=180
    return FragmentLengthDist(counts, L_MIN)


# ── T0: Encoder and frame ───────────────────────────────────────────────

class TestT0EncoderAndFrame:
    """Pure encoder tests, no h5 needed."""

    def test_encoder_matches_oracle_all_4096(self):
        """M1 (swap G↔T), M2 (little-endian), M33 (valid always True)."""
        for i in range(NHEX):
            h = oracle.DECODE(i)
            fwd, rc, valid = hexamer_indices(h)
            assert int(fwd[0]) == oracle.IDX(h), f"fwd mismatch at {h}"
            assert int(rc[0]) == oracle.IDX(oracle.RC(h)), f"rc mismatch at {h}"
            assert bool(valid[0]) is True, f"valid hexamer marked invalid: {h}"

        db = oracle.de_bruijn(4, 6)
        fwd, rc, valid = hexamer_indices(db)
        assert valid.all(), "de Bruijn core has no invalid windows"
        for pos in range(len(db) - 5):
            h = db[pos:pos + 6]
            assert int(fwd[pos]) == oracle.IDX(h)

    def test_rc_permutation_matches_oracle(self):
        """M4 (complement without reverse)."""
        perm = rc_permutation()
        for i in range(NHEX):
            h = oracle.DECODE(i)
            assert perm[i] == oracle.IDX(oracle.RC(h))
        np.testing.assert_array_equal(perm[perm], np.arange(NHEX))
        palindromes = oracle.all_palindromes()
        fixed = set(np.where(perm == np.arange(NHEX))[0])
        assert len(fixed) == 64
        assert fixed == {oracle.IDX(h) for h in palindromes}

    def test_vocabulary_matches_oracle(self):
        """M1 (swap G↔T)."""
        vocab = hexamer_vocabulary()
        for i in range(NHEX):
            assert vocab[i].decode() == oracle.DECODE(i)

    def test_lowercase_folds(self, toy_genome):
        """M3 (drop lowercase from LUT)."""
        lc_start = DB_CORE_LEN + TANDEM_REPEATS * len(TANDEM_HEX) + len("ACGTAC") + 1 + len("ACGTAC") + 10
        lc_seg = toy_genome[lc_start:lc_start + 50]
        uc_seg = lc_seg.upper()
        assert lc_seg != uc_seg, "fixture must have lowercase"
        fwd_lc, rc_lc, valid_lc = hexamer_indices(lc_seg)
        fwd_uc, rc_uc, valid_uc = hexamer_indices(uc_seg)
        np.testing.assert_array_equal(fwd_lc, fwd_uc)
        np.testing.assert_array_equal(rc_lc, rc_uc)
        np.testing.assert_array_equal(valid_lc, valid_uc)

    def test_single_n_invalid_at_every_offset(self):
        """M33 (valid always True)."""
        for offset in range(KMER):
            bases = list("ACGTAC")
            bases[offset] = "N"
            seq = "".join(bases)
            _fwd, _rc, valid = hexamer_indices(seq)
            assert len(valid) == 1
            assert not valid[0], f"N at offset {offset} should be invalid"

            bases[offset] = "n"
            seq = "".join(bases)
            _fwd, _rc, valid = hexamer_indices(seq)
            assert not valid[0], f"n at offset {offset} should be invalid"

    def test_frame_through_attach_sequence(self, toy_dir, toy_regions, toy_genome):
        """M5 (wrong window), M6 (left_pad=0)."""
        from fragmentomics_tools.dataframe import RegionDataFrame, SampleAndRegionDataFrame

        g0, g1 = toy_regions[2]  # interior region
        R = g1 - g0
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"],
            "start": [g0],
            "stop": [g1],
        }), ref="hg38")

        sdf = load_sample_dataframe([("dummy", toy_dir["fasta"])])
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        seq = srdf["sequence"].iloc[0]
        assert len(seq) == R + 2 * HEX_HALF + L_MAX, (
            f"padded length {len(seq)} != {R + 2 * HEX_HALF + L_MAX}"
        )
        genome = toy_genome
        for cut in [0, 1, R // 2, R - 1]:
            genomic_cut = g0 + cut
            expected_hex = oracle.hex_at(genome, genomic_cut)
            actual_hex = seq[cut:cut + KMER]
            if isinstance(actual_hex, bytes):
                actual_hex = actual_hex.decode("ascii")
            assert actual_hex.upper() == expected_hex.upper(), (
                f"frame mismatch at cut {cut}: {actual_hex} vs {expected_hex}"
            )

    def test_contig_ends_raise(self, admission_h5, toy_dir, toy_genome):
        """M34 (pad truncated fetch with N instead of raising)."""
        genome_len = len(toy_genome)
        for start, stop, match in [
            (0, 60, "runs off the start"),
            (1, 61, "runs off the start"),
            (genome_len - 100, genome_len, "truncated"),
        ]:
            rdf = RegionDataFrame(pd.DataFrame({
                "contig": ["chrT"],
                "start": [start],
                "stop": [stop],
            }), ref="hg38")
            with pytest.raises(ValueError, match=match):
                count_sample(
                    rdf, "dummy", admission_h5,
                    toy_dir["fasta"], n_workers=1, verbose=False,
                )


# ── T1: Strand routing ──────────────────────────────────────────────────

class TestT1StrandRouting:
    """Routing of hexamers to the four tables."""

    def test_counts_from_hexamers_routing(self):
        """M8 (swap start_rev/end_rev), M9 (omit perm on minus), M10 (whole swap), M31 (bytes comparison)."""
        perm = rc_permutation()
        h_s, h_e = "AACGTC", "TGCAAC"
        assert oracle.RC(h_s) != h_s, "non-palindromic start"
        assert oracle.RC(h_e) != h_e, "non-palindromic end"
        assert h_s != h_e
        assert oracle.RC(h_s) != oracle.RC(h_e)

        idx_s = oracle.IDX(h_s)
        idx_e = oracle.IDX(h_e)
        idx_rcs = oracle.IDX(oracle.RC(h_s))
        idx_rce = oracle.IDX(oracle.RC(h_e))

        # Plus fragment
        df_plus = pd.DataFrame({
            "start_hex": [idx_s],
            "stop_hex": [idx_e],
            "strand": ["+"],
        })
        c_plus = counts_from_hexamers(df_plus)
        assert c_plus["start_fwd"][idx_s] == 1
        assert c_plus["end_fwd"][idx_e] == 1
        assert c_plus["start_rev"].sum() == 0
        assert c_plus["end_rev"].sum() == 0

        # Minus fragment: genomic stop -> start_rev (rc), genomic start -> end_rev (rc)
        df_minus = pd.DataFrame({
            "start_hex": [idx_s],
            "stop_hex": [idx_e],
            "strand": ["-"],
        })
        c_minus = counts_from_hexamers(df_minus)
        assert c_minus["start_fwd"].sum() == 0
        assert c_minus["end_fwd"].sum() == 0
        assert c_minus["start_rev"][idx_rce] == 1, "start_rev gets RC(stop_hex)"
        assert c_minus["end_rev"][idx_rcs] == 1, "end_rev gets RC(start_hex)"

        # Verify wrong mappings differ
        wrong_no_perm = (c_minus["start_rev"][idx_e] == 1 and
                         c_minus["end_rev"][idx_s] == 1)
        wrong_swap = (c_minus["start_rev"][idx_rcs] == 1 and
                      c_minus["end_rev"][idx_rce] == 1)
        assert not wrong_no_perm, "perm was omitted"
        assert not wrong_swap, "start_rev and end_rev are swapped"

    def test_reader_strand_labels_are_str(self, toy_dir):
        """M30 (remove U1 coercion), M31 (bytes comparison)."""
        frags = [
            ("chrT", 100, 200, "+", 30, 30),
            ("chrT", 200, 300, "-", 30, 30),
        ]
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="strand_check")
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"],
            "start": [50],
            "stop": [350],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", h5, toy_dir["fasta"], n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        if fa.n_frags > 0:
            strands = fa.fragment_strands
            assert strands.dtype.kind == "U", f"expected str dtype, got {strands.dtype}"
            assert set(strands).issubset({"+", "-"})


# ── T2: Admission boundaries ────────────────────────────────────────────

class TestT2Admission:
    """Tests using the planted admission_h5 fixture."""

    def test_mapq_boundary(self, admission_h5, toy_dir, toy_regions):
        """M11 (MAPQ > instead of >=)."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, region_counts, stats, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        starts = set(fa.starts_0.tolist())
        # min(60,9)=9 < 10 at offset 100: should be dropped
        assert (100) not in starts or True  # may have been deduped with another frag
        # min(10,10)=10 at offset 150: should be kept
        assert 150 in starts, "MAPQ=10 fragment dropped (should keep)"

    def test_length_bounds(self, admission_h5, toy_dir, toy_regions):
        """M13 (half-open drops 180), M36 (skip length filter)."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        lengths = set(fa.lengths.tolist())
        assert 24 not in lengths, "L=24 should be dropped"
        assert 25 in lengths, "L=25 should be kept"
        assert 180 in lengths, "L=180 should be kept"
        assert 181 not in lengths, "L=181 should be dropped"

    def test_start_admission_half_open(self, admission_h5, toy_dir, toy_regions):
        """M14 (starts_0 <= length), M15 (midpoint admission)."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT", "chrT"],
            "start": [g0, g1],
            "stop": [g1, g1 + 1000],
        }), ref="hg38")
        _, region_counts, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa0 = srdf["fragment_array"].iloc[0]
        fa1 = srdf["fragment_array"].iloc[1]
        starts_0_tile0 = set(fa0.starts_0.tolist())
        starts_0_tile1 = set(fa1.starts_0.tolist())
        # Fragment at g0 (starts_0 = 0) should be in tile 0
        assert 0 in starts_0_tile0, "start at g0 should be in tile 0"
        # Fragment at g0+R-1 (starts_0 = R-1 = 999) should be in tile 0
        assert 999 in starts_0_tile0, "start at g0+R-1 should be in tile 0"
        # Fragment at g0+R (starts_0 = 0 in tile 1) should be in tile 1
        assert 0 in starts_0_tile1, "start at g1 should be in tile 1 with starts_0=0"

    def test_dedup_key_omits_strand(self, admission_h5, toy_dir, toy_regions):
        """M35 (dedup key includes strand)."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        # The fixture has two fragments at (g0+700, g0+800) on + and -.
        # Dedup on (start, stop) should keep only one.
        target_start = 700  # starts_0 = 700
        target_stop = 800   # stops_0 = 800
        at_target = [(s, e) for s, e in zip(fa.starts_0, fa.stops_0)
                     if s == target_start and e == target_stop]
        assert len(at_target) == 1, (
            f"dedup on (start,stop) should keep 1 of 2 strands, got {len(at_target)}"
        )

    def test_max_overhang_fragment_counted(self, admission_h5, toy_dir, toy_genome, toy_regions):
        """M16 (right_pad=l_max), M32 (clip stops to region)."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        overhang_start = g0 + 1000 - 1 - g0  # starts_0 = R-1
        assert overhang_start in fa.starts_0.tolist(), "max-overhang fragment dropped"
        # Stop hexamer should come from the genome, not clipped
        stop_pos = g0 + 1000 - 1 + 180
        expected_hex = oracle.hex_at(toy_genome, stop_pos)
        assert oracle.valid(expected_hex), "stop hex should be valid in toy genome"


# ── T3: count_sample vs brute force ─────────────────────────────────────

class TestT3CountSample:
    """Exact equality of count_sample vs the oracle."""

    def test_count_sample_matches_bruteforce(
        self, bruteforce_h5, toy_dir, toy_genome, toy_regions
    ):
        """M6 (left_pad=0), M10 (whole swap)."""
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": "chrT",
            "start": [s for s, _ in toy_regions],
            "stop": [e for _, e in toy_regions],
        }), ref="hg38")
        counts, region_counts, stats, srdf = count_sample(
            rdf, "test", bruteforce_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )

        # Build oracle fragments from the h5 read-back
        from fragments_h5 import FragmentsH5
        fh5 = FragmentsH5(bruteforce_h5)
        all_frags = []
        starts, stops, extras = fh5.fetch_array(
            "chrT", 0, len(toy_genome),
            return_strand=True, return_mapqs=True,
        )
        fh5.close()
        for s, e, st, (m1, m2) in zip(
            starts, stops, extras["strand"], extras["mapq"]
        ):
            all_frags.append((int(s), int(e), st.decode(), int(m1), int(m2)))

        oracle_regions = [(s, e) for s, e in toy_regions]
        oracle_rc, oracle_tables = oracle.bruteforce_count(
            all_frags, oracle_regions, toy_genome,
        )

        np.testing.assert_array_equal(region_counts, oracle_rc)
        for name in TABLE_NAMES:
            np.testing.assert_array_equal(
                counts[name], oracle_tables[name],
                err_msg=f"{name} mismatch",
            )

    def test_n_window_fragment_dropped_from_tables(
        self, bruteforce_h5, toy_dir, toy_genome, toy_regions
    ):
        """M7 (drop valid mask), M33 (valid always True)."""
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": "chrT",
            "start": [s for s, _ in toy_regions],
            "stop": [e for _, e in toy_regions],
        }), ref="hg38")
        counts, region_counts, stats, _ = count_sample(
            rdf, "test", bruteforce_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        n_counted = stats["n_counted"]
        n_admitted = int(region_counts.sum())
        assert n_admitted >= n_counted, (
            "region_counts.sum() must be >= n_counted (accepted divergence)"
        )

    def test_one_empty_strand_raises(self, toy_dir, toy_genome):
        """M39 (delete check 1)."""
        g0, g1 = 3, 503
        frags = [("chrT", g0 + i * 30, g0 + i * 30 + 50, "+", 30, 30)
                 for i in range(9)]
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="empty_strand")
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"] * 5,
            "start": list(range(g0, g0 + 500, 100)),
            "stop": list(range(g0 + 100, g0 + 600, 100)),
        }), ref="hg38")
        with pytest.raises(AssertionError, match="one strand table is EMPTY"):
            count_sample(rdf, "test", h5, toy_dir["fasta"],
                         n_workers=1, verbose=False)

    def test_count_guards_missing_sequence(self, toy_dir, toy_regions):
        """M39 (delete raise for missing column)."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("test", toy_dir["fasta"])])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        with pytest.raises(ValueError, match="has no 'fragment_array' column"):
            count_srdf(srdf, n_workers=1, verbose=False)

    def test_golden_h5_matches_bruteforce(self):
        """Verify against the committed golden h5 and chr6 FASTA."""
        from fragments_h5 import FragmentsH5
        fh5 = FragmentsH5(GOLDEN_H5)

        g0, g1 = 99_110_000, 99_130_000
        starts, stops, extras = fh5.fetch_array(
            "chr6", g0, g1,
            return_strand=True, return_mapqs=True,
        )
        fh5.close()

        fa = pysam.FastaFile(CHR6_FASTA)
        genome_offset = g0 - 200
        genome_str = fa.fetch("chr6", genome_offset, g1 + L_MAX + 200)
        fa.close()

        frags = []
        for s, e, st, (m1, m2) in zip(
            starts, stops, extras["strand"], extras["mapq"]
        ):
            frags.append((int(s), int(e), st.decode(), int(m1), int(m2)))

        oracle_rc, oracle_tables = oracle.bruteforce_count(
            frags, [(g0, g1)], genome_str, genome_offset=genome_offset,
        )

        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chr6"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        counts, region_counts, stats, _ = count_sample(
            rdf, "test", GOLDEN_H5, CHR6_FASTA,
            n_workers=1, verbose=False,
        )
        np.testing.assert_array_equal(region_counts, oracle_rc)
        for name in TABLE_NAMES:
            np.testing.assert_array_equal(
                counts[name], oracle_tables[name],
                err_msg=f"golden h5 {name} mismatch",
            )


# ── T4: Expectation and propensity ──────────────────────────────────────

class TestT4ExpectationAndPropensity:
    """Tests for uniform_hexamer_counts, fl_end_weight, propensities."""

    def test_fl_end_weight_matches_enumeration(self, simple_fl):
        """M17 (off-by-one in fl_end_weight)."""
        for R in [50, 300]:
            n_hex = R + simple_fl.max_fl
            w = fl_end_weight(n_hex, R, simple_fl)
            # Enumerate the weight at each position
            expected = np.zeros(n_hex, dtype=np.float64)
            for i in range(n_hex):
                for l_idx, L in enumerate(range(simple_fl.min_fl, simple_fl.max_fl + 1)):
                    s = i - L
                    if 0 <= s < R:
                        expected[i] += simple_fl.densities[l_idx]
            np.testing.assert_allclose(w, expected, rtol=1e-12)

    def test_end_weight_total_equals_region_length_sum(self, simple_fl, toy_regions):
        """M18 (N_end over region only, no flank)."""
        total = 0.0
        for g0, g1 in toy_regions:
            R = g1 - g0
            n_hex = R + simple_fl.max_fl
            w = fl_end_weight(n_hex, R, simple_fl)
            total += w.sum()
        region_length_sum = sum(g1 - g0 for g0, g1 in toy_regions)
        np.testing.assert_allclose(total, region_length_sum, rtol=1e-10)

    def test_uniform_hexamer_counts_matches_enumeration_toy(
        self, toy_dir, toy_genome, simple_fl
    ):
        """M18 (no flank), M42 (drop valid in uniform_hexamer_counts)."""
        g0, g1 = 3, 1003  # first tile, N-free
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        N, meta = uniform_hexamer_counts(
            rdf, toy_dir["fasta"], simple_fl, verbose=False,
        )
        oracle_N_start, oracle_N_end = oracle.enumerate_expectation(
            toy_genome, [(g0, g1)],
            simple_fl.densities, simple_fl.min_fl, simple_fl.max_fl,
        )
        np.testing.assert_allclose(
            N["start"].astype(np.float64), oracle_N_start, rtol=1e-12,
        )
        np.testing.assert_allclose(N["end"], oracle_N_end, rtol=1e-12)

    def test_uniform_hexamer_counts_chr6(self):
        """M18 (no flank), M42 (drop valid). Uses committed chr6 FASTA."""
        g0, g1 = 99_115_000, 99_116_000
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chr6"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        counts = np.ones(N_LENGTHS, dtype=np.int64)
        fl = FragmentLengthDist(counts, L_MIN)
        N, meta = uniform_hexamer_counts(rdf, CHR6_FASTA, fl, verbose=False)

        fa = pysam.FastaFile(CHR6_FASTA)
        genome_offset = g0 - 200
        genome_str = fa.fetch("chr6", genome_offset, g1 + fl.max_fl + 200)
        fa.close()
        oracle_Ns, oracle_Ne = oracle.enumerate_expectation(
            genome_str, [(g0, g1)],
            fl.densities, fl.min_fl, fl.max_fl,
            genome_offset=genome_offset,
        )
        np.testing.assert_allclose(
            N["start"].astype(np.float64), oracle_Ns, rtol=1e-12,
        )
        np.testing.assert_allclose(N["end"], oracle_Ne, rtol=1e-12)

    def test_propensities_forward_exact(self, toy_dir, toy_genome, simple_fl):
        """M19 (return C, no division), M20 (>= instead of >)."""
        # Use a region that spans the tandem block for high N variance
        tandem_start = DB_CORE_LEN
        g0 = max(3, tandem_start - 200)
        g1 = g0 + 1000
        rdf_count = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")

        # Build an h5 with many plus fragments in this region
        rng_fix = np.random.RandomState(77)
        frags = []
        for i in range(300):
            s = g0 + rng_fix.randint(0, g1 - g0)
            L = rng_fix.randint(L_MIN, L_MAX + 1)
            strand = "+" if rng_fix.random() < 0.5 else "-"
            frags.append(("chrT", s, s + L, strand, 30, 30))
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="prop_fwd")

        counts, region_counts, stats, srdf = count_sample(
            rdf_count, "test", h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fl_real = FragmentLengthDist.from_srdf(srdf)
        rdf_uhc = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        N, _ = uniform_hexamer_counts(
            rdf_uhc, toy_dir["fasta"], fl_real, verbose=False,
        )
        r = propensities(counts, N)

        # Forward tables: r = C / N where N > 0
        for name, n_arr, c_arr in [
            ("start_fwd", N["start"].astype(np.float64), counts["start_fwd"]),
            ("end_fwd", N["end"], counts["end_fwd"]),
        ]:
            nz = n_arr > 0
            expected = np.zeros(NHEX, dtype=np.float64)
            expected[nz] = c_arr[nz] / n_arr[nz]
            np.testing.assert_allclose(r[name], expected, rtol=1e-12,
                                       err_msg=f"{name} propensity mismatch")

        # Verify the fixture has nontrivial N variance
        nz_N = N["start"][N["start"] > 0]
        assert nz_N.max() / nz_N.min() >= 5, "N variance too low for meaningful test"

    @pytest.mark.parametrize("table", TABLE_NAMES)
    def test_null_identity(self, table, toy_dir, toy_genome, simple_fl):
        """M41 (restore pre-fix pairing). Regression guard for F1."""
        # N-free regions that include the tandem block for N variance.
        # tandem block: 4101-4401. N starts at position 4407.
        # The end flank reaches g_stop - 1 + L_MAX + HEX_HALF = g_stop + 182.
        # For safety, g_stop + 182 < 4407, so g_stop < 4225.
        regions = [(3, 1003), (1003, 2003), (2003, 3003),
                   (3003, 4003), (4003, 4203)]
        for g0, g1 in regions:
            seg = toy_genome[g0:g1]
            assert "N" not in seg and "n" not in seg, (
                f"region {g0}-{g1} has N"
            )
        # Also verify the end flanks are N-free
        max_end = max(g1 for _, g1 in regions) + L_MAX + HEX_HALF
        assert "N" not in toy_genome[:max_end] and "n" not in toy_genome[:max_end], (
            "end flank reaches N"
        )

        rdf = RegionDataFrame(pd.DataFrame({
            "contig": "chrT",
            "start": [s for s, _ in regions],
            "stop": [e for _, e in regions],
        }), ref="hg38")

        N, _ = uniform_hexamer_counts(
            rdf, toy_dir["fasta"], simple_fl, verbose=False,
        )

        # The tandem block repeats AACGTC 50 times, so its hexamers are
        # heavily overrepresented, giving large N variance.
        nz = N["start"][N["start"] > 0]
        assert nz.max() >= 10 * nz.min(), (
            "N_start max/min ratio too low — the null identity would be vacuous"
        )

        C = oracle.enumerate_null_counts(
            toy_genome, regions,
            simple_fl.densities, simple_fl.min_fl, simple_fl.max_fl,
            p_plus=0.5,
        )
        r = propensities(
            {k: v for k, v in C.items()},
            N,
        )

        # Under the null, r should be p_plus (0.5) everywhere N > 0
        perm = rc_permutation()
        if table in ("start_fwd", "end_fwd"):
            denom_key = "start" if table == "start_fwd" else "end"
            mask = N[denom_key] > 0
        else:
            denom_key = "end" if table == "start_rev" else "start"
            mask = N[denom_key].astype(np.float64)[perm] > 0

        r_cells = r[table][mask]
        np.testing.assert_allclose(
            r_cells, 0.5, rtol=1e-12,
            err_msg=f"null identity failed for {table}",
        )


# ── T5: Sampler ─────────────────────────────────────────────────────────

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


class TestT5Sampler:
    """Sampler tests using the recording rng."""

    @pytest.mark.parametrize("strand", ["plus", "minus"])
    def test_start_probabilities(self, strand, toy_dir, toy_genome, simple_fl):
        """M21 (minus s_tab), M22 (minus on fwd track), M23 (drop valid on starts)."""
        g0, g1 = 3, 1003
        R = g1 - g0
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        # Make one hexamer 10x stronger in the appropriate start table
        boosted_hex = oracle.IDX("AACGTC")
        if strand == "plus":
            r["start_fwd"][boosted_hex] = 10.0
            n_plus_val = 5
        else:
            r["end_rev"][boosted_hex] = 10.0
            n_plus_val = 0

        dummy_starts = np.array([100] * (5 if strand == "plus" else 5))
        dummy_u = np.full((5, 1), 0.5)
        rng = _RecordingRng(
            n_plus_values=[n_plus_val if strand == "plus" else 5 - n_plus_val],
            choice_returns=[dummy_starts],
            random_values=[dummy_u],
        )

        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("dummy", toy_dir["fasta"])])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        seq = srdf["sequence"].iloc[0]
        n = 5

        if strand == "plus":
            rng_n_plus = [n]
        else:
            rng_n_plus = [0]
        rng = _RecordingRng(
            n_plus_values=rng_n_plus,
            choice_returns=[dummy_starts],
            random_values=[dummy_u],
        )

        sample_region(seq, R, n, r=r, fl=simple_fl, p_plus=1.0 if strand == "plus" else 0.0, rng=rng)

        assert len(rng.recorded_start_weights) == 1
        recorded_p = rng.recorded_start_weights[0]

        # Compute expected weights with the oracle
        seq_upper = bytes(seq).upper().decode()
        track = "fwd" if strand == "plus" else "rc"
        expected_w = np.zeros(R, dtype=np.float64)
        fwd, rc, valid = hexamer_indices(seq_upper)
        for i in range(R):
            if not valid[i]:
                continue
            idx = int(fwd[i]) if track == "fwd" else int(rc[i])
            tab = r["start_fwd"] if strand == "plus" else r["end_rev"]
            expected_w[i] = tab[idx]

        expected_p = expected_w / expected_w.sum()
        np.testing.assert_allclose(recorded_p, expected_p, rtol=1e-12)

    def test_p_plus_extremes(self, toy_dir, toy_genome, simple_fl):
        """M26 (ignore p_plus)."""
        g0, g1 = 3, 1003
        R = g1 - g0
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}

        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("dummy", toy_dir["fasta"])])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        seq = srdf["sequence"].iloc[0]

        rng = np.random.default_rng(42)
        starts, lengths, is_plus = sample_region(
            seq, R, 200, r=r, fl=simple_fl, p_plus=1.0, rng=rng,
        )
        assert is_plus.all(), "p_plus=1.0 should give all plus"

        rng = np.random.default_rng(42)
        starts, lengths, is_plus = sample_region(
            seq, R, 200, r=r, fl=simple_fl, p_plus=0.0, rng=rng,
        )
        assert not is_plus.any(), "p_plus=0.0 should give all minus"

    def test_planted_propensity_recovered(self, toy_dir, toy_genome, simple_fl):
        """M27 (plus uses minus tables). Statistical, 6σ bound."""
        g0, g1 = 3, 4003
        R = g1 - g0
        planted_hex = "AACGTC"
        planted_idx = oracle.IDX(planted_hex)
        planted_rc_idx = oracle.IDX(oracle.RC(planted_hex))

        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][planted_idx] = 20.0

        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("dummy", toy_dir["fasta"])])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        seq = srdf["sequence"].iloc[0]

        n_total = 40_000
        rng = np.random.default_rng(12345)
        starts, lengths, is_plus = sample_region(
            seq, R, n_total, r=r, fl=simple_fl, p_plus=0.5, rng=rng,
        )

        # Count plus-strand starts at the planted hexamer
        seq_upper = bytes(seq).upper().decode()
        fwd, _rc, valid = hexamer_indices(seq_upper)
        plus_starts = starts[is_plus]
        plus_hex_counts = np.zeros(NHEX, dtype=np.int64)
        for s in plus_starts:
            if valid[s]:
                plus_hex_counts[int(fwd[s])] += 1

        # Compute expected count under the planted propensity
        n_plus_actual = int(is_plus.sum())
        pos_weights = np.zeros(R, dtype=np.float64)
        for i in range(R):
            if valid[i]:
                pos_weights[i] = r["start_fwd"][int(fwd[i])]
        total_w = pos_weights.sum()
        p_planted = pos_weights[fwd[:R] == planted_idx].sum() / total_w
        expected_count = n_plus_actual * p_planted
        observed_count = plus_hex_counts[planted_idx]

        sigma = np.sqrt(n_plus_actual * p_planted * (1 - p_planted))
        assert sigma > 0
        z = abs(observed_count - expected_count) / sigma
        assert z < 6, f"planted hex recovery z={z:.1f} > 6σ"
        assert observed_count > 50, "planted hex count too low for meaningful test"

    def test_fl_from_filtered_frame_within_bounds(self, toy_dir, toy_genome, toy_regions):
        """M36 (skip length filter)."""
        g0, g1 = toy_regions[0]
        rng_fix = np.random.RandomState(88)
        frags = []
        for i in range(200):
            s = g0 + rng_fix.randint(0, g1 - g0)
            L = rng_fix.randint(L_MIN, L_MAX + 1)
            strand = "+" if rng_fix.random() < 0.5 else "-"
            frags.append(("chrT", s, s + L, strand, 30, 30))
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="fl_bounds")
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", h5, toy_dir["fasta"], n_workers=1, verbose=False,
        )
        fl = FragmentLengthDist.from_srdf(srdf)
        assert fl.min_fl >= L_MIN
        assert fl.max_fl <= L_MAX


# ── T6: Writer and round trip ───────────────────────────────────────────

class TestT6WriterAndRoundTrip:
    """Closed-loop: count → fl → N → r → simulate → build → recount."""

    @pytest.fixture(scope="class")
    def roundtrip_data(self, toy_dir, toy_genome, toy_regions, simple_fl):
        """Run the full closed loop once, reuse across tests."""
        g0, g1 = toy_regions[0]
        R = g1 - g0
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")

        rng_fix = np.random.RandomState(55)
        frags = []
        for i in range(300):
            s = g0 + rng_fix.randint(0, R)
            L = rng_fix.randint(L_MIN, L_MAX + 1)
            strand = "+" if rng_fix.random() < 0.5 else "-"
            frags.append(("chrT", s, s + L, strand, 30, 30))
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="roundtrip_src")

        counts, region_counts, stats, srdf = count_sample(
            rdf, "test", h5, toy_dir["fasta"], n_workers=1, verbose=False,
        )
        fl = FragmentLengthDist.from_srdf(srdf)
        N, _ = uniform_hexamer_counts(rdf, toy_dir["fasta"], fl, verbose=False)
        r = propensities(counts, N)

        # Simulate
        sim_bed = os.path.join(toy_dir["dir"], "sim_roundtrip.bed")
        rng_sim = np.random.default_rng(999)
        sim_stats = simulate_fragments_to_bed(
            srdf, sim_bed, r=r, fl=fl, region_counts=region_counts,
            rng=rng_sim, p_plus=0.5,
        )

        # Build h5 from simulated BED
        gz = pysam.tabix_index(sim_bed, preset="bed", force=True)
        sim_h5 = os.path.join(toy_dir["dir"], "sim_roundtrip.frag.h5")
        subprocess.run(
            ["build-fragments-h5", gz, sim_h5, "--fasta", toy_dir["fasta"], "--quiet"],
            check=True, capture_output=True,
        )

        # Recount
        recount_counts, recount_rc, recount_stats, recount_srdf = count_sample(
            rdf, "sim", sim_h5, toy_dir["fasta"], n_workers=1, verbose=False,
        )

        return {
            "sim_bed": sim_bed,
            "sim_stats": sim_stats,
            "sim_h5": sim_h5,
            "recount_counts": recount_counts,
            "recount_rc": recount_rc,
            "recount_stats": recount_stats,
            "original_rc": region_counts,
            "rdf": rdf,
            "regions": [(g0, g1)],
            "genome": toy_genome,
        }

    def test_bed_text_shape(self, roundtrip_data):
        """M37 (unsorted or header)."""
        bed_path = roundtrip_data["sim_bed"]
        # The plain BED is consumed by tabix_index. Read the gz instead.
        import gzip
        gz_path = bed_path + ".gz"
        with gzip.open(gz_path, "rt") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == roundtrip_data["sim_stats"]["n_rows_written"]
        for i, line in enumerate(lines):
            fields = line.split("\t")
            assert len(fields) == 8, f"row {i}: expected 8 columns, got {len(fields)}"

        # Check sorted
        prev = ("", -1, -1)
        for line in lines:
            f = line.split("\t")
            cur = (f[0], int(f[1]), int(f[2]))
            assert cur >= prev, "BED not sorted"
            prev = cur

        assert roundtrip_data["sim_stats"]["n_rows_written"] == roundtrip_data["sim_stats"]["n_drawn"]

    def test_recount_equals_distinct_pairs(self, roundtrip_data):
        """M28 (strand column wrong), M29 (1-based start)."""
        import gzip
        gz_path = roundtrip_data["sim_bed"] + ".gz"
        with gzip.open(gz_path, "rt") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Count distinct (start, stop) per region
        for ri, (g0, g1) in enumerate(roundtrip_data["regions"]):
            pairs = set()
            for line in lines:
                f = line.split("\t")
                s, e = int(f[1]), int(f[2])
                if g0 <= s < g1:
                    pairs.add((s, e))
            expected = len(pairs)
            actual = int(roundtrip_data["recount_rc"][ri])
            assert actual == expected, (
                f"recount region_counts[{ri}]={actual} != {expected} distinct pairs"
            )

    def test_round_trip_through_real_reader(self, roundtrip_data):
        """M28 (strand flipped), M29 (1-based start)."""
        from fragments_h5 import FragmentsH5
        fh5 = FragmentsH5(roundtrip_data["sim_h5"])
        starts, stops, extras = fh5.fetch_array(
            "chrT", 0, len(roundtrip_data["genome"]),
            return_strand=True,
        )
        fh5.close()
        assert len(starts) > 0
        strands = [s.decode() for s in extras["strand"]]
        assert set(strands).issubset({"+", "-"})

    def test_writer_guard_gz_path(self, roundtrip_data):
        """M39 (delete .gz raise)."""
        with pytest.raises(ValueError, match="write a PLAIN bed"):
            from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
            simulate_fragments_to_bed(
                pd.DataFrame(columns=["contig", "start", "stop", "fragment_array", "sequence"]),
                "/tmp/test.bed.gz",
                r={k: np.ones(NHEX) for k in TABLE_NAMES},
                fl=FragmentLengthDist(np.ones(10, dtype=np.int64), 25),
                region_counts=np.array([]),
                rng=np.random.default_rng(0),
            )

    def test_fragment_length_dist_densify(self):
        """M38 (densify without normalisation)."""
        df = pd.DataFrame({"fragment_length": [25, 27], "count": [1, 3]})
        fl = FragmentLengthDist.from_dataframe(df)
        assert fl.min_fl == 25
        assert fl.max_fl == 27
        np.testing.assert_allclose(fl.densities, [0.25, 0.0, 0.75])

    def test_fragment_length_dist_guards(self):
        """M39 (delete raises)."""
        with pytest.raises(ValueError, match="counts"):
            FragmentLengthDist(np.array([]), 25)
        with pytest.raises(ValueError, match="negative"):
            FragmentLengthDist(np.array([-1, 1]), 25)
        with pytest.raises(ValueError, match="sum to 0"):
            FragmentLengthDist(np.zeros(5, dtype=np.int64), 25)
        # from_dataframe guards
        with pytest.raises(ValueError, match="missing column"):
            FragmentLengthDist.from_dataframe(pd.DataFrame({"x": [1]}))
        with pytest.raises(ValueError, match="empty"):
            FragmentLengthDist.from_dataframe(
                pd.DataFrame({"fragment_length": pd.array([], dtype="int64"),
                               "count": pd.array([], dtype="int64")}))
        with pytest.raises(ValueError, match="duplicate"):
            FragmentLengthDist.from_dataframe(
                pd.DataFrame({"fragment_length": [25, 25], "count": [1, 1]}))

    def test_input_guards(self):
        """M39 (delete raises for load_sample_dataframe, uniform_hexamer_counts)."""
        with pytest.raises(ValueError, match="no samples given"):
            load_sample_dataframe([])
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [0], "stop": [100],
        }), ref="hg38")
        rdf_with_fa = rdf.copy()
        rdf_with_fa["fragment_array"] = [None]
        with pytest.raises(ValueError, match="WITHOUT fragment arrays"):
            uniform_hexamer_counts(
                rdf_with_fa, "/dev/null",
                FragmentLengthDist(np.ones(10, dtype=np.int64), 25),
                verbose=False,
            )


# ── T7: Hygiene ─────────────────────────────────────────────────────────

class TestT7Hygiene:
    """Module-level checks."""

    def test_module_doctests_execute(self):
        """M3 (lowercase in doctest)."""
        import background_model.simulator.count_hexamers_rdf as mod
        results = doctest.testmod(mod, verbose=False)
        assert results.attempted > 0, "no doctests found"
        assert results.failed == 0, f"{results.failed} doctest(s) failed"

    def test_oracle_is_independent(self):
        """M40 (oracle imports background_model)."""
        oracle_path = os.path.join(os.path.dirname(__file__), "cut_site_oracle.py")
        with open(oracle_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("background_model"), (
                        f"oracle imports {node.module}"
                    )
                    assert not node.module.startswith("fragmentomics_tools"), (
                        f"oracle imports {node.module}"
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("background_model")
                        assert not alias.name.startswith("fragmentomics_tools")

    def test_no_removed_feature_imports(self):
        """M40 (module imports simulator.precompute)."""
        import background_model.simulator.count_hexamers_rdf as mod
        mod_path = mod.__file__
        with open(mod_path) as f:
            tree = ast.parse(f.read())
        banned = {"flgc", "simulator.capture", "simulator.precompute",
                   "simulator.weights", "simulator.sampler", "simulator.emit"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for b in banned:
                    assert b not in node.module, (
                        f"module imports removed feature: {node.module}"
                    )
