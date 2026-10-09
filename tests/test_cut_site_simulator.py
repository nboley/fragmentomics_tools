"""Tests for the cut-site simulator: ``background_model/hexamers.py``,
``background_model/cut_site_stats.py`` and ``background_model/simulator/draw.py``.

These three were one module, ``count_hexamers_rdf.py``, until owner decision
169 split it along its dependency layers; the tests were kept together because
most of them exercise the chain end to end.

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

from background_model.hexamers import (
    HEX_HALF,
    KMER,
    NHEX,
    hexamer_indices,
    hexamer_vocabulary,
    rc_permutation,
)
from background_model.cut_site_stats import (
    FragmentLengthDist,
    L_MAX,
    L_MIN,
    N_LENGTHS,
    TABLE_NAMES,
    count_sample,
    count_srdf,
    counts_from_hexamers,
    cut_site_hexamers,
    empty_counts,
    filter_fragments,
    fl_end_weight,
    load_sample_dataframe,
    propensities,
    uniform_hexamer_counts,
)
from background_model.simulator.draw import (
    oracle_nll,
    sample_region,
    simulate_fragments_to_bed,
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

    # Fragments with an N in a cut-site window, one per side.
    #
    # The window for a cut site at genomic `gc` is genome[gc-HEX_HALF : gc+HEX_HALF],
    # so it contains the N at `n_pos` iff gc - 3 <= n_pos < gc + 3, i.e.
    # gc in [n_pos-2, n_pos+3]. This was `n_pos - 3`, whose window is
    # [n_pos-6, n_pos) and EXCLUDES the N -- off by one, so the fixture planted
    # nothing and `test_n_window_fragment_dropped_from_tables` had nothing to
    # detect. That, not a hexamer collision, is why mutation M7 (dropping the
    # validity gate) went uncaught by all 47 tests.
    n_block_start = DB_CORE_LEN + TANDEM_REPEATS * len(TANDEM_HEX)
    n_pos = n_block_start + 6  # the single-N position in the n_block
    # invalid START window, valid end (the end lands in the soft-masked copy,
    # which is ACGT and case-folded, so it stays valid)
    frags.append(("chrT", n_pos, n_pos + 50, "+", 30, 30))
    # invalid END window, valid start (the start sits in the tandem repeat,
    # pure ACGT) -- covers the `e_ok` half of `ok = s_ok & e_ok` separately,
    # so a mutation dropping only one side is still caught
    frags.append(("chrT", n_pos - 50, n_pos, "+", 30, 30))

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
        # min(60, 9) = 9 < 10 at offset 100: must be dropped.
        #
        # This was `assert (100) not in starts or True`, i.e. vacuous -- the
        # `or True` made it pass unconditionally, so it never tested the lower
        # side of the MAPQ boundary at all. The comment excusing it said the
        # fragment "may have been deduped with another frag"; if that were
        # true the right fix would be the fixture, not disarming the check.
        assert 100 not in starts, (
            "MAPQ=9 fragment at offset 100 survived; min(mapq1, mapq2) >= 10 "
            "must drop it. If the fixture now collides at (start, stop) with "
            "another fragment, fix the fixture -- do not weaken this."
        )
        # min(10,10)=10 at offset 150: should be kept. The boundary is
        # inclusive, so this is the case an `>` instead of `>=` would break.
        assert 150 in starts, "MAPQ=10 fragment dropped (should keep)"

    def test_mapq_filter_precedes_dedup(self, admission_h5, toy_dir, toy_regions):
        """M12: dedup moved before the MAPQ filter.

        This is the ONE admission ordering that is load-bearing (spec §3);
        everything else commutes. The fixture plants two fragments sharing
        ``(start, stop)`` at g0+300, differing only in strand and MAPQ, with the
        LOW-mapq one written first:

            (g0+300, g0+400, '+',  5,  5)   <- first
            (g0+300, g0+400, '-', 30, 30)

        Correct order: MAPQ is applied at fetch, so the '+' fragment is gone
        before dedup ever runs; dedup is then a no-op and '-' survives.

        Under M12: dedup runs first and keeps the FIRST occurrence -- the '+'
        fragment -- and the MAPQ filter then drops it, so the pair vanishes
        entirely and the position yields nothing.

        Order is deterministic and fixture-controlled, which had to be settled
        before this test could exist: ``_build_h5`` uses a STABLE sort on
        ``(contig, start, stop)``, so ties keep fixture order, and the h5
        preserves it. Measured both ways round -- reversing the two rows
        reverses which strand survives dedup -- so this is a property of the
        fixture, not luck.
        """
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        at_300 = np.flatnonzero(fa.starts_0 == 300)

        assert len(at_300) == 1, (
            f"expected exactly 1 surviving fragment at offset 300, got "
            f"{len(at_300)}. Zero means dedup ran BEFORE the MAPQ filter: it "
            f"kept the first occurrence (mapq 5) and MAPQ then dropped it, "
            f"losing the pair."
        )
        strand = str(np.asarray(fa.fragment_strands)[at_300[0]])
        assert strand == "-", (
            f"the surviving fragment at offset 300 is on strand {strand!r}, "
            f"expected '-'. The '+' one carries mapq 5 and must be removed at "
            f"fetch, before dedup can prefer it for being first."
        )

    def test_straddler_counted_in_start_tile(self, admission_h5, toy_dir,
                                             toy_regions):
        """M15: midpoint admission instead of start-in-region.

        Midpoint is the rule the rewrite REVERSED, and two agents have already
        drawn wrong conclusions from stale docs still asserting it, so a
        regression here reintroduces the whole pre-rewrite geometry.

        The fixture plants a straddler at (g0+R-1, g0+R+79): its START is the
        last position of tile 0, while its MIDPOINT falls inside tile 1. Under
        start-in-region it belongs to tile 0 and nowhere else. Under midpoint
        admission it moves to tile 1, so BOTH assertions below flip.
        """
        (g0, g1), (g1b, g2) = toy_regions[0], toy_regions[1]
        assert g1 == g1b, "tiles 0 and 1 must be contiguous for this test"
        R = g1 - g0
        straddler_start = g0 + R - 1          # last position of tile 0
        straddler_stop = g0 + R + 79
        midpoint = (straddler_start + straddler_stop) // 2
        # Guard the guard: if the fixture drifts so the midpoint no longer
        # lands in the next tile, the two assertions below stop discriminating.
        assert straddler_start < g1 <= midpoint, (
            f"fixture no longer straddles: start {straddler_start}, midpoint "
            f"{midpoint}, boundary {g1}. This test cannot tell start-in-region "
            f"from midpoint admission unless they disagree."
        )

        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT", "chrT"], "start": [g0, g1b], "stop": [g1, g2],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa0, fa1 = srdf["fragment_array"].iloc[0], srdf["fragment_array"].iloc[1]

        assert (straddler_start - g0) in set(fa0.starts_0.tolist()), (
            f"the straddler starting at {straddler_start} is absent from tile 0 "
            f"({g0}-{g1}), which contains its START. Under midpoint admission "
            f"it would have moved to tile 1 instead."
        )
        assert (straddler_start - g1b) not in set(fa1.starts_0.tolist()), (
            f"the straddler appears in tile 1 ({g1b}-{g2}), which contains only "
            f"its MIDPOINT. Admission is start-in-region; counting it here "
            f"double-counts it across the region set."
        )

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
        """M16 (right_pad=l_max), M32 (clip stops to region).

        The fixture plants a max-overhang fragment at (g0+R-1, g0+R-1+180, '+')
        with starts_0=999 and L=180. A DIFFERENT fragment (the straddler at
        (g0+R-1, g0+R+79, '-'), L=80) also has starts_0=999, so asserting
        ``999 in starts_0`` is a tautology — the straddler guarantees it
        regardless of whether the max-overhang fragment survived. Assert the
        specific (start, stop) pair instead.
        """
        g0, g1 = toy_regions[0]
        R = g1 - g0
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        _, _, _, srdf = count_sample(
            rdf, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fa = srdf["fragment_array"].iloc[0]
        overhang_start = R - 1       # starts_0 = 999
        overhang_stop = R - 1 + 180  # stops_0 = 1179
        pairs = list(zip(fa.starts_0.tolist(), fa.stops_0.tolist()))
        assert (overhang_start, overhang_stop) in pairs, (
            f"max-overhang fragment (starts_0={overhang_start}, "
            f"stops_0={overhang_stop}, L=180) not found. Under M16 "
            f"(right_pad=l_max instead of l_max+HEX_HALF) the sequence "
            f"is too short to cover this fragment's stop hexamer."
        )
        stop_pos = g0 + R - 1 + 180
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
        counts, region_counts, stats, srdf = count_sample(
            rdf, "test", bruteforce_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        n_counted = stats["n_counted"]
        n_admitted = int(region_counts.sum())

        # Derive the expected drop INDEPENDENTLY, from the genome string rather
        # than from the module: a fragment is excluded from the tables iff
        # either of its cut-site windows holds a non-ACGT base. With
        # left_pad == HEX_HALF the window for a cut site at genomic `gc` is
        # genome[gc - HEX_HALF : gc + HEX_HALF].
        expected_drop = 0
        for fa, (g_start, _g_stop) in zip(srdf["fragment_array"], toy_regions):
            for s0, e0 in zip(fa.starts_0.tolist(), fa.stops_0.tolist()):
                for gc in (g_start + s0, g_start + e0):
                    window = toy_genome[gc - HEX_HALF:gc + HEX_HALF]
                    if any(b not in "ACGTacgt" for b in window):
                        expected_drop += 1
                        break

        # Guard the guard: if the fixture ever stops planting an N-window
        # fragment, every assertion below passes trivially and this test goes
        # quiet. That is how M7 survived the first version of it.
        assert expected_drop > 0, (
            "fixture plants no admitted fragment with an N in a cut-site "
            "window, so this test cannot detect a missing validity gate"
        )

        # EXACT, not `>=`. The previous version asserted
        # `n_admitted >= n_counted`, which is true BY CONSTRUCTION -- the gap is
        # non-negative however the code behaves -- so mutation M7 (dropping
        # `ok = s_ok & e_ok` in cut_site_hexamers) collapsed the gap to 0 and
        # the assertion still held. Zero of 47 tests caught it.
        assert n_admitted - n_counted == expected_drop, (
            f"{n_admitted - n_counted} fragments were dropped from the tables, "
            f"expected exactly {expected_drop} (the ones with a non-ACGT base "
            f"in a cut-site window). A gap of 0 means the validity gate in "
            f"cut_site_hexamers is not being applied, and N-containing cut "
            f"sites are being miscounted into neighbouring hexamers."
        )

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
        """M21 (minus s_tab), M22 (minus on fwd track), M23 (drop valid on starts).

        The minus path in sample_region reads ``w_s = end_rev[rc[pos]] * valid[pos]``.
        The boosted hexamer must therefore appear in the RC track at a position
        INSIDE the region. The original used ``IDX("AACGTC")`` whose RC "GACGTT"
        occurs only once in the de Bruijn core — at a position outside ``[3, 1003)``.
        Fix: compute the actual rc indices inside the region and boost one that is
        present, then assert as a precondition that the boosted cell is reached.
        """
        g0, g1 = 3, 1003
        R = g1 - g0
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
        n = 1

        # For the minus case, plant an N inside the region so valid[] is not
        # all-True. Without it, dropping ``* valid[pos]`` (M23) has no effect
        # and the test cannot detect the missing mask. One N invalidates six
        # consecutive cut sites because the window spans [c, c+KMER).
        if strand == "minus":
            n_inject_pos = 500 + HEX_HALF
            seq_bytes = bytearray(seq)
            seq_bytes[n_inject_pos] = ord(b"N")
            seq = bytes(seq_bytes)

        seq_upper = bytes(seq).upper().decode()
        fwd, rc_arr, valid = hexamer_indices(seq_upper)

        if strand == "minus":
            # Verify the injected N actually invalidates positions in the region.
            n_invalid_in_region = int((~valid[:R]).sum())
            assert n_invalid_in_region >= 1, (
                "injected N did not invalidate any position in [0, R) — "
                "the test cannot detect a missing valid mask"
            )

        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        if strand == "plus":
            boosted_hex = oracle.IDX("AACGTC")
            r["start_fwd"][boosted_hex] = 10.0
            n_reached = int((fwd[:R][valid[:R]] == boosted_hex).sum())
            assert n_reached >= 1, (
                f"boosted hexamer AACGTC (idx {boosted_hex}) not reachable on "
                f"fwd track within region [0, {R})"
            )
        else:
            # Pick a hexamer that appears in the RC track inside the region.
            # Position 100 is well inside [0, 1000) and valid in the de Bruijn
            # core, far from the injected N at 500.
            probe_pos = 100
            assert valid[probe_pos], "probe position must be valid"
            boosted_hex = int(rc_arr[probe_pos])
            r["end_rev"][boosted_hex] = 10.0
            n_reached = int((rc_arr[:R][valid[:R]] == boosted_hex).sum())
            assert n_reached >= 1, (
                f"boosted hexamer (rc idx {boosted_hex}) not reachable on "
                f"rc track within region [0, {R})"
            )

        dummy_starts = np.array([100])
        dummy_u = np.full((1, 1), 0.5)

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

        track_name = "fwd" if strand == "plus" else "rc"
        expected_w = np.zeros(R, dtype=np.float64)
        for i in range(R):
            if not valid[i]:
                continue
            idx = int(fwd[i]) if track_name == "fwd" else int(rc_arr[i])
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
        starts, lengths, is_plus, _probs = sample_region(
            seq, R, 200, r=r, fl=simple_fl, p_plus=1.0, rng=rng,
        )
        assert is_plus.all(), "p_plus=1.0 should give all plus"

        rng = np.random.default_rng(42)
        starts, lengths, is_plus, _probs = sample_region(
            seq, R, 200, r=r, fl=simple_fl, p_plus=0.0, rng=rng,
        )
        assert not is_plus.any(), "p_plus=0.0 should give all minus"

    def test_planted_propensity_recovered(self, toy_dir, toy_genome):
        """M27 (plus uses minus tables). Statistical, 6σ bound.

        Draws are spread across five 1000-bp regions rather than one large
        region, because ``n > region_len`` now raises.  The region containing
        the tandem AACGTC block dominates the expected count.
        """
        planted_hex = "AACGTC"
        planted_idx = oracle.IDX(planted_hex)

        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][planted_idx] = 20.0

        regions = [(3 + i * 1000, 3 + (i + 1) * 1000) for i in range(5)]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": "chrT",
            "start": [s for s, _ in regions],
            "stop": [e for _, e in regions],
        }), ref="hg38")
        sdf = load_sample_dataframe([("dummy", toy_dir["fasta"])])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )

        fl_uniform = FragmentLengthDist(
            np.ones(N_LENGTHS, dtype=np.int64), L_MIN)
        n_per_region = 800
        rng = np.random.default_rng(12345)

        plus_hex_counts = np.zeros(NHEX, dtype=np.int64)
        expected_count = np.float64(0.0)
        n_plus_total = 0

        for idx, (g0, g1) in enumerate(regions):
            R = g1 - g0
            seq = srdf["sequence"].iloc[idx]
            starts, lengths, is_plus, _probs = sample_region(
                seq, R, n_per_region, r=r, fl=fl_uniform, p_plus=0.5, rng=rng,
            )
            seq_upper = bytes(seq).upper().decode()
            fwd, _rc, valid = hexamer_indices(seq_upper)
            plus_starts = starts[is_plus]
            n_plus_region = len(plus_starts)
            n_plus_total += n_plus_region
            for s in plus_starts:
                if valid[s]:
                    plus_hex_counts[int(fwd[s])] += 1

            pos_weights = np.zeros(R, dtype=np.float64)
            for i in range(R):
                if valid[i]:
                    pos_weights[i] = r["start_fwd"][int(fwd[i])]
            total_w = pos_weights.sum()
            if total_w > 0:
                p_planted_region = (
                    pos_weights[fwd[:R] == planted_idx].sum() / total_w
                )
                expected_count += n_plus_region * p_planted_region

        observed_count = plus_hex_counts[planted_idx]
        p_agg = expected_count / n_plus_total if n_plus_total else 0
        sigma = np.sqrt(n_plus_total * p_agg * (1 - p_agg))
        assert sigma > 0
        z = abs(observed_count - expected_count) / sigma
        assert z < 6, f"planted hex recovery z={z:.1f} > 6σ"
        assert observed_count > 50, "planted hex count too low for meaningful test"

    @staticmethod
    def _single_start_seq(region_len=400, seed=7):
        """Random ACGT sequence in the real frame.

        Length is exactly ``region_len + 2*HEX_HALF + L_MAX``, matching what
        ``attach_sequence`` produces, so a hexamer index equals its
        region-local coordinate.
        """
        rng = np.random.RandomState(seed)
        return "".join("ACGT"[b] for b in
                       rng.randint(0, 4, region_len + 2 * HEX_HALF + L_MAX))

    @staticmethod
    def _unique_start(fwd, region_len, near):
        """A region-local start whose hexamer occurs exactly ONCE in the region.

        A point mass in ``start_fwd`` pins the HEXAMER, not the position --
        `w_s = s_tab[track[pos]]`, so every position carrying that hexamer
        shares the weight. With 400 positions drawn from 4096 hexamers,
        collisions are common (measured: position 100 repeated at 335), and a
        non-unique choice silently gives two admissible starts.
        """
        in_region = fwd[:region_len]
        counts = np.bincount(in_region, minlength=NHEX)
        unique = np.flatnonzero(counts[in_region] == 1)
        assert unique.size, "no region-unique hexamer; raise region_len or reseed"
        return int(unique[np.argmin(np.abs(unique - near))])

    @pytest.mark.parametrize("cause", ["f_zero", "r_zero", "non_acgt"])
    def test_zero_weight_cause(self, cause):
        """M24: drop one factor of ``w[l] = f(l) * r_end * valid``.

        §4 says the code "cannot distinguish the three causes" -- it only tests
        ``w.sum() > 0``. These are the only tests that pin each factor
        independently: each construction zeroes the weight of ONE length by ONE
        factor and asserts that length has zero probability, while a control
        length 10 away carries positive probability.

        All starts are pinned to a single position by a point-mass
        ``start_fwd``, so the assertion is about ``P(l | i)`` alone.
        A scripted RNG forces the inverse-CDF draw onto the control quantile,
        making the result deterministic rather than probabilistic.
        """
        R = 400
        seq0 = self._single_start_seq(region_len=400)
        START = self._unique_start(hexamer_indices(seq0)[0], 400, near=100)
        # CONTROL is 10 away, not adjacent: one N invalidates SIX consecutive
        # cut sites, since the window for c spans seq offsets [c, c+KMER). An
        # adjacent control is hit by the same N and the test fails for the
        # wrong reason (measured).
        BLOCKED, CONTROL = 50, 60
        seq = seq0

        if cause == "non_acgt":
            # Put an N inside the end window for L=BLOCKED only. The window for
            # a cut site at c is seq[c-HEX_HALF : c+HEX_HALF] in region-local
            # coords, i.e. seq offsets [c, c+KMER) in the padded frame.
            c = START + BLOCKED
            seq = seq[:c + HEX_HALF] + "N" + seq[c + HEX_HALF + 1:]

        fwd, _rc, valid = hexamer_indices(seq)
        r = {k: np.zeros(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][int(fwd[START])] = 1.0      # point mass -> one start
        r["end_fwd"][:] = 1.0                      # flat end propensity

        counts = np.ones(L_MAX - L_MIN + 1, dtype=np.int64)
        if cause == "f_zero":
            counts[BLOCKED - L_MIN] = 0            # f(BLOCKED) = 0
        fl = FragmentLengthDist(counts, L_MIN)

        if cause == "r_zero":
            r["end_fwd"][int(fwd[START + BLOCKED])] = 0.0

        # Guard the guard: the construction only discriminates if the blocked
        # and control end hexamers are DIFFERENT, otherwise zeroing r_end for
        # one zeroes both and the control assertion fails for the wrong reason.
        if cause == "r_zero":
            assert int(fwd[START + BLOCKED]) != int(fwd[START + CONTROL]), (
                "blocked and control lengths share an end hexamer; pick "
                "different offsets or this test cannot separate them"
            )
        if cause == "non_acgt":
            assert not valid[START + BLOCKED], "planted N did not invalidate"
            assert valid[START + CONTROL], "planted N also hit the control"

        # ── Assert on P(l | i) directly ──────────────────────────────────
        Ls = np.arange(fl.min_fl, fl.max_fl + 1)
        w_at_start = np.array([
            r["end_fwd"][int(fwd[START + L_val])]
            * float(valid[START + L_val])
            * fl.densities[l_idx]
            for l_idx, L_val in enumerate(Ls)
        ], dtype=np.float64)

        blocked_idx = BLOCKED - fl.min_fl
        control_idx = CONTROL - fl.min_fl
        assert w_at_start[blocked_idx] == 0.0, (
            f"BLOCKED length {BLOCKED} has nonzero weight "
            f"{w_at_start[blocked_idx]} under cause {cause!r}"
        )
        assert w_at_start[control_idx] > 0.0, (
            f"CONTROL length {CONTROL} has zero weight — something other "
            f"than {cause!r} is suppressing it"
        )

        # No quantile can land on BLOCKED: the CDF is flat at that index.
        cdf = np.cumsum(w_at_start / w_at_start.sum())
        prev_cdf = cdf[blocked_idx - 1] if blocked_idx > 0 else 0.0
        assert cdf[blocked_idx] == prev_cdf, (
            "CDF is not flat at BLOCKED — its weight should be zero"
        )

        # Script a deterministic draw onto the CONTROL quantile.
        u_lo = cdf[control_idx - 1] if control_idx > 0 else 0.0
        u_control = (u_lo + cdf[control_idx]) / 2.0

        rng = _RecordingRng(
            n_plus_values=[1],
            choice_returns=[[START]],
            random_values=[u_control],
        )
        starts, lengths, is_plus, _probs = sample_region(
            seq.encode(), R, 1, r=r, fl=fl, p_plus=1.0, rng=rng,
        )
        assert is_plus.all(), "p_plus=1.0 must give plus-strand draws only"
        assert starts[0] == START, (
            f"point-mass start_fwd should pin start to {START}"
        )
        assert lengths[0] == CONTROL, (
            f"scripted quantile should select CONTROL length {CONTROL}, "
            f"got {lengths[0]}"
        )

    def test_end_hexamer_offset_is_exact(self):
        """M25: end hexamer read at ``i + l - 1`` instead of ``i + l``.

        A one-position shift in the end lookup leaves every total plausible and
        every marginal nearly right, which is the definition of a silent
        failure. Point masses on BOTH sides make the draw deterministic: one
        admissible start, one admissible end position, so the drawn length can
        only be their difference. An off-by-one shifts every draw by exactly 1.
        """
        R = 400
        seq = self._single_start_seq(region_len=R, seed=11)
        fwd, _rc, valid = hexamer_indices(seq)
        # Both point masses pin a HEXAMER, so both positions must be unique.
        START = self._unique_start(fwd, R, near=100)
        END = START + 80
        expected_len = END - START
        assert L_MIN <= expected_len <= L_MAX

        start_hex, end_hex = int(fwd[START]), int(fwd[END])
        # The end hexamer must be UNIQUE over the reachable window, or several
        # lengths satisfy the point mass and the draw stops being deterministic.
        reachable = fwd[START + L_MIN:START + L_MAX + 1]
        assert (reachable == end_hex).sum() == 1, (
            "the planted end hexamer is not unique over the reachable range, "
            "so more than one length carries the point mass"
        )
        assert start_hex != end_hex, "start and end point masses must differ"

        r = {k: np.zeros(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][start_hex] = 1.0
        r["end_fwd"][end_hex] = 1.0
        fl = FragmentLengthDist(
            np.ones(L_MAX - L_MIN + 1, dtype=np.int64), L_MIN)

        starts, lengths, _, _probs = sample_region(
            seq.encode(), R, 1, r=r, fl=fl, p_plus=1.0,
            rng=np.random.default_rng(3),
        )
        assert set(starts.tolist()) == {START}
        assert set(lengths.tolist()) == {expected_len}, (
            f"drew lengths {sorted(set(lengths.tolist()))}, expected exactly "
            f"[{expected_len}]. A single off-by-one value means the end hexamer "
            f"is being read at i+l-1 or i+l+1 rather than i+l."
        )

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

    def test_p_sums_to_one(self, toy_dir, toy_genome, simple_fl):
        """Sum of p over all live cells equals 1."""
        g0, g1 = 3, 1003
        R = g1 - g0
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

        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][oracle.IDX("AACGTC")] = 10.0

        p_plus = 0.5
        Ls = np.arange(simple_fl.min_fl, simple_fl.max_fl + 1)
        seq_arr = np.frombuffer(bytes(seq).upper(), dtype=np.uint8)
        fwd, rc_arr, valid = hexamer_indices(seq_arr)
        pos = np.arange(R)
        ends_all = pos[:, None] + Ls[None, :]

        total = np.float64(0.0)
        for is_plus in (True, False):
            track = fwd if is_plus else rc_arr
            s_tab = r["start_fwd"] if is_plus else r["end_rev"]
            e_tab = r["end_fwd"] if is_plus else r["start_rev"]
            p_strand = np.float64(p_plus if is_plus else (1.0 - p_plus))

            W_s = e_tab[track[ends_all]] * valid[ends_all] * simple_fl.densities[None, :]
            t_s = W_s.sum(axis=1, dtype=np.float64)
            live = t_s > 0
            a_s = s_tab[track[pos]] * valid[pos]
            a_s_live = a_s * live
            tot = a_s_live.sum(dtype=np.float64)
            if tot <= 0:
                continue
            start_probs = a_s_live / tot

            for i in range(R):
                if start_probs[i] <= 0:
                    continue
                for l_idx in range(len(Ls)):
                    if W_s[i, l_idx] > 0:
                        p_L = W_s[i, l_idx] / t_s[i]
                        total += p_strand * start_probs[i] * p_L

        np.testing.assert_allclose(float(total), 1.0, rtol=1e-12)

    def test_n_exceeds_region_len_raises(self):
        """Requesting more fragments than positions raises."""
        R = 100
        seq = self._single_start_seq(region_len=R)
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        fl = FragmentLengthDist(np.ones(N_LENGTHS, dtype=np.int64), L_MIN)

        with pytest.raises(ValueError, match=r"requested n=101.*region has only 100"):
            sample_region(
                seq.encode(), R, R + 1, r=r, fl=fl, p_plus=0.5,
                rng=np.random.default_rng(0),
            )
        # n == region_len must NOT raise
        sample_region(
            seq.encode(), R, R, r=r, fl=fl, p_plus=0.5,
            rng=np.random.default_rng(1),
        )

    def test_2n_bound_fires(self):
        """The 2N redraw bound fires when the live space is exhausted.

        A point-mass start with only M=5 admissible lengths gives 5
        distinct (start, length) pairs.  Requesting n=7 forces the dedup
        loop to spin past the 2N budget.
        """
        R = 400
        seq = self._single_start_seq(region_len=R, seed=7)
        fwd, _rc, valid = hexamer_indices(seq)
        START = self._unique_start(fwd, R, near=100)

        M = 5
        counts = np.zeros(L_MAX - L_MIN + 1, dtype=np.int64)
        n_allowed = 0
        for l_idx in range(len(counts)):
            end_pos = START + L_MIN + l_idx
            if valid[end_pos] and n_allowed < M:
                counts[l_idx] = 1
                n_allowed += 1
        assert n_allowed == M, f"need {M} valid lengths, found {n_allowed}"
        fl = FragmentLengthDist(counts, L_MIN)

        r = {k: np.zeros(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][int(fwd[START])] = 1.0
        r["end_fwd"][:] = 1.0

        n = M + 2
        with pytest.raises(
            RuntimeError,
            match=rf"exceeded n={n}.*more than 2n={2 * n}",
        ):
            sample_region(
                seq.encode(), R, n, r=r, fl=fl, p_plus=1.0,
                rng=np.random.default_rng(42),
            )

    def test_2n_bound_comfortable_ratio(self):
        """A comfortable n/M ratio does NOT trigger the 2N bound.

        Same point-mass construction as ``test_2n_bound_fires`` but with
        M=10 admissible lengths and n=3 requests, giving n/M = 0.3.
        """
        R = 400
        seq = self._single_start_seq(region_len=R, seed=7)
        fwd, _rc, valid = hexamer_indices(seq)
        START = self._unique_start(fwd, R, near=100)

        M = 10
        counts = np.zeros(L_MAX - L_MIN + 1, dtype=np.int64)
        n_allowed = 0
        for l_idx in range(len(counts)):
            end_pos = START + L_MIN + l_idx
            if valid[end_pos] and n_allowed < M:
                counts[l_idx] = 1
                n_allowed += 1
        assert n_allowed == M, f"need {M} valid lengths, found {n_allowed}"
        fl = FragmentLengthDist(counts, L_MIN)

        r = {k: np.zeros(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        r["start_fwd"][int(fwd[START])] = 1.0
        r["end_fwd"][:] = 1.0

        n = 3
        starts, lengths, _, _ = sample_region(
            seq.encode(), R, n, r=r, fl=fl, p_plus=1.0,
            rng=np.random.default_rng(42),
        )
        assert len(starts) == n

    def test_no_duplicate_fragments(self, toy_dir, toy_genome, simple_fl):
        """Every drawn (start, stop) pair is unique."""
        g0, g1 = 3, 1003
        R = g1 - g0
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

        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        rng = np.random.default_rng(42)
        starts, lengths, is_plus, probs = sample_region(
            seq, R, 200, r=r, fl=simple_fl, p_plus=0.5, rng=rng,
        )
        pairs = set()
        for s, L in zip(starts.tolist(), lengths.tolist()):
            key = (s, s + L)
            assert key not in pairs, f"duplicate (start, stop) = {key}"
            pairs.add(key)


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
        sim_stats = simulate_fragments_to_bed(
            srdf, sim_bed, r=r, fl=fl, region_counts=region_counts,
            seed=999, p_plus=0.5,
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

        # n_rows_written == n_drawn was here but is a tautology: both are
        # counted from the same arrays in simulate_fragments_to_bed with no
        # filtering between them, so the equality holds by construction.
        # The meaningful check above — len(lines) == n_rows_written — verifies
        # the file was actually written, so that one stays.

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

    def test_writer_absolute_coordinates(self, toy_dir, toy_genome, simple_fl):
        """M29 (writer start is 1-based).

        The existing test_recount_equals_distinct_pairs is self-referential:
        both the expected and actual distinct-pair count derive from the same
        BED, so a constant +1 on every coordinate cancels out. This test
        re-derives expected absolute coordinates from the in-memory draw
        (starts_0 + gstart) and compares them to the post-round-trip read-back.

        Built on the real round-trip chain: count_sample -> sample_region ->
        simulate_fragments_to_bed -> tabix -> build_fragments_h5 -> fetch_array.
        """
        from fragments_h5 import FragmentsH5

        g0, g1 = 503, 1003
        R = g1 - g0
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")

        rng_fix = np.random.RandomState(77)
        frags = []
        for i in range(200):
            s = g0 + rng_fix.randint(0, R)
            L = rng_fix.randint(L_MIN, L_MAX + 1)
            strand = "+" if rng_fix.random() < 0.5 else "-"
            frags.append(("chrT", s, s + L, strand, 30, 30))
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="abs_coord_src")

        counts, region_counts, stats, srdf = count_sample(
            rdf, "test", h5, toy_dir["fasta"], n_workers=1, verbose=False,
        )
        fl = FragmentLengthDist.from_srdf(srdf)
        N, _ = uniform_hexamer_counts(rdf, toy_dir["fasta"], fl, verbose=False)
        r = propensities(counts, N)

        # Run sample_region directly to capture the raw draws, on the stream
        # the writer gives region 0 under seed 42 (owner decision 166).
        seq = srdf["sequence"].iloc[0]
        n = int(region_counts[0])
        rng_sim = np.random.default_rng([42, 0])
        starts_0, lengths, is_plus, _probs = sample_region(
            seq, R, n, r=r, fl=fl, p_plus=0.5, rng=rng_sim,
        )
        assert len(starts_0) > 0, "no fragments drawn"

        # Derive absolute coordinates from the draw.
        expected_starts = g0 + starts_0
        expected_stops = expected_starts + lengths
        expected_strands = np.where(is_plus, "+", "-")

        # Write through the real chain.
        sim_bed = os.path.join(toy_dir["dir"], "abs_coord.bed")
        simulate_fragments_to_bed(
            srdf, sim_bed, r=r, fl=fl, region_counts=region_counts,
            seed=42, p_plus=0.5,
        )
        gz = pysam.tabix_index(sim_bed, preset="bed", force=True)
        sim_h5 = os.path.join(toy_dir["dir"], "abs_coord.frag.h5")
        subprocess.run(
            ["build-fragments-h5", gz, sim_h5, "--fasta", toy_dir["fasta"],
             "--quiet"],
            check=True, capture_output=True,
        )

        # Read back from h5 and compare.
        fh5 = FragmentsH5(sim_h5)
        h5_starts, h5_stops, extras = fh5.fetch_array(
            "chrT", 0, len(toy_genome), return_strand=True,
        )
        fh5.close()

        # Sort both sides by (start, stop) for comparison. The h5 reader
        # returns sorted by start; the draw is in draw order.
        draw_order = np.lexsort((expected_stops, expected_starts))
        h5_order = np.lexsort((h5_stops, h5_starts))

        np.testing.assert_array_equal(
            expected_starts[draw_order], h5_starts[h5_order],
            err_msg="absolute start coordinates do not match the in-memory draw"
        )
        np.testing.assert_array_equal(
            expected_stops[draw_order], h5_stops[h5_order],
            err_msg="absolute stop coordinates do not match the in-memory draw"
        )

    def test_round_trip_strand_per_fragment(self, toy_dir, toy_genome, simple_fl):
        """M28 (writer strand column flipped).

        The original test_round_trip_through_real_reader only checked
        set(strands).issubset({"+","-"}), which a wholesale +/- swap still
        satisfies. This test checks per-fragment strand correctness by
        comparing the in-memory draw's strand labels to the h5 read-back.
        """
        from fragments_h5 import FragmentsH5

        g0, g1 = 503, 1003
        R = g1 - g0
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")

        rng_fix = np.random.RandomState(88)
        frags = []
        for i in range(200):
            s = g0 + rng_fix.randint(0, R)
            L = rng_fix.randint(L_MIN, L_MAX + 1)
            strand = "+" if rng_fix.random() < 0.5 else "-"
            frags.append(("chrT", s, s + L, strand, 30, 30))
        h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="strand_rt_src")

        counts, region_counts, stats, srdf = count_sample(
            rdf, "test", h5, toy_dir["fasta"], n_workers=1, verbose=False,
        )
        fl = FragmentLengthDist.from_srdf(srdf)
        N, _ = uniform_hexamer_counts(rdf, toy_dir["fasta"], fl, verbose=False)
        r = propensities(counts, N)

        seq = srdf["sequence"].iloc[0]
        n = int(region_counts[0])
        rng_sim = np.random.default_rng([77, 0])
        starts_0, lengths, is_plus, _probs = sample_region(
            seq, R, n, r=r, fl=fl, p_plus=0.5, rng=rng_sim,
        )
        assert len(starts_0) > 0

        expected_starts = g0 + starts_0
        expected_stops = expected_starts + lengths
        expected_strands = np.where(is_plus, "+", "-")

        sim_bed = os.path.join(toy_dir["dir"], "strand_rt.bed")
        simulate_fragments_to_bed(
            srdf, sim_bed, r=r, fl=fl, region_counts=region_counts,
            seed=77, p_plus=0.5,
        )
        gz = pysam.tabix_index(sim_bed, preset="bed", force=True)
        sim_h5 = os.path.join(toy_dir["dir"], "strand_rt.frag.h5")
        subprocess.run(
            ["build-fragments-h5", gz, sim_h5, "--fasta", toy_dir["fasta"],
             "--quiet"],
            check=True, capture_output=True,
        )

        fh5 = FragmentsH5(sim_h5)
        h5_starts, h5_stops, extras = fh5.fetch_array(
            "chrT", 0, len(toy_genome), return_strand=True,
        )
        fh5.close()

        h5_strands = np.array([s.decode() for s in extras["strand"]])

        # Sort both sides by (start, stop, strand) for deterministic comparison.
        draw_sort = np.lexsort((expected_strands, expected_stops, expected_starts))
        h5_sort = np.lexsort((h5_strands, h5_stops, h5_starts))

        # Guard: both strands must be present, otherwise a swap is undetectable.
        assert "+" in set(expected_strands) and "-" in set(expected_strands), (
            "fixture must produce both strands for this test to detect a swap"
        )
        np.testing.assert_array_equal(
            expected_strands[draw_sort], h5_strands[h5_sort],
            err_msg="per-fragment strand labels do not match the in-memory draw"
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
                seed=0,
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

    def test_count_srdf_all_empty_raises(self, toy_dir, toy_genome):
        """C1: count_srdf raises when every fragment is removed before counting."""
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        from fragmentomics_tools import RegionFragmentArray
        from fragmentomics_tools.region import Region

        g0, g1 = 3, 103
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("test", toy_dir["fasta"])])
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        # Replace fragment_array with an empty one so n_after_filters == 0.
        srdf["fragment_array"] = [
            RegionFragmentArray([], [], Region("chrT", g0, g1), L_MAX)
        ]
        with pytest.raises(ValueError, match="EVERY fragment was removed"):
            count_srdf(srdf, n_workers=1, verbose=False)

    def test_count_srdf_missing_sequence_raises(self, admission_h5, toy_dir, toy_regions):
        """C2: count_srdf raises when 'sequence' column is missing."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("test", admission_h5)])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_fragment_arrays(
            min_mapq=10,
            fragment_array_callback=filter_fragments,
            verbose=False,
        )
        with pytest.raises(ValueError, match="has no 'sequence' column"):
            count_srdf(srdf, n_workers=1, verbose=False)

    def test_simulate_guard_region_counts_shape(self, admission_h5, toy_dir, toy_regions, simple_fl, tmp_path):
        """C3: simulate_fragments_to_bed raises on region_counts shape mismatch."""
        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("test", admission_h5)])
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_fragment_arrays(
            min_mapq=10, fragment_array_callback=filter_fragments, verbose=False,
        )
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        with pytest.raises(ValueError, match="region_counts has shape"):
            simulate_fragments_to_bed(
                srdf, str(tmp_path / "test.bed"),
                r=r, fl=simple_fl,
                region_counts=np.array([10, 20]),
                seed=0,
            )

    def test_simulate_guard_missing_column(self, simple_fl, tmp_path):
        """C4: simulate_fragments_to_bed raises when a required column is missing."""
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        srdf_no_seq = pd.DataFrame({
            "contig": ["chrT"], "start": [3], "stop": [103],
            "fragment_array": [None],
        })
        with pytest.raises(ValueError, match="has no 'sequence' column"):
            simulate_fragments_to_bed(
                srdf_no_seq, str(tmp_path / "test.bed"),
                r=r, fl=simple_fl,
                region_counts=np.array([10]),
                seed=0,
            )

    def test_simulate_guard_fa_length_mismatch(self, admission_h5, toy_dir, toy_regions, simple_fl, tmp_path):
        """C5: simulate_fragments_to_bed raises when fa.length != stop - start."""
        from fragmentomics_tools import RegionFragmentArray
        from fragmentomics_tools.region import Region
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame

        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("test", admission_h5)])
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_fragment_arrays(
            min_mapq=10, fragment_array_callback=filter_fragments, verbose=False,
        )
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        wrong_len = (g1 - g0) + 50
        srdf["fragment_array"] = [
            RegionFragmentArray([], [], Region("chrT", g0, g0 + wrong_len), L_MAX)
        ]
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        with pytest.raises(AssertionError, match="fragment_array.length"):
            simulate_fragments_to_bed(
                srdf, str(tmp_path / "test.bed"),
                r=r, fl=simple_fl,
                region_counts=np.array([10]),
                seed=0,
            )

    def test_simulate_guard_sequence_length_mismatch(self, admission_h5, toy_dir, toy_regions, simple_fl, tmp_path):
        """C6: simulate_fragments_to_bed raises on sequence length mismatch."""
        from fragmentomics_tools import RegionFragmentArray
        from fragmentomics_tools.region import Region
        from fragmentomics_tools.dataframe import SampleAndRegionDataFrame

        g0, g1 = toy_regions[0]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT"], "start": [g0], "stop": [g1],
        }), ref="hg38")
        sdf = load_sample_dataframe([("test", admission_h5)])
        srdf = SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)
        srdf = srdf.attach_fragment_arrays(
            min_mapq=10, fragment_array_callback=filter_fragments, verbose=False,
        )
        srdf = srdf.attach_sequence(
            toy_dir["fasta"], left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF,
            verbose=False,
        )
        original_seq = srdf["sequence"].iloc[0]
        srdf["sequence"] = [original_seq[:-10]]
        r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
        with pytest.raises(AssertionError, match="sequence is .* b, expected"):
            simulate_fragments_to_bed(
                srdf, str(tmp_path / "test.bed"),
                r=r, fl=simple_fl,
                region_counts=np.array([10]),
                seed=0,
            )


# ── T6b: Functional chain over multiple regions ────────────────────────

class TestT6bFunctionalChain:
    """End-to-end: count → fl → N → r → simulate → tabix → build_h5 → read back.

    Runs the REAL writer chain over ~10 regions.  Asserts the three properties
    the redraw work exists to guarantee: exact draw count, no duplicate
    ``(start, stop)``, and sidecar 1:1 correspondence.
    """

    def test_full_chain_multi_region(self, toy_dir, toy_genome):
        from fragments_h5 import FragmentsH5

        regions = [(3 + i * 500, 3 + (i + 1) * 500) for i in range(10)]
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": "chrT",
            "start": [s for s, _ in regions],
            "stop": [e for _, e in regions],
        }), ref="hg38")

        rng_fix = np.random.RandomState(42)
        frags = []
        for g0, g1 in regions:
            for _ in range(50):
                s = g0 + rng_fix.randint(0, g1 - g0)
                L = rng_fix.randint(L_MIN, L_MAX + 1)
                strand = "+" if rng_fix.random() < 0.5 else "-"
                frags.append(("chrT", s, s + L, strand, 30, 30))
        source_h5 = _build_h5(
            frags, toy_dir["fasta"], toy_dir["dir"], name="func_src",
        )

        counts, region_counts, stats, srdf = count_sample(
            rdf, "src", source_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        fl = FragmentLengthDist.from_srdf(srdf)
        N, _ = uniform_hexamer_counts(rdf, toy_dir["fasta"], fl, verbose=False)
        r = propensities(counts, N)

        sim_bed = os.path.join(toy_dir["dir"], "func_chain.bed")
        sim_stats = simulate_fragments_to_bed(
            srdf, sim_bed, r=r, fl=fl, region_counts=region_counts,
            p_plus=0.5, seed=777,
        )

        # ── Assert 1: n_drawn == n_requested ──
        assert sim_stats["n_drawn"] == sim_stats["n_requested"], (
            f"n_drawn={sim_stats['n_drawn']} != "
            f"n_requested={sim_stats['n_requested']}"
        )

        # ── tabix + build h5 ──
        gz = pysam.tabix_index(sim_bed, preset="bed", force=True)
        sim_h5 = os.path.join(toy_dir["dir"], "func_chain.frag.h5")
        subprocess.run(
            ["build-fragments-h5", gz, sim_h5,
             "--fasta", toy_dir["fasta"], "--quiet"],
            check=True, capture_output=True,
        )

        fh5 = FragmentsH5(sim_h5)
        h5_starts, h5_stops, extras = fh5.fetch_array(
            "chrT", 0, len(toy_genome), return_strand=True,
        )
        fh5.close()

        # ── Assert 2: no duplicate (start, stop) ──
        pairs = set()
        for s, e in zip(h5_starts.tolist(), h5_stops.tolist()):
            assert (s, e) not in pairs, f"duplicate (start, stop) = ({s}, {e})"
            pairs.add((s, e))

        # ── Assert 3: sidecar rows map 1:1 to h5 rows ──
        sidecar_path = sim_bed.replace(".bed", ".p.tsv.gz")
        sidecar = pd.read_csv(sidecar_path, sep="\t", comment="#")
        assert len(sidecar) == len(h5_starts), (
            f"sidecar has {len(sidecar)} rows, h5 has {len(h5_starts)}"
        )

        h5_strands = np.array([s.decode() for s in extras["strand"]])
        h5_keys = set(zip(
            h5_starts.tolist(), h5_stops.tolist(), h5_strands.tolist(),
        ))
        sidecar_keys = set(zip(
            sidecar["start"].tolist(),
            sidecar["stop"].tolist(),
            sidecar["strand"].tolist(),
        ))
        assert h5_keys == sidecar_keys, (
            f"sidecar keys do not match h5 keys; "
            f"in sidecar not h5: {sidecar_keys - h5_keys}, "
            f"in h5 not sidecar: {h5_keys - sidecar_keys}"
        )


# ── T8: Seeding and parallel determinism (owner decision 166) ──────────

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


class TestT8SeedingAndParallelDeterminism:
    """Owner decision 166: per-region streams ``default_rng([seed, i])``,
    parallel draw and N(h), byte-identical across worker counts.

    Nothing here pins a ``Generator`` draw: every assertion compares two runs.
    """

    SEED = 20261008

    @pytest.fixture(scope="class")
    def draw_setup(self, toy_genome):
        # 16 contiguous 300 bp tiles. A narrow 3-length f(L) and 120 fragments
        # per tile make the live (start, L) space ~900, so duplicate redraws
        # are frequent -- the n_dup_redraws assertions need them nonzero.
        regions = [(3 + 300 * i, 3 + 300 * (i + 1)) for i in range(16)]
        assert regions[-1][1] + L_MAX + HEX_HALF <= len(toy_genome)
        rs = np.random.RandomState(166)
        r = {k: rs.uniform(0.5, 2.0, NHEX) for k in TABLE_NAMES}
        fl = FragmentLengthDist(np.array([1, 2, 1], dtype=np.int64), 60)
        counts = np.full(len(regions), 120, dtype=np.int64)
        counts[3] = 0   # an empty region must not shift anyone's stream
        return dict(regions=regions, r=r, fl=fl, counts=counts,
                    frame=_draw_frame(toy_genome, regions))

    def _simulate(self, setup, frame, counts, out, *, seed=None, n_workers=1):
        return simulate_fragments_to_bed(
            frame, str(out), r=setup["r"], fl=setup["fl"],
            region_counts=counts, seed=self.SEED if seed is None else seed,
            p_plus=0.5, n_workers=n_workers,
        )

    def test_draw_identical_across_worker_counts(self, draw_setup, tmp_path):
        """D2 (shared rng), D4 (dup counts lost in workers), D4b (dropped)."""
        s = draw_setup
        st1 = self._simulate(s, s["frame"], s["counts"],
                             tmp_path / "w1.bed", n_workers=1)
        st3 = self._simulate(s, s["frame"], s["counts"],
                             tmp_path / "w3.bed", n_workers=3)

        assert st1["n_dup_redraws"] > 0, (
            "fixture produced no duplicate redraws, so the dup-count "
            "assertions below would be vacuous"
        )
        bed1, side1 = _read_outputs(tmp_path / "w1.bed", st1)
        bed3, side3 = _read_outputs(tmp_path / "w3.bed", st3)
        assert bed1 == bed3, "BED differs between n_workers=1 and 3"
        assert side1 == side3, "p sidecar differs between n_workers=1 and 3"
        drop = lambda st: {k: v for k, v in st.items() if k != "p_sidecar"}
        assert drop(st1) == drop(st3), (
            f"stats differ between n_workers=1 and 3: {drop(st1)} vs {drop(st3)}"
        )

        # n_dup_redraws against an INDEPENDENT count: each region redrawn on
        # its own stream, counted in this process. Catches a count dropped on
        # every path, which the cross-worker equality alone cannot.
        expected = 0
        for k, ((g0, g1), n) in enumerate(zip(s["regions"], s["counts"])):
            if n == 0:
                continue
            ctr = [0]
            sample_region(
                s["frame"]["sequence"].iloc[k], g1 - g0, int(n), r=s["r"],
                fl=s["fl"], p_plus=0.5,
                rng=np.random.default_rng([self.SEED, k]), _dup_counter=ctr,
            )
            expected += ctr[0]
        assert st1["n_dup_redraws"] == expected

    def test_uniform_counts_identical_across_worker_counts(
        self, toy_dir, toy_genome
    ):
        """D3 (reduction grouping follows n_workers)."""
        regions = [(3 + 300 * i, 3 + 300 * (i + 1)) for i in range(18)]
        assert regions[-1][1] + L_MAX + HEX_HALF <= len(toy_genome)
        rdf = RegionDataFrame(pd.DataFrame({
            "contig": "chrT",
            "start": [g0 for g0, _ in regions],
            "stop": [g1 for _, g1 in regions],
        }), ref="hg38")
        # Every length populated with irregular counts, so the end-weight
        # ramps are non-dyadic and a regrouped float64 sum moves last bits.
        fl = FragmentLengthDist(
            np.random.RandomState(7).randint(1, 50, N_LENGTHS), L_MIN,
        )
        runs = {
            w: uniform_hexamer_counts(rdf, toy_dir["fasta"], fl, n_workers=w,
                                      block_size=4, verbose=False)
            for w in (1, 3, 5)
        }
        N1, meta1 = runs[1]
        for w in (3, 5):
            Nw, metaw = runs[w]
            assert N1["start"].tobytes() == Nw["start"].tobytes(), w
            assert N1["end"].tobytes() == Nw["end"].tobytes(), (
                f"N_end differs in its bits between n_workers=1 and {w}"
            )
            assert meta1 == metaw, w
        # Regrouping may move only the last bits, never the value.
        N_one_block, _ = uniform_hexamer_counts(
            rdf, toy_dir["fasta"], fl, n_workers=1, block_size=len(regions),
            verbose=False,
        )
        np.testing.assert_array_equal(N1["start"], N_one_block["start"])
        np.testing.assert_allclose(N1["end"], N_one_block["end"], rtol=1e-12)

    def test_subset_draws_the_same_fragments(self, draw_setup, tmp_path):
        """D2 (shared rng). Function level, identical (r, f, counts).

        NOT a driver-level claim: a ``--n-regions k`` run re-estimates r(h)
        and f(L) from k regions, so its draws legitimately differ.
        """
        s = draw_setup
        st_full = self._simulate(s, s["frame"], s["counts"],
                                 tmp_path / "full.bed")
        # Kept rows given OUT of order: the stream follows region_index, not
        # the row position.
        keep = [10, 4, 1, 3, 7]
        st_sub = self._simulate(
            s, s["frame"].iloc[keep].reset_index(drop=True),
            s["counts"][keep], tmp_path / "sub.bed",
        )
        bed_full, side_full = _read_outputs(tmp_path / "full.bed", st_full)
        bed_sub, side_sub = _read_outputs(tmp_path / "sub.bed", st_sub)

        spans = [s["regions"][k] for k in keep]
        in_keep = lambda start: any(g0 <= start < g1 for g0, g1 in spans)
        want_bed = [l for l in bed_full if in_keep(int(l.split("\t")[1]))]
        want_side = [l for l in side_full[2:]
                     if in_keep(int(l.split("\t")[1]))]
        assert len(want_bed) == int(s["counts"][keep].sum()) > 0
        assert bed_sub == want_bed
        assert side_sub[2:] == want_side

    def test_stream_is_keyed_on_the_seed_pair(self, draw_setup, toy_genome,
                                              tmp_path):
        """D1 (seed + i), D2 (shared rng)."""
        s = draw_setup
        g0, g1 = s["regions"][0]
        n = 120
        # Two rows with IDENTICAL sequence and n, so any difference between
        # their draws comes from the stream alone.
        frame = _draw_frame(toy_genome, [(g0, g1), (g0, g1)],
                            contigs=["chrA", "chrB"], region_index=[0, 1])
        counts = np.array([n, n])

        def by_contig(seed, tag):
            out = tmp_path / f"{tag}.bed"
            self._simulate(s, frame, counts, out, seed=seed)
            rows = {"chrA": [], "chrB": []}
            for line in open(out).read().splitlines():
                f = line.split("\t")
                rows[f[0]].append((int(f[1]), int(f[2]), f[5]))
            return rows

        a = by_contig(self.SEED, "seed_s")
        b = by_contig(self.SEED + 1, "seed_s1")
        assert a["chrA"] != a["chrB"], "regions 0 and 1 share a stream"
        assert a["chrA"] != b["chrA"], "seed does not reach the stream"
        # The seed+i collision: region 1 of seed s vs region 0 of seed s+1.
        assert a["chrB"] != b["chrA"], (
            "region 1 under seed s drew exactly what region 0 drew under "
            "seed s+1 -- the stream is keyed on seed + i, not the pair"
        )

        # And the stream IS default_rng([seed, region_index]): the writer's
        # draw for each row equals sample_region on that stream.
        seq = frame["sequence"].iloc[0]
        for contig, idx in (("chrA", 0), ("chrB", 1)):
            st, L, plus, _ = sample_region(
                seq, g1 - g0, n, r=s["r"], fl=s["fl"], p_plus=0.5,
                rng=np.random.default_rng([self.SEED, idx]),
            )
            want = sorted(zip((g0 + st).tolist(), (g0 + st + L).tolist(),
                              np.where(plus, "+", "-").tolist()))
            assert a[contig] == want, contig

    def test_seed_and_index_guards(self, draw_setup, tmp_path):
        s = draw_setup
        frame, counts = s["frame"], s["counts"]
        out = tmp_path / "guard.bed"
        with pytest.raises(TypeError, match="seed must be an int"):
            self._simulate(s, frame, counts, out, seed=1.5)
        with pytest.raises(TypeError, match="seed must be an int"):
            simulate_fragments_to_bed(
                frame, str(out), r=s["r"], fl=s["fl"], region_counts=counts,
                seed=None,
            )
        for bad in (-1, 2 ** 32):
            with pytest.raises(ValueError, match="outside"):
                self._simulate(s, frame, counts, out, seed=bad)
        with pytest.raises(ValueError, match="no 'region_index' column"):
            self._simulate(s, frame.drop(columns="region_index"), counts, out)
        dup = frame.assign(region_index=np.zeros(len(frame), dtype=np.int64))
        with pytest.raises(ValueError, match="duplicate"):
            self._simulate(s, dup, counts, out)

    def test_count_sample_carries_index_labels(self, admission_h5, toy_dir,
                                               toy_rdf):
        """D5 (region_index taken from row position, not the index label)."""
        sub = toy_rdf.iloc[[1, 0]]
        _, _, _, srdf = count_sample(
            sub, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        assert srdf["region_index"].tolist() == [1, 0]

    def test_explicit_region_index_overrides_frame_column(
        self, draw_setup, tmp_path
    ):
        """The ``region_index=`` argument overrides the frame's own column.

        Guards a mutation that ignores the explicit argument and reads the
        frame's column regardless.
        """
        s = draw_setup
        frame = s["frame"]  # region_index column is arange(16)
        offset_idx = (np.arange(len(frame)) + 100).astype(np.int64)

        out_override = tmp_path / "override.bed"
        st_override = simulate_fragments_to_bed(
            frame, str(out_override), r=s["r"], fl=s["fl"],
            region_counts=s["counts"], seed=self.SEED, p_plus=0.5,
            region_index=offset_idx, n_workers=1,
        )

        # A frame whose COLUMN holds the same offset values, no override.
        frame_with_col = frame.assign(region_index=offset_idx)
        out_col = tmp_path / "col.bed"
        st_col = self._simulate(
            s, frame_with_col, s["counts"], out_col,
        )

        bed_override, side_override = _read_outputs(out_override, st_override)
        bed_col, side_col = _read_outputs(out_col, st_col)
        assert bed_override == bed_col, (
            "overriding region_index did not match a frame whose column "
            "holds the same values"
        )
        assert side_override == side_col

        # Guard the guard: the override must actually have taken effect,
        # i.e. differ from what the ORIGINAL (unoverridden) column would draw.
        out_plain = tmp_path / "plain.bed"
        st_plain = self._simulate(s, frame, s["counts"], out_plain)
        bed_plain, _ = _read_outputs(out_plain, st_plain)
        assert bed_override != bed_plain, (
            "region_index= had no effect -- the frame's own column "
            "(arange(16)) was used instead of the explicit argument"
        )

    def test_region_index_out_of_range_and_wrong_dtype(
        self, draw_setup, tmp_path
    ):
        """-1 and 2**32 raise ValueError('outside'); a float value raises
        TypeError -- region_index shares _as_seed_word with seed."""
        s = draw_setup
        frame, counts = s["frame"], s["counts"]
        out = tmp_path / "bad_region_index.bed"
        n = len(frame)

        for bad in (-1, 2 ** 32):
            bad_idx = np.arange(n, dtype=np.int64)
            bad_idx[0] = bad
            with pytest.raises(ValueError, match="outside"):
                simulate_fragments_to_bed(
                    frame, str(out), r=s["r"], fl=s["fl"],
                    region_counts=counts, seed=self.SEED, p_plus=0.5,
                    region_index=bad_idx, n_workers=1,
                )

        float_idx = np.arange(n, dtype=np.float64)
        float_idx[0] = 1.5
        with pytest.raises(TypeError, match="must be an int"):
            simulate_fragments_to_bed(
                frame, str(out), r=s["r"], fl=s["fl"],
                region_counts=counts, seed=self.SEED, p_plus=0.5,
                region_index=float_idx, n_workers=1,
            )

    def test_region_index_wrong_shape_raises(self, draw_setup, tmp_path):
        """``region_index`` with len != len(srdf) raises ValueError('shape')."""
        s = draw_setup
        frame, counts = s["frame"], s["counts"]
        out = tmp_path / "shape.bed"
        short_idx = np.arange(len(frame) - 1, dtype=np.int64)
        with pytest.raises(ValueError, match="shape"):
            simulate_fragments_to_bed(
                frame, str(out), r=s["r"], fl=s["fl"], region_counts=counts,
                seed=self.SEED, p_plus=0.5, region_index=short_idx,
                n_workers=1,
            )

    def test_seed_bool_rejected(self, draw_setup, tmp_path):
        """``seed=True`` raises TypeError -- bools are deliberately rejected
        even though ``bool`` is an ``int`` subclass."""
        s = draw_setup
        frame, counts = s["frame"], s["counts"]
        out = tmp_path / "bool_seed.bed"
        with pytest.raises(TypeError, match="must be an int"):
            simulate_fragments_to_bed(
                frame, str(out), r=s["r"], fl=s["fl"], region_counts=counts,
                seed=True, p_plus=0.5, n_workers=1,
            )

    def test_uniform_hexamer_counts_block_size_and_empty_rdf(
        self, toy_dir, toy_rdf, simple_fl, monkeypatch
    ):
        """``block_size=0`` raises; an empty rdf returns all-zero tables with
        ``n_regions == 0``, without ever forking."""
        with pytest.raises(ValueError, match="block_size"):
            uniform_hexamer_counts(
                toy_rdf, toy_dir["fasta"], simple_fl, block_size=0,
                verbose=False,
            )

        empty_rdf = RegionDataFrame(pd.DataFrame({
            "contig": pd.Series([], dtype=object),
            "start": pd.Series([], dtype=np.int64),
            "stop": pd.Series([], dtype=np.int64),
        }), ref="hg38")

        from fragmentomics_tools.dataframe import DataFrameBase

        def _forbid_parallel_apply(self, *a, **k):
            raise AssertionError(
                "parallel_apply was called on an empty region set -- the "
                "empty case must return before forking"
            )
        monkeypatch.setattr(DataFrameBase, "parallel_apply", _forbid_parallel_apply)

        N, meta = uniform_hexamer_counts(
            empty_rdf, toy_dir["fasta"], simple_fl, n_workers=None,
            verbose=False,
        )
        assert meta["n_regions"] == 0
        np.testing.assert_array_equal(N["start"], np.zeros(NHEX, dtype=np.int64))
        np.testing.assert_array_equal(N["end"], np.zeros(NHEX, dtype=np.float64))

    def test_count_sample_index_guard_and_region_index_passthrough(
        self, admission_h5, toy_dir, toy_rdf
    ):
        """A non-integer index with no ``region_index`` column raises; an
        EXISTING ``region_index`` column passes through rather than being
        overwritten from the frame's (positional) index labels."""
        bad = RegionDataFrame(pd.DataFrame({
            "contig": ["chrT", "chrT"],
            "start": [int(toy_rdf["start"].iloc[0]), int(toy_rdf["start"].iloc[1])],
            "stop": [int(toy_rdf["stop"].iloc[0]), int(toy_rdf["stop"].iloc[1])],
        }, index=["a", "b"]), ref="hg38")
        with pytest.raises(ValueError, match="unique integer index"):
            count_sample(
                bad, "test", admission_h5, toy_dir["fasta"],
                n_workers=1, verbose=False,
            )

        # Index labels here are [1, 0]; the explicit column is [7, 3] --
        # different values, so a passthrough and an index-derived column are
        # distinguishable.
        with_idx = toy_rdf.iloc[[1, 0]].assign(
            region_index=np.array([7, 3], dtype=np.int64)
        )
        _, _, _, srdf = count_sample(
            with_idx, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        assert srdf["region_index"].tolist() == [7, 3]


# ── T7: Hygiene ─────────────────────────────────────────────────────────

class TestT7Hygiene:
    """Module-level checks."""

    def test_module_doctests_execute(self):
        """M3 (lowercase in doctest).

        ``make test`` does not collect ``background_model/``, so this is the
        ONLY place these modules' doctests run.  It covers all three modules
        of the split.  ``hexamers`` carries every example today; the other
        two must still pass if one is added, and an example vanishing from
        ``hexamers`` fails here rather than silently dropping out.
        """
        import background_model.cut_site_stats as stats_mod
        import background_model.hexamers as hex_mod
        import background_model.simulator.draw as draw_mod
        for mod, must_have_examples in ((hex_mod, True), (stats_mod, False),
                                        (draw_mod, False)):
            results = doctest.testmod(mod, verbose=False)
            if must_have_examples:
                assert results.attempted > 0, f"no doctests found in {mod.__name__}"
            assert results.failed == 0, (
                f"{results.failed} doctest(s) failed in {mod.__name__}"
            )

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
        import background_model.cut_site_stats as stats_mod
        import background_model.hexamers as hex_mod
        import background_model.simulator.draw as draw_mod
        banned = {"flgc", "simulator.capture", "simulator.precompute",
                   "simulator.weights", "simulator.sampler", "simulator.emit"}
        for mod in (hex_mod, stats_mod, draw_mod):
            with open(mod.__file__) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for b in banned:
                        assert b not in node.module, (
                            f"{mod.__name__} imports removed feature: "
                            f"{node.module}"
                        )
