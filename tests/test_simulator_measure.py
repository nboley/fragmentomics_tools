"""Tests for ``background_model/simulator/measure.py``: admission, ``C(h)``,
``N(h)``, ``f(L)`` and ``r(h)``.

Split out of ``tests/test_cut_site_simulator.py`` by owner decision 188; the
test bodies are unchanged.  Classes keep their T-numbers; where a T-class held
both measure and draw tests, the measure half is here under a new name and the
draw half is in ``tests/test_simulator_draw.py``.  Shared fixtures are in
``tests/conftest.py`` and shared helpers in ``tests/cut_site_helpers.py``.

Every oracle imports NOTHING from the module under test.  The AST check
``test_oracle_is_independent`` enforces this.

Mutations each test must catch are documented in-line as comments.
"""

import numpy as np
import pandas as pd
import pysam
import pytest

import cut_site_oracle as oracle
from cut_site_helpers import CHR6_FASTA, DB_CORE_LEN, GOLDEN_H5, _build_h5

from background_model.constants import (
    HEX_HALF,
    KMER,
    L_MAX,
    L_MIN,
    N_LENGTHS,
    NHEX,
)
from background_model.hexamers import rc_permutation
from background_model.simulator.measure import (
    FragmentLengthDist,
    TABLE_NAMES,
    count_sample,
    count_srdf,
    counts_from_hexamers,
    filter_fragments,
    fl_end_weight,
    load_sample_dataframe,
    propensities,
    uniform_hexamer_counts,
)
from fragmentomics_tools.dataframe import RegionDataFrame


# ── T0: Frame ───────────────────────────────────────────────────────────────

class TestT0Frame:
    """The padded sequence frame ``count_sample`` counts against."""

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


# ── T1: Strand routing ──────────────────────────────────────────────────────

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


# ── T2: Admission boundaries ────────────────────────────────────────────────

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


# ── T3: count_sample vs brute force ─────────────────────────────────────────

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


# ── T4: Expectation and propensity ──────────────────────────────────────────

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


# ── T5: f(L) from the filtered frame ────────────────────────────────────────

class TestT5LengthFromFrame:
    """f(L) built from the filtered frame. Moved from ``TestT5Sampler``."""

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


# ── T6: Measure guards ──────────────────────────────────────────────────────

class TestT6MeasureGuards:
    """measure's input guards. Moved from ``TestT6WriterAndRoundTrip``."""

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


# ── T8: N(h) determinism and region_index (owner decision 166) ──────────────

class TestT8MeasureDeterminism:
    """Owner decision 166, measure side: N(h) byte-identical across
    worker counts, and the ``region_index`` labels ``count_sample``
    carries to the draw. Moved from
    ``TestT8SeedingAndParallelDeterminism``.
    """

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

    def test_count_sample_carries_index_labels(self, admission_h5, toy_dir,
                                               toy_rdf):
        """D5 (region_index taken from row position, not the index label)."""
        sub = toy_rdf.iloc[[1, 0]]
        _, _, _, srdf = count_sample(
            sub, "test", admission_h5, toy_dir["fasta"],
            n_workers=1, verbose=False,
        )
        assert srdf["region_index"].tolist() == [1, 0]

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
