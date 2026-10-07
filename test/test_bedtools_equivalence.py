"""Differential tests: run bedtools and our API on the same input, compare.

Every mapping documented in ``docs/architecture/bedtools_equivalence.md`` is tested
here by running the real ``bedtools`` binary and comparing the result against
our ``intervals`` module.  This is the regression net that hand-written
expectations cannot provide — six real bugs survived 81 synthetic tests and
two reviews, and every one was caught by executing against real input.

Marked ``requires_bedtools``: the tests **skip** when bedtools is not on PATH
by default, and **fail** when ``--bedtools`` is given.  ``make test-equivalence``
passes that flag, so a CI or developer machine without bedtools degrades
gracefully, while the entry point that exists to catch regressions cannot
silently pass.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fragmentomics_tools.dataframe import RegionDataFrame
from fragmentomics_tools.intervals import (
    cluster,
    merge,
    nearest,
    overlap_indices,
    overlaps,
)

# ── bedtools discovery ──────────────────────────────────────────────

BEDTOOLS = shutil.which(
    "bedtools",
    path=os.environ.get(
        "PATH",
        "/home/nathanboley/miniconda3/envs/biomarker_env/bin:"
        + os.environ.get("PATH", ""),
    ),
)
if BEDTOOLS is None:
    # Try the known conda env path directly.
    _candidate = "/home/nathanboley/miniconda3/envs/biomarker_env/bin/bedtools"
    if os.path.isfile(_candidate) and os.access(_candidate, os.X_OK):
        BEDTOOLS = _candidate


def _bedtools_available():
    return BEDTOOLS is not None


# ── Real-data paths ─────────────────────────────────────────────────

CTCF_BED = Path("/efs/analytics/nathanboley/interval_fixtures/ctcf.hg38.bed6.bed")
BLACKLIST_BED = Path(
    "/efs/analytics/nathanboley/data_resources/genome/hg38-blacklist.v2.bed.gz"
)

_REAL_DATA_AVAILABLE = None


def _real_data_available():
    global _REAL_DATA_AVAILABLE
    if _REAL_DATA_AVAILABLE is None:
        _REAL_DATA_AVAILABLE = CTCF_BED.exists() and BLACKLIST_BED.exists()
    return _REAL_DATA_AVAILABLE


# ── Helpers ──────────────────────────────────────────────────────────


def _rdf(rows, ref="hg38"):
    return RegionDataFrame(pd.DataFrame(rows), ref=ref)


def _write_bed3(rdf, path):
    """Write a RegionDataFrame as strict BED3 (no extra columns)."""
    with open(path, "w") as f:
        for _, row in rdf[["contig", "start", "stop"]].iterrows():
            f.write(f"{row.contig}\t{row.start}\t{row.stop}\n")


def _run_bedtools(args, timeout=300):
    """Run bedtools and return stdout lines."""
    r = subprocess.run(
        [BEDTOOLS] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert r.returncode == 0, f"bedtools failed (exit {r.returncode}):\n{r.stderr[:2000]}"
    return r.stdout.strip().split("\n") if r.stdout.strip() else []


def _sort_bed(unsorted_path, sorted_path):
    """Sort a BED file with bedtools sort (required for closest/cluster)."""
    r = subprocess.run(
        f"{BEDTOOLS} sort -i {unsorted_path}",
        shell=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"bedtools sort failed: {r.stderr[:500]}"
    with open(sorted_path, "w") as f:
        f.write(r.stdout)


def _parse_bed3_lines(lines):
    """Parse BED3 output lines into a set of (contig, start, stop) tuples."""
    result = set()
    for line in lines:
        parts = line.split("\t")
        result.add((parts[0], int(parts[1]), int(parts[2])))
    return result


def _parse_wa_wb_bed3(lines):
    """Parse -wa -wb output from two BED3 files (6 columns)."""
    result = set()
    for line in lines:
        p = line.split("\t")
        result.add((p[0], int(p[1]), int(p[2]), p[3], int(p[4]), int(p[5])))
    return result


def _require_bedtools(request):
    """Skip or fail depending on --bedtools flag."""
    if _bedtools_available():
        return
    if request.config.getoption("--bedtools", default=False):
        pytest.fail(
            "--bedtools was requested but bedtools binary was not found. "
            "This target exists so a bedtools equivalence check cannot silently skip."
        )
    pytest.skip("bedtools binary not found (pass --bedtools to fail instead of skip)")


def _require_real_data(request):
    """Skip or fail depending on --bedtools flag and data availability."""
    _require_bedtools(request)
    if _real_data_available():
        return
    missing = []
    if not CTCF_BED.exists():
        missing.append(str(CTCF_BED))
    if not BLACKLIST_BED.exists():
        missing.append(str(BLACKLIST_BED))
    if request.config.getoption("--bedtools", default=False):
        pytest.fail(
            "--bedtools was requested but real-data inputs are missing:\n  "
            + "\n  ".join(missing)
        )
    pytest.skip(f"real-data inputs unavailable: {missing[0]}")


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_data():
    """Load the real CTCF and blacklist data (module-scoped for performance)."""
    if not _real_data_available():
        pytest.skip("real data not available")
    ctcf = RegionDataFrame.from_bed(str(CTCF_BED), ref="hg38")
    bl = RegionDataFrame.from_bed(str(BLACKLIST_BED), ref="hg38")
    return ctcf, bl


@pytest.fixture(scope="module")
def real_bed_files(real_data):
    """Write real data as BED3 and provide sorted versions."""
    ctcf, bl = real_data
    d = tempfile.mkdtemp(prefix="bedtools_equiv_")
    ctcf_bed = os.path.join(d, "ctcf.bed")
    bl_bed = os.path.join(d, "bl.bed")
    bl_sorted = os.path.join(d, "bl_sorted.bed")
    _write_bed3(ctcf, ctcf_bed)
    _write_bed3(bl, bl_bed)
    _sort_bed(bl_bed, bl_sorted)
    yield ctcf_bed, bl_bed, bl_sorted
    shutil.rmtree(d, ignore_errors=True)


# ── Small synthetic fixtures ─────────────────────────────────────────

# Deliberately includes: overlapping, book-ended (gap=0), gap=1, large gap.
_SMALL_A = _rdf(
    {
        "contig": ["chr1", "chr1", "chr1", "chr1", "chr1"],
        "start": [100, 500, 1000, 2000, 3000],
        "stop": [200, 600, 1100, 2100, 3100],
    }
)
_SMALL_B = _rdf(
    {
        "contig": ["chr1", "chr1", "chr1", "chr1"],
        "start": [150, 600, 1100, 1101],
        "stop": [250, 700, 1200, 1201],
    }
)
# A[0] overlaps B[0], A[1] is book-ended with B[1] (gap=0),
# A[2] is book-ended with B[2] (gap=0), B[3] has gap=1 to A[2],
# A[3] has large gap to everything, A[4] has large gap to everything.


# ═══════════════════════════════════════════════════════════════════════
# Synthetic differential tests — small fixtures, every boundary case
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_bedtools
class TestIntersectSynthetic:
    """Compare intersect variants on small fixtures."""

    def test_intersect_u(self, request, tmp_path):
        """bedtools intersect -u == overlaps(a, b)."""
        _require_bedtools(request)
        _write_bed3(_SMALL_A, tmp_path / "a.bed")
        _write_bed3(_SMALL_B, tmp_path / "b.bed")
        bt = _run_bedtools(
            ["intersect", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-u"]
        )
        mask = overlaps(_SMALL_A, _SMALL_B)
        assert len(bt) == mask.sum(), (
            f"intersect -u: bedtools={len(bt)}, ours={mask.sum()}"
        )

    def test_intersect_v(self, request, tmp_path):
        """bedtools intersect -v == overlap_indices(how='anti')."""
        _require_bedtools(request)
        _write_bed3(_SMALL_A, tmp_path / "a.bed")
        _write_bed3(_SMALL_B, tmp_path / "b.bed")
        bt = _run_bedtools(
            ["intersect", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-v"]
        )
        anti = overlap_indices(_SMALL_A, _SMALL_B, how="anti")
        bt_set = _parse_bed3_lines(bt)
        our_set = set(
            (r.contig, int(r.start), int(r.stop))
            for _, r in _SMALL_A.iloc[anti.a_pos.values.astype(int)].iterrows()
        )
        assert bt_set == our_set

    def test_intersect_wa_wb(self, request, tmp_path):
        """bedtools intersect -wa -wb == overlap_indices (inner)."""
        _require_bedtools(request)
        _write_bed3(_SMALL_A, tmp_path / "a.bed")
        _write_bed3(_SMALL_B, tmp_path / "b.bed")
        bt = _run_bedtools(
            ["intersect", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-wa", "-wb"]
        )
        idx = overlap_indices(_SMALL_A, _SMALL_B)
        bt_set = _parse_wa_wb_bed3(bt)
        our_set = set()
        for _, row in idx.iterrows():
            ar = _SMALL_A.iloc[int(row.a_pos)]
            br = _SMALL_B.iloc[int(row.b_pos)]
            our_set.add((
                ar.contig, int(ar.start), int(ar.stop),
                br.contig, int(br.start), int(br.stop),
            ))
        assert bt_set == our_set

    def test_intersect_f_half(self, request, tmp_path):
        """bedtools intersect -f 0.5 == overlap_indices(min_frac_a=0.5)."""
        _require_bedtools(request)
        _write_bed3(_SMALL_A, tmp_path / "a.bed")
        _write_bed3(_SMALL_B, tmp_path / "b.bed")
        bt = _run_bedtools(
            ["intersect", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-wa", "-wb", "-f", "0.5"]
        )
        idx = overlap_indices(_SMALL_A, _SMALL_B, min_frac_a=0.5)
        assert len(bt) == len(idx)


@pytest.mark.requires_bedtools
class TestSubtractSynthetic:
    """bedtools subtract -A == overlap_indices(how='anti')."""

    def test_subtract_A(self, request, tmp_path):
        _require_bedtools(request)
        _write_bed3(_SMALL_A, tmp_path / "a.bed")
        _write_bed3(_SMALL_B, tmp_path / "b.bed")
        bt = _run_bedtools(
            ["subtract", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-A"]
        )
        anti = overlap_indices(_SMALL_A, _SMALL_B, how="anti")
        bt_set = _parse_bed3_lines(bt)
        our_set = set(
            (r.contig, int(r.start), int(r.stop))
            for _, r in _SMALL_A.iloc[anti.a_pos.values.astype(int)].iterrows()
        )
        assert bt_set == our_set


# ═══════════════════════════════════════════════════════════════════════
# Window boundary tests — gap=0 is the critical boundary
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_bedtools
class TestWindowBoundary:
    """bedtools window -w P == overlap_indices(pad=P), verified at boundaries.

    Gap=0 (book-ended) is tested explicitly: it is the boundary that three
    prior reviews missed, because tests asserted the interior (param=G matches,
    G-1 does not) which holds for G >= 1 but not for G=0.
    """

    @pytest.mark.parametrize(
        "gap,w,expect_match",
        [
            # gap=0: book-ended. window -w 0 = strict overlap = no match.
            (0, 0, False),
            (0, 1, True),
            # gap=1: one-base gap.
            (1, 0, False),
            (1, 1, False),
            (1, 2, True),
            # gap=5.
            (5, 5, False),
            (5, 6, True),
            # gap=10.
            (10, 10, False),
            (10, 11, True),
        ],
    )
    def test_window_boundary(self, request, tmp_path, gap, w, expect_match):
        """Gap G first matches at w=G+1 (gap < w), including G=0."""
        _require_bedtools(request)
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200 + gap], "stop": [300 + gap]})
        _write_bed3(a, tmp_path / "a.bed")
        _write_bed3(b, tmp_path / "b.bed")

        bt = _run_bedtools(
            ["window", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-w", str(w)]
        )
        bt_matched = len(bt) > 0

        idx = overlap_indices(a, b, pad=w)
        our_matched = len(idx) > 0

        assert bt_matched == our_matched == expect_match, (
            f"gap={gap}, w={w}: bedtools={bt_matched}, ours={our_matched}, "
            f"expected={expect_match}"
        )

    def test_genuinely_overlapping_at_w0(self, request, tmp_path):
        """Genuinely overlapping intervals match at window -w 0 / pad=0."""
        _require_bedtools(request)
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [300]})
        _write_bed3(a, tmp_path / "a.bed")
        _write_bed3(b, tmp_path / "b.bed")

        bt = _run_bedtools(
            ["window", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-w", "0"]
        )
        idx = overlap_indices(a, b, pad=0)
        assert len(bt) == len(idx) == 1


# ═══════════════════════════════════════════════════════════════════════
# Merge boundary tests — gap=0 is the critical boundary
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_bedtools
class TestMergeBoundary:
    """bedtools merge -d D == merge(min_dist=D), verified at boundaries."""

    @pytest.mark.parametrize(
        "gap,d,expect_merged",
        [
            # gap=0: book-ended. merge -d 0 joins them.
            (0, 0, True),
            # gap=1. merge -d 0 does NOT reach, -d 1 does.
            (1, 0, False),
            (1, 1, True),
            # gap=5.
            (5, 4, False),
            (5, 5, True),
            # gap=10.
            (10, 9, False),
            (10, 10, True),
        ],
    )
    def test_merge_boundary(self, request, tmp_path, gap, d, expect_merged):
        """Gap G first merges at d=G (gap <= d), including G=0."""
        _require_bedtools(request)
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 200 + gap],
            "stop": [200, 300 + gap],
        })
        _write_bed3(a, tmp_path / "a.bed")

        bt = _run_bedtools(
            ["merge", "-i", str(tmp_path / "a.bed"), "-d", str(d)]
        )
        bt_n = len(bt)

        merged = merge(a, min_dist=d)
        our_n = len(merged)

        if expect_merged:
            assert bt_n == our_n == 1, (
                f"gap={gap}, d={d}: expected merged. bt={bt_n}, ours={our_n}"
            )
        else:
            assert bt_n == our_n == 2, (
                f"gap={gap}, d={d}: expected separate. bt={bt_n}, ours={our_n}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Cluster tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_bedtools
class TestClusterSynthetic:
    """bedtools cluster grouping == cluster() grouping."""

    def test_cluster_grouping(self, request, tmp_path):
        """Same intervals are grouped together by both."""
        _require_bedtools(request)
        a = _rdf({
            "contig": ["chr1", "chr1", "chr1", "chr1"],
            "start": [100, 150, 500, 510],
            "stop": [200, 300, 600, 700],
        })
        _write_bed3(a, tmp_path / "a.bed")

        bt = _run_bedtools(["cluster", "-i", str(tmp_path / "a.bed")])
        bt_labels = [int(line.split("\t")[-1]) for line in bt]
        our_labels = list(cluster(a).values)

        # Labels may differ (bedtools is 1-based, ours is 0-based), but
        # the grouping must be identical.
        bt_groups = {}
        for i, lbl in enumerate(bt_labels):
            bt_groups.setdefault(lbl, []).append(i)
        our_groups = {}
        for i, lbl in enumerate(our_labels):
            our_groups.setdefault(lbl, []).append(i)

        assert sorted(bt_groups.values()) == sorted(our_groups.values())

    def test_cluster_book_ended(self, request, tmp_path):
        """Book-ended intervals cluster together (gap=0, default d=0)."""
        _require_bedtools(request)
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 200],
            "stop": [200, 300],
        })
        _write_bed3(a, tmp_path / "a.bed")

        bt = _run_bedtools(["cluster", "-i", str(tmp_path / "a.bed")])
        bt_labels = [int(line.split("\t")[-1]) for line in bt]
        assert bt_labels[0] == bt_labels[1], "bedtools should cluster book-ended"

        our_labels = cluster(a)
        assert our_labels.iloc[0] == our_labels.iloc[1], "ours should cluster book-ended"


# ═══════════════════════════════════════════════════════════════════════
# Closest tests — document the distance convention difference
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_bedtools
class TestClosestSynthetic:
    """bedtools closest -d == nearest(), column-for-column.

    Distance follows the bedtools convention: 0 for overlapping,
    gap + 1 for non-overlapping (including book-ended).  This test
    asserts EXACT AGREEMENT, not a known offset.

    This test is designed to FAIL against commit 76a2c6f, which used
    the bioframe gap convention (0 for book-ended).  That is the proof
    that the previous behaviour was wrong.
    """

    @pytest.mark.parametrize(
        "gap,expected_dist",
        [
            (-50, 0),    # overlapping
            (0, 1),      # book-ended (gap=0) — the critical boundary
            (1, 2),      # gap=1
            (10, 11),    # gap=10
            (100, 101),  # gap=100
        ],
    )
    def test_closest_distance_exact_agreement(
        self, request, tmp_path, gap, expected_dist
    ):
        """Our distance matches bedtools column-for-column."""
        _require_bedtools(request)
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b_start = 200 + gap
        if b_start < 0:
            b_start = 150  # overlapping case
        b = _rdf({"contig": ["chr1"], "start": [b_start], "stop": [b_start + 100]})
        _write_bed3(a, tmp_path / "a.bed")
        _write_bed3(b, tmp_path / "b.bed")

        bt = _run_bedtools(
            ["closest", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-d"]
        )
        bt_dist = int(bt[0].split("\t")[-1])

        near = nearest(a, b)
        our_dist = int(near.distance.iloc[0])

        assert bt_dist == expected_dist, (
            f"gap={gap}: bedtools distance={bt_dist}, expected={expected_dist}"
        )
        assert our_dist == expected_dist, (
            f"gap={gap}: our distance={our_dist}, expected={expected_dist}"
        )
        assert bt_dist == our_dist, (
            f"gap={gap}: distances disagree: bedtools={bt_dist}, ours={our_dist}"
        )

    def test_closest_finds_same_target(self, request, tmp_path):
        """Both find the same nearest B interval, even when there are several."""
        _require_bedtools(request)
        a = _rdf({"contig": ["chr1"], "start": [500], "stop": [600]})
        b = _rdf({
            "contig": ["chr1", "chr1", "chr1"],
            "start": [100, 700, 900],
            "stop": [200, 800, 1000],
        })
        _write_bed3(a, tmp_path / "a.bed")
        _write_bed3(b, tmp_path / "b.bed")

        bt = _run_bedtools(
            ["closest", "-a", str(tmp_path / "a.bed"),
             "-b", str(tmp_path / "b.bed"), "-d"]
        )
        bt_b_start = int(bt[0].split("\t")[4])
        bt_dist = int(bt[0].split("\t")[-1])

        near = nearest(a, b)
        our_b = b.iloc[int(near.b_pos.iloc[0])]
        our_dist = int(near.distance.iloc[0])

        assert bt_b_start == int(our_b.start), (
            f"bedtools found b_start={bt_b_start}, ours found {int(our_b.start)}"
        )
        assert bt_dist == our_dist, (
            f"distances disagree: bedtools={bt_dist}, ours={our_dist}"
        )

    def test_closest_gap_table_differential(self, request, tmp_path):
        """Full gap table comparison including overlapping and gap=0.

        This is the DIFFERENTIAL TEST required by the task: it compares our
        distance against the real CLI column-for-column across a gap table.
        This test FAILS against commit 76a2c6f (which used bioframe's gap
        convention) at the gap=0 (book-ended) case: bedtools reports 1,
        the old code reported 0.
        """
        _require_bedtools(request)
        # Build A: one interval per gap value.  B: one target interval.
        gaps = [-50, 0, 1, 5, 10, 50, 100]
        a_starts = []
        a_stops = []
        b_target_start = 10000
        b_target_stop = 10100
        for g in gaps:
            if g < 0:
                # Overlapping: A extends into B.
                a_s = b_target_start + g
                a_e = b_target_start + 100
            else:
                # Non-overlapping: A ends at b_start - gap.
                a_e = b_target_start - g
                a_s = a_e - 100
            a_starts.append(a_s)
            a_stops.append(a_e)

        a = _rdf({
            "contig": ["chr1"] * len(gaps),
            "start": a_starts,
            "stop": a_stops,
        })
        b = _rdf({
            "contig": ["chr1"],
            "start": [b_target_start],
            "stop": [b_target_stop],
        })
        a_bed = tmp_path / "a.bed"
        b_bed = tmp_path / "b.bed"
        a_sorted = tmp_path / "a_sorted.bed"
        _write_bed3(a, a_bed)
        _write_bed3(b, b_bed)
        _sort_bed(a_bed, a_sorted)

        bt_lines = _run_bedtools(
            ["closest", "-a", str(a_sorted), "-b", str(b_bed), "-d"]
        )
        bt_dists = {}
        for line in bt_lines:
            parts = line.split("\t")
            a_key = (parts[0], int(parts[1]), int(parts[2]))
            bt_dists[a_key] = int(parts[-1])

        near = nearest(a, b)
        our_dists = {}
        for _, row in near.iterrows():
            ar = a.iloc[int(row.a_pos)]
            a_key = (ar.contig, int(ar.start), int(ar.stop))
            our_dists[a_key] = int(row.distance)

        for key in bt_dists:
            assert key in our_dists, f"Missing from our result: {key}"
            assert bt_dists[key] == our_dists[key], (
                f"Distance mismatch at {key}: "
                f"bedtools={bt_dists[key]}, ours={our_dists[key]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Real-data differential tests — the main regression net
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.requires_bedtools
class TestRealDataIntersect:
    """Run bedtools intersect variants on real data and compare row sets."""

    def test_intersect_u_count(self, request, real_data, real_bed_files):
        """bedtools intersect -u row count matches overlaps()."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(["intersect", "-a", ctcf_bed, "-b", bl_bed, "-u"])
        mask = overlaps(ctcf, bl)
        assert len(bt) == mask.sum()

    def test_intersect_v_count(self, request, real_data, real_bed_files):
        """bedtools intersect -v row count matches anti-join."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(["intersect", "-a", ctcf_bed, "-b", bl_bed, "-v"])
        anti = overlap_indices(ctcf, bl, how="anti")
        assert len(bt) == len(anti)

    def test_intersect_u_plus_v_equals_total(self, request, real_data, real_bed_files):
        """Overlapping + non-overlapping = total rows."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt_u = _run_bedtools(["intersect", "-a", ctcf_bed, "-b", bl_bed, "-u"])
        bt_v = _run_bedtools(["intersect", "-a", ctcf_bed, "-b", bl_bed, "-v"])
        assert len(bt_u) + len(bt_v) == len(ctcf)

    def test_intersect_wa_wb_row_set(self, request, real_data, real_bed_files):
        """bedtools intersect -wa -wb row SET matches overlap_indices."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(
            ["intersect", "-a", ctcf_bed, "-b", bl_bed, "-wa", "-wb"]
        )
        bt_set = _parse_wa_wb_bed3(bt)

        idx = overlap_indices(ctcf, bl)
        our_set = set()
        for _, row in idx.iterrows():
            ar = ctcf.iloc[int(row.a_pos)]
            br = bl.iloc[int(row.b_pos)]
            our_set.add((
                ar.contig, int(ar.start), int(ar.stop),
                br.contig, int(br.start), int(br.stop),
            ))

        assert bt_set == our_set, (
            f"Row sets differ: {len(bt_set - our_set)} bedtools-only, "
            f"{len(our_set - bt_set)} ours-only"
        )

    def test_intersect_f_half(self, request, real_data, real_bed_files):
        """bedtools intersect -f 0.5 count matches."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(
            ["intersect", "-a", ctcf_bed, "-b", bl_bed, "-wa", "-wb", "-f", "0.5"]
        )
        idx = overlap_indices(ctcf, bl, min_frac_a=0.5)
        assert len(bt) == len(idx)

    def test_intersect_f_half_reciprocal(self, request, real_data, real_bed_files):
        """bedtools intersect -f 0.5 -r count matches."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(
            ["intersect", "-a", ctcf_bed, "-b", bl_bed, "-wa", "-wb",
             "-f", "0.5", "-r"]
        )
        idx = overlap_indices(ctcf, bl, min_frac_a=0.5, reciprocal=True)
        assert len(bt) == len(idx)


@pytest.mark.requires_bedtools
class TestRealDataSubtract:
    def test_subtract_A(self, request, real_data, real_bed_files):
        """bedtools subtract -A matches anti-join on real data."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(
            ["subtract", "-a", ctcf_bed, "-b", bl_bed, "-A"]
        )
        anti = overlap_indices(ctcf, bl, how="anti")
        assert len(bt) == len(anti)


@pytest.mark.requires_bedtools
class TestRealDataWindow:
    def test_window_w10(self, request, real_data, real_bed_files):
        """bedtools window -w 10 matches overlap_indices(pad=10)."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, bl_bed, _ = real_bed_files

        bt = _run_bedtools(
            ["window", "-a", ctcf_bed, "-b", bl_bed, "-w", "10"]
        )
        idx = overlap_indices(ctcf, bl, pad=10)
        assert len(bt) == len(idx)


@pytest.mark.requires_bedtools
class TestRealDataMerge:
    def test_merge_d0_row_set(self, request, real_data, real_bed_files):
        """bedtools merge -d 0 row set matches merge(min_dist=0)."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, _, _ = real_bed_files

        bt = _run_bedtools(["merge", "-i", ctcf_bed, "-d", "0"])
        bt_set = _parse_bed3_lines(bt)

        merged = merge(ctcf, min_dist=0)
        our_set = set(
            (r.contig, int(r.start), int(r.stop))
            for _, r in merged.iterrows()
        )
        assert bt_set == our_set

    def test_merge_d10(self, request, real_data, real_bed_files):
        """bedtools merge -d 10 count matches merge(min_dist=10)."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, _, _ = real_bed_files

        bt = _run_bedtools(["merge", "-i", ctcf_bed, "-d", "10"])
        merged = merge(ctcf, min_dist=10)
        assert len(bt) == len(merged)


@pytest.mark.requires_bedtools
class TestRealDataCluster:
    def test_cluster_count(self, request, real_data, real_bed_files):
        """Same number of clusters as bedtools."""
        _require_real_data(request)
        ctcf, bl = real_data
        ctcf_bed, _, _ = real_bed_files

        bt = _run_bedtools(["cluster", "-i", ctcf_bed])
        bt_n = len(set(line.split("\t")[-1] for line in bt))

        labels = cluster(ctcf)
        our_n = labels.nunique()
        assert bt_n == our_n
