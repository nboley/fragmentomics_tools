"""Tests for scripts/count_cut_site_hexamers.py.

These are written to FAIL if the cut-site geometry is wrong, which is the whole
risk: a hexamer table misaligned with the simulator's indexing errors nowhere
and simply parameterises the wrong thing.  So the hexamer index is recomputed
here from an *independent* encoder (plain Python string arithmetic, no import of
the production LUT) and every strand/parity case asserts both that the expected
bin is hot and that the bin a plausible bug would have hit is cold.

All synthetic -- no FASTA and no h5 except the one test that cross-checks
``region_hexamers`` against ``simulator.precompute.precompute_region``.
"""

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, "..", "scripts", "count_cut_site_hexamers.py")
_spec = importlib.util.spec_from_file_location("count_cut_site_hexamers", _PATH)
csh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csh)

NHEX = csh.NHEX
TABLE_NAMES = csh.TABLE_NAMES


# ── independent hexamer encoder (deliberately NOT the production one) ─────

_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def hidx(s: str) -> int:
    """Big-endian base-4 index of a 6-mer, computed from scratch."""
    v = 0
    for ch in s:
        v = v * 4 + _CODE[ch]
    return v


def rc(s: str) -> str:
    return "".join(_COMP[c] for c in reversed(s))


def hexamer_at(genome: str, cut: int) -> str:
    """The 6-mer spanning genomic cut site ``cut``: 3 bases in, 3 bases out."""
    return genome[cut - 3:cut + 3]


# ── helpers ──────────────────────────────────────────────────────────────

def tracks(genome: str, gstart: int, gstop: int):
    """``(hex_fwd, hex_rc, valid)`` for region ``[gstart, gstop)`` of *genome*."""
    sub = genome[gstart - csh.HEX_HALF:gstop + csh.HEX_HALF]
    return csh.hexamer_indices(np.frombuffer(sub.encode("ascii"), dtype=np.uint8))


def empty_tables(n_bands=1):
    return {k: np.zeros((n_bands, NHEX), dtype=np.int64) for k in TABLE_NAMES}


def count_one(genome, gstart, gstop, p, L, plus, n_bands=1, l_min=0, band_width=10_000):
    """Run one fragment through ``accumulate_observed``; return ``(obs, n)``."""
    hex_fwd, hex_rc, valid = tracks(genome, gstart, gstop)
    obs = empty_tables(n_bands)
    n = csh.accumulate_observed(
        obs, hex_fwd, hex_rc, valid,
        np.array([p]), np.array([p + L]), np.array([plus]),
        gstart, l_min, band_width, n_bands,
    )
    return obs, n


# An 800 bp pseudo-random-but-fixed genome with no long repeats, so that the
# hexamers at two different cut sites are essentially never equal and a
# mis-indexing bug cannot pass by coincidence.  Long enough to hold a region
# that fits the real L range (25..180) with room for fragments to start
# anywhere in it.
GENOME = "".join(
    "ACGT"[(i * 7 + (i * i) // 3) % 4] for i in range(800)
)


# ── band edges ───────────────────────────────────────────────────────────

def test_band_edges_tile_the_range_and_truncate_the_last():
    edges = csh.make_band_edges(25, 180, 10)
    assert len(edges) == 16
    assert edges[0].tolist() == [25, 35]
    assert edges[-1].tolist() == [175, 181]
    # contiguous, no gap, no overlap
    assert (edges[1:, 0] == edges[:-1, 1]).all()
    # every length in range lands in exactly one band
    covered = sum(int(hi - lo) for lo, hi in edges)
    assert covered == 180 - 25 + 1


def test_band_index_matches_the_edges():
    edges = csh.make_band_edges(25, 180, 10)
    lengths = np.arange(25, 181)
    bidx = csh.band_index(lengths, 25, 10)
    assert bidx.min() == 0 and bidx.max() == len(edges) - 1
    for L, b in zip(lengths, bidx):
        lo, hi = edges[b]
        assert lo <= L < hi, f"L={L} landed in band {b} = [{lo},{hi})"


def test_band_edges_reject_nonsense():
    with pytest.raises(ValueError):
        csh.make_band_edges(25, 180, 0)
    with pytest.raises(ValueError):
        csh.make_band_edges(180, 25, 10)


# ── the expected index, both strands, both parities of L ─────────────────

@pytest.mark.parametrize("L", [30, 31])  # even and odd: odd L is where an
                                         # off-by-one in p+L vs p+L-1 hides
def test_plus_strand_hits_the_expected_fwd_index(L):
    gstart, gstop, p = 20, 120, 40
    obs, n = count_one(GENOME, gstart, gstop, p, L, plus=True)
    assert n == 1

    # Cut sites, NOT endpoint bases: p and p+L (not p+L-1).
    expect_start = hidx(hexamer_at(GENOME, p))
    expect_end = hidx(hexamer_at(GENOME, p + L))
    assert obs["start_fwd"][0, expect_start] == 1
    assert obs["end_fwd"][0, expect_end] == 1
    assert obs["start_fwd"].sum() == 1 and obs["end_fwd"].sum() == 1
    # the minus tables must be untouched
    assert obs["start_rev"].sum() == 0 and obs["end_rev"].sum() == 0

    # The off-by-one a "last base" reading would produce must be a DIFFERENT
    # index -- otherwise this test proves nothing.
    off_by_one = hidx(hexamer_at(GENOME, p + L - 1))
    assert off_by_one != expect_end
    assert obs["end_fwd"][0, off_by_one] == 0


@pytest.mark.parametrize("L", [30, 31])
def test_minus_strand_swaps_the_cut_sites_and_reads_reverse_complement(L):
    """The two ways to get the minus strand wrong, each asserted to be wrong.

    1. Failing to swap: ``c5`` must be ``p+L`` (the HIGHER coordinate), not ``p``.
    2. Failing to reverse-complement: the minus tables index ``hex_rc``, not
       ``hex_fwd``.

    Both alternatives are shown to give a *different* index, and the bin they
    would have hit is asserted empty.
    """
    gstart, gstop, p = 20, 120, 40
    obs, n = count_one(GENOME, gstart, gstop, p, L, plus=False)
    assert n == 1

    hex_at_low = hexamer_at(GENOME, p)        # c3 for a minus fragment
    hex_at_high = hexamer_at(GENOME, p + L)   # c5 for a minus fragment

    expect_start = hidx(rc(hex_at_high))
    expect_end = hidx(rc(hex_at_low))
    assert obs["start_rev"][0, expect_start] == 1
    assert obs["end_rev"][0, expect_end] == 1
    assert obs["start_fwd"].sum() == 0 and obs["end_fwd"].sum() == 0

    # (1) no-swap bug: start would read the LOW coordinate.
    no_swap_start = hidx(rc(hex_at_low))
    assert no_swap_start != expect_start
    assert obs["start_rev"][0, no_swap_start] == 0

    # (2) fwd/rc mix-up: start would read the forward index at the same site.
    #     Guard that the 6-mer is not its own reverse complement, or the
    #     assertion below would be vacuous.
    assert hex_at_high != rc(hex_at_high), "palindromic 6-mer makes this vacuous"
    mixup_start = hidx(hex_at_high)
    assert mixup_start != expect_start
    assert obs["start_rev"][0, mixup_start] == 0


def test_plus_and_minus_at_identical_coordinates_land_in_different_bins():
    """A fwd/rc mix-up would make the two strands agree. They must not."""
    gstart, gstop, p, L = 20, 120, 40, 31
    obs_p, _ = count_one(GENOME, gstart, gstop, p, L, plus=True)
    obs_m, _ = count_one(GENOME, gstart, gstop, p, L, plus=False)
    hot_p = np.flatnonzero(obs_p["start_fwd"][0])
    hot_m = np.flatnonzero(obs_m["start_rev"][0])
    assert len(hot_p) == len(hot_m) == 1
    assert hot_p[0] != hot_m[0]


def test_counts_land_in_the_right_length_band():
    edges = csh.make_band_edges(25, 180, 10)
    gstart, gstop, p = 20, 620, 40
    for L in (25, 34, 35, 100, 180):
        obs, n = count_one(
            GENOME, gstart, gstop, p, L, plus=True,
            n_bands=len(edges), l_min=25, band_width=10,
        )
        assert n == 1
        hot_band = np.flatnonzero(obs["start_fwd"].sum(axis=1))
        assert hot_band.tolist() == [int(csh.band_index(np.array(L), 25, 10))]
        lo, hi = edges[hot_band[0]]
        assert lo <= L < hi


# ── nothing is silently dropped ──────────────────────────────────────────

def test_total_observed_is_two_per_counted_fragment():
    gstart, gstop = 20, 620
    rng = np.random.default_rng(0)
    p = rng.integers(gstart, gstop - 180, size=200)
    L = rng.integers(25, 181, size=200)
    plus = rng.integers(0, 2, size=200).astype(bool)
    hex_fwd, hex_rc, valid = tracks(GENOME, gstart, gstop)
    obs = empty_tables(16)
    n = csh.accumulate_observed(
        obs, hex_fwd, hex_rc, valid, p, p + L, plus, gstart, 25, 10, 16,
    )
    assert n == 200  # no Ns in GENOME, so every fragment is countable
    assert sum(int(v.sum()) for v in obs.values()) == 2 * n
    # and each fragment contributes exactly one start and one end
    assert obs["start_fwd"].sum() + obs["start_rev"].sum() == n
    assert obs["end_fwd"].sum() + obs["end_rev"].sum() == n
    assert obs["start_fwd"].sum() == int(plus.sum())


# ── the stated rules at the edges ────────────────────────────────────────

def test_region_edge_rule_is_containment_not_overlap():
    gstart, gstop = 100, 200
    starts = np.array([100, 150, 99, 190, 170])
    stops = np.array([150, 200, 149, 210, 201])
    keep = csh.contained_in_region(starts, stops, gstart, gstop)
    # flush at the left edge, flush at the right edge -> both kept
    assert keep.tolist() == [True, True, False, False, False]


def test_a_fragment_straddling_a_tile_boundary_is_counted_in_neither_tile():
    """The boundary fragment must not be smuggled into one side."""
    left, right = (100, 200), (200, 300)
    starts, stops = np.array([190]), np.array([230])
    assert not csh.contained_in_region(starts, stops, *left).any()
    assert not csh.contained_in_region(starts, stops, *right).any()


def test_a_cut_site_in_an_N_window_drops_the_whole_fragment():
    """Stated rule: count only when BOTH cut sites have valid hexamers."""
    genome = list(GENOME)
    gstart, gstop, p, L = 20, 120, 40, 31
    # An N two bases upstream of the 5' cut site is inside its 3-in/3-out
    # window, so that cut site is invalid and the fragment must be dropped.
    genome[p - 2] = "N"
    g = "".join(genome)
    _, _, valid = tracks(g, gstart, gstop)
    assert not valid[p - gstart], "test setup: the 5' cut site should be invalid"
    assert valid[p + L - gstart], "test setup: the 3' cut site should be valid"

    for plus in (True, False):
        obs, n = count_one(g, gstart, gstop, p, L, plus=plus)
        assert n == 0
        assert sum(int(v.sum()) for v in obs.values()) == 0

    # ...and with the N gone the same fragment IS counted, so the test above
    # is about the N and not about the coordinates.
    obs, n = count_one(GENOME, gstart, gstop, p, L, plus=True)
    assert n == 1


def test_an_N_outside_both_hexamer_windows_does_not_drop_the_fragment():
    genome = list(GENOME)
    gstart, gstop, p, L = 20, 120, 40, 31
    genome[p + 10] = "N"  # interior, far from either cut-site window
    obs, n = count_one("".join(genome), gstart, gstop, p, L, plus=True)
    assert n == 1


# ── filtering and de-duplication ─────────────────────────────────────────

def _supp(mapqs, strands):
    return np.array(mapqs, dtype=np.int32), np.array(strands, dtype="S1")


def test_mapq_filter_is_inclusive_on_the_pair_minimum():
    starts = np.array([1, 2, 3, 4])
    stops = np.array([51, 52, 53, 54])
    mapq, strand = _supp(
        [[60, 60], [10, 60], [9, 60], [60, 9]], [b"+", b"+", b"+", b"+"],
    )
    s, _, _, n_pass, _, _ = csh.filter_and_dedup(starts, stops, mapq, strand, 10)
    assert n_pass == 2  # 60/60 and 10/60; both 9s fail
    assert s.tolist() == [1, 2]


def test_unknown_mapq_minus_one_is_removed():
    """The '-1 >= 10' trap: an h5 without MAPQ loses everything."""
    starts = np.array([1, 2])
    stops = np.array([51, 52])
    mapq, strand = _supp([[-1, -1], [-1, -1]], [b"+", b"-"])
    s, _, _, n_pass, _, _ = csh.filter_and_dedup(starts, stops, mapq, strand, 10)
    assert n_pass == 0 and len(s) == 0


def test_dedup_collapses_identical_coordinates_and_ignores_strand():
    starts = np.array([10, 10, 10, 20])
    stops = np.array([60, 60, 60, 70])
    mapq, strand = _supp(
        [[60, 60]] * 4, [b"+", b"+", b"-", b"+"],
    )
    s, e, is_plus, n_pass, n_dup, n_dup_s = csh.filter_and_dedup(
        starts, stops, mapq, strand, 10,
    )
    assert n_pass == 4
    assert len(s) == 2                  # (10,60) and (20,70)
    assert n_dup == 2                   # applied key: coordinates only
    assert n_dup_s == 1                 # a strand-aware key would keep the (10,60,-)
    assert sorted(s.tolist()) == [10, 20]


def test_mapq_filter_runs_before_dedup():
    """Order matters: a low-MAPQ copy must not be the survivor of a dup pair."""
    starts = np.array([10, 10])
    stops = np.array([60, 60])
    mapq, strand = _supp([[5, 5], [60, 60]], [b"+", b"+"])
    s, _, _, n_pass, n_dup, _ = csh.filter_and_dedup(
        starts, stops, mapq, strand, 10,
    )
    assert n_pass == 1 and n_dup == 0 and len(s) == 1


# ── background: candidate cut sites ──────────────────────────────────────

def brute_background(hex_fwd, hex_rc, valid, region_len, lo, hi):
    """Enumerate the generative domain Omega, one fragment at a time."""
    t = {k: np.zeros(NHEX, dtype=np.float64) for k in TABLE_NAMES}
    for L in range(lo, hi + 1):
        for c5 in range(0, region_len - L + 1):      # plus: c3 = c5 + L
            c3 = c5 + L
            if valid[c5] and valid[c3]:
                t["start_fwd"][hex_fwd[c5]] += 1
                t["end_fwd"][hex_fwd[c3]] += 1
        for c5 in range(L, region_len + 1):          # minus: c3 = c5 - L
            c3 = c5 - L
            if valid[c5] and valid[c3]:
                t["start_rev"][hex_rc[c5]] += 1
                t["end_rev"][hex_rc[c3]] += 1
    return t


@pytest.mark.parametrize("genome_mutator", ["clean", "with_Ns"])
def test_closed_form_background_equals_brute_force_enumeration(genome_mutator):
    """The cumsum shortcut must reproduce a literal walk over Omega."""
    gstart, gstop = 20, 100
    region_len = gstop - gstart
    g = GENOME
    if genome_mutator == "with_Ns":
        gl = list(GENOME)
        for i in (25, 26, 70, 95):
            gl[i] = "N"
        g = "".join(gl)
    hex_fwd, hex_rc, valid = tracks(g, gstart, gstop)
    if genome_mutator == "with_Ns":
        assert not valid.all(), "test setup: the Ns should invalidate cut sites"

    edges = np.array([[5, 13]])  # L in 5..12
    bg = {k: np.zeros((1, NHEX), dtype=np.float64) for k in TABLE_NAMES}
    csh.accumulate_background(bg, hex_fwd, hex_rc, valid, edges, region_len)

    want = brute_background(hex_fwd, hex_rc, valid, region_len, 5, 12)
    for k in TABLE_NAMES:
        np.testing.assert_array_equal(bg[k][0], want[k], err_msg=k)
    assert bg["start_fwd"].sum() > 0


def test_background_total_matches_the_generative_domain_size():
    """With no Ns, the total equals the containment-rule domain size.

    ``accumulate_background`` uses the containment rule (c3 stays in
    [0, region_len]), not the midpoint rule.  ``generative_domain_size``
    was changed to return the midpoint-rule count, so this test computes
    the containment-rule count directly.
    """
    gstart, gstop = 20, 100
    region_len = gstop - gstart
    hex_fwd, hex_rc, valid = tracks(GENOME, gstart, gstop)
    assert valid.all()

    edges = np.array([[5, 13]])
    bg = {k: np.zeros((1, NHEX), dtype=np.float64) for k in TABLE_NAMES}
    csh.accumulate_background(bg, hex_fwd, hex_rc, valid, edges, region_len)

    # Containment-rule domain: 2 * sum_{L=lo}^{hi} (region_len - L + 1)
    omega_containment = 2 * sum(region_len - L + 1 for L in range(5, 13))
    # each member of Omega contributes one start and one end
    assert sum(float(v.sum()) for v in bg.values()) == 2 * omega_containment
    # ...split evenly between the strands
    assert bg["start_fwd"].sum() + bg["end_fwd"].sum() == omega_containment


def test_background_start_end_symmetry_across_strands():
    """plus-start candidates == minus-end candidates (different track only)."""
    gl = list(GENOME)
    gl[40] = "N"
    hex_fwd, hex_rc, valid = tracks("".join(gl), 20, 100)
    edges = np.array([[5, 13]])
    bg = {k: np.zeros((1, NHEX), dtype=np.float64) for k in TABLE_NAMES}
    csh.accumulate_background(bg, hex_fwd, hex_rc, valid, edges, 80)
    assert bg["start_fwd"].sum() == bg["end_rev"].sum()
    assert bg["end_fwd"].sum() == bg["start_rev"].sum()


def test_background_bands_partition_the_whole_length_range():
    """Summing narrow bands must equal one wide band -- no double count, no gap."""
    hex_fwd, hex_rc, valid = tracks(GENOME, 20, 100)
    narrow = csh.make_band_edges(5, 20, 4)
    wide = np.array([[5, 21]])
    bg_n = {k: np.zeros((len(narrow), NHEX)) for k in TABLE_NAMES}
    bg_w = {k: np.zeros((1, NHEX)) for k in TABLE_NAMES}
    csh.accumulate_background(bg_n, hex_fwd, hex_rc, valid, narrow, 80)
    csh.accumulate_background(bg_w, hex_fwd, hex_rc, valid, wide, 80)
    for k in TABLE_NAMES:
        np.testing.assert_array_equal(bg_n[k].sum(axis=0), bg_w[k][0], err_msg=k)


def test_observed_is_a_subset_of_the_background_support():
    """Every hexamer bin an observation lands in must be a candidate bin.

    A non-zero observation against a zero background is the signature of a
    geometry mismatch between the two halves of the output.
    """
    gstart, gstop, region_len = 20, 620, 600
    hex_fwd, hex_rc, valid = tracks(GENOME, gstart, gstop)
    edges = csh.make_band_edges(25, 180, 10)

    rng = np.random.default_rng(7)
    p = rng.integers(gstart, gstop - 180, size=300)
    L = rng.integers(25, 181, size=300)
    plus = rng.integers(0, 2, size=300).astype(bool)

    obs = empty_tables(len(edges))
    csh.accumulate_observed(
        obs, hex_fwd, hex_rc, valid, p, p + L, plus, gstart, 25, 10, len(edges),
    )
    bg = {k: np.zeros((len(edges), NHEX)) for k in TABLE_NAMES}
    csh.accumulate_background(bg, hex_fwd, hex_rc, valid, edges, region_len)

    for k in TABLE_NAMES:
        orphan = (obs[k] > 0) & (bg[k] == 0)
        assert not orphan.any(), f"{k}: {orphan.sum()} observed bins have zero background"


# ── the shared primitive really is shared ────────────────────────────────

def test_region_hexamers_matches_precompute_region_exactly(tmp_path):
    """``region_hexamers`` must not have drifted from the simulator's version.

    It exists only to avoid re-opening the 3 GB FASTA per region; if it ever
    disagrees with ``precompute_region`` the observed tables silently stop
    matching the tracks they parameterise.
    """
    pysam = pytest.importorskip("pysam")
    from background_model.simulator.precompute import precompute_region

    fa_path = tmp_path / "tiny.fa"
    seq = GENOME[:60] + "N" * 4 + GENOME[64:]
    with open(fa_path, "w") as fh:
        fh.write(">chrT\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")
    pysam.faidx(str(fa_path))

    gstart, gstop = 20, 180
    want = precompute_region("chrT", gstart, gstop, str(fa_path), pad=0)

    fa = pysam.FastaFile(str(fa_path))
    try:
        hex_fwd, hex_rc, valid = csh.region_hexamers(fa, "chrT", gstart, gstop)
    finally:
        fa.close()

    np.testing.assert_array_equal(hex_fwd, want.hex_fwd)
    np.testing.assert_array_equal(hex_rc, want.hex_rc)
    np.testing.assert_array_equal(valid, want.valid)
    assert len(hex_fwd) == gstop - gstart + 1
    assert not valid.all(), "test setup: the N run should invalidate windows"


def test_hexamer_track_matches_the_independent_encoder():
    """Cross-check the production LUT against plain string arithmetic."""
    gstart, gstop = 20, 100
    hex_fwd, hex_rc, valid = tracks(GENOME, gstart, gstop)
    assert valid.all()
    for c in (0, 1, 37, gstop - gstart):
        six = hexamer_at(GENOME, gstart + c)
        assert len(six) == 6
        assert hex_fwd[c] == hidx(six), f"fwd at cut site {c}"
        assert hex_rc[c] == hidx(rc(six)), f"rc at cut site {c}"


# ── the docstring examples are tests, and nothing else runs them ─────────

def test_every_docstring_example_in_the_script_runs():
    """``make test --doctest-modules`` covers ``test/`` and
    ``fragmentomics_tools/``, not ``scripts/`` -- and ``scripts/`` cannot simply
    be added, because three unrelated scripts there fail to even import.  So
    this module's examples were never executed, and one of them was stale:
    numpy 2.x reprs a bytes scalar as ``np.bytes_(b'AAAAAA')``, so the
    ``hexamer_vocabulary`` example could not have passed.  Run them here.
    """
    import doctest

    results = doctest.testmod(csh, verbose=False)
    assert results.attempted >= 3, "the examples stopped being collected"
    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} doctests failed"
    )


# ── the hexamer vocabulary ───────────────────────────────────────────────

def test_vocabulary_inverts_the_independent_encoder():
    """``vocab[i]`` must be the 6-mer the test's own encoder sends to ``i``.

    This is the property the whole output format rests on. It is checked
    against ``hidx`` -- plain string arithmetic written here -- rather than
    against the production indexer, so agreement is evidence and not a
    tautology.
    """
    vocab = csh.hexamer_vocabulary()
    assert vocab.shape == (NHEX,)
    assert len(set(vocab.tolist())) == NHEX, "vocabulary is not a bijection"
    for i, word in enumerate(vocab.astype("U")):
        assert hidx(word) == i, f"vocab[{i}] = {word!r}, but hidx says {hidx(word)}"


# ── round trip: write, read, join on the STRING ──────────────────────────

def _distinct_tables(n_bands, seed=11):
    """``(obs, bg)`` whose every element is distinct, so a permutation shows.

    Counts that were all equal, or equal within a band, would let a scrambled
    hexamer column round-trip "successfully". Every one of the
    ``4 x n_bands x 4096`` entries here is a different number.
    """
    rng = np.random.default_rng(seed)
    flat = rng.permutation(len(TABLE_NAMES) * n_bands * NHEX).reshape(
        len(TABLE_NAMES), n_bands, NHEX
    )
    obs = {k: flat[i].astype(np.int64) for i, k in enumerate(TABLE_NAMES)}
    # background is float64 upstream (np.bincount weights), but integral.
    bg = {k: (flat[i] * 7 + 3).astype(np.float64) for i, k in enumerate(TABLE_NAMES)}
    return obs, bg


def _consumer_rebuild(df, n_bands, band_edges):
    """Rebuild index-ordered ``(n_bands, NHEX)`` arrays the way a consumer must.

    Deliberately uses this file's own ``hidx`` encoder and nothing from the
    producer: each row's hexamer *string* is looked up to get a position. If
    the producer ever changed its internal ordering this still lands every
    count in the right bin; if it changed its RC or string convention, the
    lookup misses and the assertions below fail.
    """
    hexamers = df["hexamer"].to_numpy()
    hex_lut = {s: hidx(s) for s in set(hexamers.tolist())}
    idx = np.fromiter(
        (hex_lut[s] for s in hexamers), dtype=np.int64, count=len(hexamers),
    )

    band_lut = {(int(lo), int(hi)): b for b, (lo, hi) in enumerate(band_edges)}
    band = np.fromiter(
        (band_lut[(lo, hi)]
         for lo, hi in zip(df["band_lo"].tolist(), df["band_hi"].tolist())),
        dtype=np.int64, count=len(df),
    )

    names = df["table"].to_numpy()
    out = {}
    for k in TABLE_NAMES:
        m = names == k
        cells = {}
        for col in ("observed", "background"):
            # -1 fill: a cell no row reached stays negative, so a dropped row
            # leaves a hole rather than a plausible zero.
            a = np.full((n_bands, NHEX), -1, dtype=np.int64)
            a[band[m], idx[m]] = df[col].to_numpy()[m]
            cells[col] = a
        out[k] = cells
    return out


def test_round_trip_joins_on_the_hexamer_string(tmp_path):
    """Write, read back, join by string: every count must survive exactly.

    Not "the totals agree" -- element for element, through a rebuild that
    never sees the producer's integer code.
    """
    band_edges = csh.make_band_edges(25, 180, 10)
    n_bands = len(band_edges)
    obs, bg = _distinct_tables(n_bands)
    meta = {"sample_name": "synthetic", "min_mapq": 10, "stats": {"n_counted": 7}}

    path = tmp_path / "rt.parquet"
    csh.write_output(str(path), obs, bg, band_edges, meta)
    df, got_meta = csh.read_output(str(path))

    assert got_meta == meta
    assert list(df.columns) == [
        "hexamer", "table", "band_lo", "band_hi", "observed", "background",
    ]
    assert len(df) == len(TABLE_NAMES) * n_bands * NHEX
    # the key really is a key: no (hexamer, table, band) appears twice
    assert not df.duplicated(["hexamer", "table", "band_lo", "band_hi"]).any()

    rebuilt = _consumer_rebuild(df, n_bands, band_edges)
    for k in TABLE_NAMES:
        np.testing.assert_array_equal(
            rebuilt[k]["observed"], obs[k], err_msg=f"observed {k}",
        )
        np.testing.assert_array_equal(
            rebuilt[k]["background"], bg[k].astype(np.int64),
            err_msg=f"background {k}",
        )
        # -1 is the fill: every cell must have been reached by some row.
        assert (rebuilt[k]["observed"] >= 0).all()


def test_round_trip_detects_a_permuted_hexamer_column(tmp_path):
    """The failure the string key exists to make detectable, made to happen.

    A relabelling of the table is invisible to totals and to any normalisation
    -- so it is asserted here that the *join* catches it. Without this, the
    round-trip test above only proves that writing and reading are mutually
    consistent, which they would be under a scrambled convention too.
    """
    band_edges = csh.make_band_edges(25, 180, 10)
    n_bands = len(band_edges)
    obs, bg = _distinct_tables(n_bands)
    path = tmp_path / "rt.parquet"
    csh.write_output(str(path), obs, bg, band_edges, {"sample_name": "synthetic"})
    df, _ = csh.read_output(str(path))

    # Roll the hexamer column by one within each (table, band) block: totals
    # and every normalisation are untouched, the file still parses, and the
    # row count is identical.
    scrambled = df.copy()
    blocks = []
    for start in range(0, len(df), NHEX):
        block = df["hexamer"].to_numpy()[start:start + NHEX]
        blocks.append(np.roll(block, 1))
    scrambled["hexamer"] = np.concatenate(blocks)
    assert scrambled["observed"].sum() == df["observed"].sum()
    assert len(scrambled) == len(df)

    rebuilt = _consumer_rebuild(scrambled, n_bands, band_edges)
    mismatched = [
        k for k in TABLE_NAMES
        if not np.array_equal(rebuilt[k]["observed"], obs[k])
    ]
    assert mismatched == list(TABLE_NAMES), (
        "a permuted hexamer column round-tripped undetected -- the string key "
        "is not actually load-bearing"
    )


def test_round_trip_detects_a_mislabelled_table(tmp_path):
    """Swapping two table labels must also fail the join, not just the totals."""
    band_edges = csh.make_band_edges(25, 180, 10)
    n_bands = len(band_edges)
    obs, bg = _distinct_tables(n_bands)
    path = tmp_path / "rt.parquet"
    csh.write_output(str(path), obs, bg, band_edges, {"sample_name": "synthetic"})
    df, _ = csh.read_output(str(path))

    swap = {"start_fwd": "start_rev", "start_rev": "start_fwd"}
    mislabelled = df.copy()
    mislabelled["table"] = df["table"].map(lambda t: swap.get(t, t))
    assert mislabelled["observed"].sum() == df["observed"].sum()

    rebuilt = _consumer_rebuild(mislabelled, n_bands, band_edges)
    assert not np.array_equal(rebuilt["start_fwd"]["observed"], obs["start_fwd"])
    assert not np.array_equal(rebuilt["start_rev"]["observed"], obs["start_rev"])
    # ...and the untouched tables still match, so the assertion above is about
    # the swap and not about the rebuild being broken generally.
    np.testing.assert_array_equal(rebuilt["end_fwd"]["observed"], obs["end_fwd"])


def test_round_trip_detects_a_dropped_row(tmp_path):
    """A short table must be visible, not silently padded."""
    band_edges = csh.make_band_edges(25, 180, 10)
    n_bands = len(band_edges)
    obs, bg = _distinct_tables(n_bands)
    path = tmp_path / "rt.parquet"
    csh.write_output(str(path), obs, bg, band_edges, {"sample_name": "synthetic"})
    df, _ = csh.read_output(str(path))

    truncated = df.drop(index=df.index[3]).reset_index(drop=True)
    rebuilt = _consumer_rebuild(truncated, n_bands, band_edges)
    missing = [k for k in TABLE_NAMES if (rebuilt[k]["observed"] < 0).any()]
    assert missing, "a dropped row left no hole -- the rebuild is not complete"


# ── provenance ───────────────────────────────────────────────────────────

def test_every_provenance_field_survives_the_round_trip(tmp_path):
    """Nothing the npz carried may be lost by the format change."""
    band_edges = csh.make_band_edges(25, 40, 5)
    obs, bg = _distinct_tables(len(band_edges))
    meta = {
        "script": "count_cut_site_hexamers.py",
        "sample_name": "RD-00000-Lib1",
        "fragments_h5": "/efs/x/y.h5",
        "regions_bed": "/repo/data/regions.bed",
        "regions_bed_md5": "0" * 32,
        "n_regions_in_bed": 11505,
        "fasta": "/efs/hg38.fa",
        "min_mapq": 10,
        "mapq_rule": "min(mapq_read1, mapq_read2) >= min_mapq (inclusive)",
        "dedup_rule": "collapse (contig, start, stop); strand NOT in the key",
        "in_region_rule": "fully contained: gstart <= start and stop <= gstop",
        "l_min": 25, "l_max_inclusive": 180, "band_width": 5,
        "n_bands": len(band_edges), "n_hexamers": NHEX,
        "background_included": True,
        "duplicate_rate": 0.7863620858508819,
        "stats": {"n_fetched": 5991138, "n_counted": 857675, "elapsed_s": 187.2},
    }
    path = tmp_path / "prov.parquet"
    csh.write_output(str(path), obs, bg, band_edges, meta)
    _, got = csh.read_output(str(path))
    assert got == meta
    # floats must not have been stringified or rounded on the way through
    assert got["duplicate_rate"] == meta["duplicate_rate"]
    assert got["stats"]["n_fetched"] == 5991138


def test_reading_a_parquet_without_provenance_raises(tmp_path):
    """Counts of unknown origin must not be returned as if they were ours."""
    band_edges = csh.make_band_edges(25, 40, 5)
    obs, bg = _distinct_tables(len(band_edges))
    path = tmp_path / "bare.parquet"
    csh.write_output(str(path), obs, bg, band_edges, {"sample_name": "x"})

    stripped = tmp_path / "stripped.parquet"
    table = csh.pq.read_table(str(path)).replace_schema_metadata({})
    csh.pq.write_table(table, str(stripped))
    with pytest.raises(ValueError, match="provenance"):
        csh.read_output(str(stripped))


def test_the_band_columns_carry_the_edges_as_values(tmp_path):
    """The band scheme must be recoverable from the data, not from the schema."""
    band_edges = csh.make_band_edges(25, 180, 10)
    obs, bg = _distinct_tables(len(band_edges))
    path = tmp_path / "bands.parquet"
    csh.write_output(str(path), obs, bg, band_edges, {"sample_name": "x"})
    df, _ = csh.read_output(str(path))
    got = (
        df[["band_lo", "band_hi"]].drop_duplicates()
        .sort_values("band_lo").to_numpy()
    )
    np.testing.assert_array_equal(got, band_edges)


def test_a_non_integral_background_is_refused_not_rounded(tmp_path):
    """int64 columns must not silently truncate a float that is not a count."""
    band_edges = csh.make_band_edges(25, 40, 5)
    obs, bg = _distinct_tables(len(band_edges))
    bg["end_rev"][0, 17] += 0.5
    with pytest.raises(ValueError, match="non-integral"):
        csh.write_output(
            str(tmp_path / "bad.parquet"), obs, bg, band_edges, {"sample_name": "x"},
        )
