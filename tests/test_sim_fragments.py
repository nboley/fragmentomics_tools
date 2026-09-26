"""Fast unit tests for the cfDNA fragment simulator (scripts/sim_fragments.py).

Covers hexamer extraction incl. the reverse-complement far end, GC computation,
2-D GC x length bias interpolation (hold-at-edges + the short/long sign flip),
acceptance composition, determinism under a seed, that a strong synthetic w6
measurably skews endpoint composition, and that the emitted (start, stop,
strand) representation round-trips through RegionFragmentArray / build_coverage_counts
(the store + apply_fragment_weights entry points).

All synthetic and tiny -- no FASTA, no h5, no torch.
"""

import gzip
import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_PATH = os.path.join(_HERE, "..", "scripts", "sim_fragments.py")
_spec = importlib.util.spec_from_file_location("sim_fragments", _SIM_PATH)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


# ── the real fitted grid (from gc_bias_grid.json) so tests don't touch EFS ──
_LENGTHS = [24, 32, 42, 52, 75]
_GC = [30, 40, 50, 60, 70]
_GRID = [
    [1.2576767369459334, 0.848070753226077, 1.2833273004283872, 0.7888614955201368, 0.8220637138794653],
    [0.6313120489778389, 1.1575445451438464, 0.9490041298298351, 1.2024262127524794, 1.059713063296],
    [0.7614817543684849, 0.9943358499489585, 1.1255820178984428, 1.1881908818582059, 0.9304094959259078],
    [0.7505752099802561, 0.7579033702391661, 0.7852378679674301, 1.3199742300983923, 1.3863093217147548],
    [0.4585500697483137, 1.110766850395932, 0.7224134136131811, 0.8375684795249504, 1.8707011867176224],
]
# published GC slopes (70/30) per length -- the interaction the plan requires
_TRUE_SLOPES = [0.6536367332958034, 1.6785883700648319, 1.2218408262421452,
                1.8469958816668377, 4.079600702589364]


def _gcbias():
    return sim.GCBias2D(_LENGTHS, _GC, _GRID)


# ── hexamer extraction + reverse complement ───────────────────────────────

def test_hexamer_index_forward():
    seq = np.frombuffer(b"AAAAAC", dtype=np.uint8)
    fwd, rc, valid = sim.hexamer_indices(seq)
    assert fwd.tolist() == [1]           # AAAAAC = 0..0,C(1) big-endian
    assert valid.all()


def test_hexamer_reverse_complement_far_end():
    # RC(AAAAAC) = GTTTTT ; the FAR end of a fragment uses this index.
    seq = np.frombuffer(b"AAAAAC", dtype=np.uint8)
    fwd, rc, _ = sim.hexamer_indices(seq)
    # GTTTTT = 2*4^5 + 3*(4^4+4^3+4^2+4^1+4^0) = 3071
    assert rc.tolist() == [3071]
    # rc must equal the RC permutation applied to the forward index
    assert rc[0] == sim.RC_PERM[fwd[0]]


def test_rc_perm_is_involution():
    assert np.array_equal(sim.RC_PERM[sim.RC_PERM], np.arange(sim.NHEX))


def test_hexamer_N_marked_invalid():
    seq = np.frombuffer(b"AANAAA", dtype=np.uint8)
    fwd, rc, valid = sim.hexamer_indices(seq)
    assert not valid[0]


def test_sliding_window_count():
    seq = np.frombuffer(b"ACGTACGT", dtype=np.uint8)  # 8 bases -> 3 hexamers
    fwd, rc, valid = sim.hexamer_indices(seq)
    assert len(fwd) == 3 and len(rc) == 3 and len(valid) == 3


# ── GC computation (via the cumulative-GC convention used in precompute) ───

def test_gc_cumsum_fraction():
    core = np.frombuffer(b"GCGCATATGC", dtype=np.uint8)  # 10 bp, 6 are G/C
    is_gc = (core == ord("G")) | (core == ord("C"))
    cum = np.concatenate([[0], np.cumsum(is_gc)])
    # whole fragment [0,10): 6 GC of 10 = 60%
    assert 100.0 * (cum[10] - cum[0]) / 10 == 60.0
    # sub-fragment [0,4) = GCGC = 100%
    assert 100.0 * (cum[4] - cum[0]) / 4 == 100.0


# ── 2-D bias interpolation: grid points, hold-at-edges, sign flip ─────────

def test_bias_grid_points_exact():
    gb = _gcbias()
    for i, L in enumerate(_LENGTHS):
        for j, G in enumerate(_GC):
            assert np.isclose(gb(L, G), _GRID[i][j])


def test_bias_hold_at_edges():
    gb = _gcbias()
    # length below/above the grid holds the edge row
    assert np.isclose(gb(10, 50), gb(24, 50))
    assert np.isclose(gb(167, 50), gb(75, 50))
    # GC below/above holds the edge column
    assert np.isclose(gb(42, 10), gb(42, 30))
    assert np.isclose(gb(42, 95), gb(42, 70))
    # corner: both out of range -> grid corner
    assert np.isclose(gb(5, 5), _GRID[0][0])
    assert np.isclose(gb(300, 100), _GRID[-1][-1])


def test_bias_sign_flip_short_vs_long():
    gb = _gcbias()
    slope_short = float(gb(24, 70) / gb(24, 30))
    slope_long = float(gb(75, 70) / gb(75, 30))
    # short fragments GC-disfavouring (<1), long strongly GC-favouring (>1)
    assert slope_short < 1.0 < slope_long
    assert np.isclose(slope_short, _TRUE_SLOPES[0])
    assert np.isclose(slope_long, _TRUE_SLOPES[-1])


def test_bias_bilinear_midpoint():
    gb = _gcbias()
    # midpoint in GC between grid cols 30 and 40 at grid length 24
    expected = 0.5 * (_GRID[0][0] + _GRID[0][1])
    assert np.isclose(gb(24, 35), expected)


def test_bias_max_is_grid_max():
    gb = _gcbias()
    assert np.isclose(gb.max_bias, np.max(_GRID))


# ── w6 construction ───────────────────────────────────────────────────────

def test_w6_normalised_and_deterministic():
    a = sim.build_w6(seed=7, dynamic_range=4.0)
    b = sim.build_w6(seed=7, dynamic_range=4.0)
    assert a.shape == (sim.NHEX,)
    assert np.isclose(a.max(), 1.0)
    assert np.array_equal(a, b)
    c = sim.build_w6(seed=8, dynamic_range=4.0)
    assert not np.array_equal(a, c)


def test_w6_dynamic_range_monotone():
    lo = sim.build_w6(seed=1, dynamic_range=2.0)
    hi = sim.build_w6(seed=1, dynamic_range=8.0)
    # wider dynamic range -> larger p95/p5 spread
    def spread(w):
        return np.percentile(w, 95) / np.percentile(w, 5)
    assert spread(hi) > spread(lo)


# ── acceptance composition + a strong w6 skews endpoints ──────────────────

class _FlatBias:
    """A GC bias that is identically 1 (isolates the w6 contribution)."""
    max_bias = 1.0

    def __call__(self, length, gc):
        return np.ones_like(np.asarray(length, dtype=float))


def _synthetic_region(region_len, fav_pos):
    """Region arrays where every position has a distinct hexamer index == position."""
    n = region_len + 1
    fwd = np.arange(n, dtype=np.int64)
    rc = np.zeros(n, dtype=np.int64)          # far end always hexamer index 0
    valid = np.ones(n, dtype=bool)
    cum_gc = (np.arange(n) // 2).astype(np.int64)   # ~50% GC everywhere
    return {"fwd_cut": fwd, "rc_cut": rc, "valid": valid, "cum_gc": cum_gc}


def test_strong_w6_skews_endpoint_composition():
    region_len = 60
    pre = _synthetic_region(region_len, fav_pos=5)
    w6 = np.full(sim.NHEX, 0.01)
    w6[0] = 1.0            # far-end hexamer (rc_cut==0) so fragments can accept
    w6[5] = 1.0            # strongly favoured LEFT-cut hexamer at position 5
    rng = np.random.default_rng(0)
    ridx, start, stop, strand = sim.simulate_sample(
        [pre], [4000], w6, _FlatBias(), np.array([20, 30]),
        np.array([0.5, 0.5]), region_len, rng
    )
    # mean w6 at the accepted left-cut hexamers >> mean w6 over all positions
    mean_acc = w6[pre["fwd_cut"][start]].mean()
    mean_all = w6[pre["fwd_cut"][:region_len]].mean()
    assert mean_acc > 5 * mean_all
    # position 5 (the favoured hexamer) is over-represented among starts
    frac5 = np.mean(start == 5)
    assert frac5 > 0.1


def test_far_end_uses_rc_weight():
    # Make ONLY the far-end (rc) weight discriminate: left weight constant.
    region_len = 40
    n = region_len + 1
    pre = {
        "fwd_cut": np.zeros(n, dtype=np.int64),      # left hexamer always index 0
        "rc_cut": np.arange(n, dtype=np.int64),      # far hexamer index == position
        "valid": np.ones(n, dtype=bool),
        "cum_gc": (np.arange(n) // 2).astype(np.int64),
    }
    w6 = np.full(sim.NHEX, 0.01)
    w6[0] = 1.0
    w6[20] = 1.0          # favoured FAR-end hexamer at stop position 20
    rng = np.random.default_rng(1)
    _, start, stop, _ = sim.simulate_sample(
        [pre], [4000], w6, _FlatBias(), np.array([10, 15]),
        np.array([0.5, 0.5]), region_len, rng
    )
    mean_acc = w6[pre["rc_cut"][stop]].mean()
    mean_all = w6[pre["rc_cut"][1:region_len + 1]].mean()
    assert mean_acc > 5 * mean_all


def test_simulate_sample_deterministic():
    region_len = 50
    pre = _synthetic_region(region_len, fav_pos=3)
    w6 = sim.build_w6(seed=3, dynamic_range=4.0)
    out1 = sim.simulate_sample([pre], [500], w6, _FlatBias(),
                               np.array([10, 20, 30]), np.array([0.2, 0.5, 0.3]),
                               region_len, np.random.default_rng(42))
    out2 = sim.simulate_sample([pre], [500], w6, _FlatBias(),
                               np.array([10, 20, 30]), np.array([0.2, 0.5, 0.3]),
                               region_len, np.random.default_rng(42))
    for a, b in zip(out1, out2):
        assert np.array_equal(a, b)


def test_fragments_fit_region_and_valid_coords():
    region_len = 50
    pre = _synthetic_region(region_len, fav_pos=3)
    w6 = sim.build_w6(seed=5, dynamic_range=4.0)
    _, start, stop, strand = sim.simulate_sample(
        [pre], [300], w6, _FlatBias(), np.array([10, 20]),
        np.array([0.5, 0.5]), region_len, np.random.default_rng(0)
    )
    assert (start >= 0).all()
    assert (stop <= region_len).all()
    assert (start < stop).all()
    assert set(np.unique(strand)).issubset({"+", "-"})


# ── representation round-trip: npz coords -> RegionFragmentArray ───────────

def test_representation_builds_rfa_and_coverage_counts():
    """The emitted (start, stop, strand) suffices to build store counts and to
    construct an RFA that meets apply_fragment_weights preconditions."""
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
    from fragmentomics_tools.region import Region
    from background_model.preprocess import FL_BANDS, TRACK_INDEX

    region_len = 200
    n = region_len + 1
    pre = {
        "fwd_cut": np.arange(n, dtype=np.int64),
        "rc_cut": np.zeros(n, dtype=np.int64),
        "valid": np.ones(n, dtype=bool),
        "cum_gc": (np.arange(n) // 2).astype(np.int64),
    }
    w6 = sim.build_w6(seed=2, dynamic_range=4.0)
    w6[0] = 1.0
    # lengths spanning both fl bands so counts land in tracks
    _, start, stop, strand = sim.simulate_sample(
        [pre], [500], w6, _FlatBias(),
        np.array([50, 130]), np.array([0.5, 0.5]), region_len,
        np.random.default_rng(0)
    )
    region = Region(chrom="chr1", start=1_000, stop=1_000 + region_len, strand=".")
    rfa = RegionFragmentArray(
        starts_0=start.astype(np.int64),
        stops_0=stop.astype(np.int64),
        region=region,
        max_frag_len=511,
        fragment_strands=strand,
    )
    # (a) store counts path (mirrors background_model/preprocess._worker_inner)
    sparse = rfa.build_coverage_counts(
        fl_bands=list(FL_BANDS), split_strand=True, return_sparse=True
    )
    assert len(sparse) == len(TRACK_INDEX) == 12
    # (b) apply_fragment_weights preconditions: strandless, not flipped
    assert rfa.region.strand in (None, ".", "+")
    assert rfa.is_flipped is False


# ── Change 1: centred carve offset derived from the tile width ─────────────

def test_center_offset_default_byte_identical():
    # Default path (training_tiles.bed, all 16,384 bp) must reproduce the old
    # hardcoded (16_384 - region_len) // 2 exactly, for every plausible width.
    for region_len in (1, 2, 2047, 2048, 2049, 2304, 8192, 16383, 16384):
        assert sim._center_offset(16_384, region_len) == (16_384 - region_len) // 2


def test_center_offset_narrow_tiles_are_centred():
    # The two runs ahead: 2,304 bp carve of a 2,560 tile, 1,280 bp carve of 1,536.
    assert sim._center_offset(2560, 2304) == 128
    assert sim._center_offset(1536, 1280) == 128
    # exact-fit tile has zero offset
    assert sim._center_offset(2304, 2304) == 0


def test_center_offset_fails_when_tile_narrower_than_region():
    with pytest.raises(ValueError):
        sim._center_offset(1536, 2048)
    with pytest.raises(ValueError):
        sim._center_offset(2303, 2304)


# ── Change 3: count rescaling from tile width to the central carve ─────────

def test_scale_counts_to_region_factor():
    counts = np.array([100, 200, 1000], dtype=np.int64)
    # 2,304 carve of a 2,560 tile -> x0.900
    scaled, scale = sim._scale_counts_to_region(counts, 2560, 2304)
    assert np.isclose(scale, 0.9)
    assert scaled.tolist() == [90, 180, 900]
    # 1,280 carve of a 1,536 tile -> x0.8333
    scaled2, scale2 = sim._scale_counts_to_region(counts, 1536, 1280)
    assert np.isclose(scale2, 1280 / 1536)
    # 100*0.8333=83.33->83, 200->167 (166.67 rounds to 167), 1000->833
    assert scaled2.tolist() == [83, 167, 833]


def test_scale_counts_drops_nonpositive_after_rounding():
    # a count that rounds to 0 (0*scale, and 1*0.4 -> 0) is dropped, like the
    # --real-store pool drops empty tiles
    counts = np.array([0, 1, 3], dtype=np.int64)
    scaled, _ = sim._scale_counts_to_region(counts, 5, 2)  # scale 0.4
    # 0->0 dropped, 1*0.4=0.4->0 dropped, 3*0.4=1.2->1 kept
    assert scaled.tolist() == [1]


def _write_count_tsv(path, rows):
    """rows: iterable of (sample, region_set, contig, start, stop, count)."""
    with gzip.open(path, "wt") as fh:
        fh.write("# mapq_filter: >30, dedup: by (start,stop)\n")
        fh.write("# NOT comparable to bg-model store (min_mapq=10)\n")
        fh.write("sample\tregion_set\tcontig\tstart\tstop\tcount\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def test_per_region_count_pool_from_tsv(tmp_path):
    tsv = tmp_path / "counts.tsv.gz"
    _write_count_tsv(tsv, [
        ("S", "removed_tile2560", "chr1", 0, 2560, 100),
        ("S", "removed_tile2560", "chr1", 3000, 5560, 200),
        # a different region set, must be excluded
        ("S", "removed_tile1536", "chr2", 0, 1536, 999),
    ])
    pool, tile_width, scale = sim.per_region_count_pool_from_tsv(
        str(tsv), "removed_tile2560", 2304)
    assert tile_width == 2560
    assert np.isclose(scale, 0.9)
    assert sorted(pool.tolist()) == [90, 180]  # 999-row excluded


def test_per_region_count_pool_missing_region_set_raises(tmp_path):
    tsv = tmp_path / "counts.tsv.gz"
    _write_count_tsv(tsv, [("S", "removed_tile2560", "chr1", 0, 2560, 100)])
    with pytest.raises(ValueError):
        sim.per_region_count_pool_from_tsv(str(tsv), "does_not_exist", 2304)


def test_per_region_count_pool_nonuniform_width_raises(tmp_path):
    tsv = tmp_path / "counts.tsv.gz"
    _write_count_tsv(tsv, [
        ("S", "mixed", "chr1", 0, 2560, 100),
        ("S", "mixed", "chr1", 0, 2000, 100),  # different width
    ])
    with pytest.raises(ValueError):
        sim.per_region_count_pool_from_tsv(str(tsv), "mixed", 1280)


# ── Change 3: --real-store and --real-count-tsv are mutually exclusive ─────

def test_real_store_and_real_count_tsv_mutually_exclusive():
    # the guard is the first thing run() checks, before any FASTA/EFS access
    with pytest.raises(ValueError, match="mutually exclusive"):
        sim.run("A", 1, 0, 4.0, 1, "/tmp/none", "/tmp/none.h5",
                real_store="/tmp/store.zarr", real_count_tsv="/tmp/c.tsv.gz")
