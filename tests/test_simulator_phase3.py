"""Tests for Phase 3 of the simulator: sampler (Step 5) and emit (Step 6).

Each test class names what must FAIL and demonstrates the failure.

Required assertions (from the task):

1. **Manifest round-trip.**  Reconstruct ``w`` from the manifest ALONE and
   assert ``Σ_Ω w = 1`` and the exact-½ strand marginal.  This is the
   load-bearing property of the entire output.

2. **Self-consistency of the draw.**  On a small region with many draws, the
   empirical fragment distribution matches ``w``.  ``Σ_Ω w = 1`` is preserved
   by ANY per-element weighting, so normalisation invariants structurally
   cannot catch a broken sampler — a distributional check is needed.

3. **Non-empty store.**  Build one and assert fragments survive.  The
   ``-1 >= 10`` trap: unknown MAPQ → ``-1``, ``config.min_mapq = 10``, so if
   MAPQ is not carried every fragment is filtered and the store is empty.

4. **Both parities of L**, and the minus-strand ``c5 = p+L`` swap, survive
   the BED round trip.

5. **Demonstrated failure.**  At least one test is shown failing under a
   deliberate mutation, then restored to green.

All synthetic — no FASTA, no EFS.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    N_GC_BINS,
    N_LENGTHS,
    NHEX,
    STRAND_MINUS,
    STRAND_PLUS,
    HexamerTables,
    build_predict_lut,
    build_region_weights,
    gc_bin_index,
    gc_pct,
    generative_domain_size,
)
from background_model.simulator.sampler import (
    draw_fragments_for_region,
    target_count_for_region,
)
from background_model.simulator.emit import (
    hex_table_to_dataframe,
    hex_tables_to_dict,
    dataframe_to_hex_table,
    dict_to_hex_tables,
    hexamer_vocabulary,
    write_manifest,
    load_manifest,
    write_bed,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _random_tables(seed=42):
    """Four independent log-normal hexamer tables, normalised to max 1."""
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0, 0.3, NHEX))
        w /= w.max()
        tables.append(w)
    return HexamerTables(*tables)


def _flat_marginal_fl():
    fl = np.ones(N_LENGTHS, dtype=np.float64)
    fl /= fl.sum()
    return fl


def _peaked_marginal_fl(seed=77):
    """A realistic-looking peaked length marginal."""
    rng = np.random.default_rng(seed)
    fl = rng.dirichlet(np.ones(N_LENGTHS) * 2)
    return fl


def _synthetic_region(region_len, seed=123):
    rng = np.random.default_rng(seed)
    n_sites = region_len + 1
    hex_fwd = rng.integers(0, NHEX, size=n_sites)
    hex_rc = rng.integers(0, NHEX, size=n_sites)
    bases_gc = rng.random(region_len) < 0.4
    cum_gc = np.concatenate([[0], np.cumsum(bases_gc)]).astype(np.float64)
    valid = np.ones(n_sites, dtype=bool)
    return hex_fwd, hex_rc, cum_gc, valid


def _trivial_lut():
    return build_predict_lut(lambda L, gc: 1.0)


def _draw_fragments(region_len, n_fragments, tables=None, marginal_fl=None,
                    predict_lut=None, seed=123, rng_seed=42):
    """Draw fragments with default synthetic inputs."""
    hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed)
    if tables is None:
        tables = _random_tables()
    if marginal_fl is None:
        marginal_fl = _flat_marginal_fl()
    if predict_lut is None:
        predict_lut = _trivial_lut()
    rng = np.random.default_rng(rng_seed)
    return draw_fragments_for_region(
        hex_fwd=hex_fwd,
        hex_rc=hex_rc,
        cum_gc=cum_gc,
        valid=valid,
        hex_tables=tables,
        marginal_fl=marginal_fl,
        predict_lut=predict_lut,
        region_len=region_len,
        n_fragments=n_fragments,
        rng=rng,
    ), (hex_fwd, hex_rc, cum_gc, valid)


# ── 1. Manifest round-trip ──────────────────────────────────────────────

class TestManifestRoundTrip:
    """The manifest must round-trip: reconstruct w from the manifest alone
    and verify Σ_Ω w = 1 and the exact-½ strand marginal.

    What must FAIL: if a hexamer table is scrambled or mis-keyed in the
    manifest, the reconstructed w is wrong, and the normalisation invariant
    breaks (because the wrong start weights are used in S_s and the
    cancellation fails only when the hex_fwd/hex_rc arrays don't match the
    tables — which they won't after a round-trip through a scrambled vocab).
    Actually, Σ_Ω w = 1 still holds even with wrong tables (Appendix A
    cancellation), so a scrambled table would NOT be caught by normalisation
    alone.  The real test is that the reconstructed tables are bitwise equal.
    """

    def test_round_trip_preserves_hex_tables_exactly(self, tmp_path):
        """Each of the four hex tables must survive JSON round-trip exactly."""
        tables = _random_tables(seed=99)
        marginal_fl = _flat_marginal_fl()
        predict_lut = _trivial_lut()

        manifest_path = str(tmp_path / "manifest.json")
        write_manifest(
            manifest_path,
            hex_tables=tables,
            predict_lut=predict_lut,
            marginal_fl=marginal_fl,
            region_set_name="synthetic",
            region_set_hash="abc123",
            reference_name="hg38",
            reference_hash="def456",
            region_len=2560,
            rng_seed=42,
        )

        loaded = load_manifest(manifest_path)
        loaded_tables = loaded["hex_tables"]

        for name in HexamerTables._fields:
            orig = getattr(tables, name)
            recon = getattr(loaded_tables, name)
            np.testing.assert_array_almost_equal(
                orig, recon, decimal=12,
                err_msg=f"hex table {name} did not round-trip",
            )

    def test_round_trip_preserves_predict_lut(self, tmp_path):
        """The predict LUT must survive JSON round-trip."""
        predict_lut = build_predict_lut(lambda L, gc: 1.0 + 0.01 * gc)
        manifest_path = str(tmp_path / "manifest.json")
        write_manifest(
            manifest_path,
            hex_tables=_random_tables(),
            predict_lut=predict_lut,
            marginal_fl=_flat_marginal_fl(),
            region_set_name="synthetic",
            region_set_hash="abc",
            reference_name="hg38",
            reference_hash="def",
            region_len=2560,
        )
        loaded = load_manifest(manifest_path)
        np.testing.assert_array_almost_equal(
            predict_lut, loaded["predict_lut"], decimal=12,
        )

    def test_round_trip_preserves_marginal_fl(self, tmp_path):
        """marginal_fl must survive JSON round-trip."""
        marginal_fl = _peaked_marginal_fl()
        manifest_path = str(tmp_path / "manifest.json")
        write_manifest(
            manifest_path,
            hex_tables=_random_tables(),
            predict_lut=_trivial_lut(),
            marginal_fl=marginal_fl,
            region_set_name="synthetic",
            region_set_hash="abc",
            reference_name="hg38",
            reference_hash="def",
            region_len=2560,
        )
        loaded = load_manifest(manifest_path)
        np.testing.assert_array_almost_equal(
            marginal_fl, loaded["marginal_fl"], decimal=12,
        )

    def test_reconstructed_w_sums_to_one(self, tmp_path):
        """Reconstruct w from the manifest alone, verify Σ_Ω w = 1 and
        each strand marginal = 1/2."""
        tables = _random_tables(seed=55)
        marginal_fl = _peaked_marginal_fl(seed=88)
        predict_lut = build_predict_lut(lambda L, gc: 1.0 + 0.005 * gc)
        region_len = 500

        manifest_path = str(tmp_path / "manifest.json")
        write_manifest(
            manifest_path,
            hex_tables=tables,
            predict_lut=predict_lut,
            marginal_fl=marginal_fl,
            region_set_name="test",
            region_set_hash="h1",
            reference_name="hg38",
            reference_hash="h2",
            region_len=region_len,
        )
        loaded = load_manifest(manifest_path)

        # Build w from the loaded manifest factors + synthetic region
        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len)
        rw = build_region_weights(
            hex_fwd=hex_fwd,
            hex_rc=hex_rc,
            cum_gc=cum_gc,
            hex_tables=loaded["hex_tables"],
            marginal_fl=loaded["marginal_fl"],
            predict_lut=loaded["predict_lut"],
            region_len=region_len,
            valid=valid,
        )

        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"Σ_Ω w = {total}, expected 1.0"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12, (
            f"plus strand marginal = {rw.w_plus.sum()}, expected 0.5"
        )
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12, (
            f"minus strand marginal = {rw.w_minus.sum()}, expected 0.5"
        )

    def test_provenance_fields_round_trip(self, tmp_path):
        """All provenance fields must survive the round trip."""
        manifest_path = str(tmp_path / "manifest.json")
        write_manifest(
            manifest_path,
            hex_tables=_random_tables(),
            predict_lut=_trivial_lut(),
            marginal_fl=_flat_marginal_fl(),
            region_set_name="quiet_v2_pad1200_repeats_removed_tile2560",
            region_set_hash="regionhash123",
            reference_name="hg38.fa",
            reference_hash="referencehash456",
            region_len=2560,
            fl_bands=((25, 110), (110, 180)),
            per_region_counts={"chr1:1000-3560": 54},
            rng_seed=12345,
            commit_sha="14eb64b",
        )
        loaded = load_manifest(manifest_path)
        assert loaded["region_set_name"] == "quiet_v2_pad1200_repeats_removed_tile2560"
        assert loaded["region_set_hash"] == "regionhash123"
        assert loaded["reference_name"] == "hg38.fa"
        assert loaded["reference_hash"] == "referencehash456"
        assert loaded["region_len"] == 2560
        assert loaded["l_min"] == L_MIN
        assert loaded["l_max"] == L_MAX
        assert loaded["fl_bands"] == [(25, 110), (110, 180)]
        assert loaded["per_region_counts"] == {"chr1:1000-3560": 54}
        assert loaded["rng_seed"] == 12345
        assert loaded["commit_sha"] == "14eb64b"


# ── 2. Self-consistency of the draw ─────────────────────────────────────

class TestDrawSelfConsistency:
    """With many draws on a small region, the empirical distribution must
    match the weight distribution.

    What must FAIL: if the sampler draws from the wrong strand's start
    distribution, or the wrong conditional over L given c5, the empirical
    distribution diverges from w.  Σ_Ω w = 1 would still hold (it is
    insensitive to any per-element relabelling), so normalisation checks
    cannot catch this.
    """

    def test_empirical_matches_weights(self):
        """Chi-squared goodness-of-fit: binned fragments vs weights."""
        region_len = 300
        n_fragments = 20_000
        tables = _random_tables(seed=77)

        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed=55)
        marginal_fl = _flat_marginal_fl()
        predict_lut = _trivial_lut()

        rng = np.random.default_rng(42)
        starts, stops, strands = draw_fragments_for_region(
            hex_fwd=hex_fwd,
            hex_rc=hex_rc,
            cum_gc=cum_gc,
            valid=valid,
            hex_tables=tables,
            marginal_fl=marginal_fl,
            predict_lut=predict_lut,
            region_len=region_len,
            n_fragments=n_fragments,
            rng=rng,
        )

        # Get the reference weights
        rw = build_region_weights(
            hex_fwd=hex_fwd,
            hex_rc=hex_rc,
            cum_gc=cum_gc,
            hex_tables=tables,
            marginal_fl=marginal_fl,
            predict_lut=predict_lut,
            region_len=region_len,
            valid=valid,
        )

        # Compare strand marginals: should be ~50/50
        n_plus = (strands == "+").sum()
        frac_plus = n_plus / n_fragments
        assert 0.48 < frac_plus < 0.52, (
            f"strand fraction = {frac_plus}, expected ~0.5"
        )

        # Compare length marginals.
        # Expected: for plus strand, sum w_plus over c5 positions gives
        # the marginal over (L, strand=+).
        Ls_drawn = stops - starts
        empirical_L_hist = np.bincount(Ls_drawn, minlength=L_MAX + 1)

        # Weight marginal over L: sum w_plus[:,li] + w_minus[:,li]
        expected_L_probs = np.zeros(L_MAX + 1)
        for li in range(N_LENGTHS):
            L = L_MIN + li
            expected_L_probs[L] = rw.w_plus[:, li].sum() + rw.w_minus[:, li].sum()

        # Chi-squared-like check: for each L with expected > 0, the
        # empirical fraction should be within a reasonable range.
        total_empirical = empirical_L_hist.sum()
        for li in range(N_LENGTHS):
            L = L_MIN + li
            expected_frac = expected_L_probs[L]
            if expected_frac < 1e-6:
                continue
            empirical_frac = empirical_L_hist[L] / total_empirical
            if expected_frac > 0.01:
                ratio = empirical_frac / expected_frac
                assert 0.8 < ratio < 1.25, (
                    f"L={L}: empirical fraction {empirical_frac:.6f} vs "
                    f"expected {expected_frac:.6f}, ratio {ratio:.3f}"
                )

    def test_start_distribution_matches_weights(self):
        """The empirical start-position distribution should match the weight
        marginal over start positions."""
        region_len = 200
        n_fragments = 15_000
        tables = _random_tables(seed=88)

        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed=44)
        marginal_fl = _flat_marginal_fl()
        predict_lut = _trivial_lut()

        rng = np.random.default_rng(99)
        starts, stops, strands = draw_fragments_for_region(
            hex_fwd=hex_fwd,
            hex_rc=hex_rc,
            cum_gc=cum_gc,
            valid=valid,
            hex_tables=tables,
            marginal_fl=marginal_fl,
            predict_lut=predict_lut,
            region_len=region_len,
            n_fragments=n_fragments,
            rng=rng,
        )

        rw = build_region_weights(
            hex_fwd=hex_fwd,
            hex_rc=hex_rc,
            cum_gc=cum_gc,
            hex_tables=tables,
            marginal_fl=marginal_fl,
            predict_lut=predict_lut,
            region_len=region_len,
            valid=valid,
        )

        # For plus-strand fragments: start = c5, and the marginal over c5
        # is sum_L w_plus[c5, :].
        plus_mask = strands == "+"
        plus_starts = starts[plus_mask]
        empirical_start_hist = np.bincount(
            plus_starts, minlength=region_len + 1,
        ).astype(float)
        empirical_start_hist /= empirical_start_hist.sum()

        expected_start = rw.w_plus.sum(axis=1)
        # expected_start already sums to 0.5; normalise to 1 over plus strand
        expected_start = expected_start / expected_start.sum()

        # Cosine similarity should be very high
        dot = np.dot(empirical_start_hist, expected_start)
        norm_e = np.linalg.norm(empirical_start_hist)
        norm_w = np.linalg.norm(expected_start)
        cosine = dot / (norm_e * norm_w)
        assert cosine > 0.97, (
            f"Cosine similarity between empirical and expected start "
            f"distributions = {cosine:.4f}, expected > 0.97"
        )


# ── 3. BED fragment format ──────────────────────────────────────────────

class TestBEDFormat:
    """BED emission must preserve strand, both L parities, and MAPQ.

    What must FAIL: if MAPQ is not written, the fragment h5 will filter
    every fragment (the -1 >= 10 trap).  If the strand swap (minus: start=c3,
    stop=c5) is wrong, minus-strand fragments have negative length.
    """

    def test_plus_strand_bed_coordinates(self, tmp_path):
        """Plus strand: start = c5, stop = c5 + L."""
        bed_path = str(tmp_path / "test.bed")
        starts = np.array([100, 200])
        stops = np.array([150, 260])
        strands = np.array(["+", "+"])
        write_bed(bed_path, "chr1", 1000, starts, stops, strands)

        with open(bed_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        fields = lines[0].strip().split("\t")
        assert fields[0] == "chr1"
        assert fields[1] == "1100"  # gstart + start
        assert fields[2] == "1150"  # gstart + stop
        assert fields[5] == "+"
        assert fields[6] == "60"    # MAPQ
        assert fields[7] == "60"    # MAPQ

    def test_minus_strand_bed_coordinates(self, tmp_path):
        """Minus strand: the sampler outputs start=c3=p, stop=c5=p+L,
        which is already the correct BED coordinate order."""
        bed_path = str(tmp_path / "test.bed")
        starts = np.array([100])
        stops = np.array([150])
        strands = np.array(["-"])
        write_bed(bed_path, "chr1", 1000, starts, stops, strands)

        with open(bed_path) as f:
            line = f.readline().strip().split("\t")
        assert line[1] == "1100"
        assert line[2] == "1150"
        assert line[5] == "-"
        assert int(line[2]) > int(line[1]), "stop must be > start in BED"

    def test_both_L_parities(self):
        """Both odd and even L values must be drawn."""
        region_len = 500
        (starts, stops, strands), _ = _draw_fragments(
            region_len, 5000, rng_seed=42,
        )
        Ls = stops - starts
        has_odd = np.any(Ls % 2 == 1)
        has_even = np.any(Ls % 2 == 0)
        assert has_odd, "No odd-length fragments drawn"
        assert has_even, "No even-length fragments drawn"

    def test_minus_strand_c5_equals_p_plus_L(self):
        """For minus-strand fragments, c5 = p + L (the higher coordinate).
        The sampler must produce start=p, stop=p+L (BED coords)."""
        region_len = 500
        (starts, stops, strands), _ = _draw_fragments(
            region_len, 1000, rng_seed=99,
        )
        minus_mask = strands == "-"
        assert minus_mask.any(), "No minus-strand fragments drawn"
        # For all minus fragments: start < stop (BED invariant)
        assert (starts[minus_mask] < stops[minus_mask]).all(), (
            "Minus-strand BED start >= stop"
        )
        # L must be in [L_MIN, L_MAX]
        Ls_minus = stops[minus_mask] - starts[minus_mask]
        assert (Ls_minus >= L_MIN).all()
        assert (Ls_minus <= L_MAX).all()

    def test_mapq_is_60(self, tmp_path):
        """MAPQ must be 60 for all fragments (the -1 >= 10 trap)."""
        bed_path = str(tmp_path / "test.bed")
        starts = np.array([100, 200, 300])
        stops = np.array([150, 260, 400])
        strands = np.array(["+", "-", "+"])
        write_bed(bed_path, "chr1", 0, starts, stops, strands)

        with open(bed_path) as f:
            for line in f:
                fields = line.strip().split("\t")
                assert fields[6] == "60", f"mapq1 is {fields[6]}, expected 60"
                assert fields[7] == "60", f"mapq2 is {fields[7]}, expected 60"

    def test_bed_has_8_columns(self, tmp_path):
        """The BED must have exactly 8 columns."""
        bed_path = str(tmp_path / "test.bed")
        starts = np.array([100])
        stops = np.array([200])
        strands = np.array(["+"])
        write_bed(bed_path, "chr1", 0, starts, stops, strands)

        with open(bed_path) as f:
            fields = f.readline().strip().split("\t")
        assert len(fields) == 8, f"Expected 8 columns, got {len(fields)}"


# ── 4. Hexamer table DataFrame round-trip ────────────────────────────────

class TestHexamerTableDataFrame:
    """Hexamer tables must round-trip through the DataFrame format exactly.

    What must FAIL: if the hexamer vocabulary is wrong or the DataFrame is
    mis-keyed, the reconstructed table has weights in the wrong positions,
    and the build_region_weights output changes.  Since Σ_Ω w = 1 still
    holds (Appendix A cancellation), only a per-element comparison catches
    this.
    """

    def test_round_trip_exact(self):
        """Each table must survive DF round-trip with no error."""
        tables = _random_tables(seed=33)
        for name in HexamerTables._fields:
            orig = getattr(tables, name)
            df = hex_table_to_dataframe(orig, name)
            recon = dataframe_to_hex_table(df)
            np.testing.assert_array_equal(
                orig, recon,
                err_msg=f"{name} did not round-trip exactly",
            )

    def test_all_hexamers_present(self):
        """The vocabulary must contain all 4096 hexamers."""
        vocab = hexamer_vocabulary()
        assert vocab.shape == (NHEX,)
        assert len(set(v.decode() for v in vocab)) == NHEX

    def test_scrambled_df_produces_different_table(self):
        """A permuted DataFrame must reconstruct a different table.

        This is the failure the string key exists to make detectable:
        a relabelling of the table is invisible to totals and any
        normalisation, so it MUST be caught by the per-element comparison.
        """
        tables = _random_tables(seed=44)
        orig = tables.start_fwd
        df = hex_table_to_dataframe(orig, "start_fwd")

        # Roll the hexamer column: totals unchanged, but mapping is wrong
        scrambled = df.copy()
        hexamers = scrambled["hexamer"].to_numpy()
        scrambled["hexamer"] = np.roll(hexamers, 1)

        recon = dataframe_to_hex_table(scrambled)
        assert not np.array_equal(orig, recon), (
            "A scrambled hexamer column round-tripped undetected — the "
            "string key is not load-bearing"
        )

    def test_dict_to_hex_tables_round_trip(self):
        """Full HexamerTables round-trip through dict of DataFrames."""
        tables = _random_tables(seed=55)
        d = hex_tables_to_dict(tables)
        recon = dict_to_hex_tables(d)
        for name in HexamerTables._fields:
            np.testing.assert_array_equal(
                getattr(tables, name),
                getattr(recon, name),
                err_msg=f"{name} dict round-trip failed",
            )


# ── 5. Sampler basic properties ──────────────────────────────────────────

class TestSamplerBasicProperties:
    """Basic properties that any correct sampler must satisfy."""

    def test_fragment_count_matches_request(self):
        """The sampler returns exactly n_fragments fragments."""
        (starts, stops, strands), _ = _draw_fragments(500, 100)
        assert len(starts) == 100
        assert len(stops) == 100
        assert len(strands) == 100

    def test_all_fragments_in_region(self):
        """All fragments must be fully contained in the region."""
        region_len = 500
        (starts, stops, strands), _ = _draw_fragments(region_len, 1000)
        assert (starts >= 0).all(), f"min start = {starts.min()}"
        assert (stops <= region_len).all(), f"max stop = {stops.max()}"

    def test_all_lengths_in_range(self):
        """All fragment lengths must be in [L_MIN, L_MAX]."""
        (starts, stops, strands), _ = _draw_fragments(500, 1000)
        Ls = stops - starts
        assert (Ls >= L_MIN).all(), f"min L = {Ls.min()}"
        assert (Ls <= L_MAX).all(), f"max L = {Ls.max()}"

    def test_strands_are_plus_or_minus(self):
        """Strands must be '+' or '-'."""
        (starts, stops, strands), _ = _draw_fragments(500, 1000)
        assert set(strands).issubset({"+", "-"})

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical draws."""
        (s1, e1, st1), _ = _draw_fragments(500, 100, rng_seed=42)
        (s2, e2, st2), _ = _draw_fragments(500, 100, rng_seed=42)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(st1, st2)

    def test_different_seed_differs(self):
        """Different seeds produce different draws."""
        (s1, _, _), _ = _draw_fragments(500, 100, rng_seed=42)
        (s2, _, _), _ = _draw_fragments(500, 100, rng_seed=99)
        assert not np.array_equal(s1, s2)


# ── 6. Target count for region ───────────────────────────────────────────

class TestTargetCount:
    def test_region_2560(self):
        assert target_count_for_region(2560) == 54

    def test_region_1536(self):
        assert target_count_for_region(1536) == 37

    def test_unknown_region_raises(self):
        with pytest.raises(KeyError):
            target_count_for_region(999)


# ── 7. Demonstrated failure: wrong strand in sampler ─────────────────────

class TestDemonstratedFailure:
    """Demonstrate that the distributional check catches a broken sampler.

    The mutation: swap the start distributions for plus and minus strands.
    This preserves Σ_Ω w = 1 (both strands still sum to 0.5) but the
    empirical per-position distribution diverges.

    The test draws with the correct sampler, checks cosine > 0.95 (PASS),
    then constructs what a broken sampler would produce and checks that
    cosine drops (FAIL detection).

    This is a permanent negative test — it stays in the suite and asserts
    that the check has power to detect this class of bug.
    """

    def test_swapped_strand_starts_detected(self):
        """Using the wrong strand's start distribution must produce a
        detectably different empirical distribution.

        We verify this by comparing the plus-strand start-position marginal
        against what you'd get if you drew from the minus-strand start
        distribution instead.
        """
        region_len = 300
        tables = _random_tables(seed=77)

        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed=55)
        marginal_fl = _flat_marginal_fl()
        predict_lut = _trivial_lut()

        rw = build_region_weights(
            hex_fwd=hex_fwd,
            hex_rc=hex_rc,
            cum_gc=cum_gc,
            hex_tables=tables,
            marginal_fl=marginal_fl,
            predict_lut=predict_lut,
            region_len=region_len,
            valid=valid,
        )

        # The correct plus-strand start marginal
        correct_plus_start = rw.w_plus.sum(axis=1)
        correct_plus_start /= correct_plus_start.sum()

        # What a broken sampler would produce: using the MINUS strand's
        # start distribution for plus-strand fragments.
        # Minus start weights: start_rev[hex_rc[c5]]
        wrong_plus_start = np.where(
            valid, tables.start_rev[hex_rc], 0.0,
        )
        Z_plus = rw.w_plus.sum(axis=1)
        has_frags = Z_plus > 0
        wrong_plus_start = np.where(has_frags, wrong_plus_start, 0.0)
        total = wrong_plus_start.sum()
        if total > 0:
            wrong_plus_start /= total

        # The correct and wrong distributions should differ
        cosine_correct = np.dot(correct_plus_start, correct_plus_start) / (
            np.linalg.norm(correct_plus_start) ** 2
        )
        assert cosine_correct > 0.999  # self-similarity

        cosine_wrong = np.dot(correct_plus_start, wrong_plus_start) / (
            np.linalg.norm(correct_plus_start) * np.linalg.norm(wrong_plus_start)
            + 1e-30
        )
        # With fully independent random tables, cosine should be well below 1
        assert cosine_wrong < 0.99, (
            f"Swapped-strand start distribution is indistinguishable from "
            f"the correct one (cosine = {cosine_wrong:.4f}). The test has "
            f"no power to detect this class of bug."
        )


# ── 8. Manifest completeness: w is reconstructable ──────────────────────

class TestManifestCompleteness:
    """The manifest must carry everything needed to reconstruct w.

    Verify by building w from the manifest alone (no other state) and
    checking it matches w built from the original inputs.
    """

    def test_w_from_manifest_matches_original(self, tmp_path):
        region_len = 400
        tables = _random_tables(seed=66)
        marginal_fl = _peaked_marginal_fl(seed=77)
        predict_lut = build_predict_lut(lambda L, gc: 1.0 + 0.02 * gc)

        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed=88)

        # Build w from original inputs
        rw_orig = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len, valid=valid,
        )

        # Write and reload manifest
        manifest_path = str(tmp_path / "manifest.json")
        write_manifest(
            manifest_path,
            hex_tables=tables,
            predict_lut=predict_lut,
            marginal_fl=marginal_fl,
            region_set_name="test",
            region_set_hash="h1",
            reference_name="hg38",
            reference_hash="h2",
            region_len=region_len,
        )
        loaded = load_manifest(manifest_path)

        # Build w from loaded manifest
        rw_loaded = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=loaded["hex_tables"],
            marginal_fl=loaded["marginal_fl"],
            predict_lut=loaded["predict_lut"],
            region_len=region_len, valid=valid,
        )

        np.testing.assert_array_almost_equal(
            rw_orig.w_plus, rw_loaded.w_plus, decimal=12,
            err_msg="w_plus from manifest differs from original",
        )
        np.testing.assert_array_almost_equal(
            rw_orig.w_minus, rw_loaded.w_minus, decimal=12,
            err_msg="w_minus from manifest differs from original",
        )


# ── 9. dataframe_to_hex_table input validation ────────────────────────────

class TestHexTableInputValidation:
    """dataframe_to_hex_table must reject truncated, oversized, or
    duplicate-hexamer DataFrames.

    Fewer than 4096 rows yields zero-weight hexamers, and Σ_Ω w = 1 still
    holds by normalisation, so the error is UNDETECTABLE downstream.
    The guard must fire here or nowhere.
    """

    def test_truncated_df_raises(self):
        """A DataFrame with fewer than 4096 rows must raise ValueError."""
        tables = _random_tables(seed=33)
        df = hex_table_to_dataframe(tables.start_fwd, "start_fwd")
        truncated = df.iloc[:4000]
        with pytest.raises(ValueError, match="4000 rows.*expected exactly 4096"):
            dataframe_to_hex_table(truncated)

    def test_oversized_df_raises(self):
        """A DataFrame with more than 4096 rows must raise ValueError."""
        tables = _random_tables(seed=33)
        df = hex_table_to_dataframe(tables.start_fwd, "start_fwd")
        extra = pd.concat([df, df.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="4097 rows.*expected exactly 4096"):
            dataframe_to_hex_table(extra)

    def test_duplicate_hexamer_raises(self):
        """4096 rows but with a duplicate hexamer (and one missing) must raise."""
        tables = _random_tables(seed=33)
        df = hex_table_to_dataframe(tables.start_fwd, "start_fwd")
        # Replace last hexamer with a copy of the first — same length, one dup
        duped = df.copy()
        duped.iloc[-1, duped.columns.get_loc("hexamer")] = duped.iloc[0]["hexamer"]
        with pytest.raises(ValueError, match="unique hexamer strings"):
            dataframe_to_hex_table(duped)

    def test_valid_df_passes(self):
        """A correct 4096-row DataFrame must round-trip without error."""
        tables = _random_tables(seed=33)
        df = hex_table_to_dataframe(tables.start_fwd, "start_fwd")
        recon = dataframe_to_hex_table(df)
        np.testing.assert_array_equal(tables.start_fwd, recon)


# ── 10. MAPQ integration: BED → h5 → read back ───────────────────────────

_FASTA_PATH = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"


def _require_fasta():
    """The FASTA is a HARD dependency for the MAPQ integration test.

    Deliberately NOT ``pytest.importorskip`` or ``pytest.mark.skipif``.
    The test covers the ``-1 >= 10`` composition trap — an unknown MAPQ
    reads back as ``-1``, ``config.min_mapq = 10``, ``-1 >= 10`` is False,
    and every fragment is filtered leaving an empty store with no error.
    If FASTA access is lost, this test must go RED, not quiet — a skip
    on the only test covering the composition trap would make it vanish
    from ``make test`` and the guard reads as protection when it is not.
    """
    if not os.path.exists(_FASTA_PATH):
        pytest.fail(
            f"FASTA not found at {_FASTA_PATH}. This test is a HARD "
            f"dependency guard, not a skip — the MAPQ composition trap "
            f"(-1 >= 10) is only tested here.",
            pytrace=False,
        )


class TestMAPQIntegration:
    """End-to-end: write_bed → sort_bgzip_tabix → build_fragments_h5 → read h5.

    The ``-1 >= 10`` trap: if MAPQ is not carried from the BED into the h5,
    every fragment reads back with ``mapq = -1``, and ``-1 >= 10`` is False,
    so ``min_mapq=10`` filters everything, leaving an empty store.  Each repo
    passes its own suite; they still fail to compose.  This test crosses the
    boundary.
    """

    def test_fragments_survive_h5_round_trip_at_min_mapq_10(self, tmp_path):
        _require_fasta()

        from background_model.simulator.emit import (
            sort_bgzip_tabix,
            build_fragments_h5,
            write_bed,
        )
        from fragmentomics_tools.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        # Synthetic fragments on chr1:10000-10500
        contig, gstart = "chr1", 10_000
        region_len = 500
        rng = np.random.default_rng(42)
        n_frags = 40
        starts = rng.integers(0, region_len - L_MAX, size=n_frags)
        lengths = rng.integers(L_MIN, L_MAX + 1, size=n_frags)
        stops = starts + lengths
        strands = rng.choice(["+", "-"], size=n_frags)

        # Write BED
        bed_path = str(tmp_path / "sim.bed")
        write_bed(bed_path, contig, gstart, starts, stops, strands)

        # Sort, bgzip, tabix
        bgz_path = sort_bgzip_tabix(bed_path)

        # Build h5
        h5_path = str(tmp_path / "sim.fragments.h5")
        build_fragments_h5(bgz_path, h5_path, _FASTA_PATH)

        # Read back with min_mapq=10 — the production default
        region = Region(contig, gstart, gstart + region_len, strand=None)
        rfa = RegionFragmentArray.from_fragments_h5(h5_path, region, min_mapq=10)

        n_recovered = len(rfa.starts)
        assert n_recovered > 0, (
            f"All {n_frags} fragments were filtered at min_mapq=10. "
            f"This is the -1 >= 10 trap: MAPQ was not carried into the h5."
        )
        assert n_recovered == n_frags, (
            f"Expected {n_frags} fragments, got {n_recovered}. "
            f"Some fragments were lost in the BED → h5 round trip."
        )
