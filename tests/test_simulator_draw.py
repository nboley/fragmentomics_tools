"""Tests for ``background_model/simulator/draw.py``: the sampler, the
BED/sidecar writer and the seeding scheme, plus the end-to-end chain that ends
in the draw.

Split out of ``tests/test_cut_site_simulator.py`` by owner decision 188; the
test bodies are unchanged.  Classes keep their T-numbers; the measure half of a
split T-class is in ``tests/test_simulator_measure.py``.  Shared fixtures are
in ``tests/conftest.py`` and shared helpers (``_RecordingRng``,
``_draw_frame``, ...) in ``tests/cut_site_helpers.py``.

Every oracle imports NOTHING from the module under test.  The AST check
``test_oracle_is_independent`` enforces this.

Mutations each test must catch are documented in-line as comments.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pysam
import pytest

import cut_site_oracle as oracle
from cut_site_helpers import (
    _RecordingRng,
    _build_h5,
    _draw_frame,
    _read_outputs,
)

from background_model.constants import HEX_HALF, L_MAX, L_MIN, N_LENGTHS, NHEX
from background_model.hexamers import hexamer_indices
from background_model.simulator.measure import (
    FragmentLengthDist,
    TABLE_NAMES,
    count_sample,
    filter_fragments,
    load_sample_dataframe,
    propensities,
    uniform_hexamer_counts,
)
from background_model.simulator.draw import (
    sample_region,
    simulate_fragments_to_bed,
)
from fragmentomics_tools.dataframe import RegionDataFrame


# ── T5: Sampler ─────────────────────────────────────────────────────────────

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


# ── T6: Writer and round trip ───────────────────────────────────────────────

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


# ── T6b: Functional chain over multiple regions ─────────────────────────────

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


# ── T8: Seeding and parallel determinism (owner decision 166) ───────────────

class TestT8SeedingAndParallelDeterminism:
    """Owner decision 166: per-region streams ``default_rng([seed, i])``
    and a parallel draw, byte-identical across worker counts. The
    N(h) half is ``TestT8MeasureDeterminism`` in
    ``tests/test_simulator_measure.py``.

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
