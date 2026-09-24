"""The fragment-weight callback interface.

The protocol is ``weight_fn(fa) -> ndarray`` of one weight per fragment. The
weight is applied uniformly to every coverage getter -- there are no longer
separate endpoint weight vectors.

These tests pin the three properties that are easy to regress silently:
the length check on assignment, that ``set_fragment_array_weights`` mutates
in place rather than returning a copy, and that the old v1 signature fails
with a message naming its replacement.
"""

import os
import sys

import numpy
import pandas as pd
import pytest

from fragmentomics_tools.fragment_array.fragment_array import FragmentArray
from fragmentomics_tools.fragment_array.weights import UniformWeights, GCFlWeights
from fragmentomics_tools.dataframe import SampleAndRegionDataFrame

BIOMARKER_SRC = "/home/nathanboley/src/biomarker"


def _fitted_gcfl_model(fit=True):
    """A REAL GCFlDistModel, optionally fitted on a small synthetic cell map.

    Skips rather than silently passing when the biomarker source is absent, so
    this never degrades into a test that proves nothing.
    """
    if not os.path.isdir(BIOMARKER_SRC):
        pytest.skip("biomarker source not available")
    if BIOMARKER_SRC not in sys.path:
        sys.path.insert(0, BIOMARKER_SRC)
    pytest.importorskip("flgc.model")
    from flgc.model import GCFlDistModel

    model = GCFlDistModel()
    if not fit:
        return model

    # (length, gc%) -> (duplicate counts k, observations at each k, kmax, seen)
    cell_map = {
        (35, 35.0): (numpy.array([1, 2, 3]), numpy.array([60, 30, 10]), 3, 100),
        (45, 45.0): (numpy.array([1, 2]), numpy.array([70, 30]), 2, 100),
    }
    model.fit(
        cell_map,
        length_bins=[(20, 40), (41, 60)],
        gc_bins=[(30, 40), (41, 50)],
        min_cell_size=10,
    )
    return model


def _fragment_array(n=4, gc=None):
    return FragmentArray(
        starts_0=[10, 20, 30, 40][:n],
        stops_0=[60, 90, 130, 180][:n],
        length=1000,
        max_frag_len=511,
        fragment_strands=numpy.array(list("+-+-")[:n], dtype="<U1"),
        gc=gc,
    )


class TestUniformWeights:
    def test_returns_ones_of_correct_length(self):
        fa = _fragment_array()
        w = UniformWeights()(fa)
        assert len(w) == fa.n_fragments
        assert (w == 1.0).all()

    def test_empty_fragment_array(self):
        fa = FragmentArray(starts_0=[], stops_0=[], length=100, max_frag_len=511)
        assert len(UniformWeights()(fa)) == 0


class TestAssignWeights:
    def test_assigns(self):
        fa = _fragment_array()
        fa.assign_weights(numpy.array([1.0, 2.0, 3.0, 4.0]))
        assert (fa.weights == numpy.array([1.0, 2.0, 3.0, 4.0])).all()

    def test_too_short_raises(self):
        """The length check is the whole reason this method exists -- a missing
        one is how the sibling `gc` misalignment survived undetected."""
        fa = _fragment_array()
        with pytest.raises(ValueError, match="must match fragment count"):
            fa.assign_weights(numpy.array([1.0, 2.0]))

    def test_too_long_raises(self):
        fa = _fragment_array()
        with pytest.raises(ValueError, match="must match fragment count"):
            fa.assign_weights(numpy.ones(99))


class TestSingleWeightDrivesEveryGetter:
    """One weight per fragment now feeds first/last/midpoint alike."""

    def test_all_three_getters_reflect_the_single_vector(self):
        fa = _fragment_array()
        fa.assign_weights(numpy.array([2.0, 2.0, 2.0, 2.0]))

        # each getter sums weights at its own position array, so with a uniform
        # weight of 2.0 every total is 2x the fragment count landing in-region
        first = fa.get_first_covered_base_array(return_sparse=False).sum()
        last = fa.get_last_covered_base_array(return_sparse=False).sum()
        mid = fa.get_midpoint_coverage_array(return_sparse=False).sum()

        for name, total in (("first", first), ("last", last), ("midpoint", mid)):
            assert total == pytest.approx(2.0 * fa.n_fragments), (
                f"{name} getter did not use the single weights vector"
            )

    def test_endpoint_weight_vectors_are_gone(self):
        fa = _fragment_array()
        assert not hasattr(fa, "first_covered_base_weights")
        assert not hasattr(fa, "last_covered_base_weights")


class TestGCFlWeights:
    def test_raises_without_gc(self):
        fa = _fragment_array(gc=None)

        class _Normalizer:
            def predict(self, length, gc):
                raise AssertionError("must not be reached")

        with pytest.raises(ValueError, match="no GC data"):
            GCFlWeights(_Normalizer())(fa)

    def test_converts_fraction_to_percent(self):
        """fa.gc is a FRACTION; the normalizer contract is PERCENT."""
        seen = {}

        class _Normalizer:
            def predict(self, length, gc):
                seen["gc"] = numpy.asarray(gc)
                return numpy.ones(len(numpy.atleast_1d(length)))

        fa = _fragment_array(gc=numpy.array([0.1, 0.25, 0.5, 0.75]))
        GCFlWeights(_Normalizer())(fa)
        assert seen["gc"] == pytest.approx([10.0, 25.0, 50.0, 75.0])

    def test_unfitted_real_model_refuses(self):
        """An unfitted model must raise rather than return silent garbage."""
        model = _fitted_gcfl_model(fit=False)
        fa = _fragment_array(gc=numpy.array([0.3, 0.4, 0.5, 0.6]))
        with pytest.raises(RuntimeError, match="not fitted"):
            GCFlWeights(model)(fa)

    def test_real_fitted_model_end_to_end(self):
        """A genuinely fitted GCFlDistModel, real weights, nothing stubbed.

        Cross-checked against the model's own predict() so the fraction ->
        percent conversion and the array plumbing are both verified, rather
        than merely asserting the result looks plausible.
        """
        model = _fitted_gcfl_model()

        # lengths 35 and 45 land in the two fitted length bins;
        # gc fractions 0.35/0.45 are 35%/45%, inside the two gc bins
        fa = FragmentArray(
            starts_0=[10, 100],
            stops_0=[45, 145],
            length=1000,
            max_frag_len=511,
            gc=numpy.array([0.35, 0.45]),
        )
        assert list(fa.fragment_lengths) == [35, 45]

        got = GCFlWeights(model)(fa)

        expected = numpy.array([model.predict(35, 35.0), model.predict(45, 45.0)])
        assert got == pytest.approx(expected)
        # real correction weights, not a degenerate all-ones vector
        assert ((got >= 1.0) & (got <= model.max_weight)).all()

    def test_real_model_through_the_srdf_entry_point(self):
        """The whole path: SRDF -> callback -> fitted model -> weights in place."""
        model = _fitted_gcfl_model()

        def _fa():
            return FragmentArray(
                starts_0=[10, 100],
                stops_0=[45, 145],
                length=1000,
                max_frag_len=511,
                gc=numpy.array([0.35, 0.45]),
            )

        srdf = SampleAndRegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1"],
                    "start": [0, 1000],
                    "stop": [1000, 2000],
                    "sample_id": ["s1", "s1"],
                    "frag_h5": ["/nonexistent.h5", "/nonexistent.h5"],
                    "fragment_array": [_fa(), _fa()],
                }
            ),
            ref="hg38",
        )

        srdf.set_fragment_array_weights(GCFlWeights(model), n_workers=1, verbose=False)

        expected = numpy.array([model.predict(35, 35.0), model.predict(45, 45.0)])
        for fa in srdf["fragment_array"]:
            assert fa.weights == pytest.approx(expected)


class TestSetFragmentArrayWeights:
    def _srdf(self):
        return SampleAndRegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1"],
                    "start": [0, 1000],
                    "stop": [1000, 2000],
                    "sample_id": ["s1", "s1"],
                    "frag_h5": ["/nonexistent.h5", "/nonexistent.h5"],
                    "fragment_array": [_fragment_array(), _fragment_array()],
                }
            ),
            ref="hg38",
        )

    def test_mutates_in_place_and_returns_self(self):
        """Must NOT build a new frame -- these can be large enough to OOM."""
        srdf = self._srdf()
        before = [id(fa) for fa in srdf["fragment_array"]]

        out = srdf.set_fragment_array_weights(UniformWeights(), n_workers=1)

        assert out is srdf, "must return the same object, not a copy"
        assert [id(fa) for fa in srdf["fragment_array"]] == before, (
            "fragment arrays were replaced rather than mutated"
        )

    def test_weights_actually_stick(self):
        srdf = self._srdf()

        class _Two:
            def __call__(self, fa):
                return numpy.full(fa.n_fragments, 2.0)

        srdf.set_fragment_array_weights(_Two(), n_workers=1)
        for fa in srdf["fragment_array"]:
            assert (fa.weights == 2.0).all()

    def test_weights_stick_across_forked_workers(self):
        """The parent must do the assigning.

        A forked worker mutating its own copy of the fragment array is
        discarded when the process exits, so the weights have to travel back as
        vectors and be applied here.
        """
        srdf = self._srdf()

        class _Three:
            def __call__(self, fa):
                return numpy.full(fa.n_fragments, 3.0)

        srdf.set_fragment_array_weights(_Three(), n_workers=2, verbose=False)
        for fa in srdf["fragment_array"]:
            assert (fa.weights == 3.0).all(), "weights lost crossing the process boundary"

    def test_ragged_fragment_counts(self):
        """Regions with differing fragment counts must not be NaN-padded.

        Returning a bare ndarray from the callback made pandas build a
        (n_regions, n_fragments) frame, which both loses the vectors and pads
        unequal lengths with NaN.
        """
        srdf = SampleAndRegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1"],
                    "start": [0, 1000],
                    "stop": [1000, 2000],
                    "sample_id": ["s1", "s1"],
                    "frag_h5": ["/nonexistent.h5", "/nonexistent.h5"],
                    "fragment_array": [_fragment_array(n=2), _fragment_array(n=4)],
                }
            ),
            ref="hg38",
        )

        srdf.set_fragment_array_weights(UniformWeights(), n_workers=1, verbose=False)

        counts = [fa.n_fragments for fa in srdf["fragment_array"]]
        assert counts == [2, 4]
        for fa in srdf["fragment_array"]:
            assert len(fa.weights) == fa.n_fragments
            assert not numpy.isnan(fa.weights).any()

    def test_v1_model_raises_with_migration_guidance(self):
        """The 6 notebooks calling the old signature learn what to write here."""
        srdf = self._srdf()

        class _V1Model:
            def predict_weights_from_rdf(self, rdf):
                raise AssertionError("must not be reached")

        with pytest.raises(NotImplementedError) as exc:
            srdf.set_fragment_array_weights(_V1Model())
        msg = str(exc.value)
        assert "was:" in msg and "now:" in msg, "error must name old and new usage"

    def test_non_callable_raises(self):
        srdf = self._srdf()
        with pytest.raises(NotImplementedError):
            srdf.set_fragment_array_weights("not a callback")
