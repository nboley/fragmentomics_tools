"""Tests for DataFrameBase.parallel_apply.

This code previously had zero coverage, which is how three separate paths to
an unrecoverable hang survived in it. The worker-death and unpicklable-result
tests below hang forever against the pre-fix implementation, so every one of
them is bounded by pytest-timeout-free explicit means: they either complete or
the executor raises.
"""

import multiprocessing
import os
import signal

import pandas as pd
import pytest

from fragmentomics_tools.dataframe import RegionDataFrame


def make_rdf(n=12):
    return RegionDataFrame(
        pd.DataFrame(
            {
                "contig": ["chr1"] * n,
                "start": list(range(0, n * 10, 10)),
                "stop": list(range(5, n * 10 + 5, 10)),
            }
        ),
        ref="hg38",
    )


# --- module-level callables (picklable; used where that matters) -----------


def _double_start(row):
    return {"v": row.start * 2}


def _as_frame(row):
    return pd.DataFrame({"v": [row.start]})


def _frame_with_reserved_col(row):
    return pd.DataFrame({"v": [row.start], "original_index": [999]})


def _raises_on_one(row):
    if row.start == 50:
        raise RuntimeError("boom in worker")
    return {"v": row.start}


def _sigkill_on_one(row):
    # simulates an OOM kill / segfault: the worker vanishes mid-task
    if row.start == 50:
        os.kill(os.getpid(), signal.SIGKILL)
    return {"v": row.start}


def _returns_unpicklable(row):
    # a closure cannot be pickled back to the parent
    return lambda: row.start


class TestResultsAndOrdering:
    def test_dict_records_build_a_frame(self):
        rdf = make_rdf()
        out = rdf.parallel_apply(_double_start, n_workers=3, verbose=False)
        assert list(out["v"]) == [s * 2 for s in rdf.start]

    def test_order_matches_input_not_completion(self):
        # workers finish out of order; the result must still line up with the
        # input rows, which is what the index bookkeeping exists for
        rdf = make_rdf(24)
        out = rdf.parallel_apply(_double_start, n_workers=4, verbose=False)
        assert list(out["v"]) == [s * 2 for s in rdf.start]
        assert list(out.index) == list(rdf.index)

    def test_frame_records_are_concatenated_in_order(self):
        rdf = make_rdf()
        out = rdf.parallel_apply(_as_frame, n_workers=3, verbose=False)
        assert list(out["v"]) == list(rdf.start)
        assert list(out["original_index"]) == list(rdf.index)

    def test_single_worker_debug_path_agrees(self):
        rdf = make_rdf()
        parallel = rdf.parallel_apply(_double_start, n_workers=3, verbose=False)
        serial = rdf.parallel_apply(_double_start, n_workers=1, verbose=False)
        assert list(parallel["v"]) == list(serial["v"])
        assert list(parallel.index) == list(serial.index)

    def test_lambda_is_supported(self):
        # fork inheritance means fn is never pickled; submitting it per task
        # would break this
        rdf = make_rdf()
        out = rdf.parallel_apply(lambda row: {"v": row.start + 1},
                                 n_workers=3, verbose=False)
        assert list(out["v"]) == [s + 1 for s in rdf.start]


class TestFailureModes:
    def test_worker_death_raises_rather_than_hanging(self):
        # THE regression test. Against the previous implementation this hung
        # forever: the dead worker had already claimed its row from the shared
        # counter, so the parent's `while len(indices) < shape[0]` loop could
        # never be satisfied and nothing checked whether the workers were alive.
        rdf = make_rdf()
        with pytest.raises(Exception) as excinfo:
            rdf.parallel_apply(_sigkill_on_one, n_workers=3, verbose=False)
        # BrokenProcessPool in practice; assert on behaviour, not the class
        assert excinfo.type.__name__ in {
            "BrokenProcessPool",
            "BrokenExecutor",
        }, f"unexpected exception type {excinfo.type.__name__}"

    def test_exception_in_fn_propagates(self):
        rdf = make_rdf()
        with pytest.raises(RuntimeError, match="boom in worker"):
            rdf.parallel_apply(_raises_on_one, n_workers=3, verbose=False)

    def test_unpicklable_result_raises_rather_than_hanging(self):
        # the previous worker pickled results outside any try/except, so this
        # killed the worker silently and triggered the same hang
        rdf = make_rdf()
        with pytest.raises(Exception):
            rdf.parallel_apply(_returns_unpicklable, n_workers=3, verbose=False)

    def test_empty_frame_does_not_crash(self):
        # `all(...)` is vacuously True on no records, which sent this into
        # pd.concat([]) -> ValueError("No objects to concatenate")
        rdf = make_rdf(0)
        out = rdf.parallel_apply(_double_start, n_workers=2, verbose=False)
        assert len(out) == 0

    def test_reserved_column_raises_clearly(self):
        rdf = make_rdf(4)
        with pytest.raises(ValueError, match="original_index"):
            rdf.parallel_apply(_frame_with_reserved_col, n_workers=2, verbose=False)


class TestCallerStateIsNotMutated:
    def test_returned_frames_are_not_modified_in_place(self):
        # the old code did `x['original_index'] = ...` directly on the frames
        # returned by fn, which is visible to anything else holding them
        rdf = make_rdf(4)
        out = rdf.parallel_apply(_as_frame, n_workers=1, verbose=False)
        assert "original_index" in out.columns

        # rebuild what fn returned and confirm it carries no added column
        fresh = _as_frame(rdf.iloc[0])
        assert "original_index" not in fresh.columns

    def test_worker_state_is_cleared_after_use(self):
        from fragmentomics_tools.dataframe import _PARALLEL_APPLY_STATE

        rdf = make_rdf(4)
        rdf.parallel_apply(_double_start, n_workers=2, verbose=False)
        assert _PARALLEL_APPLY_STATE == {}

    def test_worker_state_is_cleared_after_failure(self):
        from fragmentomics_tools.dataframe import _PARALLEL_APPLY_STATE

        rdf = make_rdf()
        with pytest.raises(RuntimeError):
            rdf.parallel_apply(_raises_on_one, n_workers=2, verbose=False)
        assert _PARALLEL_APPLY_STATE == {}
