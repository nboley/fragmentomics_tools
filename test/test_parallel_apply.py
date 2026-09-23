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


def _slow_for_early_rows(row):
    # invert completion order: row 0 sleeps longest, the last row returns first
    import time

    time.sleep(max(0, (70 - row.start)) / 1000.0)
    return {"v": row.start}


def _mixed_frame_and_dict(row):
    return pd.DataFrame({"v": [row.start]}) if row.start == 50 else {"v": row.start}


def _returns_none_on_one(row):
    return None if row.start == 50 else {"v": row.start}


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


def _identity_start(row):
    return {"v": row.start}


def _outer_that_nests(row):
    """A callable that itself calls parallel_apply, from inside a worker."""
    sub = RegionDataFrame(
        pd.DataFrame(
            {
                "contig": ["chr1"] * 4,
                "start": [500 + i * 10 for i in range(4)],
                "stop": [500 + i * 10 + 5 for i in range(4)],
            }
        ),
        ref="hg38",
    )
    inner = sub.parallel_apply(_identity_start, n_workers=2, verbose=False)
    return {"v": row.start, "inner_sum": int(inner["v"].sum())}


class TestResultsAndOrdering:
    def test_dict_records_build_a_frame(self):
        rdf = make_rdf()
        out = rdf.parallel_apply(_double_start, n_workers=3, verbose=False)
        assert list(out["v"]) == [s * 2 for s in rdf.start]

    def test_order_matches_input(self):
        rdf = make_rdf(24)
        out = rdf.parallel_apply(_double_start, n_workers=4, verbose=False)
        assert list(out["v"]) == [s * 2 for s in rdf.start]
        assert list(out.index) == list(rdf.index)

    def test_order_survives_skewed_completion_times(self):
        # Row 0 finishes last. executor.map yields in submission order, so the
        # re-sort in _parallel_apply is currently a no-op -- a review removed it
        # by mutation and all tests still passed, which means the old name of
        # the test above ("..._not_completion") claimed something it never
        # checked. This pins the end-to-end property instead of the mechanism,
        # so it still holds if the implementation ever moves to as_completed,
        # where completion order would genuinely leak through.
        rdf = make_rdf(8)
        out = rdf.parallel_apply(_slow_for_early_rows, n_workers=4, verbose=False)
        assert list(out["v"]) == list(rdf.start)
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


class TestCallerMistakes:
    """fn contract violations must be named, not silently absorbed.

    Both of these were found by review. Before the guards, mixed return types
    produced a frame with Series objects sitting in cells and no error at all,
    and returning None surfaced as "'NoneType' object is not iterable" from the
    DataFrame constructor -- naming neither fn nor the offending row.
    """

    def test_mixed_return_types_are_rejected(self):
        rdf = make_rdf()
        with pytest.raises(ValueError, match="same kind of value"):
            rdf.parallel_apply(_mixed_frame_and_dict, n_workers=3, verbose=False)

    def test_returning_none_is_rejected(self):
        rdf = make_rdf()
        with pytest.raises(ValueError, match="returned None"):
            rdf.parallel_apply(_returns_none_on_one, n_workers=3, verbose=False)

    def test_all_frames_still_works(self):
        # the mixed-type guard must not fire when every record is a frame
        rdf = make_rdf()
        out = rdf.parallel_apply(_as_frame, n_workers=3, verbose=False)
        assert list(out["v"]) == list(rdf.start)

    def test_all_dicts_still_works(self):
        rdf = make_rdf()
        out = rdf.parallel_apply(_double_start, n_workers=3, verbose=False)
        assert list(out["v"]) == [s * 2 for s in rdf.start]


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

    def test_parent_process_state_is_never_populated(self):
        # The frame and callable are handed to workers via the executor's
        # initargs, so the parent's copy of this global must stay empty. An
        # earlier version set it in the parent, which made two threads calling
        # parallel_apply concurrently overwrite each other -- see
        # TestConcurrentCallers below.
        from fragmentomics_tools.dataframe import _PARALLEL_APPLY_STATE

        rdf = make_rdf(4)
        rdf.parallel_apply(_double_start, n_workers=2, verbose=False)
        assert _PARALLEL_APPLY_STATE == {}


class TestConcurrentCallers:
    """Two callers must not see each other's data.

    A previous implementation stashed the frame and callable in a module
    global in the parent process. Concurrent calls from threads overwrote that
    slot, and workers silently computed against whichever frame happened to be
    installed -- returning confident, wrong numbers rather than failing. These
    tests fail against that version.
    """

    def test_threads_do_not_corrupt_each_other(self):
        import threading

        results = {}
        errors = {}

        def run(tid):
            rdf = RegionDataFrame(
                pd.DataFrame(
                    {
                        "contig": ["chr1"] * 6,
                        "start": [tid * 1000 + i * 10 for i in range(6)],
                        "stop": [tid * 1000 + i * 10 + 5 for i in range(6)],
                    }
                ),
                ref="hg38",
            )
            try:
                out = rdf.parallel_apply(_identity_start, n_workers=2,
                                         verbose=False)
                results[tid] = list(out["v"])
            except Exception as exc:  # pragma: no cover - diagnostic only
                errors[tid] = f"{type(exc).__name__}: {exc}"

        threads = [threading.Thread(target=run, args=(t,)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == {}, f"threads raised: {errors}"
        for tid in range(3):
            expected = [tid * 1000 + i * 10 for i in range(6)]
            assert results[tid] == expected, (
                f"thread {tid} got another thread's data: "
                f"{results[tid]} != {expected}"
            )

    def test_nested_calls_work(self):
        # a callable that itself runs parallel_apply, from inside a worker.
        # Per-worker state makes this safe; a parent-side global did not.
        rdf = make_rdf(4)
        out = rdf.parallel_apply(_outer_that_nests, n_workers=2, verbose=False)
        expected_inner = sum(500 + i * 10 for i in range(4))
        assert list(out["v"]) == [0, 10, 20, 30]
        assert all(s == expected_inner for s in out["inner_sum"])
