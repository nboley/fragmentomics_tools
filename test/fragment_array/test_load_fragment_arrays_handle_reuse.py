"""`load_fragment_arrays` must open each fragments h5 once per process.

It used to pass the path STRING to ``from_fragments_h5`` for every row, so the
same file was opened and closed once PER REGION. Profiling a 300-region
``attach_fragment_arrays`` put 6.5s of its 22.6s in ``FragmentsH5.__init__``
plus ``.close()``, against 4.3s for the reads themselves.

The reuse is keyed by pid and never populated in the parent on the forking
path: an HDF5 handle opened before a fork must not be used in the child, and
CLAUDE.md records a ``parallel_apply`` fork deadlock that ran 12 hours emitting
nothing. ``test_forking_path_opens_nothing_in_the_parent`` is the guard for
that, and it is the one to keep if the rest are ever trimmed.

Equivalence is checked field-by-field with ``assert_array_equal`` rather than
through ``RegionFragmentArray.__eq__``, which sorts both operands and compares
weights with ``allclose``: neither would notice a reordering or a last-bit
change.
"""

import os

import numpy
import pandas as pd
import pytest

from fragments_h5 import FragmentsH5

from fragmentomics_tools import RegionFragmentArray
from fragmentomics_tools.dataframe import SampleAndRegionDataFrame
from fragmentomics_tools.region import Region

# The fixture h5 covers chr6:99,110,000-99,130,000 (see conftest). These tile
# the fragment-bearing end of it; `test_regions_are_not_vacuous` asserts they
# actually carry fragments, so a mis-edit here cannot turn the equivalence
# checks into comparisons of empty arrays.
REGIONS = [("chr6", 99_119_500 + i * 1_000, 99_119_500 + i * 1_000 + 800)
           for i in range(8)]


def make_srdf(h5):
    return SampleAndRegionDataFrame(
        pd.DataFrame({
            "contig": [c for c, _, _ in REGIONS],
            "start": [s for _, s, _ in REGIONS],
            "stop": [e for _, _, e in REGIONS],
            "sample_id": ["s1"] * len(REGIONS),
            "frag_h5": [h5] * len(REGIONS),
        }),
        ref="hg38",
    )


@pytest.fixture
def opened(monkeypatch):
    """Every FragmentsH5 constructed IN THIS PROCESS, in order.

    Patching the method on the class rather than a module-level name catches
    opens from anywhere -- `dataframe` and `fragment_array` hold separate
    references to the same class object. A forked worker's own opens land in
    its copy of this list and are invisible here, which is exactly what
    `test_forking_path_opens_nothing_in_the_parent` relies on.
    """
    seen = []
    real_init = FragmentsH5.__init__

    def counting_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        seen.append(self)

    monkeypatch.setattr(FragmentsH5, "__init__", counting_init)
    return seen


def n_open_fds_for(path):
    """How many of this process's fds point at `path`."""
    real = os.path.realpath(path)
    n = 0
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.path.join("/proc/self/fd", name)) == real:
                n += 1
        except OSError:
            pass
    return n


def assert_identical(a, b, label):
    """Bit-identical, field by field, in the order the arrays came back."""
    assert a.region == b.region, label
    assert set(a.init_kwargs) == set(b.init_kwargs), label
    for key in a.init_kwargs:
        x, y = a.init_kwargs[key], b.init_kwargs[key]
        if x is None or y is None:
            assert x is None and y is None, f"{label}: {key} None mismatch"
            continue
        if isinstance(x, numpy.ndarray):
            assert x.dtype == y.dtype, f"{label}: {key} dtype"
            # assert_array_equal is exact and treats NaN as matching NaN,
            # which `gc` needs -- unknown GC is stored as NaN.
            numpy.testing.assert_array_equal(x, y, err_msg=f"{label}: {key}")
        else:
            assert x == y, f"{label}: {key}"


def reference_arrays(h5_path):
    """What the pre-change implementation produced: a fresh open per region."""
    return [
        RegionFragmentArray.from_fragments_h5(
            h5_path,
            region=Region(c, s, e, ".", ref="hg38"),
            max_frag_len=511,
            generate_weights_callback=None,
            fetch_array_kwargs=None,
            min_mapq=None,
        )
        for c, s, e in REGIONS
    ]


def test_regions_are_not_vacuous(small_h5_path):
    """Guard the guards: the fixture regions must carry fragments."""
    total = sum(fa.n_frags for fa in reference_arrays(small_h5_path))
    assert total > 0, "REGIONS hold no fragments; equivalence checks are empty"


def test_opens_the_h5_once_not_once_per_region(small_h5_path, opened):
    """The regression guard. Pre-change this opened the file 8 times."""
    make_srdf(small_h5_path).load_fragment_arrays(n_workers=1, verbose=0)
    assert len(REGIONS) > 1
    assert len(opened) == 1, (
        f"{len(opened)} opens for {len(REGIONS)} regions -- the per-process "
        "handle is not being reused"
    )


def test_in_process_path_leaks_no_handle(small_h5_path, opened):
    before = n_open_fds_for(small_h5_path)
    make_srdf(small_h5_path).load_fragment_arrays(n_workers=1, verbose=0)
    assert opened, "nothing was opened; the probe is not wired up"
    for h5 in opened:
        # h5py.File is falsy once closed.
        assert not h5._f, "load_fragment_arrays returned with an open handle"
    assert n_open_fds_for(small_h5_path) == before


def test_handle_is_released_even_when_fn_raises(small_h5_path, opened):
    def boom(fa):
        raise RuntimeError("callback failed")

    before = n_open_fds_for(small_h5_path)
    with pytest.raises(RuntimeError, match="callback failed"):
        make_srdf(small_h5_path).load_fragment_arrays(
            n_workers=1, verbose=0, fragment_array_callback=boom
        )
    for h5 in opened:
        assert not h5._f
    assert n_open_fds_for(small_h5_path) == before


def test_in_process_result_is_bit_identical(small_h5_path):
    got = make_srdf(small_h5_path).load_fragment_arrays(n_workers=1, verbose=0)
    expected = reference_arrays(small_h5_path)
    assert len(got) == len(expected)
    for i, (a, b) in enumerate(zip(got, expected)):
        assert_identical(a, b, f"n_workers=1 region {i}")


def test_forking_path_opens_nothing_in_the_parent(small_h5_path, opened):
    """No handle may exist in the parent at fork time, for the child to inherit.

    An inherited HDF5 handle used in a child is a known cause of silent
    corruption and hangs. On this path `get_fa` only ever runs in a worker, so
    the parent must open nothing at all.
    """
    before = n_open_fds_for(small_h5_path)
    make_srdf(small_h5_path).load_fragment_arrays(n_workers=2, verbose=0)
    assert opened == [], "the parent opened a handle the workers could inherit"
    assert n_open_fds_for(small_h5_path) == before


def test_forking_path_result_is_bit_identical(small_h5_path):
    rv = make_srdf(small_h5_path).load_fragment_arrays(n_workers=2, verbose=0)
    got = list(rv["fragment_array"])
    expected = reference_arrays(small_h5_path)
    assert len(got) == len(expected)
    for i, (a, b) in enumerate(zip(got, expected)):
        assert_identical(a, b, f"n_workers=2 region {i}")


def test_a_live_handle_in_the_column_is_not_closed(small_h5_path, opened):
    """A handle the caller put in `frag_h5` stays the caller's to close.

    `detach_h5`/`close_handles` exist because a live FragmentsH5 can sit in
    that column, shared by reference across rows and slices. Closing one here
    would invalidate it everywhere.
    """
    h5 = FragmentsH5(small_h5_path, cache_pointers=False)
    try:
        got = make_srdf(h5).load_fragment_arrays(n_workers=1, verbose=0)
        assert h5._f, "load_fragment_arrays closed a handle it did not open"
        # Only the caller's handle was ever opened -- no second one was made.
        assert len(opened) == 1
        for i, (a, b) in enumerate(zip(got, reference_arrays(small_h5_path))):
            assert_identical(a, b, f"live handle region {i}")
    finally:
        h5.close()
