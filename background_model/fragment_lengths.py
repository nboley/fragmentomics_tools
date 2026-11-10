"""Fragment-length distribution ``f(L)`` and the end-position weight it implies.

Shared definitions, like ``constants`` and ``hexamers``: numpy and stdlib
only, and nothing from ``background_model.simulator``.  The simulator
measures ``f(L)`` and draws from it (``background_model.simulator.measure``,
``background_model.simulator.draw``).  Model-side code that builds its own
tables from ``f(L)``, such as the cut-site table counter, needs the same CODE,
and decision 187 forbids model code from importing ``simulator/``.  So the
code lives here, where either side may import it.  Moved out of
``simulator/measure.py`` unchanged (owner decision 2026-10-10).
Authority: ``docs/pending/simulator_spec.md``.
"""

from __future__ import annotations

import numpy as np


class FragmentLengthDist:
    """Fragment-length distribution over ``[min_fl, max_fl]``.

    Attributes: ``counts`` (int64), ``densities`` (float64, sums to 1),
    ``min_fl``, ``max_fl``.  Both vectors have length ``max_fl - min_fl + 1``
    and are indexed by ``L - min_fl``.

    The support is whatever the caller supplies.  Built from an SRDF after
    the caller's admission filter (the simulator's is
    ``simulator.measure.filter_fragments``) it is bounded by that filter, but
    the class does not assume that and does not impose ``L_MIN`` / ``L_MAX``.
    """

    __slots__ = ("counts", "densities", "min_fl", "max_fl", "_cdf")

    def __init__(self, counts, min_fl: int):
        # copy=True so the instance owns its buffer: the arrays are frozen
        # below, and freezing a caller's array would be a surprising side
        # effect.
        counts = np.array(counts, dtype=np.int64, copy=True)
        if counts.ndim != 1 or counts.size == 0:
            raise ValueError(f"counts must be 1-D and non-empty, got {counts.shape}")
        if (counts < 0).any():
            raise ValueError("counts has negative entries")
        total = int(counts.sum())
        if total == 0:
            raise ValueError("counts sum to 0; densities would be undefined")
        self.counts = counts
        self.min_fl = int(min_fl)
        self.max_fl = int(min_fl) + counts.size - 1
        self.densities = counts / np.float64(total)

        # Cumulative density, built once. cdf_at is called twice per region by
        # fl_end_weight, so recomputing here would mean ~133k cumsums over the
        # 66,649-region set, inside the loop.
        #
        # _cdf[k] is the mass of the first k lengths, so _cdf[0] == 0 and
        # _cdf[-1] == 1. Both vectors are then made READ-ONLY: an in-place
        # edit of counts or densities would leave this cache stale, and a
        # stale CDF fails silently.
        self._cdf = np.concatenate([[0.0], np.cumsum(self.densities)])
        self.counts.setflags(write=False)
        self.densities.setflags(write=False)
        self._cdf.setflags(write=False)

    @classmethod
    def from_dataframe(cls, df) -> "FragmentLengthDist":
        """From a frame with columns ``fragment_length`` and ``count``.

        Lengths absent from the frame between the min and max become zero
        counts.  Densifying here is the point: a caller that indexed a frame
        with gaps by ``L - min_fl`` would read a neighbouring length's count
        and never find out.
        """
        missing = {"fragment_length", "count"} - set(df.columns)
        if missing:
            raise ValueError(f"frame is missing column(s) {sorted(missing)}")
        fl = np.asarray(df["fragment_length"], dtype=np.int64)
        ct = np.asarray(df["count"], dtype=np.int64)
        if fl.size == 0:
            raise ValueError("frame is empty")
        if np.unique(fl).size != fl.size:
            raise ValueError(
                "fragment_length has duplicate values; aggregate before "
                "constructing, since silently summing them would hide a "
                "double-counted population"
            )
        lo, hi = int(fl.min()), int(fl.max())
        counts = np.zeros(hi - lo + 1, dtype=np.int64)
        counts[fl - lo] = ct
        return cls(counts, lo)

    @classmethod
    def from_srdf(cls, srdf) -> "FragmentLengthDist":
        """From an SRDF that already carries filtered ``fragment_array``s.

        Call AFTER the caller's admission filter (the simulator's is
        ``simulator.measure.filter_fragments``), so this is the same
        population the caller's hexamer tables count.
        """
        if "fragment_array" not in srdf.columns:
            raise ValueError(
                "srdf has no 'fragment_array' column -- call "
                "attach_fragment_arrays first"
            )
        fas = [fa for fa in srdf["fragment_array"] if fa.n_frags]
        if not fas:
            raise ValueError("no fragments in any region")
        hi = max(int(fa.lengths.max()) for fa in fas)
        counts = np.zeros(hi + 1, dtype=np.int64)
        for fa in fas:
            counts += np.bincount(fa.lengths, minlength=hi + 1)
        nz = np.nonzero(counts)[0]
        lo = int(nz[0])
        return cls(counts[lo:], lo)

    def cdf_at(self, L):
        """Total density with ``L' <= L``. Reads the cached CDF.

        Clipped outside the support: below ``min_fl`` returns 0, at or above
        ``max_fl`` returns 1.  Accepts a scalar or an array.
        """
        return self._cdf[np.clip(np.asarray(L) - self.min_fl + 1, 0,
                                 self.counts.size)]

    def __repr__(self):
        return (
            f"FragmentLengthDist(min_fl={self.min_fl}, max_fl={self.max_fl}, "
            f"n={int(self.counts.sum())})"
        )


def fl_end_weight(n_hex: int, region_len: int, fl: "FragmentLengthDist"):
    """Expected end-position weight per hexamer index, under a uniform start.

    Start and end expectations are NOT symmetric.  With starts uniform over
    ``[gstart, gstop)``, an end at offset ``i`` is reachable from ``s = i - L``
    for every ``L`` whose start still lands in the region, and each route
    carries FL mass ``f(L)``.  Interior ends are reachable by every length and
    so carry total weight 1; ends within ``max_fl`` of either edge carry a
    partial sum.  Weighting every end equally would understate the denominator
    at the edges and inflate ``r_end`` there.

    One closed form rather than one window per length:

        w(i) = F(min(i, max_fl)) - F(max(min_fl - 1, i - region_len))

    for FL CDF ``F``, so the cost is ``O(region_len + n_lengths)``.
    """
    i = np.arange(n_hex)
    return (
        fl.cdf_at(np.minimum(i, fl.max_fl))
        - fl.cdf_at(np.maximum(fl.min_fl - 1, i - region_len))
    )
