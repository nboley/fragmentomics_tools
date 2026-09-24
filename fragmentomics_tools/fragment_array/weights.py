"""Fragment weight callbacks for bias correction.

This module provides callable weight functions for use with
``FragmentArray.assign_weights`` and ``SampleAndRegionDataFrame.set_fragment_array_weights``.

The callback protocol is::

    weight_fn(fa) -> numpy.ndarray   # shape (n_fragments,)

Each callable takes a FragmentArray and returns a 1-D array of per-fragment
weights. These weights are applied uniformly to all coverage types (first,
last, midpoint).

Available weight functions:

- ``UniformWeights()``: All ones, the identity weight (each fragment counts once).
- ``GCFlWeights(normalizer)``: GC and fragment-length bias correction.

Example usage::

    from fragmentomics_tools.fragment_array.weights import UniformWeights, GCFlWeights

    # Reset to uniform weights
    srdf.set_fragment_array_weights(UniformWeights())

    # Apply GC/FL correction
    from flgc.model import GCFlDistModel
    normalizer = GCFlDistModel.load(...)
    srdf.set_fragment_array_weights(GCFlWeights(normalizer))
"""

import numpy as np


class UniformWeights:
    """Uniform (identity) fragment weights.

    All fragments receive weight 1.0, meaning each fragment counts once.
    This is the default uncorrected state.

    Example::

        srdf.set_fragment_array_weights(UniformWeights())
    """

    def __call__(self, fa) -> np.ndarray:
        """Return all-ones weights for the fragment array.

        Args:
            fa: A FragmentArray or RegionFragmentArray.

        Returns:
            1-D array of ones with length n_fragments.
        """
        return np.ones(fa.n_fragments, dtype=np.float64)

    def __repr__(self):
        return "UniformWeights()"


class GCFlWeights:
    """GC and fragment-length bias correction weights.

    Corrects for the joint GC content and fragment length bias in cfDNA
    fragmentation. The normalizer must have a ``predict(length, gc)`` method
    where length is fragment length and gc is GC content in PERCENT (0-100).

    Fragment arrays store GC as a FRACTION (0-1), so conversion is handled
    internally.

    Args:
        normalizer: A model with a ``predict(length, gc)`` method, e.g.
            ``flgc.model.GCFlDistModel``. The model should return
            inverse-probability weights for bias correction.

    Raises:
        ValueError: If the fragment array has no GC data (``fa.gc is None``).
            This occurs when fragments were not loaded with ``return_gc=True``.

    Example::

        from flgc.model import GCFlDistModel
        normalizer = GCFlDistModel.load("/path/to/model.pkl")
        srdf.set_fragment_array_weights(GCFlWeights(normalizer))
    """

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, fa) -> np.ndarray:
        """Compute GC/FL bias correction weights.

        Args:
            fa: A FragmentArray or RegionFragmentArray with gc data.

        Returns:
            1-D array of correction weights with length n_fragments.

        Raises:
            ValueError: If fa.gc is None.
        """
        if fa.gc is None:
            raise ValueError(
                "Fragment array has no GC data. Reload with return_gc=True.\n"
                "  was:  RegionFragmentArray.from_fragments_h5(...)\n"
                "  now:  RegionFragmentArray.from_fragments_h5(..., return_gc=True)"
            )
        # gc is stored as fraction (0-1), normalizer expects percent (0-100)
        gc_percent = fa.gc * 100.0
        weights = np.atleast_1d(self.normalizer.predict(fa.fragment_lengths, gc_percent))
        return weights.astype(np.float64)

    def __repr__(self):
        return f"GCFlWeights({self.normalizer!r})"
