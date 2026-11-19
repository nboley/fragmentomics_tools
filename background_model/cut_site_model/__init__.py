"""Cut-site fragment model — per-fragment (start, length, strand) multinomial.

The model emits one logit per `(m, L, s)` cell and trains on the fragment
multinomial, so fragment length is a continuous axis rather than a banded
output channel.  Per-fragment probabilities become correction weights, letting
downstream biomarkers be read with cut-site preference removed.

    logit(m, L, s) = log p0(m, L, s) + f_theta(sequence)
    log p          = log_softmax over D,  |D| = tile_size * n_L * 2

`log p0` is an analytical hexamer prior with no free parameters beyond counted
tables; `f_theta` is a learned convolutional correction.

Three modules, mirroring `band_model/`:

- `model.py`   — `CutSiteFragmentModel`, `compute_log_p0`, the window geometry
                 (`WINDOW_HALF`, `CUT_OFFSET_IN_KMER`) and the registration
                 assertions.
- `store.py`   — `CutSiteStoreConfig` and the zarr store schema/validation.
                 numpy/zarr only, no torch: the AWS Batch builder depends on
                 it, which is why `rf_budget`'s relation to `WINDOW_HALF` is
                 pinned by a test rather than an import.
- `dataset.py` — `CutSiteFragmentDataset` and its collate function.  Owns the
                 crop and the start-based admission rule.

Import submodules explicitly, as callers of `band_model` do:

    from background_model.cut_site_model.model import CutSiteFragmentModel

Shared length bounds and k-mer geometry come from `background_model.constants`
and are not restated here.
"""
