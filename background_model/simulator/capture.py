"""Capture-surface fitting and marginal fragment-length distribution (Steps 1-2).

Step 1: Fit a GCFlDistModel from a paired-end duplicate histogram, producing
the capture-surface inverse ``predict(L, gc)`` = ``min(1/P(seen), max_weight)``.

Step 2: Build ``marginal_fl(L)`` — the empirical, unweighted length marginal
from the same duphist's deduplicated ``molecule_keys`` column, restricted to
L = 25..180, normalised to sum 1.  Nothing is deconvolved out of it (owner
decision 67).

Both functions reuse ``load_duphist`` and ``build_cell_map`` from
``scripts/ztnb_from_duphist.py`` (design doc §1, sanctioned reuse).

Runtime dependency: ``flgc.model`` needs
``PYTHONPATH=/home/nathanboley/src/biomarker``, including in the Batch container.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    N_GC_BINS,
    N_LENGTHS,
    GC_BIN_WIDTH,
    build_predict_lut,
)

# ── paths ─────────────────────────────────────────────────────────────────

DUPHIST_DIR = "/efs/analytics/nathanboley/ctcf_fa_cache/duphist_merged"

# ── bins for the capture-surface fit (Appendix F) ─────────────────────────
#
# Length bins: 1-bp bins over 25..180 — one bin per integer length, so the
# surface resolves every length the simulator draws.
# GC bins: 5% contiguous bins [0,5), [5,10), ..., [95,100], last inclusive.
# These are the same bins used by gc_bin_index in weights.py.
#
# Do NOT use the flgc defaults: their top length bin (101,200) would collapse
# the whole high band [110,180) into one bin, and their GC range stops short
# of 0-100 (Appendix F).

SIM_LENGTH_BINS = [(L, L) for L in range(L_MIN, L_MAX + 1)]

SIM_GC_BINS = [
    (int(i * GC_BIN_WIDTH), int((i + 1) * GC_BIN_WIDTH - 1))
    for i in range(N_GC_BINS - 1)
] + [(int((N_GC_BINS - 1) * GC_BIN_WIDTH), 100)]
# Result: [(0,4), (5,9), (10,14), ..., (90,94), (95,100)]
# _bin_index uses lo <= v <= hi (inclusive), so 4.5 falls in (0,4)? No —
# 4.5 > 4, so it misses (0,4); and 4.5 < 5, so it also misses (5,9) —
# it falls between bins.  BUT: duphist GC values are integers
# (whole-number percent), so non-integer values don't arise in the fit.
# The LUT's gc_bin_index (floor-based) handles non-integer GC at predict
# time.  The fit bins only need to cover the integer grid.

MAX_SANE_LENGTH = 1000


# ── duphist loading (reuse from scripts/ztnb_from_duphist.py) ─────────────

def load_duphist(sample: str, duphist_dir: str = DUPHIST_DIR) -> pd.DataFrame:
    """Load and filter a paired-end duplicate histogram.

    Returns rows with ``1 <= length <= MAX_SANE_LENGTH``.

    Note: the ``{sample}__duphist_wg.tsv.gz`` filename pattern is hardcoded.
    Phase 5 should wire this into the full pipeline's path resolution.
    """
    path = f"{duphist_dir}/{sample}__duphist_wg.tsv.gz"
    df = pd.read_csv(path, sep="\t")
    return df[(df.length >= 1) & (df.length <= MAX_SANE_LENGTH)]


def build_cell_map(df: pd.DataFrame) -> dict:
    """Build the ``(length, gc) -> (k_vals, obs, kmax, seen_unique)`` map
    that ``GCFlDistModel.fit()`` expects."""
    cell_map = {}
    for (length, gc), g in df.groupby(["length", "gc"], sort=False):
        k = g.multiplicity.to_numpy()
        obs = g.molecule_keys.to_numpy()
        cell_map[(int(length), float(gc))] = (k, obs, int(k.max()), int(obs.sum()))
    return cell_map


# ── Step 1: fit the capture surface ───────────────────────────────────────

def fit_capture_surface(
    sample: str,
    *,
    duphist_dir: str = DUPHIST_DIR,
    min_cell_size: int = 200,
    save_path: str | None = None,
    df: pd.DataFrame | None = None,
) -> "GCFlDistModel":
    """Fit a ``GCFlDistModel`` from a paired-end duphist.

    Uses 1-bp length bins (25..180) and 5% GC bins (0-100, contiguous,
    last inclusive) per Appendix F.

    Parameters
    ----------
    sample : str
        Sample name (e.g. ``"RD-56670"``).
    duphist_dir : str
        Directory containing ``<sample>__duphist_wg.tsv.gz`` files.
    min_cell_size : int
        Minimum number of unique molecules per (length, gc) cell for fitting.
    save_path : str, optional
        If provided, serialize the fitted model to this JSON path.
    df : DataFrame, optional
        Pre-loaded duphist.  If provided, ``sample`` and ``duphist_dir`` are
        not used for loading (avoids a redundant gzipped-TSV read when the
        caller already has the data).

    Returns
    -------
    GCFlDistModel
        The fitted model.  ``model.predict(L, gc)`` returns
        ``min(1/P(seen), max_weight)`` with GC in percent.
    """
    from flgc.model import GCFlDistModel

    if df is None:
        df = load_duphist(sample, duphist_dir=duphist_dir)
    cell_map = build_cell_map(df)
    model = GCFlDistModel()
    model.fit(
        cell_map,
        length_bins=SIM_LENGTH_BINS,
        gc_bins=SIM_GC_BINS,
        min_cell_size=min_cell_size,
    )
    if save_path is not None:
        model.save(save_path)
    return model


def predict_lut_from_model(model) -> np.ndarray:
    """Build the ``(N_LENGTHS, N_GC_BINS)`` predict LUT from a fitted model.

    Wraps ``build_predict_lut`` with the model's ``predict`` method.

    The LUT evaluates ``predict`` at bin midpoints (2.5, 7.5, ..., 97.5).
    This is exact **only if** the model's ``predict`` is piecewise-constant
    on those bins — true for the ZTNB method (which bins GC internally via
    ``_bin_index``), but NOT for ``spike_grid`` (which interpolates).  The
    assertions below enforce this precondition.

    Raises
    ------
    ValueError
        If the model uses ``spike_grid`` (interpolated, not piecewise-constant)
        or if the model's ``gc_bins`` do not match ``SIM_GC_BINS``.
    """
    # Guard 1: the model must be ZTNB, not spike_grid.
    # GCFlDistModel._method is private — checked flgc/model.py (biomarker repo):
    # there is no public equivalent.  _method is "ztnb" after .fit() and
    # "spike_grid" after .from_spike_grid(); it is serialised as "method" in
    # save()/load().  Using the private attr is the lesser evil vs. no check.
    if getattr(model, "_method", None) == "spike_grid":
        raise ValueError(
            "predict_lut_from_model requires a ZTNB model (piecewise-constant "
            "predict on GC bins), but got a spike_grid model (interpolated). "
            "The LUT evaluates predict at bin midpoints, which is only exact "
            "for piecewise-constant models."
        )

    # Guard 2: the model's gc_bins must match SIM_GC_BINS.
    # If they differ, the LUT's bin midpoints don't align with the model's
    # internal binning, and the gather silently approximates.
    model_gc_bins = getattr(model, "gc_bins", None)
    if model_gc_bins is not None:
        sim_gc_tuples = [tuple(b) for b in SIM_GC_BINS]
        model_gc_tuples = [tuple(b) for b in model_gc_bins]
        if model_gc_tuples != sim_gc_tuples:
            raise ValueError(
                f"Model gc_bins do not match SIM_GC_BINS. "
                f"Model: {model_gc_tuples[:3]}...{model_gc_tuples[-1:]}, "
                f"Expected: {sim_gc_tuples[:3]}...{sim_gc_tuples[-1:]}. "
                f"Mismatched bins make the LUT midpoint evaluation inexact."
            )

    return build_predict_lut(model.predict)


# ── Step 2: marginal fragment-length distribution ─────────────────────────

def build_marginal_fl(df: pd.DataFrame) -> np.ndarray:
    """Empirical unweighted fragment-length marginal from a duphist.

    Sums ``molecule_keys`` across all GC values for each length L, restricts
    to L = 25..180, and normalises to sum 1.  Nothing is deconvolved out of
    it (owner decision 67).

    Parameters
    ----------
    df : DataFrame
        Duphist with columns ``length`` and ``molecule_keys``.

    Returns
    -------
    ndarray, shape (N_LENGTHS,)
        ``marginal_fl[li]`` = P(L = L_MIN + li), sum = 1.
    """
    # Sum molecule_keys per length
    counts_by_length = df.groupby("length")["molecule_keys"].sum()
    fl = np.zeros(N_LENGTHS, dtype=np.float64)
    for li in range(N_LENGTHS):
        L = L_MIN + li
        if L in counts_by_length.index:
            fl[li] = counts_by_length[L]
    total = fl.sum()
    if total <= 0:
        raise ValueError(
            f"No molecule_keys in L={L_MIN}..{L_MAX}; "
            f"duphist has {len(df)} rows, length range "
            f"{df.length.min()}-{df.length.max()}"
        )
    fl /= total
    return fl


# ── convenience: full pipeline ────────────────────────────────────────────

def fit_and_build(
    sample: str,
    *,
    duphist_dir: str = DUPHIST_DIR,
    min_cell_size: int = 200,
    save_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the capture surface and build marginal_fl in one call.

    Returns ``(predict_lut, marginal_fl)`` — the two data inputs that
    ``build_region_weights`` needs (beyond the per-region precompute).
    """
    df = load_duphist(sample, duphist_dir=duphist_dir)
    model = fit_capture_surface(
        sample,
        duphist_dir=duphist_dir,
        min_cell_size=min_cell_size,
        save_path=save_path,
        df=df,
    )
    lut = predict_lut_from_model(model)
    fl = build_marginal_fl(df)
    return lut, fl
