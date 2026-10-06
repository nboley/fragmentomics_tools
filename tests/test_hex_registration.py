"""Pin the hexamer registration relative to the cut site.

There was no guard on this and it drifted: the table builder registered the
k-mer at cut-2 while the model looked it up at cut+0, so the tables were keyed
on one hexamer and read with another.  Nothing caught it because every test
asserted shapes, gradients or self-consistent NLLs -- none tied a sequence to
the specific k-mer index it should produce.

The convention is junction-centred: for a cut at base ``s`` the k-mer spans
``s - CUT_OFFSET_IN_KMER .. s - CUT_OFFSET_IN_KMER + k - 1``, i.e. the cut base
sits at offset ``CUT_OFFSET_IN_KMER`` within the k-mer.

These tests assert the *registration*, not a stored number, so they stay valid
across table rebuilds.
"""
import numpy as np
import pytest
import torch

from background_model.cut_site_model import (
    CUT_OFFSET_IN_KMER,
    HEX_K,
    WINDOW_HALF,
    _onehot_to_hex_indices,
    compute_log_p0,
)


def _random_onehot(width, seed=0):
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 4, size=width)
    oh = np.zeros((4, width), dtype=np.float64)
    oh[bases, np.arange(width)] = 1.0
    return torch.tensor(oh)[None], bases


def test_cut_offset_is_junction_centred():
    """k=6 is even, so 'middle' is ambiguous; this pins which one we mean.

    Offset 3 puts the cut junction (between s-1 and s) exactly mid-k-mer.
    """
    assert HEX_K == 6
    assert CUT_OFFSET_IN_KMER == 3


def test_compute_log_p0_reads_kmer_starting_cut_offset_before_the_cut():
    """Probe which k-mer index compute_log_p0 actually gathers.

    Trick: make start_fwd an identity table (``table[j] == j``) and zero the
    others, so log_p0 for strand 0 returns the gathered index verbatim.
    """
    tile_size = 64
    L_min, L_max = 25, 40
    n_L = L_max - L_min + 1
    l_in = tile_size + 2 * WINDOW_HALF

    x, _ = _random_onehot(l_in, seed=7)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()   # hex_idx[i] = k-mer at x pos i

    n_hex = 4 ** HEX_K
    identity = torch.arange(n_hex, dtype=torch.float64)
    zeros_tab = torch.zeros(n_hex, dtype=torch.float64)
    log_pL = torch.zeros(n_L, dtype=torch.float64)

    log_p0 = compute_log_p0(
        x, identity, zeros_tab, zeros_tab, zeros_tab, log_pL,
        tile_size, L_min, L_max,
    )

    for m in (0, 17, tile_size - 1):
        for L in (L_min, 33, L_max):
            li = L - L_min
            got = int(round(float(log_p0[0, m, li, 0])))
            cut_in_x = WINDOW_HALF + m - L // 2
            expected = int(hex_idx[cut_in_x])
            assert got == expected, (
                f"m={m} L={L}: gathered k-mer {got}, expected {expected} "
                f"(k-mer starting {CUT_OFFSET_IN_KMER} before the cut)"
            )


def test_stop_endpoint_uses_the_same_registration():
    """The 3' cut must be registered identically to the 5' cut."""
    tile_size = 64
    L_min, L_max = 25, 40
    n_L = L_max - L_min + 1
    l_in = tile_size + 2 * WINDOW_HALF

    x, _ = _random_onehot(l_in, seed=11)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()

    n_hex = 4 ** HEX_K
    identity = torch.arange(n_hex, dtype=torch.float64)
    zeros_tab = torch.zeros(n_hex, dtype=torch.float64)
    log_pL = torch.zeros(n_L, dtype=torch.float64)

    # identity on stop_fwd this time; start_fwd zeroed
    log_p0 = compute_log_p0(
        x, zeros_tab, identity, zeros_tab, zeros_tab, log_pL,
        tile_size, L_min, L_max,
    )

    for m in (5, 30):
        for L in (L_min, L_max):
            li = L - L_min
            got = int(round(float(log_p0[0, m, li, 0])))
            stop_in_x = WINDOW_HALF + m - L // 2 + L - 1
            expected = int(hex_idx[stop_in_x])
            assert got == expected, (
                f"m={m} L={L}: stop k-mer {got}, expected {expected}"
            )


def test_builder_and_model_agree_on_registration():
    """The builder's window helper must produce the same k-mer the model reads.

    This is the invariant that actually broke.  Testing each side against the
    constant separately would still pass if both drifted together; this ties
    them to each other.
    """
    from scripts.stage1_hex_baseline import hex_window_for_tile

    tile_size = 64
    l_seq = tile_size + 2 * WINDOW_HALF      # builder tolerates any wider frame
    x, bases = _random_onehot(l_seq, seed=23)

    builder_hex = hex_window_for_tile(bases.astype(np.uint8), tile_size, HEX_K)
    assert len(builder_hex) == tile_size

    model_hex = _onehot_to_hex_indices(x)[0].numpy()
    tile_start = (l_seq - tile_size) // 2

    for p in (0, 1, 31, tile_size - 1):
        cut_in_seq = tile_start + p
        expected = int(model_hex[cut_in_seq - CUT_OFFSET_IN_KMER])
        assert int(builder_hex[p]) == expected, (
            f"p={p}: builder gave {builder_hex[p]}, model frame gives {expected}"
        )


# ---------------------------------------------------------------------------
# The invariant that would have caught the real bug
# ---------------------------------------------------------------------------
#
# The two probes above build their own `l_in = tile_size + 2*WINDOW_HALF` and
# assert compute_log_p0 against that.  Both passed while the registration was
# wrong, because the error lived in the *contract between* the model's geometry
# and the dataset's crop, and neither probe touches either.  The dataset added
# a k-mer margin that calc_input_region_size already contained, so x arrived 6
# bases wide; centred, that shifted the tile 3 bases right of where
# compute_log_p0 indexes it.
#
# These two tests pin that contract.  They need no store and no fixture.


def test_calc_input_region_size_implies_the_cut_offset():
    """The model's own padding must put the cut at CUT_OFFSET_IN_KMER.

    hex_idx[j] spans x[j : j+HEX_K] and the model gathers at WINDOW_HALF + q,
    so the cut's offset inside the gathered k-mer is determined entirely by how
    much padding calc_input_region_size reserves:

        pad  = (l_in - P) // 2        <- where the dataset centres the tile
        off  = pad - WINDOW_HALF      <- where the cut lands in the k-mer

    If these disagree the prior is silently misregistered, and a misregistered
    hex prior measures WORSE than uniform (12.835 vs ln|D| = 12.674) while
    still looking like a trained-model improvement.
    """
    from background_model.cut_site_model import CutSiteFragmentModel

    for tile_size in (256, 512, 1024):
        model = CutSiteFragmentModel(tile_size=tile_size)
        l_in = model.calc_input_region_size(tile_size)
        assert l_in % 2 == 0, f"tile_size={tile_size}: l_in={l_in} must be even"
        pad = (l_in - tile_size) // 2
        assert pad - WINDOW_HALF == CUT_OFFSET_IN_KMER, (
            f"tile_size={tile_size}: padding {pad} puts the cut at offset "
            f"{pad - WINDOW_HALF} inside the k-mer, want {CUT_OFFSET_IN_KMER}"
        )
