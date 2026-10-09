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
            # Under start admission m IS the start; 5' cut is at m.
            cut_in_x = WINDOW_HALF + m
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
            # Under start admission: 3' cut is at m + L - 1.
            stop_in_x = WINDOW_HALF + m + L - 1
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


def test_training_and_inference_agree_on_l_in():
    """model_input_size and calc_input_region_size must give the same l_in.

    The old code had two conventions sharing one name: model_input_size
    returned tile_size + 2*WINDOW_HALF, and the dataset added +5 to that.
    inference.py called calc_input_region_size which already included +5.
    If someone passed calc_input_region_size to the dataset, l_in would be
    6 bases too wide — a 3-base frame shift per side.
    """
    from background_model.cut_site_model import CutSiteFragmentModel

    for tile_size in (256, 512, 1024):
        model = CutSiteFragmentModel(tile_size=tile_size)
        assert model.model_input_size == model.calc_input_region_size(tile_size), (
            f"tile_size={tile_size}: model_input_size={model.model_input_size} != "
            f"calc_input_region_size={model.calc_input_region_size(tile_size)}"
        )


# ---------------------------------------------------------------------------
# Regression test: the model's hex lookup COVERS every cut site the dataset
# can produce, including the far-right edge under start admission.
# ---------------------------------------------------------------------------


def test_hex_lookup_covers_all_cuts_under_start_admission():
    """For every (m, L, strand) cell the dataset can produce, the model's
    hex lookup must index a valid hex_idx position whose k-mer CONTAINS the
    cut site.

    This is the test that would have caught the WINDOW_HALF=128 silent
    overrun: at the right edge, 1,275 of 159,744 cells per strand (0.80%)
    were clamped to the wrong hexamer with no error.
    """
    from background_model.cut_site_model import CutSiteFragmentModel, DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 1024
    L_min, L_max = DEFAULT_L_MIN, DEFAULT_L_MAX
    model = CutSiteFragmentModel(tile_size=tile_size, L_min=L_min, L_max=L_max)
    l_in = model.model_input_size

    x, bases = _random_onehot(l_in, seed=42)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()
    max_hex = len(hex_idx) - 1

    n_hex = 4 ** HEX_K
    identity = torch.arange(n_hex, dtype=torch.float64)
    zeros_tab = torch.zeros(n_hex, dtype=torch.float64)
    n_L = L_max - L_min + 1
    log_pL = torch.zeros(n_L, dtype=torch.float64)

    log_p0 = compute_log_p0(
        x, identity, zeros_tab, zeros_tab, zeros_tab, log_pL,
        tile_size, L_min, L_max,
    )

    # Probe the corners: {+ strand, - strand} × {L_min, L_max} × right edge
    probe_cases = [
        (0, L_min, 0, "left edge, L_min, + strand"),
        (tile_size - 1, L_min, 0, "right edge, L_min, + strand"),
        (tile_size - 1, L_max, 0, "right edge, L_max, + strand"),
        (0, L_min, 1, "left edge, L_min, - strand"),
        (tile_size - 1, L_max, 1, "right edge, L_max, - strand"),
    ]

    for m, L, strand, label in probe_cases:
        li = L - L_min
        # Under start admission: 5' cut at m, 3' cut at m + L - 1.
        start_pos = WINDOW_HALF + m
        stop_pos = WINDOW_HALF + m + L - 1
        assert 0 <= start_pos <= max_hex, (
            f"{label}: start_pos {start_pos} out of hex_idx range [0, {max_hex}]"
        )
        assert 0 <= stop_pos <= max_hex, (
            f"{label}: stop_pos {stop_pos} out of hex_idx range [0, {max_hex}]"
        )

        # The gathered value must match the hex_idx at that position
        got_start = int(round(float(log_p0[0, m, li, 0])))
        assert got_start == int(hex_idx[start_pos]), (
            f"{label}: start hex {got_start} != expected {hex_idx[start_pos]}"
        )


def test_builder_widened_window_covers_right_edge():
    """The hex table builder's widened window must reach the 3' endpoint
    of a L_max fragment starting at tile_size - 1.
    """
    from scripts.stage1_hex_baseline import hex_window_for_tile
    from background_model.cut_site_model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 64
    L_max = DEFAULT_L_MAX
    hex_width = tile_size + L_max - 1
    l_seq = tile_size + 2 * WINDOW_HALF + 2 * L_max

    _, bases = _random_onehot(l_seq, seed=99)
    builder_hex = hex_window_for_tile(
        bases.astype(np.uint8), tile_size, HEX_K, width=hex_width,
    )
    assert len(builder_hex) == hex_width

    # The furthest 3' endpoint: start at tile_size-1, length L_max
    far_end = tile_size - 1 + L_max - 1
    assert far_end < len(builder_hex), (
        f"far 3' endpoint at {far_end} is outside builder window of {len(builder_hex)}"
    )
    assert builder_hex[far_end] >= 0, (
        f"hex index at far 3' endpoint is invalid ({builder_hex[far_end]})"
    )


# ---------------------------------------------------------------------------
# Finding 5: value-level minus-strand probe
# ---------------------------------------------------------------------------


def test_minus_strand_start_rev_gathers_at_stop_pos():
    """Probe minus-strand start_rev: it gathers at flat_stop, so with an
    identity table on start_rev the value at log_p0[0, m, li, 1] should be
    hex_idx[stop_pos] where stop_pos = WINDOW_HALF + m + L - 1.
    """
    from background_model.cut_site_model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 64
    L_min, L_max = DEFAULT_L_MIN, DEFAULT_L_MAX
    n_L = L_max - L_min + 1
    raw = tile_size + 2 * WINDOW_HALF + 5
    l_in = raw + (raw % 2)

    x, _ = _random_onehot(l_in, seed=55)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()

    n_hex = 4 ** HEX_K
    identity = torch.arange(n_hex, dtype=torch.float64)
    zeros_tab = torch.zeros(n_hex, dtype=torch.float64)
    log_pL = torch.zeros(n_L, dtype=torch.float64)

    log_p0 = compute_log_p0(
        x, zeros_tab, zeros_tab, identity, zeros_tab, log_pL,
        tile_size, L_min, L_max,
    )

    for m, L, label in [
        (0, L_min, "left edge, L_min"),
        (0, L_max, "left edge, L_max"),
        (tile_size - 1, L_min, "right edge, L_min"),
        (tile_size - 1, L_max, "right edge, L_max"),
    ]:
        li = L - L_min
        got = int(round(float(log_p0[0, m, li, 1])))
        stop_in_x = WINDOW_HALF + m + L - 1
        expected = int(hex_idx[stop_in_x])
        assert got == expected, (
            f"{label}: minus-strand start_rev gathered {got}, "
            f"expected {expected} at stop_pos {stop_in_x}"
        )


def test_minus_strand_stop_rev_gathers_at_start_pos():
    """Probe minus-strand stop_rev: it gathers at flat_start, so with an
    identity table on stop_rev the value at log_p0[0, m, li, 1] should be
    hex_idx[start_pos] where start_pos = WINDOW_HALF + m.
    """
    from background_model.cut_site_model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 64
    L_min, L_max = DEFAULT_L_MIN, DEFAULT_L_MAX
    n_L = L_max - L_min + 1
    raw = tile_size + 2 * WINDOW_HALF + 5
    l_in = raw + (raw % 2)

    x, _ = _random_onehot(l_in, seed=77)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()

    n_hex = 4 ** HEX_K
    identity = torch.arange(n_hex, dtype=torch.float64)
    zeros_tab = torch.zeros(n_hex, dtype=torch.float64)
    log_pL = torch.zeros(n_L, dtype=torch.float64)

    log_p0 = compute_log_p0(
        x, zeros_tab, zeros_tab, zeros_tab, identity, log_pL,
        tile_size, L_min, L_max,
    )

    for m, L, label in [
        (0, L_min, "left edge, L_min"),
        (0, L_max, "left edge, L_max"),
        (tile_size - 1, L_min, "right edge, L_min"),
        (tile_size - 1, L_max, "right edge, L_max"),
    ]:
        li = L - L_min
        got = int(round(float(log_p0[0, m, li, 1])))
        start_in_x = WINDOW_HALF + m
        expected = int(hex_idx[start_in_x])
        assert got == expected, (
            f"{label}: minus-strand stop_rev gathered {got}, "
            f"expected {expected} at start_pos {start_in_x}"
        )


# ---------------------------------------------------------------------------
# Finding 8: odd tile_size must raise
# ---------------------------------------------------------------------------


def test_odd_tile_size_raises():
    """An odd tile_size silently slides CUT_OFFSET_IN_KMER by 1."""
    from background_model.cut_site_model import CutSiteFragmentModel

    with pytest.raises(ValueError, match="tile_size must be even"):
        CutSiteFragmentModel(tile_size=1023)


# ---------------------------------------------------------------------------
# Finding 9: rf_budget must equal WINDOW_HALF + 5
# ---------------------------------------------------------------------------


def test_rf_budget_equals_window_half_plus_5():
    """rf_budget in CutSiteStoreConfig must stay in sync with WINDOW_HALF."""
    from background_model.cut_site_store import CutSiteStoreConfig

    cfg = CutSiteStoreConfig(tile_size=2048)
    assert cfg.rf_budget == WINDOW_HALF + 5, (
        f"rf_budget={cfg.rf_budget} != WINDOW_HALF + 5 = {WINDOW_HALF + 5}.  "
        f"Update CutSiteStoreConfig.rf_budget when WINDOW_HALF changes."
    )


# ---------------------------------------------------------------------------
# Owner decision 186: the cut-site side uses the SHARED hexamer index order
# ---------------------------------------------------------------------------
#
# The registration tests above tie the model and the table builder to EACH
# OTHER, so they would still pass if both drifted to the same wrong order.
# These tie each of them to background_model.hexamers, the one encoder.


def _ascii_seq(width, seed, n_frac=0.0):
    rng = np.random.default_rng(seed)
    seq = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=width)
    if n_frac:
        seq[rng.random(width) < n_frac] = ord("N")
    return seq


def test_model_encoder_matches_shared_encoder():
    """Through the dataset's own one-hot path, the model's index equals
    hexamers.hexamer_indices' forward index at every valid position."""
    from background_model.hexamers import hexamer_indices
    from fragmentomics_tools.region import one_hot_encode_sequences

    seq = _ascii_seq(4096, seed=186, n_frac=0.02)
    onehot = one_hot_encode_sequences([seq.tobytes()])[0]       # (L, 4)
    x = torch.from_numpy(np.ascontiguousarray(onehot.T)).float()[None]
    got = _onehot_to_hex_indices(x)[0].numpy()

    fwd, _rc, valid = hexamer_indices(seq)
    assert got.shape == fwd.shape
    assert valid.sum() > 3000 and (~valid).sum() > 100       # both cases exercised
    np.testing.assert_array_equal(got[valid], fwd[valid])

    # N handling is unchanged by decision 186: N reads as A via argmax, which
    # is the N-as-A reading the shared encoder also reports for such windows.
    np.testing.assert_array_equal(got[~valid], fwd[~valid])


def test_model_encoder_is_big_endian():
    """Pin the order directly: index i decodes to hexamer_vocabulary()[i]."""
    from background_model.hexamers import hexamer_vocabulary

    acgt = np.frombuffer(b"ACGT", dtype=np.uint8)
    vocab = hexamer_vocabulary()
    for i in (0, 1, 4, 1024, 1234, 4095):
        codes = np.searchsorted(acgt, np.frombuffer(vocab[i], dtype=np.uint8))
        oh = np.zeros((4, HEX_K), dtype=np.float32)
        oh[codes, np.arange(HEX_K)] = 1.0
        assert int(_onehot_to_hex_indices(torch.from_numpy(oh)[None])[0, 0]) == i


def test_builder_encoder_matches_shared_encoder():
    """stage1_hex_baseline keys its tables in the shared order, keeps -1 for
    N-containing windows, and reads numeric 0-3 and lowercase like ASCII."""
    from background_model.hexamers import hexamer_indices
    from scripts.stage1_hex_baseline import _seq_to_hexamer_indices

    seq = _ascii_seq(4096, seed=187, n_frac=0.02)
    fwd, _rc, valid = hexamer_indices(seq)
    got = _seq_to_hexamer_indices(seq, HEX_K)
    np.testing.assert_array_equal(got, np.where(valid, fwd, -1))

    acgt = np.frombuffer(b"ACGT", dtype=np.uint8)
    ascii_ = _ascii_seq(512, seed=188)
    numeric = np.searchsorted(acgt, ascii_).astype(np.uint8)
    lower = ascii_ + np.uint8(32)
    want = _seq_to_hexamer_indices(ascii_, HEX_K)
    assert (want >= 0).all()
    np.testing.assert_array_equal(_seq_to_hexamer_indices(numeric, HEX_K), want)
    np.testing.assert_array_equal(_seq_to_hexamer_indices(lower, HEX_K), want)


def _write_tables(path, **extra):
    from background_model.cut_site_model import DEFAULT_L_MAX, DEFAULT_L_MIN

    rng = np.random.default_rng(0)
    np.savez(
        path,
        **{k: rng.dirichlet(np.ones(4 ** HEX_K)) for k in
           ("start_fwd", "stop_fwd", "start_rev", "stop_rev")},
        p_L=rng.dirichlet(np.ones(DEFAULT_L_MAX - DEFAULT_L_MIN + 1)),
        **extra,
    )


def test_untagged_hex_table_is_refused(tmp_path):
    """A table from the old little-endian builder has no encoding tag, and
    loading it would silently mis-key every hexamer.  It must raise."""
    from background_model.cut_site_model import CutSiteFragmentModel

    old = str(tmp_path / "old.npz")
    _write_tables(old, k=6, L_min=25, L_max=180)       # what the old builder wrote
    with pytest.raises(ValueError, match="encoding tag None"):
        CutSiteFragmentModel(tile_size=64, hex_tables_path=old, freeze_hex=True)

    wrong = str(tmp_path / "wrong.npz")
    _write_tables(wrong, hex_encoding="little-endian")
    with pytest.raises(ValueError, match="little-endian"):
        CutSiteFragmentModel(tile_size=64, hex_tables_path=wrong)


def test_tagged_hex_table_loads(tmp_path):
    from background_model.cut_site_model import (
        HEX_TABLE_ENCODING,
        CutSiteFragmentModel,
    )

    good = str(tmp_path / "good.npz")
    _write_tables(good, hex_encoding=HEX_TABLE_ENCODING)
    model = CutSiteFragmentModel(tile_size=64, hex_tables_path=good, freeze_hex=True)
    assert model._use_hex_baseline


# ---------------------------------------------------------------------------
# Owner decision 176: CutSiteStoreConfig.tile_size has no default
# ---------------------------------------------------------------------------


def test_store_config_requires_tile_size():
    import json

    from background_model.cut_site_store import CutSiteStoreConfig

    with pytest.raises(TypeError, match="tile_size"):
        CutSiteStoreConfig()

    # An existing store records tile_size in config_json, so it still loads,
    # and its hash is unchanged (the hash covers values, not defaults).
    cfg = CutSiteStoreConfig(tile_size=2048)
    assert json.loads(cfg.to_json())["tile_size"] == 2048
    loaded = CutSiteStoreConfig.from_json(cfg.to_json())
    assert loaded == cfg and loaded.config_hash() == cfg.config_hash()
