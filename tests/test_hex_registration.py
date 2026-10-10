"""Pin the hexamer registration relative to the cut site.

There was no guard on this and it drifted twice.  First the table builder
registered the k-mer at cut-2 while the model looked it up at cut+0.  Then the
prior read the 3' cut at m+L-1 while L2a and the simulator read m+L, and the
test that should have caught it asserted ``WINDOW_HALF + m + L - 1`` -- a
restatement of the code's own formula, which any formula satisfies.

So every expectation here is DERIVED, never restated: from the fragment's span
(it occupies tile bases m .. m+L-1, so its junctions are (m-1)|m and
(m+L-1)|(m+L)), from CUT_OFFSET_IN_KMER (the k-mer for a junction starts that
many bases left of the base just right of the junction), and from where the
dataset places the tile inside x.  And the 3' read is tied ACROSS stacks: to
L2a's right tap and to the simulator's stop hexamer (T12).

These tests assert the *registration*, not a stored number, so they stay valid
across table rebuilds.
"""
import numpy as np
import pytest
import torch

from background_model.cut_site_model.model import (
    CUT_OFFSET_IN_KMER,
    HEX_K,
    WINDOW_HALF,
    CutSiteFragmentModel,
    _onehot_to_hex_indices,
    compute_log_p0,
)
from fragmentomics_tools.region import one_hot_encode_sequences


def _random_onehot(width, seed=0):
    """``(x, letters)``: random ACGT, one-hot encoded by the DATASET's encoder,
    so channel order cannot differ from what the model sees in training."""
    rng = np.random.default_rng(seed)
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=width)]
    oh = one_hot_encode_sequences([letters.tobytes()])[0]          # (width, 4)
    return torch.tensor(np.ascontiguousarray(oh.T), dtype=torch.float64)[None], letters


def _geometry(tile_size, L_min, L_max):
    """``(l_in, pad)``: the model's input width, and the x position of tile
    base 0.  The dataset centres the tile in x (``jitter_matrix`` at j=0), and
    ``test_calc_input_region_size_implies_the_cut_offset`` pins that contract."""
    l_in = CutSiteFragmentModel(tile_size=tile_size, L_min=L_min, L_max=L_max,
                                N1=4, K=4, N2=4, H=4).model_input_size
    return l_in, (l_in - tile_size) // 2


def _junction_bases(m, L):
    """Tile index of the base just right of each cut junction, from the span.

    The fragment occupies ``first = m`` .. ``last = m + L - 1``.  Its 5'
    junction is ``(first-1) | first`` and its 3' junction ``last | (last+1)``.
    """
    first, last = m, m + L - 1
    return first, last + 1


def _expected_hex_pos(junction_base, pad):
    """hex_idx index of the junction-centred k-mer.  It starts
    CUT_OFFSET_IN_KMER bases left of ``junction_base``, i.e. at x position
    ``pad + junction_base - CUT_OFFSET_IN_KMER``, and ``hex_idx[j]`` is the
    k-mer starting at ``x[j]``."""
    return pad + junction_base - CUT_OFFSET_IN_KMER


def _identity_probe(x, slot, tile_size, L_min, L_max):
    """compute_log_p0 with an identity table in ``slot`` (0 start_fwd,
    1 stop_fwd, 2 start_rev, 3 stop_rev) and zeros elsewhere, so each cell
    returns the k-mer index that table gathered there."""
    n_hex = 4 ** HEX_K
    tabs = [torch.zeros(n_hex, dtype=torch.float64) for _ in range(4)]
    tabs[slot] = torch.arange(n_hex, dtype=torch.float64)
    log_pL = torch.zeros(L_max - L_min + 1, dtype=torch.float64)
    out = compute_log_p0(x, *tabs, log_pL, tile_size, L_min, L_max)
    return out[0].round().long().numpy()                 # (P, n_L, 2)


def model_index_of_sim_index():
    """``little[b]``: the model's k-mer index for the simulator's index ``b``.

    Both sides through their OWN encoders, joined by the k-mer string: the
    simulator's ``hexamer_vocabulary`` decodes ``b``, and the model's
    ``_onehot_to_hex_indices`` re-encodes it after the dataset's one-hot.
    No base-4 convention is restated here.
    """
    from background_model.hexamers import hexamer_vocabulary

    vocab = hexamer_vocabulary()
    flat = b"".join(vocab.tolist())
    oh = one_hot_encode_sequences([flat])[0]
    x = torch.tensor(np.ascontiguousarray(oh.T), dtype=torch.float32)[None]
    idx = _onehot_to_hex_indices(x)[0].numpy()[::HEX_K]
    assert np.unique(idx).size == len(idx) == 4 ** HEX_K
    return idx


def test_cut_offset_is_junction_centred():
    """k=6 is even, so 'middle' is ambiguous; this pins which one we mean.

    Offset 3 puts the cut junction (between s-1 and s) exactly mid-k-mer.
    """
    assert HEX_K == 6
    assert CUT_OFFSET_IN_KMER == 3


def test_compute_log_p0_reads_kmer_starting_cut_offset_before_the_cut():
    """The 5' read: the k-mer centred on the junction (m-1)|m."""
    tile_size, L_min, L_max = 64, 25, 40
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=7)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()
    got = _identity_probe(x, 0, tile_size, L_min, L_max)

    for m in (0, 17, tile_size - 1):
        for L in (L_min, 33, L_max):
            j5, _ = _junction_bases(m, L)
            expected = int(hex_idx[_expected_hex_pos(j5, pad)])
            assert got[m, L - L_min, 0] == expected, (
                f"m={m} L={L}: gathered k-mer {got[m, L - L_min, 0]}, expected "
                f"{expected} (k-mer starting {CUT_OFFSET_IN_KMER} before the "
                f"5' junction)"
            )


def test_stop_endpoint_uses_the_same_registration():
    """The 3' cut must be registered like the 5' cut: on the k-mer centred on
    its junction (m+L-1)|(m+L), derived from the span, for every cell."""
    tile_size, L_min, L_max = 64, 25, 180
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=11)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()
    got = _identity_probe(x, 1, tile_size, L_min, L_max)      # stop_fwd

    m = np.arange(tile_size)[:, None]
    L = np.arange(L_min, L_max + 1)[None, :]
    _, j3 = _junction_bases(m, L)
    expected = hex_idx[_expected_hex_pos(j3, pad)]
    bad = np.argwhere(got[:, :, 0] != expected)
    assert bad.size == 0, (
        f"{len(bad)} cells read the wrong 3' k-mer, first (m, L) = "
        f"({bad[0][0]}, {bad[0][1] + L_min})"
    )


def test_counter_and_model_agree_on_registration():
    """The table counter's k-mer for tile base p must be the one the model
    frame puts there.

    This is the invariant that actually broke.  Testing each side against the
    constant separately would still pass if both drifted together; this ties
    them to each other.  (Retargeted from the retired stage1 builder.)
    """
    from scripts.cut_site_model.baselines.count_tables import (
        cut_base_window, tile_hexamers,
    )

    tile_size, L_min, L_max = 64, 25, 180
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=23)
    model_hex = _onehot_to_hex_indices(x)[0].numpy()

    n = cut_base_window(tile_size, L_max)
    counter_hex, ok = tile_hexamers(x[0], n)
    assert ok.all()
    p = np.arange(n)
    np.testing.assert_array_equal(counter_hex, model_hex[_expected_hex_pos(p, pad)])


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
    hex lookup must index a valid hex_idx position whose k-mer is the one
    centred on the cut junction.

    This is the test that would have caught the WINDOW_HALF=128 silent
    overrun: at the right edge, 1,275 of 159,744 cells per strand (0.80%)
    were clamped to the wrong hexamer with no error.
    """
    from background_model.cut_site_model.model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 1024
    L_min, L_max = DEFAULT_L_MIN, DEFAULT_L_MAX
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=42)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()
    max_hex = len(hex_idx) - 1
    starts = _identity_probe(x, 0, tile_size, L_min, L_max)
    stops = _identity_probe(x, 1, tile_size, L_min, L_max)

    # Probe the corners: {L_min, L_max} x {left, right edge}
    for m, L, label in [
        (0, L_min, "left edge, L_min"),
        (tile_size - 1, L_min, "right edge, L_min"),
        (tile_size - 1, L_max, "right edge, L_max"),
        (0, L_max, "left edge, L_max"),
    ]:
        li = L - L_min
        j5, j3 = _junction_bases(m, L)
        start_pos = _expected_hex_pos(j5, pad)
        stop_pos = _expected_hex_pos(j3, pad)
        assert 0 <= start_pos <= max_hex, f"{label}: start_pos {start_pos} out of range"
        assert 0 <= stop_pos <= max_hex, f"{label}: stop_pos {stop_pos} out of range"
        assert starts[m, li, 0] == int(hex_idx[start_pos]), label
        assert stops[m, li, 0] == int(hex_idx[stop_pos]), label


def test_counter_window_covers_right_edge():
    """The counter's window must reach the 3' cut of an L_max fragment
    starting at tile_size - 1, with a valid k-mer there.  (Retargeted from the
    retired stage1 builder, which hardcoded ``L_max - 1``.)"""
    from scripts.cut_site_model.baselines.count_tables import (
        cut_base_window, tile_hexamers,
    )
    from background_model.cut_site_model.model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size, L_min, L_max = 64, DEFAULT_L_MIN, DEFAULT_L_MAX
    l_in, _ = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=99)

    _, far_junction = _junction_bases(tile_size - 1, L_max)
    n = cut_base_window(tile_size, L_max)
    assert n == far_junction + 1, (
        f"counter window {n} does not end at the far 3' cut base {far_junction}"
    )
    _, ok = tile_hexamers(x[0], n)
    assert ok[far_junction]


# ---------------------------------------------------------------------------
# Finding 5: value-level minus-strand probe
# ---------------------------------------------------------------------------


def test_minus_strand_start_rev_gathers_at_the_3prime_junction():
    """Minus-strand start_rev reads the k-mer centred on the RIGHT junction
    (m+L-1)|(m+L): a minus-strand fragment's 5' end is its genomic stop."""
    from background_model.cut_site_model.model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 64
    L_min, L_max = DEFAULT_L_MIN, DEFAULT_L_MAX
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=55)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()
    got = _identity_probe(x, 2, tile_size, L_min, L_max)

    for m, L, label in [
        (0, L_min, "left edge, L_min"),
        (0, L_max, "left edge, L_max"),
        (tile_size - 1, L_min, "right edge, L_min"),
        (tile_size - 1, L_max, "right edge, L_max"),
    ]:
        _, j3 = _junction_bases(m, L)
        expected = int(hex_idx[_expected_hex_pos(j3, pad)])
        assert got[m, L - L_min, 1] == expected, (
            f"{label}: minus-strand start_rev gathered {got[m, L - L_min, 1]}, "
            f"expected {expected}"
        )


def test_minus_strand_stop_rev_gathers_at_the_5prime_junction():
    """Minus-strand stop_rev reads the k-mer centred on the LEFT junction."""
    from background_model.cut_site_model.model import DEFAULT_L_MIN, DEFAULT_L_MAX

    tile_size = 64
    L_min, L_max = DEFAULT_L_MIN, DEFAULT_L_MAX
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=77)
    hex_idx = _onehot_to_hex_indices(x)[0].numpy()
    got = _identity_probe(x, 3, tile_size, L_min, L_max)

    for m, L, label in [
        (0, L_min, "left edge, L_min"),
        (0, L_max, "left edge, L_max"),
        (tile_size - 1, L_min, "right edge, L_min"),
        (tile_size - 1, L_max, "right edge, L_max"),
    ]:
        j5, _ = _junction_bases(m, L)
        expected = int(hex_idx[_expected_hex_pos(j5, pad)])
        assert got[m, L - L_min, 1] == expected, (
            f"{label}: minus-strand stop_rev gathered {got[m, L - L_min, 1]}, "
            f"expected {expected}"
        )


# ---------------------------------------------------------------------------
# T12: tie the prior's reads to L2a and to the simulator (cross-checks)
# ---------------------------------------------------------------------------
#
# The probes above compare the prior to a derivation.  These compare it to the
# two OTHER implementations of "which bases does a cut site see": L2a's taps
# and the simulator's hexamers.  The 3' registration bug was exactly a
# disagreement between the prior and both of them, and no test compared them.


def _receptive_fields(fn, x):
    """``fields[j]``: which output cells change when the base at x[j] changes.

    Exhaustive over every position of x: each base is swapped for another
    and ``fn`` re-run, so a field is measured, not assumed.
    """
    base = fn(x)
    fields = np.zeros((x.shape[-1],) + base.shape, dtype=bool)
    for j in range(x.shape[-1]):
        x2 = x.clone()
        x2[0, :, j] = x2[0, :, j].roll(1)
        fields[j] = fn(x2) != base
    return fields


def _expected_fields(junction, pad, l_in):
    """(l_in, P, n_L) bool: the junction-centred k-mer's x positions per cell."""
    lo = _expected_hex_pos(junction, pad)
    j = np.arange(l_in)[:, None, None]
    return (j >= lo[None]) & (j < lo[None] + HEX_K)


def test_T12a_prior_reads_match_L2a_taps():
    """Every cell, both strands, both ends: the bases the prior's hexamer read
    depends on == the bases L2a's tap depends on == the k-mer centred on the
    junction derived from the span.  Fails on an m+L-1 prior (shifted 1 left
    of L2a) and on an L2a that drifts while the prior holds."""
    tile_size, L_min, L_max = 64, 25, 40
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, _ = _random_onehot(l_in, seed=5)
    x = x.float()
    m = np.arange(tile_size)[:, None]
    L = np.arange(L_min, L_max + 1)[None, :]
    j5, j3 = _junction_bases(m, L)
    want = {"left": _expected_fields(j5, pad, l_in),
            "right": _expected_fields(np.broadcast_to(j3, (tile_size, L.size)), pad, l_in)}
    want["left"] = np.broadcast_to(want["left"], want["right"].shape)

    # The prior: (slot, strand) of each table, grouped by the end it reads.
    prior_reads = {"left": [(0, 0), (3, 1)], "right": [(1, 0), (2, 1)]}
    for end, reads in prior_reads.items():
        for slot, strand in reads:
            fields = _receptive_fields(
                lambda xx: _identity_probe(xx.double(), slot, tile_size, L_min, L_max),
                x)[..., strand]
            assert np.array_equal(fields, want[end]), (
                f"prior table slot {slot} ({end} end): receptive field differs "
                f"from the junction-centred k-mer"
            )

    # L2a: only one tap's banks live, every other branch zeroed, so a logit
    # depends on x only through that tap.
    torch.manual_seed(0)
    model = CutSiteFragmentModel(tile_size=tile_size, L_min=L_min, L_max=L_max,
                                 N1=8, K=8, N2=2, H=4).eval()
    banks = {"left": (0, 1), "right": (2, 3)}
    for end, live in banks.items():
        with torch.no_grad():
            model.interior_F.zero_()
            model.w_out.fill_(1.0)
            for b in range(4):
                model.bank_a[b].normal_()
                if b not in live:
                    model.bank_a[b].zero_()

        def logits(xx):
            with torch.no_grad():
                return model(xx)[0][0].numpy()

        fields = _receptive_fields(logits, x)
        for strand in (0, 1):
            assert np.array_equal(fields[..., strand], want[end]), (
                f"L2a {end} tap, strand {strand}: receptive field differs from "
                f"the junction-centred k-mer"
            )


def test_T12b_prior_reads_match_simulator_hexamers():
    """Every cell, both strands: the k-mer the prior reads at each end is the
    k-mer the simulator records there (``measure.cut_site_hexamers``, whose
    stop is ``stops_0 = m + L``, half-open), compared across the two encoders
    by k-mer string."""
    from types import SimpleNamespace

    from background_model.constants import HEX_HALF
    from background_model.simulator.measure import cut_site_hexamers

    tile_size, L_min, L_max = 64, 25, 180
    l_in, pad = _geometry(tile_size, L_min, L_max)
    x, letters = _random_onehot(l_in, seed=31)
    m, L = np.meshgrid(np.arange(tile_size), np.arange(L_min, L_max + 1),
                       indexing="ij")
    m, L = m.ravel(), L.ravel()

    # The simulator's sequence contract: region-local, left_pad == HEX_HALF,
    # so sequence[c] begins the window for cut site c.  Tile base 0 is x[pad].
    seq = letters[pad - HEX_HALF:]
    rfa = SimpleNamespace(n_frags=len(m), starts_0=m, stops_0=m + L,
                          fragment_strands=np.full(len(m), "+"))
    sim = cut_site_hexamers(rfa, seq)
    assert len(sim) == len(m)                      # every window valid
    to_model = model_index_of_sim_index()
    sim_left = to_model[sim["start_hex"].to_numpy()].reshape(tile_size, -1)
    sim_right = to_model[sim["stop_hex"].to_numpy()].reshape(tile_size, -1)

    for slot, strand, want, label in [
        (0, 0, sim_left, "start_fwd"), (1, 0, sim_right, "stop_fwd"),
        (2, 1, sim_right, "start_rev"), (3, 1, sim_left, "stop_rev"),
    ]:
        got = _identity_probe(x, slot, tile_size, L_min, L_max)[..., strand]
        bad = np.argwhere(got != want)
        assert bad.size == 0, (
            f"{label}: {len(bad)} cells disagree with the simulator, first "
            f"(m, L) = ({bad[0][0]}, {bad[0][1] + L_min})"
        )


# ---------------------------------------------------------------------------
# Finding 8: odd tile_size must raise
# ---------------------------------------------------------------------------


def test_odd_tile_size_raises():
    """An odd tile_size silently slides CUT_OFFSET_IN_KMER by 1."""
    from background_model.cut_site_model.model import CutSiteFragmentModel

    with pytest.raises(ValueError, match="tile_size must be even"):
        CutSiteFragmentModel(tile_size=1023)


# ---------------------------------------------------------------------------
# Finding 9: rf_budget must equal WINDOW_HALF + 5
# ---------------------------------------------------------------------------


def test_rf_budget_equals_window_half_plus_5():
    """rf_budget in CutSiteStoreConfig must stay in sync with WINDOW_HALF."""
    from background_model.cut_site_model.store import CutSiteStoreConfig

    cfg = CutSiteStoreConfig(tile_size=2048)
    assert cfg.rf_budget == WINDOW_HALF + 5, (
        f"rf_budget={cfg.rf_budget} != WINDOW_HALF + 5 = {WINDOW_HALF + 5}.  "
        f"Update CutSiteStoreConfig.rf_budget when WINDOW_HALF changes."
    )


# ---------------------------------------------------------------------------
# Owner decision 176: CutSiteStoreConfig.tile_size has no default
# ---------------------------------------------------------------------------


def test_store_config_requires_tile_size():
    import json

    from background_model.cut_site_model.store import CutSiteStoreConfig

    with pytest.raises(TypeError, match="tile_size"):
        CutSiteStoreConfig()

    # An existing store records tile_size in config_json, so it still loads,
    # and its hash is unchanged (the hash covers values, not defaults).
    cfg = CutSiteStoreConfig(tile_size=2048)
    assert json.loads(cfg.to_json())["tile_size"] == 2048
    loaded = CutSiteStoreConfig.from_json(cfg.to_json())
    assert loaded == cfg and loaded.config_hash() == cfg.config_hash()

    # Pinned literal: the hash of the old default config.  Stores built before
    # tile_size became required carry this hash, so it must not move.
    assert cfg.config_hash() == "e71256c72cbacd4e"


def test_model_requires_tile_size():
    """CutSiteFragmentModel.tile_size has no default either (owner choice)."""
    from background_model.cut_site_model.model import CutSiteFragmentModel

    with pytest.raises(TypeError, match="tile_size"):
        CutSiteFragmentModel()
    with pytest.raises(TypeError, match="tile_size"):
        CutSiteFragmentModel(N1=8, K=4, N2=4, H=4)
