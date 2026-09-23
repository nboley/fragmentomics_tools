"""Correctness (not just shape) tests for BackgroundModelHybrid.

``test_background_model_hybrid.py`` locks the output *geometry* — that the
forward pass emits ``calc_input_region_size``-many positions.  It does not lock
what those positions are *reading*.  The tests here do:

* the embedding branch's center-crop places its receptive field on the SAME
  input bases as the conv stem's (a wrong crop keeps every length identical but
  silently shifts the k-mer lookup off the position it is meant to describe);
* the embedding branch is exactly reverse-complement equivariant, which is the
  property the RC weight tying exists to provide, and which the dataset's RC
  augmentation relies on;
* the fused forward/backward is finite under bf16 autocast (the precision the
  real training runs use);
* a hybrid checkpoint round-trips through the store-free consumer path
  (``inference`` / ``correction``), and a cross-architecture load fails loudly
  rather than silently producing a differently-wired model.

Receptive fields are measured by finite difference: mutate one input base, see
which output positions move.  That gives the EXACT support (weights only affect
the magnitudes, not which entries are non-zero), so the assertions below are
closed-form and seed-independent.
"""

import copy
import os
import tempfile

import numpy as np
import pytest
import torch

import background_model_core as bmc
from background_model_core import (
    BackgroundModel,
    BackgroundModelHybrid,
    BackgroundModelKEN,
    one_hot_to_kmer_indices,
)

N_TRACKS = len(bmc.DEFAULT_OUTPUT_TRACKS)


def _one_hot(rng, length):
    idx = rng.integers(0, 4, size=length)
    x = np.zeros((1, 4, length), dtype=np.float32)
    x[0, idx, np.arange(length)] = 1.0
    return torch.from_numpy(x)


def _tiny(**kw):
    cfg = dict(k=6, kernel_size=8, d_embed=4, n_kernels=4,
               num_residual_layers=1, dropout=0.0, loss="multinomial")
    cfg.update(kw)
    return BackgroundModelHybrid(**cfg)


@torch.no_grad()
def _rf_support(model, x, j, track=0, tol=1e-6):
    """Input positions that output ``[track, j]`` depends on -> ``(lo, hi)``.

    Substitutes every alternative base at every input position and records
    where the logit moves.  Returns ``None`` if the output is insensitive to
    the whole input (i.e. that branch is disabled).
    """
    model.eval()
    base = model(x)[0][0, track, j].item()
    L_in = x.shape[-1]
    hit = np.zeros(L_in, dtype=bool)
    for q in range(L_in):
        cur = int(x[0, :, q].argmax())
        for alt in range(4):
            if alt == cur:
                continue
            xq = x.clone()
            xq[0, :, q] = 0.0
            xq[0, alt, q] = 1.0
            if abs(model(xq)[0][0, track, j].item() - base) > tol:
                hit[q] = True
                break
    nz = np.nonzero(hit)[0]
    if not len(nz):
        return None
    return int(nz.min()), int(nz.max())


def _embed_only(model):
    """Copy with the conv stem zeroed, so only the embedding branch responds."""
    m = copy.deepcopy(model)
    torch.nn.init.zeros_(m.conv_stem[0].weight)
    torch.nn.init.zeros_(m.conv_stem[0].bias)
    return m


def _conv_only(model):
    """Copy with the embedding table zeroed, so only the conv stem responds."""
    m = copy.deepcopy(model)
    torch.nn.init.zeros_(m.embed.weight)
    return m


# --------------------------------------------------------------------------
# Center-crop alignment
# --------------------------------------------------------------------------

# (k, kernel_size, num_residual_layers)
_ALIGN_CASES = [
    (6, 8, 1),    # kernel_size - k even
    (4, 8, 1),
    (8, 8, 1),    # k == kernel_size, no crop
    (5, 8, 1),    # kernel_size - k odd
    (7, 8, 1),
    (4, 8, 2),    # deeper trunk
    (6, 16, 1),
]


@pytest.mark.parametrize("k,ks,n_res", _ALIGN_CASES)
def test_conv_branch_receptive_field_is_centered(k, ks, n_res):
    """The conv branch defines the frame: output j reads input [j, j+2*margin]."""
    model = _tiny(k=k, kernel_size=ks, num_residual_layers=n_res)
    L_out = 16
    L_in = model.calc_input_region_size(L_out)
    margin = (L_in - L_out) // 2
    j = L_out // 2
    torch.manual_seed(0)
    x = _one_hot(np.random.default_rng(0), L_in)

    assert _rf_support(_conv_only(model), x, j) == (j, j + 2 * margin)


@pytest.mark.parametrize("k,ks,n_res", _ALIGN_CASES)
def test_embedding_branch_crop_is_centered_on_the_conv_frame(k, ks, n_res):
    """The center-crop must leave the embedding RF concentric with the stem's.

    The embedding unfold trims ``k-1`` where the stem trims ``kernel_size-1``,
    so the crop drops ``d = kernel_size - k`` positions.  Centered means
    ``d//2`` come off the left and ``d - d//2`` off the right; any other split
    shifts every k-mer lookup relative to the position it describes while
    leaving all output shapes untouched.
    """
    model = _tiny(k=k, kernel_size=ks, num_residual_layers=n_res)
    L_out = 16
    L_in = model.calc_input_region_size(L_out)
    margin = (L_in - L_out) // 2
    j = L_out // 2
    torch.manual_seed(0)
    x = _one_hot(np.random.default_rng(0), L_in)

    d = ks - k
    expected = (j + d // 2, j + 2 * margin - (d - d // 2))
    assert _rf_support(_embed_only(model), x, j) == expected

    # ...which is the same center as the conv branch, up to the half-base that
    # an odd d cannot resolve.
    lo, hi = expected
    center = (lo + hi) / 2.0
    assert center == pytest.approx(j + margin - 0.5 * (d % 2))


def test_embedding_and_conv_branches_share_a_center_when_trim_is_even():
    """With ``kernel_size - k`` even the two branches are exactly concentric."""
    model = _tiny(k=6, kernel_size=8, num_residual_layers=1)
    L_out = 16
    L_in = model.calc_input_region_size(L_out)
    j = L_out // 2
    x = _one_hot(np.random.default_rng(0), L_in)

    c_lo, c_hi = _rf_support(_conv_only(model), x, j)
    e_lo, e_hi = _rf_support(_embed_only(model), x, j)
    assert (c_lo + c_hi) == (e_lo + e_hi)          # identical centers
    assert (e_hi - e_lo) == (c_hi - c_lo) - (8 - 6)  # narrower by exactly k-mer trim


def test_hybrid_and_ken_place_the_kmer_within_half_a_base():
    """Both architectures read the k-mer centered on the emitted position.

    They are not bit-identical: for even ``k`` KEN's even-parity round-up trims
    the last unfold position (offset -0.5 bp), while the hybrid's symmetric
    center-crop lands on 0 when ``kernel_size - k`` is even.  Lock the fact
    that neither is off by a whole base.
    """
    L_out = 16
    j = L_out // 2
    offsets = {}

    hybrid = _tiny(k=6, kernel_size=8, num_residual_layers=1)
    L_in = hybrid.calc_input_region_size(L_out)
    margin = (L_in - L_out) // 2
    lo, hi = _rf_support(_embed_only(hybrid), _one_hot(np.random.default_rng(0), L_in), j)
    offsets["hybrid"] = (lo + hi) / 2.0 - (j + margin)

    ken = BackgroundModelKEN(k=6, n_context_layers=0, dropout=0.0,
                             loss="multinomial")
    L_in_k = ken.calc_input_region_size(L_out)
    margin_k = (L_in_k - L_out) // 2
    lo, hi = _rf_support(ken, _one_hot(np.random.default_rng(0), L_in_k), j)
    offsets["ken"] = (lo + hi) / 2.0 - (j + margin_k)

    assert offsets["hybrid"] == 0.0
    assert offsets["ken"] == -0.5
    for v in offsets.values():
        assert abs(v) <= 0.5


# --------------------------------------------------------------------------
# RC equivariance of the embedding branch
# --------------------------------------------------------------------------


def _embed_branch(model, x):
    """The embedding branch exactly as ``forward`` computes it (incl. the crop)."""
    ki = one_hot_to_kmer_indices(x, model.hparams.k, model._powers)
    h = model.embed(model._to_canonical[ki]).transpose(1, 2)
    target = x.shape[-1] - (model.hparams.kernel_size - 1)
    diff = h.shape[-1] - target
    if diff > 0:
        lo = diff // 2
        h = h[..., lo:lo + target]
    return h


def _rc(x):
    """Reverse complement of a one-hot (A,C,G,T) tensor: flip channels and positions."""
    return torch.flip(x, dims=[1, 2])


@pytest.mark.parametrize("k,ks", [(6, 8), (4, 8), (8, 8), (6, 32), (4, 32)])
def test_embedding_branch_is_rc_equivariant_when_trim_is_even(k, ks):
    """RC tying + a symmetric crop => embed(RC(x)) is the mirror of embed(x).

    This is what makes the dataset's RC augmentation coherent for the embedding
    branch (the conv stem is not tied and relies on the augmentation itself).
    """
    torch.manual_seed(0)
    model = _tiny(k=k, kernel_size=ks)
    model.eval()
    L_in = model.calc_input_region_size(32)
    x = _one_hot(np.random.default_rng(3), L_in)
    with torch.no_grad():
        h = _embed_branch(model, x)
        h_rc = _embed_branch(model, _rc(x))
    assert torch.equal(h_rc, torch.flip(h, dims=[2]))


@pytest.mark.parametrize("k,ks", [(5, 8), (7, 8)])
def test_embedding_branch_rc_equivariance_breaks_by_half_a_base_for_odd_trim(k, ks):
    """Documented consequence of an odd ``kernel_size - k``.

    The crop cannot be symmetric, so the mirror identity holds only after a
    one-position shift.  Locked so the asymmetry stays a known, bounded
    property rather than a surprise.
    """
    torch.manual_seed(0)
    model = _tiny(k=k, kernel_size=ks)
    model.eval()
    L_in = model.calc_input_region_size(32)
    x = _one_hot(np.random.default_rng(3), L_in)
    with torch.no_grad():
        h = _embed_branch(model, x)
        h_rc = _embed_branch(model, _rc(x))
    mirrored = torch.flip(h, dims=[2])
    assert not torch.equal(h_rc, mirrored)
    # off by exactly one position
    assert torch.equal(h_rc[..., 1:], mirrored[..., :-1])


def test_rc_augmented_batch_trains_without_error():
    """A whole batch of RC'd inputs steps cleanly (the dataset emits these)."""
    torch.manual_seed(0)
    model = _tiny(loss="nb_offset", dispersion_window_size=8)
    L_out = 16
    L_in = model.calc_input_region_size(L_out)
    rng = np.random.default_rng(0)
    x = torch.cat([_one_hot(rng, L_in), _rc(_one_hot(rng, L_in))], dim=0)
    y = torch.from_numpy(
        rng.poisson(2.0, size=(2, N_TRACKS, L_out)).astype(np.float32)
    )
    mask = torch.ones(2, L_out, dtype=torch.bool)
    loss = model._step((x, y, mask), "train_loss")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.embed.weight.grad).all()
    assert torch.isfinite(model.conv_stem[0].weight.grad).all()


# --------------------------------------------------------------------------
# Mixed precision
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loss", ["multinomial", "dirichlet_multinomial",
                                  "nb_offset"])
def test_bf16_autocast_forward_and_backward_are_finite(loss):
    """The fused ``cat`` mixes an autocast bf16 conv output with an fp32
    embedding output; confirm the promotion works and nothing goes non-finite.
    """
    torch.manual_seed(0)
    model = _tiny(loss=loss, dispersion_window_size=8)
    L_out = 16
    L_in = model.calc_input_region_size(L_out)
    rng = np.random.default_rng(0)
    x = torch.cat([_one_hot(rng, L_in) for _ in range(2)], dim=0)
    y = torch.from_numpy(
        rng.poisson(2.0, size=(2, N_TRACKS, L_out)).astype(np.float32)
    )
    mask = torch.ones(2, L_out, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        shape_logits, dispersion = model(x)
        assert torch.isfinite(shape_logits).all()
        if dispersion is not None:
            assert torch.isfinite(dispersion).all()
        step_loss = model._step((x, y, mask), "train_loss")
    assert torch.isfinite(step_loss)
    step_loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------------------
# Checkpoint round-trip through the store-free consumers
# --------------------------------------------------------------------------


class _FakeFasta:
    """Minimal ``pysam.FastaFile.fetch`` stand-in over an in-memory contig."""

    def __init__(self, seq):
        self._seq = seq

    def fetch(self, contig, start, stop):
        return self._seq[max(0, start):stop]


def _save_ckpt(model, path):
    import lightning as L

    trainer = L.Trainer(accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, max_steps=1,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)


@pytest.mark.parametrize("loss", ["multinomial", "dirichlet_multinomial",
                                  "nb_offset"])
def test_hybrid_checkpoint_round_trips_through_inference_and_correction(loss):
    from background_model.correction import expected_profile
    from background_model.inference import (
        WindowGeometry,
        predict_region_profiles,
    )

    tile = 64
    torch.manual_seed(0)
    model = _tiny(loss=loss, dispersion_window_size=tile)

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "hybrid.ckpt")
        _save_ckpt(model, ckpt)
        loaded = BackgroundModelHybrid.load_from_checkpoint(
            ckpt, map_location="cpu"
        )

    ref = model.state_dict()
    got = loaded.state_dict()
    assert set(ref) == set(got)
    for key in ref:
        assert torch.equal(ref[key], got[key]), key

    geom = WindowGeometry.from_model(loaded, tile)
    assert geom.model_input_size == loaded.calc_input_region_size(tile)

    rng = np.random.default_rng(0)
    contig_len = 4 * tile
    seq = "".join(rng.choice(list("ACGT"), size=contig_len + 2 * geom.margin))
    fasta = _FakeFasta(seq)
    start, stop = geom.margin, geom.margin + 2 * tile

    profile = predict_region_profiles(
        loaded, fasta, "chrT", start, stop, tile_size=tile
    )
    assert profile.probs.shape == (N_TRACKS, stop - start)
    assert np.isfinite(profile.probs).all()
    for w in range(2):
        window = profile.probs[:, w * tile:(w + 1) * tile]
        assert np.allclose(window.sum(axis=1), 1.0)

    observed = rng.poisson(3, size=(N_TRACKS, stop - start)).astype(np.float64)
    exp = expected_profile(
        loaded, fasta, "chrT", start, stop, observed, tile_size=tile
    )
    assert exp.expected.shape == (N_TRACKS, stop - start)
    for w in range(2):
        window = exp.expected[:, w * tile:(w + 1) * tile]
        assert np.allclose(np.nansum(window, axis=1), exp.N[w])


def test_cross_architecture_checkpoint_load_fails_loudly():
    """A hybrid checkpoint must not silently load as the CNN baseline.

    ``scripts/sim_evaluate.py`` and ``scripts/ctcf_pileup_run.py`` hardcode the
    non-hybrid classes; the failure mode that would matter is a SILENT partial
    load, not an exception.
    """
    torch.manual_seed(0)
    model = _tiny()
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "hybrid.ckpt")
        _save_ckpt(model, ckpt)
        with pytest.raises(Exception):
            BackgroundModel.load_from_checkpoint(ckpt, map_location="cpu")
        with pytest.raises(Exception):
            BackgroundModelKEN.load_from_checkpoint(ckpt, map_location="cpu")


# --------------------------------------------------------------------------
# Ambiguous bases
# --------------------------------------------------------------------------


def test_ambiguous_bases_are_read_as_A_by_the_embedding_branch():
    """``N`` one-hot encodes as 0.25 across all four channels.

    The conv stem sees a neutral average base, but ``one_hot_to_kmer_indices``
    takes an ``argmax``, so the embedding branch reads every ``N`` as an ``A``
    — i.e. an all-N window is looked up as poly-A, not as a neutral k-mer.
    Shared with ``BackgroundModelKEN``.  Locked as a known property so that a
    future change to it is deliberate.
    """
    from fragmentomics_tools.region import one_hot_encode_sequences

    onehot = one_hot_encode_sequences([b"NNNNNNAAAAAA"])[0].T
    assert np.allclose(onehot[:, 0], 0.25)

    model = _tiny(k=6, kernel_size=8)
    x = torch.from_numpy(np.ascontiguousarray(onehot, dtype=np.float32))[None]
    idx = one_hot_to_kmer_indices(x, 6, model._powers)
    assert idx[0, 0].item() == 0          # NNNNNN -> AAAAAA
    assert idx[0, -1].item() == 0         # AAAAAA -> AAAAAA
