"""Unit tests for BackgroundModelHybrid (k-mer embedding + conv stem -> dilated trunk).

CPU-only, seeded, tiny configs. Follows the pattern of
test_background_model_ken.py.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

import background_model_core as bmc
from background_model_core import (
    BackgroundModelHybrid,
    LOSSES,
    rc_kmer_permutation,
)

N_TRACKS = len(bmc.DEFAULT_OUTPUT_TRACKS)


def random_one_hot(rng, batch, length):
    idx = rng.integers(0, 4, size=(batch, length))
    x = np.zeros((batch, 4, length), dtype=np.float32)
    for b in range(batch):
        x[b, idx[b], np.arange(length)] = 1.0
    return torch.from_numpy(x)


def tiny(**kw):
    """Small hybrid model with dropout off for determinism."""
    cfg = dict(k=6, d_embed=16, n_kernels=32, num_residual_layers=1, dropout=0.0)
    cfg.update(kw)
    return BackgroundModelHybrid(**cfg)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("loss", LOSSES)
def test_forward_matches_calc_input_region_size(k, loss):
    """forward() emits exactly the length calc_input_region_size promises."""
    model = tiny(k=k, loss=loss)
    l_out = 256
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(0)
    x = random_one_hot(rng, 2, l_in)
    shape_logits, dispersion = model(x)
    assert shape_logits.shape == (2, N_TRACKS, l_out)
    if loss == "multinomial":
        assert dispersion is None
    else:
        assert dispersion.shape == (2, N_TRACKS, l_out)


@pytest.mark.parametrize("n_res", [1, 2, 3, 4])
def test_geometry_across_trunk_depths(n_res):
    model = tiny(num_residual_layers=n_res)
    l_out = 128
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(1)
    shape_logits, _ = model(random_one_hot(rng, 1, l_in))
    assert shape_logits.shape[-1] == l_out


def test_input_size_even_for_even_output():
    """Dataset requires an even model_input_size; every trim term is even."""
    for k in [4, 5, 6, 7, 8]:
        for n_res in [1, 2, 3]:
            model = tiny(k=k, num_residual_layers=n_res)
            assert model.calc_input_region_size(2048) % 2 == 0


def test_k_larger_than_kernel_size_rejected():
    """The embedding branch is cropped to the stem, so k must not exceed it."""
    with pytest.raises(ValueError, match="must not exceed kernel_size"):
        BackgroundModelHybrid(k=40, kernel_size=32)


# --------------------------------------------------------------------------
# Both branches contribute
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loss", LOSSES)
def test_gradients_flow_through_both_branches(loss):
    """Embedding and conv-stem branches must both receive finite, nonzero grads."""
    model = tiny(loss=loss)
    l_out = 256
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(2)
    x = random_one_hot(rng, 2, l_in)
    y = torch.from_numpy(
        rng.poisson(3.0, size=(2, N_TRACKS, l_out)).astype(np.float32)
    )
    mask = torch.ones(2, l_out, dtype=torch.bool)
    mask[:, :30] = False
    y = y * mask[:, None, :]  # model boundary: zero targets at masked positions

    loss_val = model._step((x, y, mask), "train_loss")
    loss_val.backward()

    for name, module in [
        ("embed", model.embed),
        ("conv_stem", model.conv_stem[0]),
        ("fuse", model.fuse),
        ("shape_head", model.shape_head),
    ]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name}: no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"{name}: non-finite"
        assert any(g.abs().sum() > 0 for g in grads), f"{name}: all-zero"


def test_embedding_branch_changes_output():
    """Perturbing only the embedding table must change the prediction."""
    model = tiny()
    model.eval()
    l_out = 128
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(3)
    x = random_one_hot(rng, 1, l_in)
    with torch.no_grad():
        before = model(x)[0].clone()
        model.embed.weight.add_(1.0)
        after = model(x)[0]
    assert not torch.allclose(before, after)


def test_conv_stem_branch_changes_output():
    """Perturbing only the conv stem must change the prediction."""
    model = tiny()
    model.eval()
    l_out = 128
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(4)
    x = random_one_hot(rng, 1, l_in)
    with torch.no_grad():
        before = model(x)[0].clone()
        model.conv_stem[0].weight.add_(1.0)
        after = model(x)[0]
    assert not torch.allclose(before, after)


# --------------------------------------------------------------------------
# RC weight tying
# --------------------------------------------------------------------------


def test_rc_weight_tying():
    """Each k-mer shares an embedding row with its reverse complement."""
    model = tiny(k=6)
    rc_perm = rc_kmer_permutation(6)
    to_canonical = model._to_canonical.numpy()
    np.testing.assert_array_equal(to_canonical, to_canonical[rc_perm])
    assert model.embed.num_embeddings == 2080


@pytest.mark.parametrize("k", [4, 5, 6])
def test_rc_tying_row_count(k):
    """Canonical row count equals the number of RC equivalence classes."""
    model = tiny(k=k)
    rc_perm = rc_kmer_permutation(k)
    expected = len(np.unique(np.minimum(np.arange(4 ** k), rc_perm)))
    assert model.embed.num_embeddings == expected


# --------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------


def test_predict_profile_interface():
    """Matches BackgroundModel.predict_profile: normalized probs per track."""
    model = tiny()
    l_out = 128
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(5)
    x = random_one_hot(rng, 1, l_in)[0].numpy()
    out = model.predict_profile(x, mask=np.ones(l_out, dtype=bool))
    assert out["probs"].shape == (N_TRACKS, l_out)
    np.testing.assert_allclose(out["probs"].sum(axis=-1), 1.0, rtol=1e-5)
    assert out["log_dispersion"] is None  # multinomial


def test_predict_profile_dispersion_for_nb():
    model = tiny(loss="nb_offset", dispersion_window_size=64)
    l_out = 128
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(6)
    x = random_one_hot(rng, 1, l_in)[0].numpy()
    out = model.predict_profile(x, mask=np.ones(l_out, dtype=bool))
    assert out["log_dispersion"].shape == (N_TRACKS, l_out // 64)


def test_checkpoint_roundtrip():
    model = tiny()
    model.eval()
    l_out = 128
    l_in = model.calc_input_region_size(l_out)
    rng = np.random.default_rng(7)
    x = random_one_hot(rng, 1, l_in)
    with torch.no_grad():
        before = model(x)[0]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        torch.save(model.state_dict(), path)
        restored = BackgroundModelHybrid(**dict(model.hparams))
        restored.load_state_dict(torch.load(path, weights_only=True))
        restored.eval()
        with torch.no_grad():
            after = restored(x)[0]
    torch.testing.assert_close(before, after)


def test_freeze_dispersion():
    model = tiny(loss="nb_offset", freeze_dispersion=True)
    assert all(not p.requires_grad for p in model.dispersion_head.parameters())
    assert any(p.requires_grad for p in model.embed.parameters())
    assert any(p.requires_grad for p in model.conv_stem.parameters())


def test_weight_decay_applies_only_to_embedding():
    model = tiny(weight_decay=0.01)
    result = model.configure_optimizers()
    groups = result["optimizer"].param_groups
    decayed = [g for g in groups if g["weight_decay"] > 0]
    assert len(decayed) == 1
    embed_ids = {id(p) for p in model.embed.parameters()}
    assert {id(p) for p in decayed[0]["params"]} == embed_ids


def test_dispersion_lr_scale_creates_separate_group():
    model = tiny(loss="nb_offset", dispersion_lr_scale=0.1, learning_rate=1e-3)
    result = model.configure_optimizers()
    groups = result["optimizer"].param_groups
    assert len(groups) == 3
    lrs = sorted(g["lr"] for g in groups)
    assert lrs[0] == pytest.approx(1e-4)
