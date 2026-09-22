"""Unit tests for the K-mer Embedding Network (BackgroundModelKEN).

Tests T1-T8 from kmer_embedding_plan.md. All tests are CPU-only, seeded,
tiny configs. Follow the pattern of test_background_model_core.py.
"""

import numpy as np
import pytest
import torch

import background_model_core as bmc
from background_model_core import (
    BackgroundModelKEN,
    LOSSES,
    one_hot_to_kmer_indices,
    rc_kmer_permutation,
)

N_TRACKS = len(bmc.DEFAULT_OUTPUT_TRACKS)


def random_one_hot(rng, batch, length):
    idx = rng.integers(0, 4, size=(batch, length))
    x = np.zeros((batch, 4, length), dtype=np.float32)
    for b in range(batch):
        x[b, idx[b], np.arange(length)] = 1.0
    return torch.from_numpy(x)


# --------------------------------------------------------------------------
# T1: K-mer index computation correctness
# --------------------------------------------------------------------------


def test_kmer_indices_known_sequence():
    """Verify KEN index encoding for a known sequence."""
    # ACGTAC: A=0, C=1, G=2, T=3
    # 0*4^5 + 1*4^4 + 2*4^3 + 3*4^2 + 0*4^1 + 1*4^0
    # = 0 + 256 + 128 + 48 + 0 + 1 = 433
    seq_bases = [0, 1, 2, 3, 0, 1, 2, 3, 0, 0]  # ACGTACGTAA
    one_hot = torch.zeros(1, 4, 10, dtype=torch.float32)
    for i, b in enumerate(seq_bases):
        one_hot[0, b, i] = 1.0
    powers = 4 ** torch.arange(5, -1, -1, dtype=torch.long)
    idx = one_hot_to_kmer_indices(one_hot, 6, powers)
    assert idx.shape == (1, 5)  # 10 - 6 + 1 = 5 hexamers
    assert idx[0, 0].item() == 433  # ACGTAC


def test_kmer_indices_all_same_base():
    """All-A sequence should produce index 0 for all k-mers."""
    one_hot = torch.zeros(1, 4, 20, dtype=torch.float32)
    one_hot[0, 0, :] = 1.0  # all A
    powers = 4 ** torch.arange(5, -1, -1, dtype=torch.long)
    idx = one_hot_to_kmer_indices(one_hot, 6, powers)
    assert (idx == 0).all()


def test_kmer_indices_all_T():
    """All-T (base=3) should produce index 4^6 - 1 = 4095."""
    one_hot = torch.zeros(1, 4, 20, dtype=torch.float32)
    one_hot[0, 3, :] = 1.0  # all T
    powers = 4 ** torch.arange(5, -1, -1, dtype=torch.long)
    idx = one_hot_to_kmer_indices(one_hot, 6, powers)
    assert (idx == 4095).all()


def test_kmer_indices_parametric_k():
    """Index computation works for various k values."""
    for k in [4, 5, 6, 7, 8]:
        one_hot = torch.zeros(1, 4, k + 5, dtype=torch.float32)
        one_hot[0, 0, :] = 1.0
        powers = 4 ** torch.arange(k - 1, -1, -1, dtype=torch.long)
        idx = one_hot_to_kmer_indices(one_hot, k, powers)
        assert idx.shape == (1, 6)  # (k+5) - k + 1 = 6
        assert (idx == 0).all()


# --------------------------------------------------------------------------
# T2: Geometry -- forward matches calc_input_region_size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("loss", LOSSES)
def test_geometry_forward_matches(k, loss):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = BackgroundModelKEN(
        k=k, d_embed=16, d_context=32,
        n_context_layers=1, context_kernel_size=5,
        loss=loss, dropout=0.0,
    )
    model.eval()
    L_out = 512
    L_in = model.calc_input_region_size(L_out)
    assert L_in % 2 == 0, "model_input_size must be even"
    x = random_one_hot(rng, 2, L_in)
    with torch.no_grad():
        shape_logits, disp = model(x)
    assert shape_logits.shape == (2, N_TRACKS, L_out)
    if loss != "multinomial":
        assert disp.shape == (2, N_TRACKS, L_out)
    else:
        assert disp is None


@pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
def test_geometry_no_context_layers(k):
    """Geometry works with n_context_layers=0."""
    torch.manual_seed(43)
    rng = np.random.default_rng(43)
    model = BackgroundModelKEN(
        k=k, d_embed=16, d_context=32,
        n_context_layers=0, loss="multinomial", dropout=0.0,
    )
    model.eval()
    L_out = 256
    L_in = model.calc_input_region_size(L_out)
    assert L_in % 2 == 0
    x = random_one_hot(rng, 1, L_in)
    with torch.no_grad():
        shape_logits, _ = model(x)
    assert shape_logits.shape == (1, N_TRACKS, L_out)


# --------------------------------------------------------------------------
# T3: Gradient finiteness with masking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loss", LOSSES)
def test_gradients_finite_with_mask(loss):
    torch.manual_seed(30)
    rng = np.random.default_rng(30)
    model = BackgroundModelKEN(
        k=6, d_embed=16, d_context=32,
        n_context_layers=1, context_kernel_size=5,
        loss=loss, dropout=0.0,
        dispersion_window_size=64,
    )
    L_out = 512
    L_in = model.calc_input_region_size(L_out)
    B = 2
    x = random_one_hot(rng, B, L_in)
    mask = np.ones((B, L_out), dtype=bool)
    mask[:, 100:150] = False
    y = rng.integers(0, 6, size=(B, N_TRACKS, L_out)).astype(np.float32)
    y *= mask[:, None, :]
    batch = (x, torch.as_tensor(y), torch.as_tensor(mask))
    loss_val = model.training_step(batch, 0)
    assert torch.isfinite(loss_val)
    loss_val.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


# --------------------------------------------------------------------------
# T4: predict_profile interface compatibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loss", LOSSES)
def test_predict_profile_shape_and_normalization(loss):
    torch.manual_seed(70)
    rng = np.random.default_rng(70)
    model = BackgroundModelKEN(
        k=6, d_embed=16, d_context=32,
        n_context_layers=1, context_kernel_size=5,
        loss=loss, dropout=0.0,
        dispersion_window_size=64,
    )
    L_out = 512
    L_in = model.calc_input_region_size(L_out)
    seq = random_one_hot(rng, 1, L_in)[0].numpy()
    out = model.predict_profile(seq)
    assert out["probs"].shape == (N_TRACKS, L_out)
    np.testing.assert_allclose(out["probs"].sum(axis=-1), 1.0, atol=1e-4)

    # With mask
    mask = np.ones(L_out, dtype=bool)
    mask[50:100] = False
    out_m = model.predict_profile(seq, mask=mask)
    assert (out_m["probs"][:, ~mask] == 0).all()
    np.testing.assert_allclose(out_m["probs"][:, mask].sum(axis=-1), 1.0, atol=1e-4)

    if loss == "multinomial":
        assert out["log_dispersion"] is None
    elif loss == "dirichlet_multinomial":
        assert out["log_dispersion"].shape == (N_TRACKS,)
    else:  # nb_offset
        assert out["log_dispersion"].shape == (N_TRACKS, L_out // 64)
    for o in (out, out_m):
        if o["log_dispersion"] is not None:
            assert np.isfinite(o["log_dispersion"]).all()


# --------------------------------------------------------------------------
# T5: Embedding gradient sparsity
# --------------------------------------------------------------------------


def test_embedding_gradients_are_sparse():
    """Only k-mers that appear in the batch should receive gradients."""
    torch.manual_seed(50)
    model = BackgroundModelKEN(
        k=6, d_embed=16, d_context=0,
        n_context_layers=0, loss="multinomial", dropout=0.0,
    )
    L_out = 64
    L_in = model.calc_input_region_size(L_out)
    # Constant sequence: all A's -> only one k-mer (AAAAAA = index 0)
    x = torch.zeros(1, 4, L_in)
    x[0, 0, :] = 1.0  # all A
    y = torch.ones(1, N_TRACKS, L_out)
    mask = torch.ones(1, L_out, dtype=torch.bool)
    loss = model.training_step((x, y, mask), 0)
    loss.backward()
    grad = model.embed.weight.grad
    assert grad is not None
    # Only the canonical entry for AAAAAA (index 0) should have nonzero gradient
    assert grad[0].abs().sum() > 0
    assert grad[1:].abs().sum() == 0


# --------------------------------------------------------------------------
# T6: Checkpoint save/load roundtrip
# --------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(60)
    model = BackgroundModelKEN(
        k=6, d_embed=16, d_context=32,
        n_context_layers=1, context_kernel_size=5,
        loss="multinomial",
    )
    ckpt = tmp_path / "test.ckpt"
    torch.save(model.state_dict(), ckpt)
    model2 = BackgroundModelKEN(
        k=6, d_embed=16, d_context=32,
        n_context_layers=1, context_kernel_size=5,
        loss="multinomial",
    )
    model2.load_state_dict(torch.load(ckpt, weights_only=True))
    # Verify same output
    rng = np.random.default_rng(0)
    x = random_one_hot(rng, 1, model.calc_input_region_size(128))
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = model2(x)
    torch.testing.assert_close(out1, out2)


# --------------------------------------------------------------------------
# T7: Parameter count verification
# --------------------------------------------------------------------------


def test_parameter_count():
    model = BackgroundModelKEN(
        k=6, d_embed=64, d_context=128,
        n_context_layers=2, context_kernel_size=15,
        loss="multinomial",
    )
    total = sum(p.numel() for p in model.parameters())
    # 2080*64 + (64*128*15+128) + (128*128*15+128) + (128*12+12) = ~503K
    assert 480_000 < total < 530_000, f"unexpected param count: {total}"


# --------------------------------------------------------------------------
# T8: RC weight tying correctness
# --------------------------------------------------------------------------


def test_rc_weight_tying():
    """RC partner k-mers must map to the same embedding row."""
    torch.manual_seed(80)
    model = BackgroundModelKEN(
        k=6, d_embed=16, d_context=0,
        n_context_layers=0, loss="multinomial", dropout=0.0,
    )
    rc_perm = rc_kmer_permutation(6)
    to_canonical = model._to_canonical.numpy()
    for i in range(4096):
        assert to_canonical[i] == to_canonical[rc_perm[i]], \
            f"k-mer {i} and its RC {rc_perm[i]} have different canonical indices"
    assert model.embed.num_embeddings == 2080


def test_rc_kmer_permutation_is_involution():
    """RC permutation applied twice should give the identity."""
    perm = rc_kmer_permutation(6)
    np.testing.assert_array_equal(perm[perm], np.arange(4096))


def test_rc_kmer_permutation_known_pair():
    """Verify a known RC pair: ACGTAC (idx 433) <-> GTACGT."""
    perm = rc_kmer_permutation(6)
    # ACGTAC = [0,1,2,3,0,1] -> RC = [2,3,0,1,2,3] = GTACGT
    # GTACGT = 2*4^5 + 3*4^4 + 0*4^3 + 1*4^2 + 2*4^1 + 3*4^0
    #        = 2048 + 768 + 0 + 16 + 8 + 3 = 2843
    assert perm[433] == 2843
    assert perm[2843] == 433


def test_rc_kmer_permutation_various_k():
    """RC permutation is an involution for k=4..8."""
    for k in [4, 5, 6, 7, 8]:
        n = 4 ** k
        perm = rc_kmer_permutation(k)
        assert len(perm) == n
        np.testing.assert_array_equal(perm[perm], np.arange(n))
