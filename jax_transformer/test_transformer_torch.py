import torch

from transformer_torch import TransformerClassifier, accuracy, train_epochs

torch.manual_seed(0)


def make_separable_data(n=256, seq_len=4, d_model=112, n_classes=10):
    """Class identity is encoded in the FIRST token region so first-token
    pooling can read it after attention mixes the sequence."""
    y = torch.randint(0, n_classes, (n,))
    x = torch.randn(n, seq_len * d_model) * 0.5
    for c in range(n_classes):
        x[y == c, c * 8:(c + 1) * 8] += 3.0        # class signature
    return x, y


def test_forward_shapes():
    model = TransformerClassifier()
    out = model(torch.randn(5, 4 * 112))
    assert out.shape == (5, 10)


def test_permutation_of_later_tokens_only_changes_via_attention():
    # Encoder attention is permutation-equivariant over tokens (no positional
    # embeddings in this architecture): permuting non-pooled tokens must not
    # change the first-token output.
    model = TransformerClassifier().eval()
    x = torch.randn(1, 4 * 112)
    x_tok = x.view(1, 4, 112)
    perm = x_tok[:, [0, 2, 3, 1]].reshape(1, -1)
    with torch.no_grad():
        a, b = model(x), model(perm)
    assert torch.allclose(a, b, atol=1e-5)


def test_overfits_separable_data():
    x, y = make_separable_data()
    model = TransformerClassifier()
    train_epochs(model, x, y, epochs=20)
    assert accuracy(model, x, y) > 0.95


def test_matches_jax_layer_norm_semantics():
    # Post-norm: output of every block is LayerNorm-ed => per-token feature
    # mean ~ 0 and std ~ 1 (gamma=1, beta=0 at init).
    model = TransformerClassifier(n_layers=1).eval()
    x_BLD = torch.randn(3, 4, 112)
    with torch.no_grad():
        out = model.blocks[0](x_BLD)
    assert out.mean(-1).abs().max() < 1e-5
    assert (out.std(-1, unbiased=False) - 1).abs().max() < 1e-2
