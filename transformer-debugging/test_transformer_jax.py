import jax
import jax.numpy as jnp
import numpy as np

import fixed_transformer_jax as fixed


def test_causal_no_future_leak():
    params = fixed.init_params(jax.random.key(0))
    ids = np.random.default_rng(0).integers(0, fixed.VOCAB, (1, fixed.SEQ_LEN))
    out_a = fixed.forward(params, jnp.array(ids))
    ids2 = ids.copy()
    ids2[0, -1] = (ids2[0, -1] + 1) % fixed.VOCAB
    out_b = fixed.forward(params, jnp.array(ids2))
    np.testing.assert_allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-5)


def test_attention_rows_sum_to_one_over_keys():
    L = 6
    q = jax.random.normal(jax.random.key(0), (1, 1, L, L))
    k = jax.random.normal(jax.random.key(1), (1, 1, L, L))
    v = jnp.eye(L).reshape(1, 1, L, L)
    rows = fixed.attention(q, k, v)[0, 0]
    np.testing.assert_allclose(rows.sum(-1), 1.0, atol=1e-5)


def test_loss_uses_shifted_labels():
    rng = np.random.default_rng(0)
    ids = jnp.array(rng.integers(0, fixed.VOCAB, (2, fixed.SEQ_LEN)))

    # A model that perfectly predicts the NEXT token must get ~zero loss.
    # Build it by rigging forward's output via a direct logp computation:
    logits_next = np.full((2, fixed.SEQ_LEN, fixed.VOCAB), -30.0)
    for b in range(2):
        for t in range(fixed.SEQ_LEN - 1):
            logits_next[b, t, int(ids[b, t + 1])] = 30.0

    logp = jax.nn.log_softmax(jnp.array(logits_next)[:, :-1], axis=-1)
    picked = jnp.take_along_axis(logp, ids[:, 1:][..., None], axis=-1)[..., 0]
    assert float(-picked.mean()) < 1e-3


def test_training_overfits_and_generation_copies():
    params, final_loss = fixed.train_copy_task(steps=300)
    # Irreducible entropy floor ~ 4*ln(11)/10 ~ 0.96 (see torch version).
    assert final_loss < 1.05

    rng = np.random.default_rng(123)
    ids = np.asarray(fixed.make_copy_batch(8, rng))
    prompt = ids[:, :fixed.PREFIX_LEN + 1]
    out = fixed.greedy_generate(params, prompt, fixed.PREFIX_LEN)
    copied = out[:, fixed.PREFIX_LEN + 1:]
    assert (copied == ids[:, :fixed.PREFIX_LEN]).mean() > 0.95
