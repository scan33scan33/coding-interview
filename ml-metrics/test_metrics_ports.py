"""Parametrized suite for the PyTorch and JAX metric ports."""

import numpy as np
import pytest

rng = np.random.default_rng(0)


@pytest.fixture(params=["torch", "jax"])
def impl(request):
    if request.param == "torch":
        import torch
        import metrics_torch as m
        return m, lambda a: torch.tensor(np.asarray(a))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import metrics_jax as m
    return m, jnp.asarray


def test_auc_anchors(impl):
    m, to = impl
    y = np.array([0, 0, 1, 1])
    assert m.roc_auc(to(y), to([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert m.roc_auc(to(y), to([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_equals_pairwise_probability_with_ties(impl):
    m, to = impl
    y = rng.integers(0, 2, 200)
    s = np.round(rng.uniform(size=200), 1)
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    want = (wins + 0.5 * ties) / (len(pos) * len(neg))
    assert abs(m.roc_auc(to(y), to(s)) - want) < 1e-9


def test_all_ties_is_half(impl):
    m, to = impl
    assert m.roc_auc(to(np.array([0, 1, 0, 1, 1])), to(np.ones(5))) == 0.5


def test_ap_hand_case(impl):
    m, to = impl
    y = np.array([1, 0, 1])
    s = np.array([1.0, 0.8, 0.6])
    assert abs(m.average_precision(to(y), to(s)) - (0.5 + 1 / 3)) < 1e-9


def test_ece_anchors(impl):
    m, to = impl
    probs = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < probs).astype(float)
    assert m.expected_calibration_error(to(y), to(probs)) < 0.02

    over = np.full(2000, 0.95)
    y2 = (rng.uniform(size=2000) < 0.6).astype(float)
    assert abs(m.expected_calibration_error(to(y2), to(over)) - 0.35) < 0.05


def test_matches_numpy_reference(impl):
    m, to = impl
    from metrics import average_precision as np_ap, roc_auc as np_auc
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500), 2)
    assert abs(m.roc_auc(to(y), to(s)) - np_auc(y, s)) < 1e-9
    assert abs(m.average_precision(to(y), to(s)) - np_ap(y, s)) < 1e-9
