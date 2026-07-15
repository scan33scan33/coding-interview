import torch

from grpo import grpo_advantages, grpo_loss, k3_kl, masked_mean, ppo_clip_loss

torch.manual_seed(0)


def test_advantages_standardized_per_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    adv = grpo_advantages(rewards, groups)
    for g in [0, 1]:
        m = groups == g
        assert abs(adv[m].mean().item()) < 1e-5
        assert abs(adv[m].std(unbiased=False).item() - 1.0) < 1e-2
    # Higher reward within a group => higher advantage.
    assert adv[2] > adv[1] > adv[0]


def test_group_isolation():
    # A huge reward in group 1 must not change group 0's advantages.
    rewards = torch.tensor([1.0, 2.0, 5.0, 6.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv_a = grpo_advantages(rewards, groups)
    rewards2 = rewards.clone()
    rewards2[2:] += 1000.0
    adv_b = grpo_advantages(rewards2, groups)
    assert torch.allclose(adv_a[:2], adv_b[:2])


def test_gradient_pushes_toward_good_completions():
    logp = torch.zeros(2, 3, requires_grad=True)
    logp_old = torch.zeros(2, 3)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3)
    ppo_clip_loss(logp, logp_old, adv, mask).backward()
    assert (logp.grad[0] < 0).all()   # descent raises logp of good completion
    assert (logp.grad[1] > 0).all()   # ...and lowers the bad one


def test_clipping_zeroes_gradient_outside_trust_region():
    # Positive advantage, ratio already above 1+eps: objective is at the
    # clipped constant, so the gradient must be exactly zero.
    logp = torch.full((1, 4), 0.5, requires_grad=True)   # ratio = e^0.5 ~ 1.65
    logp_old = torch.zeros(1, 4)
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 4)
    ppo_clip_loss(logp, logp_old, adv, mask, clip_eps=0.2).backward()
    assert torch.allclose(logp.grad, torch.zeros_like(logp.grad))

    # But with NEGATIVE advantage the same ratio is unclipped (min picks the
    # unclipped branch) and gradient must be nonzero — clipping is one-sided.
    logp2 = torch.full((1, 4), 0.5, requires_grad=True)
    ppo_clip_loss(logp2, logp_old, torch.tensor([-1.0]), mask).backward()
    assert not torch.allclose(logp2.grad, torch.zeros_like(logp2.grad))


def test_k3_kl_nonnegative_and_zero_at_equality():
    logp = torch.randn(4, 6)
    assert torch.allclose(k3_kl(logp, logp), torch.zeros(4, 6), atol=1e-7)
    logp_ref = logp + torch.randn(4, 6) * 0.5
    assert (k3_kl(logp, logp_ref) >= -1e-7).all()   # nonneg per sample


def test_k3_estimates_true_kl():
    # Two known categoricals: E_{x~p}[k3(logp, logq)] should approach KL(p||q).
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.4, 0.4, 0.2])
    true_kl = (p * (p / q).log()).sum()
    g = torch.Generator().manual_seed(0)
    x = torch.multinomial(p, 200000, replacement=True, generator=g)
    est = k3_kl(p.log()[x], q.log()[x]).mean()
    assert abs(est.item() - true_kl.item()) < 0.01


def test_masked_tokens_do_not_contribute():
    logp = torch.randn(2, 5, requires_grad=True)
    logp_old = logp.detach() + torch.randn(2, 5) * 0.1
    adv = torch.tensor([1.0, -0.5])
    mask = torch.ones(2, 5)
    mask[:, 3:] = 0.0                                  # padding
    ppo_clip_loss(logp, logp_old, adv, mask).backward()
    assert torch.allclose(logp.grad[:, 3:], torch.zeros(2, 2))


def test_full_grpo_loss_runs_and_kl_anchors():
    B, T = 6, 8
    logp = torch.randn(B, T, requires_grad=True)
    logp_old = logp.detach() + torch.randn(B, T) * 0.05
    logp_ref = logp.detach() + torch.randn(B, T) * 0.05
    rewards = torch.randn(B)
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    mask = torch.ones(B, T)
    loss, stats = grpo_loss(logp, logp_old, logp_ref, rewards, groups, mask)
    loss.backward()
    assert logp.grad is not None
    assert stats["kl"] >= 0
