import math

import torch

from adamw import AdamW, clip_grad_global_norm, warmup_cosine_lr

torch.manual_seed(0)


def make_twin_params():
    base = [torch.randn(4, 3), torch.randn(7)]
    a = [p.clone().requires_grad_(True) for p in base]
    b = [p.clone().requires_grad_(True) for p in base]
    return a, b


def test_matches_torch_adamw_exactly():
    mine_p, torch_p = make_twin_params()
    mine = AdamW(mine_p, lr=3e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    ref = torch.optim.AdamW(torch_p, lr=3e-3, betas=(0.9, 0.99), eps=1e-8,
                            weight_decay=0.05)
    for step in range(25):
        torch.manual_seed(step)
        grads = [torch.randn_like(p) for p in mine_p]
        for p, q, g in zip(mine_p, torch_p, grads):
            p.grad, q.grad = g.clone(), g.clone()
        mine.step()
        ref.step()
    for p, q in zip(mine_p, torch_p):
        assert torch.allclose(p, q, atol=1e-6), "diverged from torch.optim.AdamW"


def test_bias_correction_matters_at_step_one():
    # Without correction the first step would be ~ lr*(1-b1)*g / sqrt((1-b2) g^2)
    # ~ 0.03*lr for default betas; with correction it's ~ lr * sign(g).
    p = torch.ones(1, requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.0)
    p.grad = torch.tensor([2.0])
    opt.step()
    step_size = (1.0 - p.item())
    assert abs(step_size - 0.1) < 1e-3     # ~ lr, not ~ 0.003


def test_decoupled_decay_bypasses_moments():
    # With zero gradient, AdamW must still decay the parameter (decoupled),
    # and the moments must remain exactly zero.
    p = torch.tensor([10.0], requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(1)
    opt.step()
    assert abs(p.item() - 10.0 * (1 - 0.1 * 0.5)) < 1e-6
    assert opt.m[0].abs().sum() == 0 and opt.v[0].abs().sum() == 0


def test_adam_step_size_is_scale_invariant():
    # Rescaling the gradient by 100x barely changes the step: Adam divides
    # by sqrt(v), which carries the same scale.
    steps = []
    for scale in [1.0, 100.0]:
        p = torch.zeros(1, requires_grad=True)
        opt = AdamW([p], lr=0.1, weight_decay=0.0)
        p.grad = torch.tensor([scale])
        opt.step()
        steps.append(-p.item())
    assert abs(steps[0] - steps[1]) < 1e-6


def test_global_norm_clipping_preserves_direction():
    a = torch.zeros(3, requires_grad=True); a.grad = torch.tensor([3.0, 0.0, 0.0])
    b = torch.zeros(1, requires_grad=True); b.grad = torch.tensor([4.0])
    pre = clip_grad_global_norm([a, b], max_norm=1.0)
    assert abs(pre - 5.0) < 1e-6
    # All grads scaled by the SAME factor 1/5 -> direction preserved.
    assert torch.allclose(a.grad, torch.tensor([0.6, 0.0, 0.0]))
    assert torch.allclose(b.grad, torch.tensor([0.8]))


def test_clipping_is_noop_under_threshold():
    a = torch.zeros(2, requires_grad=True); a.grad = torch.tensor([0.3, 0.4])
    clip_grad_global_norm([a], max_norm=1.0)
    assert torch.allclose(a.grad, torch.tensor([0.3, 0.4]))


def test_schedule_shape():
    total, warmup, base, floor = 1000, 100, 1e-3, 1e-5
    lrs = [warmup_cosine_lr(s, total, warmup, base, floor) for s in range(total)]
    assert abs(lrs[warmup - 1] - base) < 1e-9            # warmup peaks at base
    assert all(x < y + 1e-12 for x, y in zip(lrs[:warmup - 1], lrs[1:warmup]))
    assert all(x >= y - 1e-12 for x, y in zip(lrs[warmup:], lrs[warmup + 1:]))
    assert abs(lrs[-1] - floor) < base * 1e-4            # ends near min_lr
    mid = warmup + (total - warmup) // 2                 # cosine midpoint
    assert abs(lrs[mid] - (base + floor) / 2) < base * 0.01


def test_optimizes_a_quadratic():
    p = torch.tensor([5.0, -3.0], requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.0)
    for _ in range(500):
        loss = ((p - torch.tensor([1.0, 2.0])) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.allclose(p.detach(), torch.tensor([1.0, 2.0]), atol=1e-3)
