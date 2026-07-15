"""AdamW from scratch (+ global-norm gradient clipping and a warmup-cosine
learning-rate schedule). The correctness bar: match torch.optim.AdamW
step-for-step to float precision.

Kingma & Ba 2015 (Adam); Loshchilov & Hutter 2019 (decoupled weight decay).
"""

import math

import torch


class AdamW:
    """m, v moments with bias correction; DECOUPLED weight decay.

        m = b1*m + (1-b1)*g            v = b2*v + (1-b2)*g^2
        m_hat = m/(1-b1^t)             v_hat = v/(1-b2^t)
        p *= 1 - lr*wd                 # decay first, matching torch.optim
        p -= lr * m_hat/(sqrt(v_hat)+eps)

    Decoupled means the decay term multiplies the PARAMETER and bypasses the
    moment estimates entirely — L2-in-the-gradient (classic Adam + L2) gets
    rescaled by 1/sqrt(v_hat) and stops acting like weight decay for
    parameters with large gradient history.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = list(params)
        self.lr, (self.b1, self.b2) = lr, betas
        self.eps, self.wd = eps, weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t     # bias corrections: the moments start
        bc2 = 1 - self.b2 ** self.t     # at 0 and are EMA-shrunk early on
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)         # decoupled: not through m,v
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


@torch.no_grad()
def clip_grad_global_norm(params, max_norm):
    """Scale ALL gradients by one factor so the global L2 norm <= max_norm.

    Per-parameter clipping changes the gradient DIRECTION; global-norm
    clipping preserves it. Returns the pre-clip norm (the number you log).
    """
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    scale = max_norm / (total + 1e-12)
    if scale < 1.0:
        for g in grads:
            g.mul_(scale)
    return float(total)


def warmup_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    """Linear warmup from 0 to base_lr, then cosine decay to min_lr.

    The standard LLM pretraining schedule. Warmup matters specifically for
    Adam: early steps have tiny bias-corrected v estimates from few samples,
    so the effective step is noisy; a low lr rides that out.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
