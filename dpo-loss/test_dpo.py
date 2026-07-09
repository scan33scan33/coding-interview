import math

import torch
from torch import nn

from dpo import dpo_loss, dpo_step, sequence_logprob

torch.manual_seed(0)


def test_sequence_logprob_masks_prompt_and_padding():
    B, L, V = 2, 5, 7
    logits_BLV = torch.randn(B, L, V)
    labels_BL = torch.randint(0, V, (B, L))
    mask_BL = torch.tensor([[0., 0., 1., 1., 0.],   # prompt=2, completion=2, pad=1
                            [0., 1., 1., 1., 1.]])
    got_B = sequence_logprob(logits_BLV, labels_BL, mask_BL)

    logp_BLV = torch.log_softmax(logits_BLV, dim=-1)
    manual_0 = logp_BLV[0, 2, labels_BL[0, 2]] + logp_BLV[0, 3, labels_BL[0, 3]]
    assert torch.allclose(got_B[0], manual_0, atol=1e-6)

    # Changing logits at a masked position must not change the score.
    logits2_BLV = logits_BLV.clone()
    logits2_BLV[0, 0] += 100.0
    assert torch.allclose(got_B, sequence_logprob(logits2_BLV, labels_BL, mask_BL))


def test_loss_is_log2_when_policy_equals_ref():
    # At init, policy == ref, so all log-ratios cancel: loss = -log sigmoid(0) = log 2.
    lp_B = torch.randn(4)
    loss, chosen_r, rejected_r = dpo_loss(lp_B, lp_B - 1.0, lp_B, lp_B - 1.0)
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6)
    assert torch.allclose(chosen_r, torch.zeros(4))
    assert torch.allclose(rejected_r, torch.zeros(4))


def test_gradient_widens_the_margin():
    # A gradient step must push chosen logprob up and rejected logprob down.
    policy_chosen = torch.tensor([0.0], requires_grad=True)
    policy_rejected = torch.tensor([0.0], requires_grad=True)
    ref = torch.tensor([0.0])
    loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref, ref)
    loss.backward()
    assert policy_chosen.grad.item() < 0     # descent increases chosen
    assert policy_rejected.grad.item() > 0   # descent decreases rejected


def test_loss_decreases_as_margin_grows():
    ref = torch.zeros(1)
    losses = []
    for margin in [0.0, 1.0, 5.0]:
        loss, _, _ = dpo_loss(torch.tensor([margin]), torch.tensor([0.0]), ref, ref)
        losses.append(loss.item())
    assert losses[0] > losses[1] > losses[2]


def test_beta_scales_the_implicit_reward():
    pc, pr = torch.tensor([2.0]), torch.tensor([-1.0])
    ref = torch.zeros(1)
    _, cr_lo, _ = dpo_loss(pc, pr, ref, ref, beta=0.1)
    _, cr_hi, _ = dpo_loss(pc, pr, ref, ref, beta=0.5)
    assert torch.allclose(cr_hi, 5 * cr_lo)


def test_label_smoothing_softens_confident_margins():
    ref = torch.zeros(1)
    big = torch.tensor([200.0])  # beta=0.1 -> margin 20, sigmoid fully saturated
    loss_plain, _, _ = dpo_loss(big, torch.zeros(1), ref, ref, label_smoothing=0.0)
    loss_smooth, _, _ = dpo_loss(big, torch.zeros(1), ref, ref, label_smoothing=0.1)
    assert loss_plain.item() < 1e-4          # saturated: nothing left to learn
    assert loss_smooth.item() > loss_plain.item()  # smoothing keeps pressure on


class TinyLM(nn.Module):
    def __init__(self, vocab=11, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids_BL):
        return self.head(self.emb(ids_BL))


def test_dpo_training_learns_the_preference():
    # End to end: after a few steps the policy should prefer the chosen
    # completion by a growing margin, while the reference stays fixed.
    policy, ref = TinyLM(), TinyLM()
    ref.load_state_dict(policy.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    batch = {
        "chosen_ids": torch.tensor([[1, 2, 3, 4]]),
        "chosen_mask": torch.tensor([[0., 0., 1., 1.]]),
        "rejected_ids": torch.tensor([[1, 2, 5, 6]]),
        "rejected_mask": torch.tensor([[0., 0., 1., 1.]]),
    }
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    first_loss = None
    for _ in range(50):
        loss, chosen_r, rejected_r = dpo_step(policy, ref, batch, beta=0.1)
        if first_loss is None:
            first_loss = loss.item()
            assert abs(first_loss - math.log(2.0)) < 1e-5  # policy == ref at init
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss
    assert (chosen_r > rejected_r).all()
