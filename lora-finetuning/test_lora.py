import torch
from torch import nn

from lora import LoRALinear, add_lora, lora_state_dict, mark_only_lora_trainable

torch.manual_seed(0)


def test_init_is_identity_wrt_base():
    # B == 0 at init, so the wrapped layer must equal the base layer exactly.
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4)
    x_BI = torch.randn(5, 16)
    assert torch.allclose(lora(x_BI), base(x_BI), atol=1e-7)


def test_only_lora_params_get_gradients():
    lora = LoRALinear(nn.Linear(16, 8), r=4)
    lora(torch.randn(3, 16)).sum().backward()
    assert lora.base.weight.grad is None
    assert lora.base.bias.grad is None
    assert lora.lora_A_RI.grad is not None
    assert lora.lora_B_OR.grad is not None


def test_merge_matches_unmerged_forward():
    lora = LoRALinear(nn.Linear(16, 8), r=4)
    # Give B nonzero values so the adapter actually contributes.
    with torch.no_grad():
        lora.lora_B_OR.normal_()
    x_BI = torch.randn(5, 16)
    want = lora(x_BI)
    lora.merge()
    assert torch.allclose(lora(x_BI), want, atol=1e-5)
    lora.merge()  # idempotent: merging twice must not double the update
    assert torch.allclose(lora(x_BI), want, atol=1e-5)


def test_unmerge_restores_base_weights():
    base = nn.Linear(16, 8)
    w0 = base.weight.clone()
    lora = LoRALinear(base, r=4)
    with torch.no_grad():
        lora.lora_B_OR.normal_()
    lora.merge()
    assert not torch.allclose(lora.base.weight, w0)
    lora.unmerge()
    assert torch.allclose(lora.base.weight, w0, atol=1e-6)


class ToyAttentionModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        return self.out(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


def test_add_lora_targets_only_named_modules():
    model = add_lora(ToyAttentionModel(), r=4, target_names=("q_proj", "v_proj"))
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.k_proj, nn.Linear)      # untouched
    assert isinstance(model.out, nn.Linear)


def test_lora_state_dict_is_small():
    model = add_lora(ToyAttentionModel(dim=32), r=4)
    sd = lora_state_dict(model)
    assert set(k.split(".")[-1] for k in sd) == {"lora_A_RI", "lora_B_OR"}
    lora_params = sum(v.numel() for v in sd.values())
    total_params = sum(p.numel() for p in model.state_dict().values())
    assert lora_params < total_params * 0.2


def test_finetuning_moves_adapter_not_base():
    model = add_lora(ToyAttentionModel(), r=4)
    mark_only_lora_trainable(model)
    base_w0 = model.q_proj.base.weight.clone()

    x = torch.randn(64, 32)
    y = x.sum(dim=-1, keepdim=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    first = None
    for _ in range(100):
        loss = nn.functional.mse_loss(model(x), y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.1                       # it learns
    assert torch.equal(model.q_proj.base.weight, base_w0)  # base frozen bit-exact


def test_adapter_swap_via_unmerge():
    # Serve two adapters off one base model: merge A, unmerge, merge B.
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    lora = LoRALinear(base, r=4)
    with torch.no_grad():
        lora.lora_B_OR.normal_()
    out_a = lora(x)

    lora.merge()
    lora.unmerge()
    with torch.no_grad():                                  # "load" adapter B
        lora.lora_A_RI.normal_()
    out_b = lora(x)
    assert not torch.allclose(out_a, out_b)                # different adapter
    assert torch.allclose(lora(x), out_b, atol=1e-6)       # stable after swap
