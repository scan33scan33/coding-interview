import torch
import torch.nn.functional as F

from moe import MoELayer, load_balancing_loss

torch.manual_seed(0)


def test_identical_experts_reduce_to_single_expert():
    # If every expert computes the same function, the renormalized mixture
    # must equal that function's output exactly, for any routing.
    moe = MoELayer(dim=16, n_experts=4, k=2)
    for e in moe.experts[1:]:
        e.load_state_dict(moe.experts[0].state_dict())
    x = torch.randn(10, 16)
    out, _, dropped = moe(x)
    assert dropped == 0
    assert torch.allclose(out, moe.experts[0](x), atol=1e-5)


def test_top1_routes_to_argmax_expert():
    moe = MoELayer(dim=16, n_experts=4, k=1)
    x = torch.randn(6, 16)
    gates = F.softmax(moe.router(x), dim=-1)
    out, _, _ = moe(x)
    for i in range(6):
        e = int(gates[i].argmax())
        want = moe.experts[e](x[i:i + 1])
        assert torch.allclose(out[i], want[0], atol=1e-5)


def test_mixture_weights_are_renormalized():
    # k=2 weights must sum to 1 per token: scaling both chosen experts'
    # outputs by the raw softmax gates (which sum < 1) is the classic bug.
    moe = MoELayer(dim=16, n_experts=8, k=2)
    for e in moe.experts[1:]:
        e.load_state_dict(moe.experts[0].state_dict())
    x = torch.randn(5, 16)
    out, _, _ = moe(x)
    # With equal experts, output == expert(x) iff weights sum to exactly 1.
    assert torch.allclose(out, moe.experts[0](x), atol=1e-5)


def test_aux_loss_minimized_at_uniform_routing():
    n, e = 1000, 8
    uniform_gates = torch.full((n, e), 1 / e)
    uniform_top1 = torch.arange(n).remainder(e).unsqueeze(1)
    balanced = load_balancing_loss(uniform_gates, uniform_top1)
    assert abs(balanced.item() - 1.0) < 1e-5     # E * sum(1/e * 1/e) * e = 1

    skewed_gates = torch.zeros(n, e)
    skewed_gates[:, 0] = 0.9
    skewed_gates[:, 1:] = 0.1 / (e - 1)
    skewed_top1 = torch.zeros(n, 1, dtype=torch.long)
    assert load_balancing_loss(skewed_gates, skewed_top1) > balanced


def test_aux_loss_gradient_reaches_router():
    moe = MoELayer(dim=16, n_experts=4, k=2)
    x = torch.randn(20, 16)
    _, aux, _ = moe(x)
    aux.backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0


def test_capacity_drops_overflow():
    moe = MoELayer(dim=8, n_experts=4, k=1, capacity_factor=1.0)
    # Force all tokens to expert 0: rig the router row and keep inputs
    # positive so expert 0's logit is always the largest.
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] += 10.0
    x = torch.randn(16, 8).abs()
    out, _, dropped = moe(x)
    cap = 4                                     # ceil(1.0 * 16 * 1 / 4)
    assert dropped == 16 - cap
    # Dropped tokens got no expert output at all (residual handles them).
    assert (out[cap:].abs().sum(dim=-1) == 0).all()


def test_no_capacity_no_drops():
    moe = MoELayer(dim=8, n_experts=2, k=2)
    _, _, dropped = moe(torch.randn(64, 8))
    assert dropped == 0


def test_end_to_end_trains():
    moe = MoELayer(dim=8, n_experts=4, k=2)
    opt = torch.optim.Adam(moe.parameters(), lr=1e-2)
    x = torch.randn(128, 8)
    y = torch.tanh(x) * 2
    first = None
    for _ in range(150):
        out, aux, _ = moe(x)
        loss = F.mse_loss(out, y) + 0.01 * aux
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.3
