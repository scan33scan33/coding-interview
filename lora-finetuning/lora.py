import math

import torch
from torch import nn

# Shape suffix convention:
#   B = batch, I = in features, O = out features, R = lora rank
# A tensor's name tells you its shape. If an op's output name doesn't
# follow from its input names, the op is probably wrong.


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update.

        y = W x + (alpha / r) * B (A x)

    A_RI is Kaiming-initialized, B_OR is initialized to ZERO, so at step 0
    the wrapped layer computes exactly what the base layer did — fine-tuning
    starts from the pretrained function, not a perturbed one.
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r, self.scaling = r, alpha / r
        self.lora_A_RI = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B_OR = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A_RI, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x_BI):
        y_BO = self.base(x_BI)
        if not self.merged:
            y_BO = y_BO + self.dropout(x_BI) @ self.lora_A_RI.T @ self.lora_B_OR.T * self.scaling
        return y_BO

    @torch.no_grad()
    def merge(self):
        """Fold BA into the base weight: zero inference overhead. Idempotent."""
        if not self.merged:
            self.base.weight += self.lora_B_OR @ self.lora_A_RI * self.scaling
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Exact inverse of merge() — needed to swap adapters on one base model."""
        if self.merged:
            self.base.weight -= self.lora_B_OR @ self.lora_A_RI * self.scaling
            self.merged = False


def add_lora(model, r=8, alpha=16, target_names=("q_proj", "v_proj")):
    """Replace matching nn.Linear submodules with LoRALinear wrappers, in place.

    Classic LoRA recipe: adapt attention q/v projections only. Iterates over
    named_modules and swaps via the parent, since modules can't replace
    themselves.
    """
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model


def lora_state_dict(model):
    """Only the adapter weights — this is what you ship (MBs, not GBs)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def mark_only_lora_trainable(model):
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
