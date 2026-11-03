"""Minimal LoRA for frozen nn.Linear layers."""

from __future__ import annotations

import math

import torch
from torch import nn

# Hugging Face module name -> GGUF tensor name (llama.cpp convention).
PROJ_TO_GGUF = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        d = self.dropout(x).to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        return y + (self.scale * d).to(y.dtype)


def inject(layers: nn.ModuleList, layer_ids: list[int], rank: int, alpha: float, dropout: float = 0.0) -> dict[tuple[int, str], LoRALinear]:
    """Wrap the attention and MLP projections of the chosen layers."""
    out = {}
    for i in layer_ids:
        layer = layers[i]
        for parent in (layer.self_attn, layer.mlp):
            for name in PROJ_TO_GGUF:
                mod = getattr(parent, name, None)
                if isinstance(mod, nn.Linear):
                    wrapped = LoRALinear(mod, rank, alpha, dropout).to(mod.weight.device)
                    setattr(parent, name, wrapped)
                    out[(i, name)] = wrapped
    if not out:
        raise ValueError("no projection layers found to adapt")
    return out
