# Copyright (c) Kyutai, all rights reserved.
# Derived from kyutai-labs/moshi and moshi-finetune; see NOTICE and LICENSES/.

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = x W^T + scaling * x A^T B^T; only A and B are learned."""

    def __init__(self, base: nn.Linear, rank: int, scaling: float) -> None:
        super().__init__()
        if rank < 1 or scaling <= 0:
            raise ValueError("LoRA rank and scaling must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = scaling
        self.frozen_W = base
        self.lora_A = nn.Linear(
            self.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype
        )
        self.lora_B = nn.Linear(
            rank, self.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.frozen_W.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frozen_W(x) + self.scaling * self.lora_B(self.lora_A(x))

    def merge_weight(self) -> torch.Tensor:
        return self.frozen_W.weight + self.scaling * (self.lora_B.weight @ self.lora_A.weight)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    scaling: float = 2.0,
    finetune_embeddings: bool = False,
) -> int:
    """Replace every Linear in Moshi, freeze the base, and return adapter count."""

    def replace(module: nn.Module) -> int:
        count = 0
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, scaling))
                count += 1
            else:
                count += replace(child)
        return count

    count = replace(model)
    if count == 0:
        raise ValueError("no Linear layers found for LoRA")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for parameter in module.lora_A.parameters():
                parameter.requires_grad_(True)
            for parameter in module.lora_B.parameters():
                parameter.requires_grad_(True)
    if finetune_embeddings:
        for name, parameter in model.named_parameters():
            if "emb" in name:
                parameter.requires_grad_(True)
    return count


def merge_lora(model: nn.Module) -> None:
    """Merge adapters into ordinary Linear layers for a standalone checkpoint."""

    def merge(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                merged = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.frozen_W.bias is not None,
                    device=child.frozen_W.weight.device,
                    dtype=child.frozen_W.weight.dtype,
                )
                with torch.no_grad():
                    merged.weight.copy_(child.merge_weight())
                    if child.frozen_W.bias is not None:
                        merged.bias.copy_(child.frozen_W.bias)
                merged.weight.requires_grad_(False)
                if merged.bias is not None:
                    merged.bias.requires_grad_(False)
                setattr(module, name, merged)
            else:
                merge(child)

    merge(model)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key in trainable
    }
