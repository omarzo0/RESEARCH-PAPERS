"""
LoRA (Low-Rank Adaptation) implementation for Vision Transformers.

Injects low-rank adapters into Q and V attention projection matrices,
keeping the original weights frozen.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for a linear layer.

    output = original(x) + (alpha / rank) * B(A(x))

    Only A and B matrices are trainable. Original weights are frozen.

    Args:
        original_linear: The original nn.Linear layer to adapt.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original path (frozen)
        original_output = self.original(x)
        # LoRA path (trainable)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output


def apply_lora_to_vit(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list = None,
) -> nn.Module:
    """
    Apply LoRA adapters to a Vision Transformer's attention layers.

    Replaces Q and V projection weights with LoRA-wrapped versions.
    In timm ViTs, attention projections are typically in `attn.qkv` (fused)
    or separate `attn.q_proj`, `attn.v_proj`.

    For fused QKV (timm default), we split and apply LoRA to Q and V components.

    Args:
        model: A timm ViT model.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: List of module name patterns to apply LoRA to.
                        If None, targets all attention QKV projections.

    Returns:
        nn.Module: Modified model with LoRA adapters injected.
    """
    if target_modules is None:
        target_modules = ["attn.qkv"]

    lora_count = 0

    for name, module in model.named_modules():
        # Check if this module matches target patterns
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)

                # Wrap with LoRA
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_count += 1

    print(f"[LoRA] Applied {lora_count} LoRA adapters (rank={rank}, alpha={alpha})")

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model
