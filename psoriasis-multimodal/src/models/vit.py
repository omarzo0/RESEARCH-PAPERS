"""
ViT-B/16 classifier — comparison model with LoRA fine-tuning.
"""

import torch
import torch.nn as nn
import timm

from src.models.lora import apply_lora_to_vit


class ViTB16Classifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) with LoRA adapters for psoriasis classification.

    Architecture:
        - ViT-B/16 backbone (ImageNet-21k pretrained via timm)
        - LoRA adapters on Q and V attention matrices (rank=8)
        - Classification head: FC(768→256) → ReLU → Dropout(0.3) → FC(256→2)

    Args:
        config: Full config dict (vit section is extracted).
        num_classes: Number of output classes.
    """

    def __init__(self, config: dict, num_classes: int = 2):
        super().__init__()
        model_cfg = config["models"]["vit"]

        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_cfg.get("name", "vit_base_patch16_224"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=0,  # Remove original head
        )

        # Apply LoRA adapters
        lora_rank = model_cfg.get("lora_rank", 8)
        lora_alpha = model_cfg.get("lora_alpha", 16)
        self.backbone = apply_lora_to_vit(
            self.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=["attn.qkv"],
        )

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        # Classification head
        head_dims = model_cfg.get("head_dims", [feat_dim, 256, num_classes])
        dropout = model_cfg.get("dropout", 0.3)

        self.head = nn.Sequential(
            nn.Linear(head_dims[0] if head_dims[0] != num_classes else feat_dim, head_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dims[1], head_dims[2] if len(head_dims) > 2 else num_classes),
        )

        # Unfreeze head parameters
        for param in self.head.parameters():
            param.requires_grad = True

        # Hooks for attention weight extraction
        self._attention_weights = []
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights from all layers."""
        self._hooks = []
        for name, module in self.backbone.named_modules():
            if "attn_drop" in name or (
                hasattr(module, "num_heads") and isinstance(module, nn.Module)
                and "attn" in name and "." not in name.split("blocks.")[-1].split("attn")[0]
            ):
                # Hook into attention modules
                pass

        # Alternative: use timm's built-in attention extraction
        # This works for most timm ViT models
        for block in self.backbone.blocks:
            if hasattr(block, "attn"):
                hook = block.attn.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)

    def _attention_hook(self, module, input, output):
        """Capture attention weights during forward pass."""
        # For timm ViTs, attention weights can be accessed via attn.attn_drop
        # We need to compute them manually from the QKV output
        pass  # Attention rollout will compute this separately

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        self._attention_weights = []
        features = self.backbone(x)  # (B, feat_dim)
        logits = self.head(features)
        return logits

    def get_attention_maps(self, x):
        """
        Extract attention maps from all transformer layers.

        Used for attention rollout visualization.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            list: List of attention weight tensors, one per layer.
                  Each has shape (B, num_heads, seq_len, seq_len).
        """
        attention_maps = []
        B = x.shape[0]

        # Forward through patch embedding
        x_tok = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x_tok = torch.cat((cls_token, x_tok), dim=1)
        x_tok = x_tok + self.backbone.pos_embed
        x_tok = self.backbone.pos_drop(x_tok)

        # Forward through each transformer block, capturing attention
        for block in self.backbone.blocks:
            # Compute attention weights manually
            attn = block.attn
            qkv = attn.qkv(block.norm1(x_tok) if hasattr(block, 'norm1') else x_tok)
            B_curr, N, C3 = qkv.shape
            qkv = qkv.reshape(B_curr, N, 3, attn.num_heads, C3 // (3 * attn.num_heads))
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            scale = (C3 // (3 * attn.num_heads)) ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            attention_maps.append(attn_weights.detach())

            # Complete the block forward pass
            x_tok = block(x_tok)

        return attention_maps
