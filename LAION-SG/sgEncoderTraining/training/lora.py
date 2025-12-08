"""
Minimal LoRA (Low-Rank Adaptation) for UNet cross-attention layers.

This module provides a lightweight LoRA implementation that can be injected into
the cross-attention layers of Stable Diffusion's UNet to allow fine-tuning
on new conditioning signals (like constraint tokens) without training the full model.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer that wraps a Linear layer.
    
    LoRA decomposes the weight update as: W' = W + BA
    where B is (out_features, rank) and A is (rank, in_features).
    This allows training only 2*rank*features parameters instead of features^2.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize: A with normal, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze original layer
        self.original_layer.requires_grad_(False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output + scaled LoRA output
        original_out = self.original_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return original_out + self.scaling * lora_out


def inject_lora_into_unet(
    unet: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> List[LoRALayer]:
    """
    Inject LoRA layers into UNet's cross-attention projections.
    
    By default, targets: to_q, to_k, to_v in cross-attention blocks.
    These are the layers that process the conditioning embeddings from the sgEncoder.
    
    Args:
        unet: The UNet2DConditionModel from diffusers
        rank: LoRA rank (lower = fewer params, default 4 is minimal)
        alpha: LoRA scaling factor
        dropout: Dropout for LoRA layers
        target_modules: List of module name patterns to target (default: cross-attn projections)
    
    Returns:
        List of injected LoRA layers (for accessing their parameters)
    """
    if target_modules is None:
        # Target cross-attention Q/K/V projections
        # In SDXL UNet, these are in attention blocks as to_q, to_k, to_v
        target_modules = ["to_q", "to_k", "to_v"]
    
    lora_layers = []
    modules_replaced = 0
    
    for name, module in unet.named_modules():
        # Only target attention modules (skip self-attention by checking for "attn2")
        # attn1 = self-attention, attn2 = cross-attention
        if "attn2" not in name:
            continue
            
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = unet.get_submodule(parent_name)
                
                # Create LoRA wrapper
                lora_layer = LoRALayer(
                    original_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                
                # Replace module
                setattr(parent, child_name, lora_layer)
                lora_layers.append(lora_layer)
                modules_replaced += 1
                logging.debug(f"Injected LoRA into: {name}")
    
    logging.info(f"Injected LoRA into {modules_replaced} cross-attention layers (rank={rank}, alpha={alpha})")
    return lora_layers


def get_lora_params(lora_layers: List[LoRALayer]) -> List[nn.Parameter]:
    """Get all trainable parameters from LoRA layers."""
    params = []
    for layer in lora_layers:
        params.extend(layer.lora_A.parameters())
        params.extend(layer.lora_B.parameters())
    return params


def save_lora_weights(lora_layers: List[LoRALayer], path: str):
    """Save only the LoRA weights to a file."""
    state_dict = {}
    for i, layer in enumerate(lora_layers):
        state_dict[f"lora_{i}_A"] = layer.lora_A.state_dict()
        state_dict[f"lora_{i}_B"] = layer.lora_B.state_dict()
    torch.save(state_dict, path)
    logging.info(f"Saved LoRA weights to {path}")


def load_lora_weights(lora_layers: List[LoRALayer], path: str, device: torch.device):
    """Load LoRA weights from a file."""
    state_dict = torch.load(path, map_location=device)
    for i, layer in enumerate(lora_layers):
        layer.lora_A.load_state_dict(state_dict[f"lora_{i}_A"])
        layer.lora_B.load_state_dict(state_dict[f"lora_{i}_B"])
    logging.info(f"Loaded LoRA weights from {path}")

