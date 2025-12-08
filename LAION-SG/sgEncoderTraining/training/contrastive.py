"""
Simple CLIP-style contrastive utilities for aligning sgEncoder pooled embeddings
with frozen CLIP text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Tiny MLP to map sgEncoder pooled embeddings to CLIP text dimension."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def info_nce_loss(graph_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07):
    """
    Compute symmetric InfoNCE loss between graph and text embeddings.
    graph_embeds: (B, D)
    text_embeds:  (B, D)
    """
    graph_norm = F.normalize(graph_embeds, dim=-1)
    text_norm = F.normalize(text_embeds, dim=-1)

    logits = torch.matmul(graph_norm, text_norm.t()) / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_g2t = F.cross_entropy(logits, targets)
    loss_t2g = F.cross_entropy(logits.t(), targets)
    return (loss_g2t + loss_t2g) * 0.5

