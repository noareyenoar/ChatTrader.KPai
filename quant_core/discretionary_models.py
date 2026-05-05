"""Discretionary (Multimodal) model architectures — Phase 4.

Three architectures for chart-pattern / sentiment-driven direction prediction:

  Disc_ViT_v1          — Vision Transformer on rasterized candlestick images.
  Disc_Multimodal_v1   — CNN chart encoder + momentum feature encoder, fused.
  Disc_CNNChart_v1     — Lightweight ResNet-style CNN on chart images.

Input shapes:
  ViT / CNNChart:   [Batch, Channels=4, H=32, W=32]  (rasterized chart)
  Multimodal:       image [Batch, 4, 32, 32]  +  tabular [Batch, Num_Tab_Features]

Target:       3-class logits (0=down, 1=flat, 2=up)
Loss:         CrossEntropyLoss
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .interfaces import TrendModelInterface


# ---------------------------------------------------------------------------
# Model 1: Vision Transformer (patch-based)
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class _ViTBlock(nn.Module):
    """Minimal ViT block using _ManualMHA pattern (DirectML-safe)."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        nh, dh = self.nhead, self.d_head
        Q = self.q(x).view(B, N, nh, dh).transpose(1, 2)
        K = self.k(x).view(B, N, nh, dh).transpose(1, 2)
        V = self.v(x).view(B, N, nh, dh).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, C)
        return self.attn_out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self._attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DiscretionaryViT(TrendModelInterface):
    """Vision Transformer for candlestick chart image classification.

    Splits the 32×32 chart image into non-overlapping 4×4 patches,
    linearly embeds each patch, adds learnable positional embeddings,
    then applies ViT encoder blocks.  A [CLS] token aggregates the
    global chart representation for the classification head.

    Architecture:
        Image (B,4,32,32) → PatchEmbed (B,64,embed)
                           → [CLS] token prepend → pos embed
                           → [ViTBlock × num_layers]
                           → CLS token → LayerNorm → 3 logits

    Best suited for: detecting classic chart patterns (head-and-shoulders,
    double-bottom, wedges) that human traders recognize visually.
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 4,
        embed_dim: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([_ViTBlock(embed_dim, nhead, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 3)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        patches = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, patches], dim=1) + self.pos_embed  # (B, N+1, D)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, 0]))                   # CLS token → 3 logits


# ---------------------------------------------------------------------------
# Model 2: Multimodal Fusion (CNN image + tabular momentum)
# ---------------------------------------------------------------------------

class _MiniCNN(nn.Module):
    def __init__(self, in_chans: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscretionaryMultimodal(TrendModelInterface):
    """Multimodal model fusing visual chart patterns and tabular momentum.

    Combines two encoding streams:
      1. CNN image encoder: extracts local chart pattern embeddings.
      2. Tabular MLP encoder: processes momentum/sentiment numeric features.

    The two embeddings are concatenated and passed through a fusion MLP.

    Architecture:
        Image (B,4,32,32) → _MiniCNN → img_emb (B,64)
        Tabular (B,F) → Linear → Mish → tab_emb (B,64)
        Concat → LayerNorm → Linear(→3)

    Best suited for: combining price-action context (chart) with
    quantitative momentum signals in a single prediction.
    """
    def __init__(self, tab_input_dim: int, img_embed: int = 64, tab_embed: int = 64, dropout: float = 0.1):
        super().__init__()
        self.img_encoder = _MiniCNN(in_chans=4, out_dim=img_embed, dropout=dropout)
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_input_dim, tab_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tab_embed, tab_embed),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(img_embed + tab_embed),
            nn.Linear(img_embed + tab_embed, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        img_emb = self.img_encoder(img)
        if tab is None:
            tab = img.new_zeros(img.size(0), self.tab_encoder[0].in_features)
        tab_emb = self.tab_encoder(tab)
        return self.fusion(torch.cat([img_emb, tab_emb], dim=-1))


# ---------------------------------------------------------------------------
# Model 3: Pure CNN chart classifier
# ---------------------------------------------------------------------------

class _ChartResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return self.act(h + x)


class DiscretionaryCNNChart(TrendModelInterface):
    """Lightweight ResNet-style CNN for candlestick chart pattern recognition.

    Uses spatial convolutions to detect price-action shapes:
    body/wick ratios, trend channels, and consolidation patterns.
    Significantly faster inference than ViT — suitable for near-real-time
    chart scanning across many symbols.

    Architecture:
        Image (B,4,32,32) → Stem Conv → [ResBlock × 3] → AvgPool → 3 logits
    """
    def __init__(self, in_chans: int = 4, channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            _ChartResBlock(channels, dropout),
            nn.AvgPool2d(2),
            _ChartResBlock(channels, dropout),
            nn.AvgPool2d(2),
            _ChartResBlock(channels, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))
