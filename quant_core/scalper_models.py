"""Scalping / Microstructure model architectures — Phase 4.

Three architectures targeting short-horizon order-flow sequences:

  Scalper_CNN_v1           — 1-D CNN on order-flow feature maps.
  Scalper_LinearAttn_v1    — Linear-attention Transformer (O(N) complexity).
  Scalper_GRU_v1           — Bidirectional GRU for tick-sequence modeling.

All use TrendModelInterface for predict_with_confidence().

Input shape:  [Batch, Seq_Len, Num_Features]
Target:       3-class logits  (0 = down, 1 = flat, 2 = up)
Loss:         CrossEntropyLoss
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .interfaces import TrendModelInterface


# ---------------------------------------------------------------------------
# DirectML-safe manual GRU cell
# ---------------------------------------------------------------------------

class _ManualGRUCell(nn.Module):
    """GRU cell using only primitive ops (DirectML-safe)."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_n = nn.Linear(input_size, hidden_size)
        self.U_n = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        n = torch.tanh(self.W_n(x) + r * self.U_n(h))
        return (1.0 - z) * n + z * h


# ---------------------------------------------------------------------------
# Model 1: 1-D CNN — spatial order-flow patterns
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ScalperCNN(TrendModelInterface):
    """1-D dilated CNN for order-flow microstructure patterns.

    Dilated convolutions provide exponentially growing receptive fields
    without increasing parameter count — ideal for short-latency inference.

    Architecture:
        Input (B,T,F) → transpose → Conv stack (d=1,2,4) → GlobalAvgPool → MLP → 3 logits

    Best suited for: assets with strong short-term momentum and high volume activity.
    """
    def __init__(self, input_dim: int, channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.layers = nn.Sequential(
            _ConvBlock(channels, channels, kernel=3, dilation=1, dropout=dropout),
            _ConvBlock(channels, channels, kernel=3, dilation=2, dropout=dropout),
            _ConvBlock(channels, channels * 2, kernel=3, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2, channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x.transpose(1, 2))         # (B, C, T)
        z = self.layers(z)
        return self.head(self.pool(z))            # (B, 3)


# ---------------------------------------------------------------------------
# Model 2: Linear Attention Transformer — O(N) efficient attention
# ---------------------------------------------------------------------------

class _LinearAttnBlock(nn.Module):
    """Kernel-linearized attention: Q·(K^T·V) instead of softmax(Q·K^T)·V.
    Complexity: O(N·d²) vs O(N²·d) for standard attention.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def _linear_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, dh = self.nhead, self.d_head
        # Softplus keeps a strictly non-negative kernel map without relying on ELU ops
        # that can trigger CPU fallbacks on DirectML.
        Q = torch.nn.functional.softplus(self.q(x)) + 1e-4  # (B, T, C)
        K = torch.nn.functional.softplus(self.k(x)) + 1e-4
        V = self.v(x)
        Q = Q.view(B, T, nh, dh).permute(0, 2, 1, 3)  # (B, nh, T, dh)
        K = K.view(B, T, nh, dh).permute(0, 2, 1, 3)
        V = V.view(B, T, nh, dh).permute(0, 2, 1, 3)
        # KV context: (B, nh, dh, dh)
        KV = torch.einsum("bnsd,bnsf->bndf", K, V)
        # Numerator: Q · KV  → (B, nh, T, dh)
        num = torch.einsum("bnsd,bndf->bnsf", Q, KV)
        # Denominator: Q · sum(K) → (B, nh, T, 1)
        den = torch.einsum("bnsd,bnd->bns", Q, K.sum(dim=2)).unsqueeze(-1) + 1e-6
        out = (num / den).permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self._linear_attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class ScalperLinearAttn(TrendModelInterface):
    """Linear Attention Transformer for low-latency tick sequence processing.

    Replaces softmax-attention with a kernelized linear approximation
    that scales as O(T) instead of O(T²) — critical for high-frequency
    sequences where T may be large.

    Architecture:
        Input → Linear proj → [LinearAttnBlock × N] → mean pool → 3 logits

    Best suited for: high-frequency data where order-flow momentum
    over the previous ~32 bars predicts the next tick direction.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([_LinearAttnBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h.mean(dim=1))


# ---------------------------------------------------------------------------
# Model 3: Bidirectional GRU — sequential tick patterns
# ---------------------------------------------------------------------------

class ScalperGRU(TrendModelInterface):
    """Bidirectional GRU for sequential tick-level microstructure modeling.

    GRU (Gated Recurrent Unit) uses explicit LSTMCell-equivalent gating
    without the extra cell state — faster and less prone to overfitting
    on short sequences.

    Architecture:
        Input → BiGRU(layers=2) → concat forward+backward last state → MLP → 3 logits

    Best suited for: detecting order-flow imbalance buildups that precede
    short-term price bursts.
    """
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Use manual GRUCell stack to avoid potential DirectML fused kernel issues
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_size * 2
            self.forward_cells.append(_ManualGRUCell(in_d, hidden_size))
            self.backward_cells.append(_ManualGRUCell(in_d, hidden_size))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_size, 3),
        )

    def _run_layer(self, x_seq: torch.Tensor, fwd_cell: nn.GRUCell, bwd_cell: nn.GRUCell) -> torch.Tensor:
        B, T, D = x_seq.shape
        hf = x_seq.new_zeros(B, self.hidden_size)
        hb = x_seq.new_zeros(B, self.hidden_size)
        fwd_states, bwd_states = [], []
        for t in range(T):
            hf = fwd_cell(x_seq[:, t, :], hf)
            fwd_states.append(hf)
        for t in range(T - 1, -1, -1):
            hb = bwd_cell(x_seq[:, t, :], hb)
            bwd_states.insert(0, hb)
        return torch.stack(
            [torch.cat([fwd_states[t], bwd_states[t]], dim=-1) for t in range(T)],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (fc, bc) in enumerate(zip(self.forward_cells, self.backward_cells)):
            h = self._run_layer(h, fc, bc)
            if i < self.num_layers - 1:
                h = self.drop(h)
        last = h[:, -1, :]
        return self.head(last)
