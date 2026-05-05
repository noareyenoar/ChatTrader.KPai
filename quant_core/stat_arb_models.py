"""Statistical Arbitrage model architectures — Phase 4.

Three architectures for multi-asset spread convergence prediction:

  StatArb_Autoencoder_v1  — Temporal autoencoder: encodes pair sequences
                            into a latent space; reconstruction error signals
                            mispricing.
  StatArb_GAT_v1          — Graph Attention Network (pure PyTorch, no PyG):
                            models inter-asset correlation as a learned graph.
  StatArb_LSTM_v1         — LSTM on fractionally-differenced spread Z-score.

Input shape:  [Batch, Seq_Len, Num_Assets]
Target:       Spread Z-score regression (predict next-step Z-score)
Loss:         MSELoss
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .interfaces import TrendModelInterface


# ---------------------------------------------------------------------------
# DirectML-safe manual GRU cell
# ---------------------------------------------------------------------------

class _ManualGRUCell(nn.Module):
    """GRU cell using only primitive Linear/sigmoid/tanh ops (DirectML-safe)."""
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


class _ManualLSTMCell(nn.Module):
    """LSTM cell using only primitive Linear/sigmoid/tanh ops (DirectML-safe)."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_g = nn.Linear(input_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        g = torch.tanh(self.W_g(x) + self.U_g(h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# ---------------------------------------------------------------------------
# Model 1: Temporal Autoencoder
# ---------------------------------------------------------------------------

class _TemporalEncoder(nn.Module):
    def __init__(self, num_assets: int, latent_dim: int, dropout: float):
        super().__init__()
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(3):
            in_d = num_assets if i == 0 else latent_dim
            self.cells.append(_ManualGRUCell(in_d, latent_dim))
            self.norms.append(nn.LayerNorm(latent_dim))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = [x.new_zeros(B, self.cells[0].hidden_size) for _ in self.cells]
        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = self.norms[i](cell(inp, h[i]))
                inp = self.drop(h[i]) if i < len(self.cells) - 1 else h[i]
        return self.act(h[-1])  # (B, latent_dim)


class _TemporalDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_assets: int, seq_len: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.cell = _ManualGRUCell(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(latent_dim, num_assets)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        h = z
        outs = []
        for _ in range(self.seq_len):
            h = self.drop(self.act(self.norm(self.cell(z, h))))
            outs.append(self.proj(h))
        return torch.stack(outs, dim=1)  # (B, T, num_assets)


class StatArbAutoencoder(TrendModelInterface):
    """Temporal GRU autoencoder for statistical arbitrage.

    Encodes multi-asset spread sequences into a low-dimensional latent
    representation, then reconstructs the input sequence.  High
    reconstruction error on a new bar indicates an anomalous (potentially
    exploitable) regime divergence from the learned manifold.

    At inference time `predict_with_confidence` returns:
      - prediction: the mean predicted next-bar spread (from reconstruction)
      - confidence: 1 / (1 + reconstruction_loss) — inversely proportional
                    to anomaly score.

    Architecture:
        Input (B,T,A) → GRU Encoder → latent z (B,L)
                       → GRU Decoder → reconstruction (B,T,A)
                       → last-step projection → regression head → (B, num_assets)
    """
    def __init__(self, num_assets: int, latent_dim: int = 32, seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = _TemporalEncoder(num_assets, latent_dim, dropout)
        self.decoder = _TemporalDecoder(latent_dim, num_assets, seq_len, dropout)
        self.reg_head = nn.Linear(latent_dim, 1)  # predict spread scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.reg_head(z)  # (B, 1) — primary regression output

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)  # (B, T, A)


# ---------------------------------------------------------------------------
# Model 2: Graph Attention Network (pure PyTorch)
# ---------------------------------------------------------------------------

class _GraphAttnLayer(nn.Module):
    """Single GAT layer operating on a fully-connected asset graph.

    Node features: [B, N, d_model]
    Attention is computed between all pairs of nodes (dense graph),
    which is valid for N ≤ ~20 assets.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Linear(d_model, 1, bias=False)
        self.a_dst = nn.Linear(d_model, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.01)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, d_model)
        h_norm = self.pre_norm(h)
        Wh = self.W(h_norm)                               # (B, N, D)
        e_src = self.a_src(Wh)                        # (B, N, 1)
        e_dst = self.a_dst(Wh)                        # (B, N, 1)
        attn_score = self.leaky(e_src + e_dst.transpose(-2, -1))  # (B, N, N)
        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.drop(attn_score)
        agg = attn_score @ Wh                         # (B, N, D)
        h2 = self.norm(h + agg)
        return self.norm(h2 + self.ff(h2))


class StatArbGAT(TrendModelInterface):
    """Graph Attention Network for modeling cross-asset correlation structure.

    Treats each asset as a node in a fully-connected graph.  GAT layers
    learn to assign higher attention weights to correlated assets, enabling
    the model to detect regime shifts in the correlation matrix that precede
    spread mean-reversion.

    Architecture:
        Input (B,T,A) → per-time-step projection → temporal pool (mean)
                       → [GATLayer × N] → readout (mean over nodes)
                       → regression head → (B, 1)
    """
    def __init__(self, num_assets: int, d_model: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(1, d_model)  # each asset's seq projected
        self.temp_pool = nn.AdaptiveAvgPool1d(1)  # collapse time dim per asset
        self.gat_layers = nn.ModuleList([_GraphAttnLayer(d_model, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, A)
        B, T, A = x.shape
        # Per-asset temporal pooling: (B, A, T) → AvgPool → (B, A, 1) → (B, A, 1)
        xp = self.temp_pool(x.transpose(1, 2)).squeeze(-1)   # (B, A)
        h = self.node_proj(xp.unsqueeze(-1))                  # (B, A, d_model)
        for layer in self.gat_layers:
            h = layer(h)
        return self.head(h.mean(dim=1))                       # (B, 1)


# ---------------------------------------------------------------------------
# Model 3: LSTM Spread
# ---------------------------------------------------------------------------

class _TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h


class StatArbLSTM(TrendModelInterface):
    """TCN substitute for legacy StatArb LSTM path.

    The class name is preserved for compatibility with existing registry keys,
    while the internals use dilated 1D convolutions for better DirectML
    parallelism and improved gradient flow.
    """

    def __init__(self, num_assets: int, hidden_size: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(num_assets, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_TemporalConvBlock(hidden_size, dilation=2 ** i, dropout=dropout) for i in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, A) -> (B, A, T)
        h = self.in_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=-1)
        return self.head(pooled)
