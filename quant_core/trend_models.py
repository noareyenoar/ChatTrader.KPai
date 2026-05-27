from __future__ import annotations

import torch
import torch.nn as nn

from .interfaces import TrendModelInterface


class _ManualLSTMCell(nn.Module):
    """LSTM cell implemented with primitive ops for DirectML compatibility."""

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


class TrendLSTMModel(TrendModelInterface):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_size
            self.cells.append(_ManualLSTMCell(in_dim, hidden_size))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
        c = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]

        for t in range(seq_len):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(inp, (h[i], c[i]))
                inp = self.dropout(h[i]) if i < self.num_layers - 1 else h[i]

        return self.head(h[-1])


class _ManualMHA(nn.Module):
    """Scaled dot-product attention using explicit matmul ops.

    nn.MultiheadAttention and nn.TransformerEncoderLayer both call
    ``torch._transformer_encoder_layer_fwd`` (a C++ fused kernel) during
    eval, which is not implemented on the DirectML backend.  This class
    implements the same computation using only basic tensor ops
    (matmul / softmax / linear) that DirectML supports natively.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, dh = self.nhead, self.d_head
        Q = self.q(x).view(B, T, nh, dh).transpose(1, 2)  # (B, nh, T, dh)
        K = self.k(x).view(B, T, nh, dh).transpose(1, 2)
        V = self.v(x).view(B, T, nh, dh).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block built from primitive ops — no fused kernels."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = _ManualMHA(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TrendTransformerModel(TrendModelInterface):
    """Transformer encoder using manual MHA blocks (DirectML-compatible).

    Replaces ``nn.TransformerEncoderLayer`` / ``nn.TransformerEncoder`` entirely
    so that ``aten::_transformer_encoder_layer_fwd`` is never dispatched.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_TransformerBlock(d_model, nhead, d_model * 4, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.positional[:, : x.size(1), :]
        for layer in self.layers:
            h = layer(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = y[..., : x.shape[-1]]
        y = self.drop(self.act(self.norm1(y)))
        y = self.conv2(y)
        y = y[..., : x.shape[-1]]
        y = self.drop(self.act(self.norm2(y)))
        return y + self.res(x)


class TrendTCNModel(TrendModelInterface):
    def __init__(self, input_dim: int, channels: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            _TemporalBlock(input_dim, channels, dilation=1, dropout=dropout),
            _TemporalBlock(channels, channels, dilation=2, dropout=dropout),
            _TemporalBlock(channels, channels, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.net(z)
        return self.head(z)
