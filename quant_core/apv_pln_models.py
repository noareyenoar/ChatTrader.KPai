"""Adaptive Price-Volume Probabilistic Learner Network (APV-PLN) — PyTorch models.

Architecture overview
---------------------
APV_Student  (deployed at inference)
    Dual 1D-CNN branches → per-branch Cross-Attention exchange →
    Adaptive Gating fusion → Probabilistic Head (num_bins logits).

APV_Oracle_Teacher  (train-only, LUPI)
    Manual-LSTM on privileged future price+volume path →
    Soft probability distribution over bins.

APVPLNModel  (composite wrapper)
    forward(x_price, x_volume)            → student_logits         [eval/inference]
    forward(x_price, x_volume, x_oracle)  → (student_logits, oracle_soft)  [train]

Oracle Isolation Contract
-------------------------
The caller (training loop) is responsible for NEVER passing x_oracle during
val/test phases.  APVPLNModel enforces this at forward() level by returning a
single tensor when x_oracle is None.

All sub-modules use primitive ops (matmul / softmax / linear) for
DirectML compatibility — no fused CUDA kernels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Reusable primitives (DirectML-safe)
# ─────────────────────────────────────────────────────────────────────────────

class _Conv1DBlock(nn.Module):
    """Conv1D (causal-padded) → LayerNorm → LeakyReLU.

    Input / output shape: [B, T, C].
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] — transpose to [B, C, T] for Conv1d
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, out_ch]
        return self.act(self.norm(out))


class _ManualCrossAttn(nn.Module):
    """Cross-attention via explicit matmul (no fused kernel, DirectML-safe).

    query → Q projections;  context → K, V projections.
    Output is residual: LayerNorm(query + Attention(Q, K, V)).
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query, context: [B, T, d_model]
        B, T, C = query.shape
        nh, dh = self.nhead, self.d_head

        Q = self.q_proj(query).view(B, T, nh, dh).transpose(1, 2)    # [B, nh, T, dh]
        K = self.k_proj(context).view(B, T, nh, dh).transpose(1, 2)
        V = self.v_proj(context).view(B, T, nh, dh).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale                 # [B, nh, T, T]
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)   # [B, T, C]
        out = self.out_proj(out)
        return self.norm(query + out)  # residual + LayerNorm


class _ManualLSTMCell(nn.Module):
    """Single LSTM cell using primitive ops (DirectML-safe)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
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

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        g = torch.tanh(self.W_g(x) + self.U_g(h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# ─────────────────────────────────────────────────────────────────────────────
# APV Student
# ─────────────────────────────────────────────────────────────────────────────

class APV_Student(nn.Module):
    """Dual-stream 1D-CNN + Cross-Attention → probabilistic bin logits.

    Parameters
    ----------
    price_dim : int     Number of price feature channels (PRICE_DIM = 5).
    vol_dim   : int     Number of volume feature channels (VOLUME_DIM = 5).
    num_bins  : int     Output dimension = number of probability bins.
    cnn_channels : int  Hidden width for CNN and cross-attention layers.
    nhead     : int     Number of attention heads (must divide cnn_channels).
    dropout   : float   Dropout probability applied before the final head.

    Inputs
    ------
    x_price  : [B, seq_len, price_dim]
    x_volume : [B, seq_len, vol_dim]

    Output
    ------
    logits   : [B, num_bins]   (unnormalized; pass through softmax at loss time)
    """

    def __init__(
        self,
        price_dim: int,
        vol_dim: int,
        num_bins: int,
        cnn_channels: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins

        # ── Price CNN branch ──────────────────────────────────────────────────
        self.price_cnn = nn.Sequential(
            _Conv1DBlock(price_dim, cnn_channels, kernel=3),
            _Conv1DBlock(cnn_channels, cnn_channels, kernel=3),
        )

        # ── Volume CNN branch ─────────────────────────────────────────────────
        self.volume_cnn = nn.Sequential(
            _Conv1DBlock(vol_dim, cnn_channels, kernel=3),
            _Conv1DBlock(cnn_channels, cnn_channels, kernel=3),
        )

        # ── Cross-Attention ───────────────────────────────────────────────────
        # Price queries Volume  → accumulation / distribution signal
        self.price_x_vol = _ManualCrossAttn(cnn_channels, nhead=nhead, dropout=dropout)
        # Volume queries Price  → volume-at-price confirmation
        self.vol_x_price = _ManualCrossAttn(cnn_channels, nhead=nhead, dropout=dropout)

        # ── Adaptive Gating Fusion ────────────────────────────────────────────
        fused_dim = cnn_channels * 2
        self.gate_linear = nn.Linear(fused_dim, fused_dim)
        self.feat_linear = nn.Linear(fused_dim, fused_dim)
        self.gate_act = nn.Sigmoid()

        # ── Probabilistic Projection Head ─────────────────────────────────────
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_bins),
        )

    def forward(self, x_price: torch.Tensor, x_volume: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction: [B, T, cnn_channels]
        p_feat = self.price_cnn(x_price)
        v_feat = self.volume_cnn(x_volume)

        # Cross-attention exchange
        p_attended = self.price_x_vol(query=p_feat, context=v_feat)   # [B, T, C]
        v_attended = self.vol_x_price(query=v_feat, context=p_feat)   # [B, T, C]

        # Temporal mean pooling → [B, C]
        p_pooled = p_attended.mean(dim=1)
        v_pooled = v_attended.mean(dim=1)

        # Adaptive gating
        combined = torch.cat([p_pooled, v_pooled], dim=-1)            # [B, 2C]
        gate = self.gate_act(self.gate_linear(combined))
        fused = gate * self.feat_linear(combined)                      # [B, 2C]

        # Probabilistic output
        return self.head(self.drop(fused))                             # [B, num_bins]


# ─────────────────────────────────────────────────────────────────────────────
# APV Oracle Teacher (LUPI — train only)
# ─────────────────────────────────────────────────────────────────────────────

class APV_Oracle_Teacher(nn.Module):
    """Privileged-information teacher that observes the future price+volume path.

    Input : x_oracle  [B, horizon, oracle_dim]  — future bar features
    Output: soft_probs [B, num_bins]             — probability distribution over bins

    Used EXCLUSIVELY during training.  The training loop must NOT pass
    x_oracle during validation or test phases.
    """

    def __init__(
        self,
        oracle_dim: int,
        num_bins: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = _ManualLSTMCell(oracle_dim, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_bins),
        )

    def forward(self, x_oracle: torch.Tensor) -> torch.Tensor:
        # x_oracle: [B, horizon, oracle_dim]
        B, H, _ = x_oracle.shape
        h = x_oracle.new_zeros(B, self.hidden_size)
        c = x_oracle.new_zeros(B, self.hidden_size)

        for t in range(H):
            h, c = self.cell(x_oracle[:, t, :], (h, c))

        logits = self.head(self.drop(h))           # [B, num_bins]
        return torch.softmax(logits, dim=-1)        # soft probability distribution


# ─────────────────────────────────────────────────────────────────────────────
# APVPLNModel — Composite wrapper
# ─────────────────────────────────────────────────────────────────────────────

class APVPLNModel(nn.Module):
    """Full APV-PLN model combining Student and Oracle Teacher.

    forward() interface
    -------------------
    Train mode  (pass x_oracle):
        returns  (student_logits [B, num_bins], oracle_soft [B, num_bins])

    Eval/inference mode  (x_oracle is None):
        returns  student_logits [B, num_bins]

    Oracle Isolation Invariant
    --------------------------
    This class NEVER calls oracle_teacher.forward() unless x_oracle is
    explicitly provided.  Callers must guarantee x_oracle=None in val/test.
    """

    def __init__(
        self,
        price_dim: int,
        vol_dim: int,
        oracle_dim: int,
        num_bins: int,
        cnn_channels: int = 64,
        nhead: int = 4,
        oracle_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins

        self.student = APV_Student(
            price_dim=price_dim,
            vol_dim=vol_dim,
            num_bins=num_bins,
            cnn_channels=cnn_channels,
            nhead=nhead,
            dropout=dropout,
        )
        self.oracle_teacher = APV_Oracle_Teacher(
            oracle_dim=oracle_dim,
            num_bins=num_bins,
            hidden_size=oracle_hidden,
            dropout=dropout,
        )

    def forward(
        self,
        x_price: torch.Tensor,
        x_volume: torch.Tensor,
        x_oracle: torch.Tensor | None = None,
    ):
        """See class docstring for return semantics."""
        student_logits = self.student(x_price, x_volume)

        if x_oracle is not None:
            # ── Train mode: Oracle Teacher provides distillation target ────────
            oracle_soft = self.oracle_teacher(x_oracle)
            return student_logits, oracle_soft

        # ── Eval / inference mode: Student runs alone ─────────────────────────
        return student_logits

    # ── Inference helpers ─────────────────────────────────────────────────────

    def predict_bin_distribution(
        self, x_price: torch.Tensor, x_volume: torch.Tensor
    ) -> torch.Tensor:
        """Return softmax probabilities over bins — [B, num_bins]."""
        logits = self.forward(x_price, x_volume, x_oracle=None)
        return torch.softmax(logits, dim=-1)

    def predict_expected_return(
        self,
        x_price: torch.Tensor,
        x_volume: torch.Tensor,
        bin_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Expected return = Σ p_i · center_i — [B]."""
        probs = self.predict_bin_distribution(x_price, x_volume)
        return (probs * bin_centers.unsqueeze(0)).sum(dim=-1)
