from __future__ import annotations

import torch


def _time_warp_batch(x: torch.Tensor, max_warp: float) -> torch.Tensor:
    """Apply mild per-sample temporal warping with linear interpolation.

    x shape: (B, T, F)
    """
    if max_warp <= 0.0:
        return x
    bsz, seq_len, feat = x.shape
    device = x.device
    center = (seq_len - 1) * 0.5

    base = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len)
    scales = 1.0 + (torch.rand(bsz, 1, device=device) * 2.0 - 1.0) * max_warp
    src = (base - center) * scales + center
    src = src.clamp(0.0, seq_len - 1.0001)

    idx0 = src.floor().long()
    idx1 = (idx0 + 1).clamp(max=seq_len - 1)
    w = (src - idx0.float()).unsqueeze(-1)

    gather0 = x.gather(1, idx0.unsqueeze(-1).expand(-1, -1, feat))
    gather1 = x.gather(1, idx1.unsqueeze(-1).expand(-1, -1, feat))
    return gather0 * (1.0 - w) + gather1 * w


def augment_time_series_batch(
    x: torch.Tensor,
    *,
    enabled: bool,
    mask_prob: float = 0.05,
    max_warp: float = 0.10,
) -> torch.Tensor:
    """Apply random masking + time warping to sequence batches."""
    if not enabled:
        return x

    out = x
    if max_warp > 0.0:
        out = _time_warp_batch(out, max_warp=max_warp)

    if mask_prob > 0.0:
        mask = torch.rand_like(out[..., :1]) < mask_prob
        out = out.masked_fill(mask.expand_as(out), 0.0)

    return out
