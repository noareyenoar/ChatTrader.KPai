from __future__ import annotations


def cleanup_cuda(*objects: object) -> None:
    """Release references and clear CUDA cache to reduce VRAM pressure."""
    for obj in objects:
        del obj

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Runtime may not have torch available during non-training operations.
        return
