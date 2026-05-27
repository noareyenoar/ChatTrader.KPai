"""Standalone TG-MNN checkpoint evaluation script."""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import yaml

from quant_core.tg_mnn_models import TGMNNModel
from quant_core.tg_mnn_data import prepare_tg_mnn_datasets
from quant_core.tg_mnn_loss import MultiTaskLoss
from quant_core.train_tg_mnn_phase4 import evaluate_model, resolve_device


def main():
    with open("configs/tg_mnn_phase4.yaml") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    device, backend = resolve_device(training_cfg.get("preferred_backend", "auto"))
    print(f"Device: {device} ({backend})")

    datasets = prepare_tg_mnn_datasets(
        Path(data_cfg.get("data_dir", "Dataset/binance_historical")),
        seq_len=model_cfg.get("seq_len", 50),
        train_ratio=0.70,
        val_ratio=0.15,
        purge_gap=20,
        max_rows_per_symbol=data_cfg.get("max_rows_per_symbol", 50000),
    )

    model = TGMNNModel(
        input_dim=datasets.input_dim,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_backbone_layers=model_cfg.get("num_backbone_layers", 3),
    ).to(device)

    ckpt = Path("models/checkpoints/tg_mnn/model_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    print(f"Loaded checkpoint: {ckpt}")

    criterion = MultiTaskLoss(state_weight=1.0, magnitude_weight=0.5, duration_weight=0.5)
    test_loader = DataLoader(datasets.test, batch_size=1024, shuffle=False, num_workers=0)
    metrics = evaluate_model(model, test_loader, criterion, device)

    print("TEST RESULTS:")
    print(f"  State Accuracy : {metrics['state_acc']:.4f}")
    print(f"  Loss           : {metrics['loss']:.4f}")
    print(f"  Magnitude MAE  : {metrics['magnitude_mae']:.6f}")
    print(f"  Duration MAE   : {metrics['duration_mae']:.6f}")

    model.cpu()
    out_path = Path("models/checkpoints/tg_mnn/TG_MNN_v1.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved final model: {out_path}")


if __name__ == "__main__":
    main()
