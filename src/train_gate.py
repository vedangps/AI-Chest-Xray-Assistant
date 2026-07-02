"""
Train the X-ray input gate (chest X-ray vs. not).

Populate ``data/xray_gate/xray`` and ``data/xray_gate/not_xray`` first (see
gate_dataset.py), then run::

    python src/train_gate.py

The checkpoint is written to ``models/xray_gate/best_model.pth``; once it
exists, ``xray_validator.validate_chest_xray`` uses it automatically and the
app requires no changes.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    WEIGHT_DECAY,
    XRAY_GATE_BATCH_SIZE,
    XRAY_GATE_EARLY_STOPPING_PATIENCE,
    XRAY_GATE_LEARNING_RATE,
    XRAY_GATE_MAX_EPOCHS,
    XRAY_GATE_MODEL_PATH,
)
from gate_dataset import create_gate_dataloader, gate_class_weights
from gate_model import XrayGateNet
from train_densenet import run_epoch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_gate_dataloader("train", XRAY_GATE_BATCH_SIZE)
    validation_loader = create_gate_dataloader("val", XRAY_GATE_BATCH_SIZE)

    model = XrayGateNet(pretrained=True).to(device)

    weights = gate_class_weights()
    class_weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(
        model.parameters(),
        lr=XRAY_GATE_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Full precision: ResNet18 is tiny, and fp16 here can overflow and poison
    # BatchNorm running stats (NaN loss / 50% val accuracy). AMP off, scaler
    # runs as a passthrough.
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    XRAY_GATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    epochs_no_improve = 0

    print("=" * 50)
    print("X-ray Gate Training Active")
    print("=" * 50)
    print(f"Device              : {device}")
    print(f"Target Output Path  : {XRAY_GATE_MODEL_PATH}")
    print(f"Class Weights       : {weights}")
    print(f"Training Batches    : {len(train_loader)}")
    print(f"Validation Batches  : {len(validation_loader)}")

    for epoch in range(XRAY_GATE_MAX_EPOCHS):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=False,
        )
        validation_loss, validation_acc = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            use_amp=False,
        )

        print("\n" + "=" * 50)
        print(f"X-ray Gate Epoch [{epoch + 1}/{XRAY_GATE_MAX_EPOCHS}]")
        print("=" * 50)
        print(f"Training Loss   : {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(
            f"Validation Loss : {validation_loss:.4f} "
            f"| Acc: {validation_acc:.2%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            # Save fp16 to halve the checkpoint (~22 MB) so it fits the repo's
            # 40 MB commit cap; load_gate_model restores fp32 for inference.
            half_state = {
                key: (value.half() if value.is_floating_point() else value)
                for key, value in model.state_dict().items()
            }
            torch.save(half_state, XRAY_GATE_MODEL_PATH)
            print(
                "[OK] New X-ray gate best model saved! "
                f"(Val Loss: {validation_loss:.4f})"
            )
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= XRAY_GATE_EARLY_STOPPING_PATIENCE:
            print(f"\n[STOP] Early stopping after {epoch + 1} epochs.")
            break

    print(f"\nTraining complete. Best validation loss: {best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
