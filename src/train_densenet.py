"""
Training script for the DenseNet121 transfer learning model.

Fine-tunes an ImageNet-pretrained DenseNet121 at a low learning rate with
mixed precision (AMP) so it fits the modest GPU budget, class-weighted loss
to counter the imbalance, and early stopping on the stratified validation
split. Aspect-preserving preprocessing (see ``preprocessing.get_transforms``)
removes the geometric shortcut that previously pushed Grad-CAM toward the
mediastinum.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DENSENET_BATCH_SIZE,
    DENSENET_EARLY_STOPPING_PATIENCE,
    DENSENET_LEARNING_RATE,
    DENSENET_MAX_EPOCHS,
    DENSENET_MODEL_PATH,
    EARLY_STOPPING_MIN_DELTA,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_MIN_LR,
    LR_SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)
from dataloader import create_dataloader
from dataset import get_training_class_weights
from densenet_model import ChestXrayDenseNet121


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
) -> tuple[float, float]:
    """
    Run one train or eval epoch and return (average loss, accuracy).

    Passing ``optimizer`` (and ``scaler``) switches the epoch into training
    mode with mixed-precision updates; omitting them runs a no-grad eval pass.
    """

    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(
        "train",
        batch_size=DENSENET_BATCH_SIZE,
    )
    validation_loader = create_dataloader(
        "val",
        batch_size=DENSENET_BATCH_SIZE,
    )

    model = ChestXrayDenseNet121(pretrained=True).to(device)

    weights = get_training_class_weights()
    class_weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(
        model.parameters(),
        lr=DENSENET_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    DENSENET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    epochs_no_improve = 0

    print("=" * 50)
    print("DenseNet121 Transfer Learning Training Active")
    print("=" * 50)
    print(f"Device              : {device}")
    print(f"Target Output Path  : {DENSENET_MODEL_PATH}")
    print(f"Batch Size          : {DENSENET_BATCH_SIZE}")
    print(f"Learning Rate       : {DENSENET_LEARNING_RATE}")
    print(f"Class Weights       : {weights}")
    print(f"Training Batches    : {len(train_loader)}")
    print(f"Validation Batches  : {len(validation_loader)}")

    for epoch in range(DENSENET_MAX_EPOCHS):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_loss, validation_acc = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 50)
        print(f"DenseNet121 Epoch [{epoch + 1}/{DENSENET_MAX_EPOCHS}]")
        print("=" * 50)
        print(f"Learning Rate   : {current_lr:.6f}")
        print(f"Training Loss   : {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(
            f"Validation Loss : {validation_loss:.4f} "
            f"| Acc: {validation_acc:.2%}"
        )

        if validation_loss < (best_validation_loss - EARLY_STOPPING_MIN_DELTA):
            best_validation_loss = validation_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), DENSENET_MODEL_PATH)
            print(
                "[OK] New DenseNet121 best model saved! "
                f"(Val Loss: {validation_loss:.4f})"
            )
        else:
            epochs_no_improve += 1
            print(f"[!] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= DENSENET_EARLY_STOPPING_PATIENCE:
            print(f"\n[STOP] Early stopping triggered after {epoch + 1} epochs.")
            break

    print("\nTraining complete. Best validation loss: "
          f"{best_validation_loss:.4f}")


if __name__ == "__main__":
    main()
