"""
Tune the DenseNet121 decision threshold without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

from calibration import (
    ThresholdCalibration,
    calculate_decision_metrics,
    save_threshold_calibration,
)
from config import (
    DEFAULT_DECISION_THRESHOLD,
    DENSENET_MODEL_PATH,
    DENSENET_TUNED_METRICS_PATH,
    THRESHOLD_SWEEP_MAX,
    THRESHOLD_SWEEP_MIN,
    THRESHOLD_SWEEP_STEPS,
)
from dataloader import create_dataloader
from predict_densenet import load_densenet_model


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdSweepResult:
    """
    Output from a threshold search across the test set.
    """

    calibration: ThresholdCalibration
    thresholds_evaluated: int
    generated_at: str


def configure_logging() -> None:
    """
    Configure console logging for the script.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def collect_test_probabilities(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a no-grad inference pass and collect labels and pneumonia probabilities.
    """

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    if not all_labels or not all_probabilities:
        raise ValueError(
            "The test dataloader produced no samples for threshold tuning."
        )

    return (
        np.concatenate(all_labels).astype(np.int64),
        np.concatenate(all_probabilities).astype(np.float32),
    )


def find_optimal_threshold(
    labels: np.ndarray,
    pneumonia_probabilities: np.ndarray,
) -> ThresholdSweepResult:
    """
    Sweep thresholds and return the best balanced-accuracy operating point.
    """

    thresholds = np.linspace(
        THRESHOLD_SWEEP_MIN,
        THRESHOLD_SWEEP_MAX,
        THRESHOLD_SWEEP_STEPS,
    )

    baseline_metrics = calculate_decision_metrics(
        labels=labels,
        pneumonia_probabilities=pneumonia_probabilities,
        threshold=DEFAULT_DECISION_THRESHOLD,
    )

    best_metrics = baseline_metrics

    for threshold in thresholds:
        metrics = calculate_decision_metrics(
            labels=labels,
            pneumonia_probabilities=pneumonia_probabilities,
            threshold=float(threshold),
        )

        is_better = (
            metrics.balanced_accuracy
            > best_metrics.balanced_accuracy
        )
        is_tie_with_higher_specificity = (
            np.isclose(
                metrics.balanced_accuracy,
                best_metrics.balanced_accuracy,
            )
            and metrics.specificity > best_metrics.specificity
        )

        if is_better or is_tie_with_higher_specificity:
            best_metrics = metrics

    calibration = ThresholdCalibration(
        baseline=baseline_metrics,
        optimal=best_metrics,
        roc_auc=float(
            roc_auc_score(labels, pneumonia_probabilities)
        ),
        calibration_path=DENSENET_TUNED_METRICS_PATH,
    )

    return ThresholdSweepResult(
        calibration=calibration,
        thresholds_evaluated=len(thresholds),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def persist_threshold_result(
    sweep_result: ThresholdSweepResult,
    model_path: Path = DENSENET_MODEL_PATH,
    output_path: Path = DENSENET_TUNED_METRICS_PATH,
) -> Path:
    """
    Save the tuned threshold JSON and append runtime metadata.
    """

    output_path = save_threshold_calibration(
        calibration=sweep_result.calibration,
        output_path=output_path,
    )

    try:
        payload = json.loads(
            output_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise OSError(
            f"Unable to re-open saved calibration file: {output_path}"
        ) from error

    payload = {
        **payload,
        "model_path": str(model_path),
        "generated_at": sweep_result.generated_at,
        "threshold_search": {
            "min": THRESHOLD_SWEEP_MIN,
            "max": THRESHOLD_SWEEP_MAX,
            "steps": sweep_result.thresholds_evaluated,
        },
    }

    try:
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(
            f"Unable to finalize calibration file: {output_path}"
        ) from error

    return output_path


def main() -> None:
    """
    Execute DenseNet121 threshold tuning over the test set.
    """

    configure_logging()

    if not DENSENET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DenseNet checkpoint not found: {DENSENET_MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    LOGGER.info(
        "Loading DenseNet121 checkpoint from %s",
        DENSENET_MODEL_PATH,
    )
    model = load_densenet_model(device)

    # Select the operating threshold on the held-out validation split so the
    # test set stays a clean estimate of deployed performance.
    LOGGER.info("Building validation dataloader")
    validation_loader = create_dataloader("val")

    LOGGER.info("Collecting validation-set prediction probabilities")
    labels, pneumonia_probabilities = collect_test_probabilities(
        model=model,
        test_loader=validation_loader,
        device=device,
    )

    LOGGER.info(
        "Sweeping thresholds from %.2f to %.2f",
        THRESHOLD_SWEEP_MIN,
        THRESHOLD_SWEEP_MAX,
    )
    sweep_result = find_optimal_threshold(
        labels=labels,
        pneumonia_probabilities=pneumonia_probabilities,
    )

    saved_path = persist_threshold_result(sweep_result)

    baseline = sweep_result.calibration.baseline
    optimal = sweep_result.calibration.optimal

    print("=" * 60)
    print("DenseNet121 Threshold Calibration")
    print("=" * 60)
    print(f"Device               : {device}")
    print(f"Thresholds Evaluated : {sweep_result.thresholds_evaluated}")
    print(f"ROC-AUC              : {sweep_result.calibration.roc_auc:.4f}")
    print("")
    print("--- Baseline @ 0.50 ---")
    print(f"Accuracy             : {baseline.accuracy:.2%}")
    print(f"Specificity          : {baseline.specificity:.2%}")
    print(f"Recall               : {baseline.recall:.2%}")
    print(f"Balanced Accuracy    : {baseline.balanced_accuracy:.2%}")
    print("")
    print("--- Tuned Threshold ---")
    print(f"Optimal Threshold    : {optimal.threshold:.3f}")
    print(f"Accuracy             : {optimal.accuracy:.2%}")
    print(f"Specificity          : {optimal.specificity:.2%}")
    print(f"Recall               : {optimal.recall:.2%}")
    print(f"Balanced Accuracy    : {optimal.balanced_accuracy:.2%}")
    print("")
    print(f"Saved Metrics        : {saved_path}")


if __name__ == "__main__":
    main()
