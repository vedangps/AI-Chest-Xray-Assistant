"""
Threshold calibration utilities for DenseNet121 inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    DEFAULT_DECISION_THRESHOLD,
    DENSENET_TUNED_METRICS_PATH,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionMetrics:
    """
    Binary classification metrics at one decision threshold.
    """

    threshold: float
    accuracy: float
    specificity: float
    recall: float
    balanced_accuracy: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class ThresholdCalibration:
    """
    Baseline and tuned metrics loaded from calibration output.
    """

    baseline: DecisionMetrics
    optimal: DecisionMetrics
    roc_auc: float | None
    calibration_path: Path

    @property
    def decision_threshold(self) -> float:
        return self.optimal.threshold


def calculate_decision_metrics(
    labels: np.ndarray,
    pneumonia_probabilities: np.ndarray,
    threshold: float,
) -> DecisionMetrics:
    """
    Compute binary metrics for the given threshold.
    """

    predictions = (
        pneumonia_probabilities >= threshold
    ).astype(np.int64)

    true_negatives = int(
        np.sum((labels == 0) & (predictions == 0))
    )
    false_positives = int(
        np.sum((labels == 0) & (predictions == 1))
    )
    false_negatives = int(
        np.sum((labels == 1) & (predictions == 0))
    )
    true_positives = int(
        np.sum((labels == 1) & (predictions == 1))
    )

    total_samples = max(labels.size, 1)
    negative_total = max(true_negatives + false_positives, 1)
    positive_total = max(true_positives + false_negatives, 1)

    accuracy = (
        true_negatives + true_positives
    ) / total_samples
    specificity = true_negatives / negative_total
    recall = true_positives / positive_total
    balanced_accuracy = (specificity + recall) / 2

    return DecisionMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy),
        specificity=float(specificity),
        recall=float(recall),
        balanced_accuracy=float(balanced_accuracy),
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
    )


def metrics_to_dict(metrics: DecisionMetrics) -> dict[str, Any]:
    """
    Convert typed metrics into a JSON-serializable dictionary.
    """

    return asdict(metrics)


def metrics_from_dict(payload: dict[str, Any]) -> DecisionMetrics:
    """
    Build typed metrics from a JSON dictionary.
    """

    return DecisionMetrics(
        threshold=float(payload["threshold"]),
        accuracy=float(payload["accuracy"]),
        specificity=float(payload["specificity"]),
        recall=float(payload["recall"]),
        balanced_accuracy=float(payload["balanced_accuracy"]),
        true_negatives=int(payload["true_negatives"]),
        false_positives=int(payload["false_positives"]),
        false_negatives=int(payload["false_negatives"]),
        true_positives=int(payload["true_positives"]),
    )


def save_threshold_calibration(
    calibration: ThresholdCalibration,
    output_path: Path = DENSENET_TUNED_METRICS_PATH,
) -> Path:
    """
    Persist calibration metrics to JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "baseline": metrics_to_dict(calibration.baseline),
        "optimal": metrics_to_dict(calibration.optimal),
        "roc_auc": calibration.roc_auc,
    }

    try:
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(
            f"Unable to write calibration file: {output_path}"
        ) from error

    return output_path


def load_threshold_calibration(
    calibration_path: Path = DENSENET_TUNED_METRICS_PATH,
    logger: logging.Logger | None = None,
) -> ThresholdCalibration | None:
    """
    Load tuned metrics from disk when available.
    """

    active_logger = logger or LOGGER

    if not calibration_path.exists():
        active_logger.warning(
            "Calibration file not found at %s. Falling back to %.2f.",
            calibration_path,
            DEFAULT_DECISION_THRESHOLD,
        )
        return None

    try:
        payload = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        active_logger.warning(
            "Unable to read calibration file %s: %s. Falling back to %.2f.",
            calibration_path,
            error,
            DEFAULT_DECISION_THRESHOLD,
        )
        return None

    try:
        baseline = metrics_from_dict(payload["baseline"])
        optimal = metrics_from_dict(payload["optimal"])
    except (KeyError, TypeError, ValueError) as error:
        active_logger.warning(
            "Calibration file %s is malformed: %s. Falling back to %.2f.",
            calibration_path,
            error,
            DEFAULT_DECISION_THRESHOLD,
        )
        return None

    roc_auc = payload.get("roc_auc")

    return ThresholdCalibration(
        baseline=baseline,
        optimal=optimal,
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        calibration_path=calibration_path,
    )


def resolve_decision_threshold(
    calibration: ThresholdCalibration | None,
) -> float:
    """
    Resolve the runtime threshold from calibration state.
    """

    if calibration is None:
        return DEFAULT_DECISION_THRESHOLD

    return calibration.decision_threshold
