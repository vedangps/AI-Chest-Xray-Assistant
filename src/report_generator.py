"""
Generate educational AI-assisted chest X-ray reports.
"""

from dataclasses import dataclass
import numpy as np


DISCLAIMER_TEXT = (
    "This AI-generated report is for educational use only and "
    "is not a medical diagnosis. Clinical decisions must be "
    "made by a qualified healthcare professional."
)


@dataclass
class MedicalReport:
    """
    Structured educational report content.
    """

    title: str
    overview: str
    findings: str
    gradcam_observation: str
    recommendation: str
    disclaimer: str

    def to_markdown(self) -> str:
        """
        Render the report for Streamlit display.
        """

        return "\n\n".join(
            [
                f"### {self.title}",
                f"**Overview**\n{self.overview}",
                f"**Educational Findings**\n{self.findings}",
                f"**Grad-CAM Observation**\n{self.gradcam_observation}",
                f"**Recommended Next Step**\n{self.recommendation}",
            ]
        )

    def to_plain_text(self) -> str:
        """
        Render the report for PDF export.
        """

        return "\n\n".join(
            [
                self.title,
                f"Overview: {self.overview}",
                f"Educational Findings: {self.findings}",
                f"Grad-CAM Observation: {self.gradcam_observation}",
                f"Recommended Next Step: {self.recommendation}",
                f"Disclaimer: {self.disclaimer}",
            ]
        )


def describe_confidence(confidence_score: float) -> str:
    """
    Convert a numeric confidence score into a plain-language label.
    """

    if confidence_score >= 85:
        return "high"

    if confidence_score >= 70:
        return "moderate"

    return "limited"


def build_findings(
    prediction: str,
    confidence_score: float,
) -> str:
    """
    Generate an educational explanation for the model output.
    """

    confidence_label = describe_confidence(confidence_score)

    if prediction == "PNEUMONIA":
        return (
            "The model associated this image with a pattern labeled "
            f"'{prediction}' and assigned {confidence_label} confidence "
            "to that pattern. In educational terms, this suggests the "
            "network detected image features that can be seen in chest "
            "X-rays with inflammatory or air-space opacity. The output "
            "does not establish the cause, severity, or clinical context."
        )

    return (
        "The model associated this image with a pattern labeled "
        f"'{prediction}' and assigned {confidence_label} confidence "
        "to that pattern. In educational terms, this suggests the "
        "network did not prioritize features it learned to link with "
        "the pneumonia class. Normal-appearing AI output still does not "
        "exclude subtle disease or non-pulmonary abnormalities."
    )


def _analyze_heatmap_spatial_distribution(heatmap: np.ndarray) -> str:
    """
    Analyze the raw Grad-CAM heatmap grid to extract structural text descriptions.
    """
    row_slices = np.array_split(
        np.arange(heatmap.shape[0]),
        3,
    )
    column_slices = np.array_split(
        np.arange(heatmap.shape[1]),
        3,
    )

    region_labels = (
        ("upper-left", "upper-center", "upper-right"),
        ("middle-left", "middle-center", "middle-right"),
        ("lower-left", "lower-center", "lower-right"),
    )

    dominant_region = "middle-center"
    dominant_score = float("-inf")

    for row_index, row_values in enumerate(row_slices):
        for column_index, column_values in enumerate(column_slices):
            region_score = float(
                heatmap[
                    row_values[0]:row_values[-1] + 1,
                    column_values[0]:column_values[-1] + 1,
                ].mean()
            )

            if region_score > dominant_score:
                dominant_score = region_score
                dominant_region = region_labels[row_index][column_index]

    peak_activation = float(heatmap.max() * 100)
    coverage_percent = float(
        (heatmap >= 0.6).mean() * 100
    )

    return (
        "Grad-CAM highlighted the "
        f"{dominant_region} region most strongly, "
        f"with a peak activation of {peak_activation:.1f}% "
        f"and elevated attention across {coverage_percent:.1f}% "
        "of the image. This visualization reflects where the "
        "model focused during classification, not a confirmed "
        "site of disease."
    )


def generate_medical_report(
    prediction: str,
    confidence_score: float,
    heatmap: np.ndarray,
) -> MedicalReport:
    """
    Create a reusable educational report from model outputs.
    """

    overview = (
        "The AI system classified the chest X-ray as "
        f"'{prediction}' with a confidence score of "
        f"{confidence_score:.2f}%."
    )

    findings = build_findings(
        prediction=prediction,
        confidence_score=confidence_score,
    )

    gradcam_observation = _analyze_heatmap_spatial_distribution(heatmap)

    recommendation = (
        "Correlate this result with symptoms, vital signs, prior "
        "imaging, laboratory data, and formal review by a licensed "
        "clinician or radiologist."
    )

    return MedicalReport(
        title="AI-Assisted Chest X-ray Educational Report",
        overview=overview,
        findings=findings,
        gradcam_observation=gradcam_observation,
        recommendation=recommendation,
        disclaimer=DISCLAIMER_TEXT,
    )
