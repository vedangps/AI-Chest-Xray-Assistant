"""
Generate educational AI-assisted chest X-ray reports.
"""

from dataclasses import dataclass

from gradcam import GradCAMSummary


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
                f"**Disclaimer**\n{self.disclaimer}",
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


def generate_medical_report(
    prediction: str,
    confidence_score: float,
    gradcam_summary: GradCAMSummary,
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

    recommendation = (
        "Correlate this result with symptoms, vital signs, prior "
        "imaging, laboratory data, and formal review by a licensed "
        "clinician or radiologist."
    )

    return MedicalReport(
        title="AI-Assisted Chest X-ray Educational Report",
        overview=overview,
        findings=findings,
        gradcam_observation=gradcam_summary.explanation,
        recommendation=recommendation,
        disclaimer=DISCLAIMER_TEXT,
    )
