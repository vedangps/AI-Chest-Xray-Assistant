"""
Streamlit application for AI-assisted chest X-ray analysis.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from config import PROJECT_ROOT
from gradcam import (
    GradCAMResult,
    GradCAMSummary,
    compute_gradcam,
    save_overlay_image,
    summarize_gradcam,
)
from pdf_generator import (
    PDFReportRequest,
    generate_pdf_report,
)
from predict import load_model
from report_generator import (
    MedicalReport,
    generate_medical_report,
)


REPORTS_DIR = PROJECT_ROOT / "reports"
TEMP_DIR = REPORTS_DIR / "tmp"


@dataclass
class AnalysisArtifacts:
    """
    Aggregated outputs for one uploaded image.
    """

    uploaded_image_path: Path
    gradcam_image_path: Path
    gradcam_result: GradCAMResult
    gradcam_summary: GradCAMSummary
    medical_report: MedicalReport


@st.cache_resource
def load_runtime():
    """
    Load the model once for the Streamlit session.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_model(device)

    return model, device


def save_uploaded_image(uploaded_file) -> Path:
    """
    Persist the uploaded file for downstream reuse.
    """

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

    suffix = Path(uploaded_file.name).suffix.lower() or ".png"

    output_path = TEMP_DIR / f"upload_{file_hash}{suffix}"

    output_path.write_bytes(file_bytes)

    return output_path


def analyze_uploaded_image(
    uploaded_image_path: Path,
    model,
    device: torch.device,
) -> AnalysisArtifacts:
    """
    Run the full non-training analysis pipeline.
    """

    gradcam_result = compute_gradcam(
        image_path=uploaded_image_path,
        model=model,
        device=device,
    )

    gradcam_summary = summarize_gradcam(
        gradcam_result
    )

    gradcam_image_path = (
        TEMP_DIR
        / f"gradcam_overlay_{uploaded_image_path.stem}.png"
    )

    save_overlay_image(
        result=gradcam_result,
        output_path=gradcam_image_path,
    )

    medical_report = generate_medical_report(
        prediction=gradcam_result.predicted_class,
        confidence_score=gradcam_result.confidence_score,
        gradcam_summary=gradcam_summary,
    )

    return AnalysisArtifacts(
        uploaded_image_path=uploaded_image_path,
        gradcam_image_path=gradcam_image_path,
        gradcam_result=gradcam_result,
        gradcam_summary=gradcam_summary,
        medical_report=medical_report,
    )


def build_pdf_for_analysis(
    artifacts: AnalysisArtifacts,
) -> Path:
    """
    Generate a PDF report for the current analysis.
    """

    request = PDFReportRequest(
        title="AI-Assisted Chest X-ray Report",
        original_image_path=artifacts.uploaded_image_path,
        gradcam_image_path=artifacts.gradcam_image_path,
        prediction=artifacts.gradcam_result.predicted_class,
        confidence_score=artifacts.gradcam_result.confidence_score,
        medical_report=artifacts.medical_report,
    )

    return generate_pdf_report(request)


def render_results(
    artifacts: AnalysisArtifacts,
) -> None:
    """
    Display the analysis outputs in the UI.
    """

    st.subheader("Prediction")

    metric_column_1, metric_column_2 = st.columns(2)

    metric_column_1.metric(
        "Predicted Class",
        artifacts.gradcam_result.predicted_class,
    )

    metric_column_2.metric(
        "Confidence",
        f"{artifacts.gradcam_result.confidence_score:.2f}%",
    )

    image_column_1, image_column_2 = st.columns(2)

    image_column_1.image(
        Image.open(artifacts.uploaded_image_path),
        caption="Original X-ray",
        use_container_width=True,
    )

    image_column_2.image(
        artifacts.gradcam_result.overlay_image,
        caption="Grad-CAM",
        use_container_width=True,
        clamp=True,
    )

    st.subheader("Medical Report")
    st.markdown(
        artifacts.medical_report.to_markdown()
    )

    st.warning(
        artifacts.medical_report.disclaimer
    )


def main() -> None:
    """
    Render the Streamlit application.
    """

    st.set_page_config(
        page_title="AI Chest X-ray Assistant",
        page_icon="X",
        layout="wide",
    )

    st.title("AI Chest X-ray Assistant")
    st.caption(
        "Upload a chest X-ray, review the model prediction, "
        "inspect Grad-CAM attention, and export an "
        "educational PDF report."
    )

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        return

    model, device = load_runtime()

    uploaded_image_path = save_uploaded_image(
        uploaded_file
    )

    upload_key = uploaded_image_path.name

    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("pdf_path", None)

    with st.spinner("Running AI analysis..."):
        artifacts = analyze_uploaded_image(
            uploaded_image_path=uploaded_image_path,
            model=model,
            device=device,
        )

    render_results(artifacts)

    if st.button("Generate PDF Report"):
        pdf_path = build_pdf_for_analysis(
            artifacts
        )
        st.session_state["pdf_path"] = str(pdf_path)

    pdf_path_value = st.session_state.get(
        "pdf_path"
    )

    if pdf_path_value:
        pdf_path = Path(pdf_path_value)

        if pdf_path.exists():
            st.success(
                f"PDF report created: {pdf_path.name}"
            )

            st.download_button(
                label="Download PDF Report",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
