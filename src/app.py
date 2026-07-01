"""
Streamlit application for AI-assisted chest X-ray analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path

from PIL import Image
import streamlit as st
import torch

from calibration import (
    ThresholdCalibration,
    load_threshold_calibration,
    resolve_decision_threshold,
)
from config import (
    PROJECT_ROOT,
    THRESHOLD_SWEEP_MIN,
)
from gradcam import (
    GradCAMResult,
    compute_gradcam,
    save_overlay_image,
)
from pdf_generator import (
    PDFReportRequest,
    generate_pdf_report,
)
from predict_densenet import load_densenet_model
from report_generator import (
    MedicalReport,
    generate_medical_report,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

TEMP_DIR = PROJECT_ROOT / "reports" / "tmp"
SCREENING_DECISION_THRESHOLD = THRESHOLD_SWEEP_MIN


@dataclass(frozen=True)
class SensitivityMode:
    """
    Runtime threshold policy exposed in the sidebar.
    """

    label: str
    threshold: float
    summary: str


SENSITIVITY_MODE_OPTIONS = (
    "Standard",
    "Screening",
)


@dataclass
class AnalysisArtifacts:
    """
    Aggregated outputs for one uploaded image.
    """

    uploaded_image_path: Path
    gradcam_image_path: Path
    gradcam_result: GradCAMResult
    medical_report: MedicalReport


@st.cache_resource
def load_runtime() -> tuple[torch.nn.Module, torch.device]:
    """
    Load the core diagnostic engine once per Streamlit session.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_densenet_model(device)
    return model, device


def inject_styles() -> None:
    """
    Apply a premium card-based visual system to the app.
    """

    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f3f7fb;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: rgba(255, 255, 255, 0.94);
            --border: rgba(15, 23, 42, 0.10);
            --text: #102033;
            --muted: #58677a;
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.12);
            --alert: #b45309;
            --alert-soft: rgba(180, 83, 9, 0.14);
            --shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg: #07131c;
                --surface: rgba(9, 21, 31, 0.84);
                --surface-strong: rgba(12, 24, 36, 0.94);
                --border: rgba(148, 163, 184, 0.18);
                --text: #e5eef8;
                --muted: #9fb0c4;
                --accent: #4fd1c5;
                --accent-soft: rgba(79, 209, 197, 0.14);
                --alert: #fb923c;
                --alert-soft: rgba(251, 146, 60, 0.16);
                --shadow: 0 24px 70px rgba(2, 6, 23, 0.42);
            }
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 32%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.14), transparent 30%),
                linear-gradient(180deg, var(--app-bg) 0%, rgba(148, 163, 184, 0.04) 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        .hero-card,
        .panel-card,
        .result-card,
        .status-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .hero-card {
            padding: 1.75rem 1.9rem;
            margin-bottom: 1.1rem;
        }

        .hero-eyebrow {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .hero-title {
            color: var(--text);
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 700;
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            margin: 0.85rem 0 0;
            max-width: 760px;
        }

        .panel-card,
        .result-card {
            padding: 1.2rem 1.25rem;
            margin-bottom: 1rem;
        }

        .status-card {
            min-height: 120px;
            padding: 1.05rem 1.15rem;
            margin-bottom: 1rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.62rem;
            border-radius: 999px;
            background: rgba(22, 163, 74, 0.14);
            border: 1px solid rgba(22, 163, 74, 0.34);
            color: #15803d;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .status-dot {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            background: #16a34a;
            box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.14);
        }

        .status-value {
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.35;
            margin-top: 0.35rem;
        }

        .status-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.45rem;
        }

        .viewer-heading {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 0.2rem;
        }

        .viewer-caption {
            color: var(--muted);
            font-size: 0.84rem;
            margin: 0 0 0.75rem;
        }

        .result-card.alert {
            border-color: rgba(180, 83, 9, 0.30);
            background: linear-gradient(135deg, var(--alert-soft), var(--surface-strong));
        }

        .result-card.clear {
            border-color: rgba(15, 118, 110, 0.26);
            background: linear-gradient(135deg, var(--accent-soft), var(--surface-strong));
        }

        .section-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-size: 0.74rem;
            margin-bottom: 0.45rem;
        }

        .result-title {
            color: var(--text);
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }

        .result-copy {
            color: var(--muted);
            margin: 0.55rem 0 0;
            line-height: 1.55;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
        }

        div[data-testid="stMetricDelta"] {
            font-weight: 600;
        }

        div[data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--accent), #3b82f6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    model: torch.nn.Module,
    device: torch.device,
    decision_threshold: float,
) -> AnalysisArtifacts:
    """
    Run inference, Grad-CAM generation, and report synthesis.
    """

    gradcam_result = compute_gradcam(
        image_path=uploaded_image_path,
        model=model,
        device=device,
        decision_threshold=decision_threshold,
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
        heatmap=gradcam_result.heatmap,
    )

    return AnalysisArtifacts(
        uploaded_image_path=uploaded_image_path,
        gradcam_image_path=gradcam_image_path,
        gradcam_result=gradcam_result,
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


def protocol_label(
    sensitivity_mode: SensitivityMode,
) -> str:
    """
    Convert the selected sensitivity mode into clinical protocol language.
    """

    if sensitivity_mode.label == "Screening":
        return "Accelerated Triage Screening"

    return "Standard Diagnostic Confirmation"


def render_hero(
    sensitivity_mode: SensitivityMode,
) -> None:
    """
    Render the top-level product framing.
    """

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">Core Diagnostic Engine</div>
            <h1 class="hero-title">Pneumonia Diagnostic AI Agent</h1>
            <p class="hero-copy">
                Upload a pediatric chest X-ray to run calibrated pneumonia classification,
                inspect Grad-CAM attention, and export a structured educational report.
                Active protocol: <strong>{protocol_label(sensitivity_mode)}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_guidance(
    sensitivity_mode: SensitivityMode,
) -> None:
    """
    Render operational readiness details without development metrics.
    """

    status_column, protocol_column, target_column = st.columns(3)

    with status_column:
        st.markdown(
            """
            <div class="status-card">
                <div class="section-label">Status</div>
                <div class="status-value">System Status</div>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    Ready
                </div>
                <div class="status-copy">
                    Diagnostic workflow is available for image review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with protocol_column:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="section-label">Active Protocol</div>
                <div class="status-value">
                    {protocol_label(sensitivity_mode)}
                </div>
                <div class="status-copy">
                    Selected from Clinical Controls for this session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with target_column:
        st.markdown(
            """
            <div class="status-card">
                <div class="section-label">Target</div>
                <div class="status-value">
                    Target Population: Pediatric (Chest X-Ray Archive)
                </div>
                <div class="status-copy">
                    Optimized for pediatric chest radiograph review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_clinical_workflow_instructions() -> None:
    """
    Render concise operational guidance before image upload.
    """

    with st.expander(
        "📋 Clinical Workflow Instructions",
        expanded=True,
    ):
        st.markdown(
            """
            1. Confirm patient imaging is in a standard PA/AP chest view.
            2. Upload the DICOM-exported JPEG/PNG study using the dropzone below.
            3. Review the side-by-side anatomical alignment and localized attention heatmap.
            4. Export the structured educational summary for the patient log if required.
            """
        )


def render_probability_bar(
    confidence_score: float,
) -> None:
    """
    Render the diagnostic certainty meter.
    """

    certainty_ratio = max(
        0.0,
        min(confidence_score / 100, 1.0),
    )
    st.markdown("##### Diagnostic Certainty")
    st.progress(int(round(certainty_ratio * 100)))
    st.caption(
        "Higher certainty indicates stronger alignment with the "
        "patterns learned by the core diagnostic engine."
    )


def resolve_sensitivity_mode(
    calibration: ThresholdCalibration | None,
) -> SensitivityMode:
    """
    Resolve the sidebar-selected threshold policy.
    """

    standard_threshold = resolve_decision_threshold(
        calibration
    )
    st.sidebar.markdown("## Clinical Controls")
    selected_label = st.sidebar.radio(
        "Clinical Sensitivity Mode",
        options=SENSITIVITY_MODE_OPTIONS,
        index=0,
        help=(
            "Standard favors calibrated operating accuracy. "
            "Screening lowers the decision boundary to maximize recall."
        ),
    )

    if selected_label == "Screening":
        st.sidebar.caption(
            "Accelerated screening prioritizes broad triage review."
        )
        return SensitivityMode(
            label="Screening",
            threshold=SCREENING_DECISION_THRESHOLD,
            summary=(
                "Accelerated triage screening prioritizes broad review "
                "when a more sensitive workflow is needed."
            ),
        )

    st.sidebar.caption(
        "Standard mode supports routine diagnostic confirmation."
    )

    return SensitivityMode(
        label="Standard",
        threshold=standard_threshold,
        summary=(
            "Standard diagnostic confirmation supports routine review "
            "with the calibrated operating policy."
        ),
    )


def render_results(
    artifacts: AnalysisArtifacts,
    sensitivity_mode: SensitivityMode,
) -> None:
    """
    Display the calibrated prediction and supporting visuals.
    """

    is_pneumonia = (
        artifacts.gradcam_result.predicted_class == "PNEUMONIA"
    )
    result_tone = "alert" if is_pneumonia else "clear"
    result_title = (
        "Pneumonia-pattern signal detected"
        if is_pneumonia
        else "Normal-pattern signal favored"
    )
    result_copy = (
        "The active clinical protocol classified this X-ray as "
        "PNEUMONIA. Review the confidence distribution and heatmap focus "
        "before using the educational report downstream."
        if is_pneumonia
        else "The active clinical protocol kept this X-ray in the "
        "NORMAL class. This is not a diagnosis and should be interpreted "
        "alongside clinical review."
    )

    st.markdown(
        f"""
        <div class="result-card {result_tone}">
            <div class="section-label">Clinical Decision Support</div>
            <h2 class="result-title">{result_title}</h2>
            <p class="result-copy">{result_copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    column_1, column_2, column_3 = st.columns(3)

    column_1.metric(
        "Predicted Class",
        artifacts.gradcam_result.predicted_class,
    )
    column_2.metric(
        "Assigned Confidence",
        f"{artifacts.gradcam_result.confidence_score:.1f}%",
    )
    column_3.metric(
        "Active Protocol",
        protocol_label(sensitivity_mode),
    )

    st.caption(
        sensitivity_mode.summary
    )

    st.markdown("#### Medical Viewing Station")
    image_column_1, image_column_2 = st.columns(
        [1, 1],
        gap="large",
    )

    with image_column_1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="viewer-heading">Original X-ray</div>
                <div class="viewer-caption">
                    Source study for anatomical alignment review.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                Image.open(artifacts.uploaded_image_path),
                use_container_width=True,
            )

    with image_column_2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="viewer-heading">Grad-CAM Attention Map</div>
                <div class="viewer-caption">
                    Localized model attention overlay for clinician review.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(
                artifacts.gradcam_result.overlay_image,
                use_container_width=True,
                clamp=True,
            )

    certainty_column, report_column = st.columns([0.95, 1.05])

    with certainty_column:
        with st.container(border=True):
            render_probability_bar(
                artifacts.gradcam_result.confidence_score
            )
            st.metric(
                "Review Status",
                "Ready for clinician review",
            )

    with report_column:
        with st.container(border=True):
            st.markdown("##### Educational Report")
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
        page_title="Pneumonia Diagnostic AI Agent",
        page_icon="X",
        layout="wide",
    )

    inject_styles()

    calibration = load_threshold_calibration(
        logger=LOGGER
    )
    sensitivity_mode = resolve_sensitivity_mode(
        calibration
    )

    render_hero(
        sensitivity_mode=sensitivity_mode,
    )
    render_system_guidance(
        sensitivity_mode=sensitivity_mode,
    )

    render_clinical_workflow_instructions()
    with st.container(border=True):
        st.markdown("##### Upload Imaging Study")
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image",
            type=["png", "jpg", "jpeg"],
            help=(
                "Accepted formats: PNG, JPG, JPEG. "
                "Inference uses the calibrated core diagnostic engine "
                "when available."
            ),
        )

    if uploaded_file is None:
        return

    model, device = load_runtime()
    uploaded_image_path = save_uploaded_image(uploaded_file)
    upload_key = uploaded_image_path.name

    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("pdf_path", None)

    try:
        with st.spinner("Running calibrated AI analysis..."):
            artifacts = analyze_uploaded_image(
                uploaded_image_path=uploaded_image_path,
                model=model,
                device=device,
                decision_threshold=sensitivity_mode.threshold,
            )
    except Exception as error:
        LOGGER.exception("Analysis failed")
        st.error(f"Unable to analyze the uploaded image: {error}")
        return

    render_results(
        artifacts=artifacts,
        sensitivity_mode=sensitivity_mode,
    )

    if st.button(
        "Generate PDF Report",
        type="primary",
    ):
        try:
            pdf_path = build_pdf_for_analysis(
                artifacts
            )
            st.session_state["pdf_path"] = str(pdf_path)
        except Exception as error:
            LOGGER.exception("PDF generation failed")
            st.error(
                f"Unable to generate the PDF report: {error}"
            )

    pdf_path_value = st.session_state.get("pdf_path")

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
