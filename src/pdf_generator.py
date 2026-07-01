"""
Generate PDF reports for AI-assisted chest X-ray analysis.
"""

from dataclasses import dataclass
from pathlib import Path

from report_generator import MedicalReport

from config import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass
class PDFReportRequest:
    """
    Input payload for PDF generation.
    """

    title: str
    original_image_path: Path
    gradcam_image_path: Path
    prediction: str
    confidence_score: float
    medical_report: MedicalReport
    output_path: Path | None = None


def build_default_output_path(
    original_image_path: str | Path,
) -> Path:
    """
    Build the default PDF output location.
    """

    original_image_path = Path(original_image_path)

    file_name = (
        f"ai_chest_xray_report_{original_image_path.stem}.pdf"
    )

    return REPORTS_DIR / file_name


def generate_pdf_report(
    request: PDFReportRequest,
) -> Path:
    """
    Generate a professional PDF report and save it to disk.
    """

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = (
        request.output_path
        if request.output_path is not None
        else build_default_output_path(
            request.original_image_path
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not request.original_image_path.exists():
        raise FileNotFoundError(
            f"Original image not found: {request.original_image_path}"
        )

    if not request.gradcam_image_path.exists():
        raise FileNotFoundError(
            f"Grad-CAM image not found: {request.gradcam_image_path}"
        )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    body_style.leading = 16

    image_width = 3.0 * inch
    image_height = 3.0 * inch

    story = [
        Paragraph(request.title, title_style),
        Spacer(1, 12),
        Paragraph("Prediction Summary", heading_style),
        Spacer(1, 6),
        Paragraph(
            f"Predicted Class: {request.prediction}",
            body_style,
        ),
        Paragraph(
            f"Confidence Score: {request.confidence_score:.2f}%",
            body_style,
        ),
        Spacer(1, 12),
        Paragraph("Imaging Review", heading_style),
        Spacer(1, 6),
    ]

    image_table = Table(
        [
            [
                Paragraph("Original X-ray", body_style),
                Paragraph("Grad-CAM Visualization", body_style),
            ],
            [
                Image(
                    str(request.original_image_path),
                    width=image_width,
                    height=image_height,
                ),
                Image(
                    str(request.gradcam_image_path),
                    width=image_width,
                    height=image_height,
                ),
            ],
        ],
        colWidths=[3.2 * inch, 3.2 * inch],
    )

    image_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 1, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.extend(
        [
            image_table,
            Spacer(1, 16),
            Paragraph("AI-generated Report", heading_style),
            Spacer(1, 6),
            Paragraph(request.medical_report.overview, body_style),
            Spacer(1, 6),
            Paragraph(request.medical_report.findings, body_style),
            Spacer(1, 6),
            Paragraph(
                request.medical_report.gradcam_observation,
                body_style,
            ),
            Spacer(1, 6),
            Paragraph(
                request.medical_report.recommendation,
                body_style,
            ),
            Spacer(1, 12),
            Paragraph("Educational Disclaimer", heading_style),
            Spacer(1, 6),
            Paragraph(
                request.medical_report.disclaimer,
                body_style,
            ),
        ]
    )

    document.build(story)

    return output_path
