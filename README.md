# Pneumonia Diagnostic AI Agent

## Project Overview

**Pneumonia Diagnostic AI Agent** is a research project that helps review pediatric chest X-ray images for pneumonia patterns.

The app lets a user:

- **Upload a chest X-ray image**
- **Run a pneumonia screening model**
- **View a heatmap showing where the model focused**
- **Generate a structured educational report**

The goal is to make chest X-ray review easier to explore, explain, and document in a local research setting.

## Key Features

### **Clinical Sensitivity Mode**

The app includes two review modes in the sidebar:

- **Standard Diagnostic Confirmation**
  - Used for routine review.
  - Applies the calibrated model setting from the project files.

- **Accelerated Triage Screening**
  - Used when a more sensitive screening workflow is preferred.
  - Helps prioritize cases that may need closer review.

### **Explainable Heatmaps**

The app uses **Grad-CAM heatmaps** to show which areas of the X-ray influenced the model output.

This helps users compare:

- The **original chest X-ray**
- The **model attention map**
- The **generated educational summary**

### **Structured Report Export**

After analysis, the app can generate a PDF report that includes:

- Prediction result
- Confidence score
- Original image
- Grad-CAM heatmap
- Educational summary
- Disclaimer

## Setup Instructions

Follow these steps from the project root folder.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Chest-Xray-Assistant
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

On macOS or Linux:

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```powershell
.\.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run src/app.py
```

The app will open in your browser. If it does not open automatically, copy the local URL shown in the terminal.

## Streamlit Cloud Deployment

The deployed app must include at least one trained checkpoint:

- Preferred: `models/densenet121/best_model.pth`
- Fallback: `models/custom_cnn/best_model.pth`

The app will use DenseNet121 when its checkpoint is available. If it is not available, it falls back to the custom CNN checkpoint.

## Quick Look Architecture

```text
AI-Chest-Xray-Assistant/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- src/
|   |-- app.py                 # Streamlit user interface
|   |-- densenet_model.py      # DenseNet model definition
|   |-- predict_densenet.py    # Model loading and prediction logic
|   |-- gradcam.py             # Explainable heatmap generation
|   |-- report_generator.py    # Educational report text
|   |-- pdf_generator.py       # PDF export
|   |-- calibration.py         # Clinical sensitivity settings
|   |-- tune_threshold.py      # Threshold tuning utility
|   |-- config.py              # Project paths and settings
|
|-- models/
|   |-- densenet121/
|       |-- best_model.pth
|       |-- tuned_metrics.json
|
|-- data/
|   |-- chest_xray/
|
|-- reports/
|   |-- tmp/
|
|-- tests/
```

## Important Note

This project is a **research and educational tool**.

It is **not a medical device** and should **not be used as a substitute for professional clinical judgment**.

Any output from this app should be reviewed by qualified healthcare professionals before being used in a real clinical setting.
