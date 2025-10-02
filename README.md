# Student Performance Classifier v4 (Streamlit) - Minimal & Clean Dashboard

This project provides a minimal, modular Streamlit app that predicts the probability (0-100%)
that a student will pass, using realistic features. The dashboard is intentionally clean:
no raw sample data or visualizations are shown by default. Visualizations (heatmap, feature importance)
are generated only after the user makes a prediction, as requested.

## Features
- Predicts pass probability (0–100%) using practical features (no grades included).
- Inputs: study_hours (1–8), gender, failed_subjects, absences, wifi, hours_wasted, grasp, parental_support, sleep, motivation, extra_classes
- Clean UI: only prediction controls and tips shown. No raw csv or data tables are visible.
- After prediction: show aggregated heatmap and a simple feature-importance bar chart to help interpretation.
- Modular code: `src/train.py`, `src/model.py`, `src/utils.py`.
- Sample synthetic dataset included in `data/` but never displayed in the app.

## Run (Quick)
1. Create venv and install:
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\activate
    pip install -r requirements.txt

2. (Optional) Train model:
    python -m src.train

3. Run app:
    streamlit run app.py