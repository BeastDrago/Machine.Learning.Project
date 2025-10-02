import joblib
from pathlib import Path
import pandas as pd
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'model.pkl'

def load_model(model_path=None):
    path = model_path or MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError('Model not found. Train first (python -m src.train).')
    return joblib.load(path)

def predict(inputs: dict, model=None):
    if model is None:
        model = load_model()
    df = pd.DataFrame([inputs])
    proba = model.predict_proba(df)[0]
    return {'prediction': int(model.predict(df)[0]), 'probability_pass': round(proba[1]*100,2)}