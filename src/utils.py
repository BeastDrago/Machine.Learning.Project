import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'student_data.csv'

def load_data(path=None):
    if path is None:
        path = DATA_PATH
    return pd.read_csv(path)