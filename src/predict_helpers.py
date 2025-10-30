# src/predict_helpers.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"

def load_pipeline():
    p = joblib.load(MODEL_DIR / "pipeline.joblib")
    with open(MODEL_DIR / "metadata.json", "r") as f:
        meta = json.load(f)
    return p, meta

def json_to_df(payload, meta):
    # payload: dict or list of dicts representing movies
    if isinstance(payload, dict):
        rows = [payload]
    else:
        rows = list(payload)
    df = pd.DataFrame(rows)
    # ensure expected columns exist
    for c in meta["num_cols"] + meta["cat_cols"]:
        if c not in df.columns:
            df[c] = np.nan
    # If original data had 'release_date' and we expect release_year, derive it
    if "release_date" in df.columns and "release_year" in meta["num_cols"]:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    return df
