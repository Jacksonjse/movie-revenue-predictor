# src/train_model.py
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def safe_load_csv(path):
    return pd.read_csv(path)

def extract_basic_features(df):
    df = df.copy()
    # standard columns we might use
    # numeric candidates
    num_cols = []
    for c in ["budget", "popularity", "runtime", "vote_average", "vote_count"]:
        if c in df.columns:
            num_cols.append(c)
        else:
            # create if missing to keep pipeline stable
            df[c] = np.nan
            num_cols.append(c)
    # release_date -> year
    if "release_date" in df.columns:
        def year_from_date(s):
            try:
                return pd.to_datetime(s, errors='coerce').dt.year
            except:
                return pd.to_datetime(s, errors='coerce').year
        df["release_year"] = pd.to_datetime(df["release_date"], errors='coerce').dt.year
    else:
        df["release_year"] = np.nan
    num_cols.append("release_year")

    # simple text/categorical candidates (take first genre name if column is JSON or string)
    cat_cols = []
    if "original_language" in df.columns:
        cat_cols.append("original_language")
    else:
        df["original_language"] = "en"
        cat_cols.append("original_language")

    # genres sometimes stored as JSON-like strings; try to extract first genre name
    if "genres" in df.columns:
        def first_genre(x):
            try:
                if pd.isna(x):
                    return "unknown"
                if isinstance(x, str) and x.startswith("["):
                    # JSON-like list of dicts
                    l = json.loads(x.replace("'", '"'))
                    if len(l) > 0 and "name" in l[0]:
                        return l[0]["name"]
                    if len(l) > 0:
                        return str(l[0])
                if isinstance(x, str):
                    return x.split(",")[0].strip()[:32]
                return str(x)[:32]
            except Exception:
                return "unknown"
        df["genre_0"] = df["genres"].apply(first_genre)
        cat_cols.append("genre_0")
    else:
        df["genre_0"] = "unknown"
        cat_cols.append("genre_0")

    # drop id if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df, num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transform, num_cols),
        ("cat", categorical_transform, cat_cols)
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return pipeline, preprocessor

def main():
    train_csv = DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found. Place train.csv in the data/ folder.")
    df = safe_load_csv(train_csv)
    # Expect a 'revenue' column (TMDB style); if absent, attempt to detect target
    if "revenue" not in df.columns:
        # fall back: try to find the numeric column most like revenue (but best to fail early)
        raise ValueError("train.csv must contain a 'revenue' column as the target.")
    y = df["revenue"].copy()
    X = df.drop(columns=["revenue"])

    X_proc, num_cols, cat_cols = extract_basic_features(X)
    pipeline, preprocessor = build_pipeline(num_cols, cat_cols)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_proc, y, test_size=0.15, random_state=42)

    pipeline.fit(X_train, y_train)

    # Evaluate with RMSLE-like metric (use safe clamp)
    preds = pipeline.predict(X_val)
    preds = np.maximum(preds, 0)
    score = np.sqrt(mean_squared_log_error(y_val.clip(0), preds.clip(0)))
    print(f"Validation RMSLE: {score:.5f}")

    # Save pipeline + metadata
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib")
    metadata = {"num_cols": num_cols, "cat_cols": cat_cols}
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Saved model pipeline to {MODEL_DIR}/pipeline.joblib and metadata.json")

if __name__ == "__main__":
    main()
