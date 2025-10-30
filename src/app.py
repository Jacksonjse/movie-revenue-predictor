from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # ✅ Add this
from predict_helpers import load_pipeline, json_to_df
import pandas as pd
from io import BytesIO
from pathlib import Path
import joblib
import os

app = Flask(__name__)

# ✅ Allow CORS for all routes (you can restrict to specific origins later)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load your trained pipeline and metadata
pipeline, meta = load_pipeline()

@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "movie-revenue-predictor"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    df = json_to_df(payload, meta)
    preds = pipeline.predict(df)
    preds = [float(max(0, p)) for p in preds]

    # If input was a single dict, return a single prediction
    if isinstance(payload, dict):
        return jsonify({"prediction": preds[0]})
    return jsonify({"predictions": preds})

@app.route("/predict_test", methods=["POST"])
def predict_test():
    # Upload test.csv; returns CSV with id,revenue similar to sample_submission
    if 'file' not in request.files:
        return jsonify({"error": "no file uploaded, include form field 'file' with test.csv"}), 400
    
    f = request.files['file']
    df_test = pd.read_csv(f)
    
    if "id" not in df_test.columns:
        return jsonify({"error": "test.csv must contain an 'id' column"}), 400
    
    # Prepare features similar to training
    payload = df_test.to_dict(orient="records")
    X_df = json_to_df(payload, meta)
    preds = pipeline.predict(X_df)
    preds = [max(0, float(p)) for p in preds]
    
    out_df = pd.DataFrame({"id": df_test["id"], "revenue": preds})
    buf = BytesIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="submission.csv")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

