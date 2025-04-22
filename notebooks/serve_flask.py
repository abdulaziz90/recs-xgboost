from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model at startup
MODEL_PATH = "model.bst"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing {MODEL_PATH}")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()              # expect JSON array of dicts
    df = pd.DataFrame(data)
    dmat = xgb.DMatrix(df)
    preds_log = bst.predict(dmat)
    preds = np.expm1(preds_log)            # invert log1p
    return jsonify(predictions=preds.tolist())

if __name__ == "__main__":
    # for notebook demos, enable reloader off or on as you like
    app.run(host="0.0.0.0", port=8080, debug=True)
