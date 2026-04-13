"""
Minimal SageMaker-compatible serving script for the XGBoost bath-time predictor.

SageMaker protocol:
  GET  /ping         → 200 (health check)
  POST /invocations  → prediction as plain text

Input format (text/csv): die_matrix,lifetime_2nd_strike_s,oee_cycle_time_s
Output format: predicted_bath_s as a plain float string
"""

import os

import xgboost as xgb
from flask import Flask, Response, request

app = Flask(__name__)
_model: xgb.Booster | None = None

# Must match the feature order used during training
FEATURE_NAMES = ["die_matrix", "lifetime_2nd_strike_s", "oee_cycle_time_s"]


def _load_model() -> xgb.Booster:
    global _model
    if _model is None:
        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        path = os.path.join(model_dir, "xgboost-model")
        _model = xgb.Booster()
        _model.load_model(path)
    return _model


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    try:
        _load_model()
        return Response(response="OK\n", status=200, mimetype="text/plain")
    except Exception as exc:
        return Response(response=str(exc), status=500, mimetype="text/plain")


@app.route("/invocations", methods=["POST"])
def invoke() -> Response:
    try:
        raw = request.data.decode("utf-8").strip()
        # Expect CSV: die_matrix,lifetime_2nd_strike_s,oee_cycle_time_s
        values = [float(x) for x in raw.split(",")]
        dm = xgb.DMatrix([values], feature_names=FEATURE_NAMES)
        prediction = float(_load_model().predict(dm)[0])
        return Response(
            response=f"{prediction}\n", status=200, mimetype="text/csv"
        )
    except Exception as exc:
        return Response(response=str(exc), status=400, mimetype="text/plain")


if __name__ == "__main__":
    _load_model()
    app.run(host="0.0.0.0", port=8080)
