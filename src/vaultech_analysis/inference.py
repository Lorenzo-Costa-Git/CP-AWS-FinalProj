"""
Inference service for predicting total piece travel time.

Loads the trained XGBoost model and provides predictions.

Usage as CLI:
    uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5

Usage as module (for Streamlit):
    from vaultech_analysis.inference import Predictor
    predictor = Predictor()
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
"""

import argparse
import json
import time
from pathlib import Path

import boto3
import pandas as pd
from xgboost import XGBRegressor


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# Fallback values matching the trained model — used by SageMakerPredictor
# when the container runs without the models/ directory.
_DEFAULT_METADATA = {
    "features":        ["die_matrix", "lifetime_2nd_strike_s", "oee_cycle_time_s"],
    "valid_matrices":  [4974, 5052, 5090, 5091],
    "oee_median":      13.8,
    "metrics":         {"rmse": 1.8716, "mae": 0.9407, "r2": 0.6706},
}


class Predictor:
    """Loads the trained model and provides predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR, gold_file: Path = GOLD_FILE):
        model_path = Path(model_dir) / "xgboost_bath_predictor.json"
        metadata_path = Path(model_dir) / "model_metadata.json"

        # Load model
        self._model = XGBRegressor()
        self._model.load_model(str(model_path))

        # Load metadata
        with open(metadata_path) as f:
            self._metadata = json.load(f)

        self._features = self._metadata["features"]
        self._metrics = self._metadata["metrics"]
        self._valid_matrices = set(self._metadata["valid_matrices"])
        self._oee_median = self._metadata.get("oee_median", 13.8)

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict total bath time from early-stage features.

        Returns a dict with predicted_bath_time_s, input values, and model_metrics.
        Returns {"error": "..."} for unknown die_matrix values.
        Missing oee_cycle_time_s defaults to the median (~13.8s).
        """
        if int(die_matrix) not in self._valid_matrices:
            return {"error": f"Unknown die_matrix {die_matrix}. Valid: {sorted(self._valid_matrices)}"}

        oee_used = oee_cycle_time_s if oee_cycle_time_s is not None else self._oee_median

        X = pd.DataFrame([{
            "die_matrix":            int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s":      float(oee_used),
        }])[self._features]

        predicted = float(self._model.predict(X)[0])

        return {
            "predicted_bath_time_s": round(predicted, 3),
            "die_matrix":            int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s":      oee_cycle_time_s,
            "model_metrics":         self._metrics,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame of pieces.

        Handle missing oee_cycle_time_s by filling with the median.
        """
        X = df[self._features].copy()
        X["oee_cycle_time_s"] = X["oee_cycle_time_s"].fillna(self._oee_median)
        predictions = self._model.predict(X)
        return pd.Series(predictions, index=df.index)


class SageMakerPredictor:
    """Calls the deployed SageMaker endpoint for inference.

    Replaces the local Predictor when SAGEMAKER_ENDPOINT_NAME is set.
    Identical public interface: predict() and predict_batch().
    """

    _BATCH_SIZE = 2000  # rows per invoke_endpoint call
    _WORKERS = 20      # parallel batch workers

    def __init__(self, endpoint_name: str, region: str = "eu-west-1"):
        self._endpoint_name = endpoint_name
        self._region = region
        self._runtime = boto3.client("sagemaker-runtime", region_name=region)

        # Load metadata from disk when available; fall back to known training
        # values so the container works without the models/ directory.
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                meta = json.load(f)
        else:
            meta = _DEFAULT_METADATA

        self._features = meta["features"]
        self._metrics = meta["metrics"]
        self._oee_median = meta.get("oee_median", 13.8)
        self._valid_matrices = set(meta["valid_matrices"])

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict:
        """Predict bath time via SageMaker endpoint.

        Returns the same dict as Predictor.predict(), plus a '_debug' key
        with payload, raw response, and round-trip latency.
        """
        if int(die_matrix) not in self._valid_matrices:
            return {"error": f"Unknown die_matrix {die_matrix}. Valid: {sorted(self._valid_matrices)}"}

        oee = oee_cycle_time_s if oee_cycle_time_s is not None else self._oee_median
        payload = f"{int(die_matrix)},{float(lifetime_2nd_strike_s)},{float(oee)}"

        t0 = time.time()
        response = self._runtime.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)
        raw = response["Body"].read().decode("utf-8").strip()
        prediction = float(raw.split("\n")[0])

        return {
            "predicted_bath_time_s": round(prediction, 3),
            "die_matrix":            int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s":      oee_cycle_time_s,
            "model_metrics":         self._metrics,
            "_debug": {
                "endpoint":     self._endpoint_name,
                "payload":      payload,
                "raw_response": raw,
                "latency_ms":   latency_ms,
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Predict bath time for a DataFrame using parallel large-batch requests.

        Sends 2000 rows per HTTP call to the SageMaker endpoint, with 20
        parallel workers — 169k rows completes in a few seconds.
        """
        import concurrent.futures
        import boto3 as _boto3

        X = df[self._features].copy()
        X["oee_cycle_time_s"] = X["oee_cycle_time_s"].fillna(self._oee_median)

        rows = [
            f"{int(r['die_matrix'])},{float(r['lifetime_2nd_strike_s'])},{float(r['oee_cycle_time_s'])}"
            for _, r in X.iterrows()
        ]
        chunks = [rows[i : i + self._BATCH_SIZE] for i in range(0, len(rows), self._BATCH_SIZE)]

        def _call_chunk(chunk: list[str]) -> list[float]:
            rt = _boto3.client("sagemaker-runtime", region_name=self._region)
            payload = "\n".join(chunk).encode("utf-8")
            resp = rt.invoke_endpoint(
                EndpointName=self._endpoint_name,
                ContentType="text/csv",
                Body=payload,
            )
            raw = resp["Body"].read().decode("utf-8").strip()
            return [float(v) for v in raw.splitlines() if v.strip()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._WORKERS) as pool:
            results = list(pool.map(_call_chunk, chunks))

        preds = [p for chunk_preds in results for p in chunk_preds]
        return pd.Series(preds, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Predict piece bath time from early-stage features.")
    parser.add_argument("--die-matrix", type=int, required=True, help="Die matrix number (4974, 5052, 5090, 5091)")
    parser.add_argument("--strike2",    type=float, required=True, help="Cumulative time at 2nd strike (seconds)")
    parser.add_argument("--oee",        type=float, default=None,  help="OEE cycle time (seconds, optional)")
    args = parser.parse_args()

    predictor = Predictor()
    result = predictor.predict(
        die_matrix=args.die_matrix,
        lifetime_2nd_strike_s=args.strike2,
        oee_cycle_time_s=args.oee,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
