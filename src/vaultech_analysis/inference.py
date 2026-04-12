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
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
GOLD_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "gold" / "pieces.parquet"


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
