#Calculations of medians from parquet file.
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent           
PARQUET = ROOT / "pieces.parquet"             
OUTPUT = Path(__file__).parent / "reference_times.json"


SEGMENT_COLUMNS = {
    "furnace_to_2nd_strike":   "partial_furnace_to_2nd_strike_s",
    "2nd_to_3rd_strike":       "partial_2nd_to_3rd_strike_s",
    "3rd_to_4th_strike":       "partial_3rd_to_4th_strike_s",
    "4th_strike_to_aux_press": "partial_4th_strike_to_auxiliary_press_s",
    "aux_press_to_bath":       "partial_auxiliary_press_to_bath_s",
}



def compute(parquet_path: Path = PARQUET) -> dict:
    df = pd.read_parquet(parquet_path)

    result = {}
    for matrix, group in df.groupby("die_matrix"):
        result[str(int(matrix))] = {
            key: round(float(group[col].median()), 4)
            for key, col in SEGMENT_COLUMNS.items()
        }
    return result


if __name__ == "__main__":
    reference_times = compute()
    OUTPUT.write_text(json.dumps(reference_times, indent=2))
    print(f"Written to {OUTPUT}")
    print(json.dumps(reference_times, indent=2))
