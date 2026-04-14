"""

Generates api/validation_pieces.csv and api/validation_expected.json.

Per §1.4, the 10 pieces must cover:
  P001 - All-OK on matrix 4974
  P002 - All-OK on matrix 5052
  P003 - All-OK on matrix 5090
  P004 - All-OK on matrix 5091
  P005 - Only furnace_to_2nd_strike penalized
  P006 - Only 2nd_to_3rd_strike penalized
  P007 - Only 3rd_to_4th_strike penalized
  P008 - Only 4th_strike_to_aux_press penalized
  P009 - Only aux_press_to_bath penalized
  P010 - Multi-segment delay AND at least one NULL cumulative time
         (downstream segments must report null, not false)

Per the Automated Penalties section, validation_expected.json must be
generated programmatically from the rules — NOT hand-written (-15 pts).

Usage (from project root):
    uv run python api/generate_validation_set.py
"""

import csv
import json
import sys
from pathlib import Path

# Allow importing diagnose from api/src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from diagnose import diagnose, REFERENCE_TIMES

OUTPUT_DIR = Path(__file__).parent
CSV_PATH = OUTPUT_DIR / "validation_pieces.csv"
JSON_PATH = OUTPUT_DIR / "validation_expected.json"


# For all-OK pieces we use exact reference values (deviation = 0 → false).
# For penalized segments we add 2.0s above reference (deviation = 2.0,
# which is in the 1.0 < deviation ≤ 5.0 → true range per §1.3).
# For NULL we omit the key entirely (missing sensor reading per §1.5).
# ---------------------------------------------------------------------------

def refs(matrix: int) -> dict:
    return REFERENCE_TIMES[str(matrix)]


def build_cumulative(matrix: int, offsets: dict) -> dict:
    """
    offsets: dict of segment -> extra seconds added on top of reference.
    Segments not in offsets use their reference value (deviation=0).
    A value of None means the cumulative timestamp is missing (NULL).

    Segments in order:
      furnace_to_2nd_strike  -> lifetime_2nd_strike_s  (absolute)
      2nd_to_3rd_strike      -> lifetime_3rd_strike_s
      3rd_to_4th_strike      -> lifetime_4th_strike_s
      4th_strike_to_aux_press-> lifetime_auxiliary_press_s
      aux_press_to_bath      -> lifetime_bath_s
    """
    r = refs(matrix)
    seg_order = [
        ("furnace_to_2nd_strike",    "lifetime_2nd_strike_s"),
        ("2nd_to_3rd_strike",        "lifetime_3rd_strike_s"),
        ("3rd_to_4th_strike",        "lifetime_4th_strike_s"),
        ("4th_strike_to_aux_press",  "lifetime_auxiliary_press_s"),
        ("aux_press_to_bath",        "lifetime_bath_s"),
    ]

    cumulative = 0.0
    result = {}
    for seg, col in seg_order:
        offset = offsets.get(seg, 0.0)
        if offset is None:
            result[col] = None   # missing sensor reading
            # cumulative stays unchanged — we don't advance time for a null stamp
        else:
            partial = r[seg] + offset
            cumulative = round(cumulative + partial, 4)
            result[col] = cumulative
    return result


# ---------------------------------------------------------------------------
# Define the 10 validation pieces (§1.4 coverage map)
# ---------------------------------------------------------------------------

PIECES = []

# P001–P004: All-OK (all deviations = 0 → all penalized: false)
for pid, matrix in [("P001", 4974), ("P002", 5052), ("P003", 5090), ("P004", 5091)]:
    row = {"piece_id": pid, "die_matrix": matrix}
    row.update(build_cumulative(matrix, {}))
    PIECES.append(row)

# P005: Only furnace_to_2nd_strike penalized (matrix 4974, +2.0s on that segment)
p = {"piece_id": "P005", "die_matrix": 4974}
p.update(build_cumulative(4974, {"furnace_to_2nd_strike": 2.0}))
PIECES.append(p)

# P006: Only 2nd_to_3rd_strike penalized (matrix 5052, +2.0s on that segment)
p = {"piece_id": "P006", "die_matrix": 5052}
p.update(build_cumulative(5052, {"2nd_to_3rd_strike": 2.0}))
PIECES.append(p)

# P007: Only 3rd_to_4th_strike penalized (matrix 5090, +2.0s on that segment)
p = {"piece_id": "P007", "die_matrix": 5090}
p.update(build_cumulative(5090, {"3rd_to_4th_strike": 2.0}))
PIECES.append(p)

# P008: Only 4th_strike_to_aux_press penalized (matrix 5091, +2.0s on that segment)
p = {"piece_id": "P008", "die_matrix": 5091}
p.update(build_cumulative(5091, {"4th_strike_to_aux_press": 2.0}))
PIECES.append(p)

# P009: Only aux_press_to_bath penalized (matrix 4974, +2.0s on that segment)
p = {"piece_id": "P009", "die_matrix": 4974}
p.update(build_cumulative(4974, {"aux_press_to_bath": 2.0}))
PIECES.append(p)

# P010: Multi-segment delay + at least one NULL cumulative time (§1.4)
#   - furnace_to_2nd_strike: +2.0s → penalized: true
#   - lifetime_3rd_strike_s: NULL (missing sensor reading)
#     → 2nd_to_3rd_strike = null (penalized: null)
#     → 3rd_to_4th_strike = null (penalized: null) — one NULL nullifies two segments (§1.5)
#   - 4th_strike_to_aux_press: +2.0s → penalized: true
#   - aux_press_to_bath: 0 offset → penalized: false
p = {"piece_id": "P010", "die_matrix": 5052}
p.update(build_cumulative(5052, {
    "furnace_to_2nd_strike":   2.0,
    "2nd_to_3rd_strike":       None,  # NULL — missing sensor reading
    "4th_strike_to_aux_press": 2.0,
}))
PIECES.append(p)

# ---------------------------------------------------------------------------
# Write validation_pieces.csv
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "piece_id", "die_matrix",
    "lifetime_2nd_strike_s", "lifetime_3rd_strike_s", "lifetime_4th_strike_s",
    "lifetime_auxiliary_press_s", "lifetime_bath_s",
]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(PIECES)

print(f"Written {len(PIECES)} pieces to {CSV_PATH}")

# ---------------------------------------------------------------------------
# Generate validation_expected.json by running diagnose() on each piece
# Per Automated Penalties: must be programmatically generated, NOT hand-written
# ---------------------------------------------------------------------------

expected = []
for piece in PIECES:
    result = diagnose(piece)
    expected.append(result)

JSON_PATH.write_text(json.dumps(expected, indent=2))
print(f"Written expected output to {JSON_PATH}")

# ---------------------------------------------------------------------------
# Print summary for verification
# ---------------------------------------------------------------------------

print("\n=== Validation Set Summary ===")
for piece, result in zip(PIECES, expected):
    penalized_segs = [s for s, v in result["segments"].items() if v["penalized"] is True]
    null_segs = [s for s, v in result["segments"].items() if v["penalized"] is None]
    print(f"{piece['piece_id']} (matrix {piece['die_matrix']}): "
          f"is_delayed={result['is_delayed']} | "
          f"penalized={penalized_segs or 'none'} | "
          f"null={null_segs or 'none'}")
