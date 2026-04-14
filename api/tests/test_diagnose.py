"""
test_diagnose.py

Per §2.2:
- 24 unit tests (4 matrices × 6 scenarios) calling diagnose() directly as a
  pure function — no HTTP server needed.
- 1 parametrized golden test loading validation_pieces.csv and asserting
  output matches validation_expected.json (rounded to 1 decimal).
"""

import csv
import json
from pathlib import Path

import pytest

from diagnose import diagnose, REFERENCE_TIMES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

API_DIR = Path(__file__).parent.parent
VALIDATION_CSV = API_DIR / "validation_pieces.csv"
VALIDATION_JSON = API_DIR / "validation_expected.json"

# ---------------------------------------------------------------------------
# Helper: build a piece dict from reference + per-segment offsets
#
# offset=0   → deviation=0   → penalized: false  (all-OK)
# offset=2.0 → deviation=2.0 → penalized: true   (1 < dev ≤ 5)
# offset=None → that cumulative timestamp is NULL (missing sensor reading)
# ---------------------------------------------------------------------------

SEGMENT_TO_CUMULATIVE = [
    ("furnace_to_2nd_strike",    "lifetime_2nd_strike_s"),
    ("2nd_to_3rd_strike",        "lifetime_3rd_strike_s"),
    ("3rd_to_4th_strike",        "lifetime_4th_strike_s"),
    ("4th_strike_to_aux_press",  "lifetime_auxiliary_press_s"),
    ("aux_press_to_bath",        "lifetime_bath_s"),
]


def build_piece(piece_id: str, matrix: int, offsets: dict) -> dict:
    """
    Build a piece dict with cumulative timestamps.
    offsets: {segment_name: extra_seconds | None}
    Segments not in offsets use their reference value (offset=0).
    A None offset means that cumulative timestamp is missing (NULL).
    """
    refs = REFERENCE_TIMES[str(matrix)]
    cumulative = 0.0
    piece = {"piece_id": piece_id, "die_matrix": matrix}

    for seg, col in SEGMENT_TO_CUMULATIVE:
        offset = offsets.get(seg, 0.0)
        if offset is None:
            piece[col] = None
            # do not advance cumulative — timestamp is missing
        else:
            partial = refs[seg] + offset
            cumulative = round(cumulative + partial, 4)
            piece[col] = cumulative

    return piece


# ---------------------------------------------------------------------------
# 24 unit tests: 4 matrices × 6 scenarios (§2.2)
# ---------------------------------------------------------------------------

MATRICES = [4974, 5052, 5090, 5091]
SEGMENTS = [seg for seg, _ in SEGMENT_TO_CUMULATIVE]


@pytest.mark.parametrize("matrix", MATRICES)
def test_all_ok(matrix):
    """Scenario 1: all segments at reference → is_delayed False, all penalized False."""
    piece = build_piece("T_OK", matrix, {})
    result = diagnose(piece)

    assert result["is_delayed"] is False
    assert result["probable_causes"] == []
    for seg, data in result["segments"].items():
        assert data["penalized"] is False, f"{seg} should not be penalized"


@pytest.mark.parametrize("matrix", MATRICES)
def test_furnace_to_2nd_strike_penalized(matrix):
    """Scenario 2: furnace_to_2nd_strike deviation=2.0 → penalized True, others False."""
    piece = build_piece("T_F2", matrix, {"furnace_to_2nd_strike": 2.0})
    result = diagnose(piece)

    assert result["is_delayed"] is True
    assert result["segments"]["furnace_to_2nd_strike"]["penalized"] is True
    for seg in SEGMENTS[1:]:
        assert result["segments"][seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_2nd_to_3rd_strike_penalized(matrix):
    """Scenario 3: 2nd_to_3rd_strike deviation=2.0 → penalized True, others False."""
    piece = build_piece("T_23", matrix, {"2nd_to_3rd_strike": 2.0})
    result = diagnose(piece)

    assert result["is_delayed"] is True
    assert result["segments"]["2nd_to_3rd_strike"]["penalized"] is True
    for seg in [s for s in SEGMENTS if s != "2nd_to_3rd_strike"]:
        assert result["segments"][seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_3rd_to_4th_strike_penalized(matrix):
    """Scenario 4: 3rd_to_4th_strike deviation=2.0 → penalized True, others False."""
    piece = build_piece("T_34", matrix, {"3rd_to_4th_strike": 2.0})
    result = diagnose(piece)

    assert result["is_delayed"] is True
    assert result["segments"]["3rd_to_4th_strike"]["penalized"] is True
    for seg in [s for s in SEGMENTS if s != "3rd_to_4th_strike"]:
        assert result["segments"][seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_4th_strike_to_aux_press_penalized(matrix):
    """Scenario 5: 4th_strike_to_aux_press deviation=2.0 → penalized True, others False."""
    piece = build_piece("T_4A", matrix, {"4th_strike_to_aux_press": 2.0})
    result = diagnose(piece)

    assert result["is_delayed"] is True
    assert result["segments"]["4th_strike_to_aux_press"]["penalized"] is True
    for seg in [s for s in SEGMENTS if s != "4th_strike_to_aux_press"]:
        assert result["segments"][seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_aux_press_to_bath_penalized(matrix):
    """Scenario 6: aux_press_to_bath deviation=2.0 → penalized True, others False."""
    piece = build_piece("T_AB", matrix, {"aux_press_to_bath": 2.0})
    result = diagnose(piece)

    assert result["is_delayed"] is True
    assert result["segments"]["aux_press_to_bath"]["penalized"] is True
    for seg in SEGMENTS[:-1]:
        assert result["segments"][seg]["penalized"] is False


# ---------------------------------------------------------------------------
# Golden test: parametrized over all 10 validation pieces (§2.2)
#
# Loads validation_pieces.csv, runs diagnose() on each piece, and asserts
# the output matches validation_expected.json exactly (rounded to 1 decimal).
# ---------------------------------------------------------------------------

def _load_validation_pieces() -> list[dict]:
    pieces = []
    with open(VALIDATION_CSV, newline="") as f:
        for row in csv.DictReader(f):
            piece = {
                "piece_id": row["piece_id"],
                "die_matrix": int(row["die_matrix"]),
            }
            for col in [
                "lifetime_2nd_strike_s", "lifetime_3rd_strike_s",
                "lifetime_4th_strike_s", "lifetime_auxiliary_press_s",
                "lifetime_bath_s",
            ]:
                v = row.get(col, "")
                piece[col] = float(v) if v not in ("", "None", "null") else None
            pieces.append(piece)
    return pieces


def _round_floats(obj, decimals: int = 1):
    """Recursively round all floats in a nested dict/list."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


_PIECES = _load_validation_pieces()
_EXPECTED = json.loads(VALIDATION_JSON.read_text())


@pytest.mark.parametrize("piece,expected", zip(_PIECES, _EXPECTED), ids=[p["piece_id"] for p in _PIECES])
def test_golden(piece, expected):
    """
    Golden test (§2.2): diagnose() output must match validation_expected.json
    exactly after rounding to 1 decimal.
    """
    result = diagnose(piece)
    assert _round_floats(result) == _round_floats(expected)
