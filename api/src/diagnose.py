import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Static data — loaded once at import time, not on every request
# ---------------------------------------------------------------------------

# Cause table from §1.1
CAUSE_TABLE = {
    "furnace_to_2nd_strike": [
        "Billet pick, gripper close, grip retries, trajectory, permissions, queues",
    ],
    "2nd_to_3rd_strike": [
        "Press retraction, gripper, press/PLC handshake, wait points, regrip",
    ],
    "3rd_to_4th_strike": [
        "Retraction, conservative trajectory, synchronization, positioning, confirmations",
    ],
    "4th_strike_to_aux_press": [
        "Pick micro-corrections, transfer, queue at Auxiliary Press entry, interlocks",
    ],
    "aux_press_to_bath": [
        "Retraction, transport, bath queues, permissions, bath deposit",
    ],
}

# Segment processing order
SEGMENTS = list(CAUSE_TABLE.keys())

# Reference times loaded from file (bundled inside the container image)
_REF_PATH = Path(__file__).parent.parent / "reference_times.json"
with open(_REF_PATH) as _f:
    REFERENCE_TIMES: dict = json.load(_f)

# ---------------------------------------------------------------------------
# Partial time computation (§1.5)
# ---------------------------------------------------------------------------

def _compute_partial_times(piece: dict) -> dict[str, float | None]:
    """Derive 5 partial times from cumulative timestamps.
    If either operand is None/missing the partial time is None."""

    def get(key):
        v = piece.get(key)
        return None if v is None else float(v)

    t2 = get("lifetime_2nd_strike_s")
    t3 = get("lifetime_3rd_strike_s")
    t4 = get("lifetime_4th_strike_s")
    ta = get("lifetime_auxiliary_press_s")
    tb = get("lifetime_bath_s")

    def diff(a, b):
        return None if (a is None or b is None) else round(b - a, 4)

    return {
        "furnace_to_2nd_strike":  t2,                # absolute — no prior timestamp
        "2nd_to_3rd_strike":      diff(t2, t3),
        "3rd_to_4th_strike":      diff(t3, t4),
        "4th_strike_to_aux_press": diff(t4, ta),
        "aux_press_to_bath":      diff(ta, tb),
    }

# ---------------------------------------------------------------------------
# Delay-detection rule (§1.3)
# ---------------------------------------------------------------------------

def _penalize(actual: float | None, reference: float) -> tuple[str | None, float | None]:
    """Return (penalized, deviation_s) for one segment."""
    if actual is None:
        return None, None
    deviation = round(actual - reference, 4)
    if deviation > 5.0:
        penalized = None          # sensor anomaly
    elif deviation > 1.0:
        penalized = True
    else:
        penalized = False         # includes negative (faster than reference)
    return penalized, deviation

# ---------------------------------------------------------------------------
# Main pure function
# ---------------------------------------------------------------------------

def diagnose(piece: dict) -> dict:
    """
    Takes a piece dict with cumulative timestamps and returns a diagnosis dict.
    Pure function — no FastAPI or HTTP dependencies.
    """
    die_matrix = str(piece.get("die_matrix", ""))
    if die_matrix not in REFERENCE_TIMES:
        raise ValueError(f"unknown die_matrix {piece.get('die_matrix')}")

    refs = REFERENCE_TIMES[die_matrix]
    partials = _compute_partial_times(piece)

    segments_out = {}
    probable_causes = []

    for seg in SEGMENTS:
        actual = partials[seg]
        reference = refs[seg]
        penalized, deviation = _penalize(actual, reference)

        segments_out[seg] = {
            "actual_s": actual,
            "reference_s": reference,
            "deviation_s": deviation,
            "penalized": penalized,
            "probable_causes": CAUSE_TABLE[seg] if penalized is True else [],
        }

        if penalized is True:
            probable_causes.extend(CAUSE_TABLE[seg])

    is_delayed = any(s["penalized"] is True for s in segments_out.values())

    return {
        "piece_id": piece.get("piece_id"),
        "die_matrix": piece.get("die_matrix"),
        "is_delayed": is_delayed,
        "probable_causes": probable_causes,
        "segments": segments_out,
    }
