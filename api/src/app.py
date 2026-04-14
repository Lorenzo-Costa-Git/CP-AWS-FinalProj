"""
app.py — Forging Line Delay Diagnostics API

Per §2.1: exposes exactly two routes:
  GET  /openapi.json  (served automatically by FastAPI)
  POST /diagnose      (receives timing data, returns structured diagnosis)

Reference times and cause table are loaded once at startup — not on every
request — as required by §2.1 ("Data loading: at startup the API must load
two files bundled inside the container image").
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diagnose import diagnose

app = FastAPI(title="Forging Line Delay Diagnostics API")


# Per §2.1: invalid or missing body fields must return HTTP 400 (not FastAPI's default 422)
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Request schema (§2.1) — 7 fields: piece_id, die_matrix, 5 cumulative times
# Any of the lifetime_* fields may be null (missing sensor reading, per §1.5)
# ---------------------------------------------------------------------------

class PieceRequest(BaseModel):
    piece_id: str
    die_matrix: int
    lifetime_2nd_strike_s: float | None = None
    lifetime_3rd_strike_s: float | None = None
    lifetime_4th_strike_s: float | None = None
    lifetime_auxiliary_press_s: float | None = None
    lifetime_bath_s: float | None = None


# ---------------------------------------------------------------------------
# POST /diagnose
# ---------------------------------------------------------------------------

@app.post("/diagnose")
def post_diagnose(piece: PieceRequest) -> dict:
    """
    Receives cumulative timing data for one piece and returns a structured
    delay diagnosis per the rules defined in §1.3 and §1.4.
    """
    try:
        return diagnose(piece.model_dump())
    except ValueError as exc:
        # Unknown die_matrix → HTTP 400 per §2.1, exact format: {"error": "unknown die_matrix <value>"}
        return JSONResponse(status_code=400, content={"error": str(exc)})
