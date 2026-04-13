"""
Forging Line — Piece Travel Time Dashboard

Displays processed pieces with predicted bath time and per-stage
timing detail.

Usage (local model):
    uv run streamlit run app/streamlit_app.py

Usage (SageMaker endpoint):
    SAGEMAKER_ENDPOINT_NAME=vaultech-bath-predictor \
    AWS_DEFAULT_REGION=eu-west-1 \
    uv run streamlit run app/streamlit_app.py
"""

import os
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vaultech_analysis.inference import Predictor, SageMakerPredictor

GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "pieces.parquet"

SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")

PARTIAL_COLS = [
    "partial_furnace_to_2nd_strike_s",
    "partial_2nd_to_3rd_strike_s",
    "partial_3rd_to_4th_strike_s",
    "partial_4th_strike_to_auxiliary_press_s",
    "partial_auxiliary_press_to_bath_s",
]
PARTIAL_LABELS = [
    "Furnace → 2nd strike",
    "2nd strike → 3rd strike",
    "3rd strike → 4th strike",
    "4th strike → Aux. press",
    "Aux. press → Bath",
]
CUMULATIVE_COLS = [
    "lifetime_2nd_strike_s",
    "lifetime_3rd_strike_s",
    "lifetime_4th_strike_s",
    "lifetime_auxiliary_press_s",
    "lifetime_bath_s",
]
CUMULATIVE_LABELS = [
    "2nd strike (1st op)",
    "3rd strike (2nd op)",
    "4th strike (drill)",
    "Auxiliary press",
    "Bath",
]


@st.cache_resource
def load_predictor():
    if SAGEMAKER_ENDPOINT:
        return SageMakerPredictor(endpoint_name=SAGEMAKER_ENDPOINT, region=AWS_REGION)
    return Predictor(
        model_dir=PROJECT_ROOT / "models",
        gold_file=GOLD_FILE,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    predictor = load_predictor()
    df = pd.read_parquet(GOLD_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["predicted_bath_s"] = predictor.predict_batch(df)
    df["prediction_error_s"] = df["predicted_bath_s"] - df["lifetime_bath_s"]
    return df


@st.cache_data
def get_reference() -> pd.DataFrame:
    df = load_data()
    return df.groupby("die_matrix")[PARTIAL_COLS + CUMULATIVE_COLS].median()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forging Line Dashboard", layout="wide")
st.title("Forging Line — Piece Travel Time Dashboard")

if SAGEMAKER_ENDPOINT:
    st.caption(f"Inference: SageMaker endpoint `{SAGEMAKER_ENDPOINT}` ({AWS_REGION})")
else:
    st.caption("Inference: local XGBoost model")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data and running predictions…"):
    df = load_data()
    ref = get_reference()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

matrices = sorted(df["die_matrix"].unique().tolist())
selected_matrices = st.sidebar.multiselect(
    "Die matrix", matrices, default=matrices
)

min_date = df["timestamp"].dt.date.min()
max_date = df["timestamp"].dt.date.max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)

slow_only = st.sidebar.checkbox("Show only slow pieces (> 90th percentile)", value=False)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df[df["die_matrix"].isin(selected_matrices)].copy()

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start_date) &
        (filtered["timestamp"].dt.date <= end_date)
    ]

if slow_only:
    p90 = df.groupby("die_matrix")["lifetime_bath_s"].quantile(0.90)
    filtered = filtered[filtered["lifetime_bath_s"] > filtered["die_matrix"].map(p90)]

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pieces", f"{len(filtered):,}")
col2.metric("Median bath time", f"{filtered['lifetime_bath_s'].median():.1f}s")
col3.metric("Median predicted", f"{filtered['predicted_bath_s'].median():.1f}s")
col4.metric("MAE (filtered)", f"{filtered['prediction_error_s'].abs().mean():.2f}s")

# ── Pieces table ──────────────────────────────────────────────────────────────
st.subheader("Pieces")

TABLE_COLS = [
    "timestamp", "piece_id", "die_matrix",
    "lifetime_bath_s", "predicted_bath_s", "prediction_error_s",
    "oee_cycle_time_s",
]
table_df = filtered[TABLE_COLS].copy()
table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
table_df = table_df.rename(columns={
    "lifetime_bath_s":     "actual_bath_s",
    "predicted_bath_s":    "predicted_bath_s",
    "prediction_error_s":  "error_s",
})

event = st.dataframe(
    table_df.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# ── Piece detail panel ────────────────────────────────────────────────────────
selected_rows = event.selection.get("rows", []) if event.selection else []

if selected_rows:
    piece = filtered.reset_index(drop=True).iloc[selected_rows[0]]
    matrix = int(piece["die_matrix"])
    matrix_ref = ref.loc[matrix] if matrix in ref.index else None

    st.subheader(f"Piece detail — {piece['piece_id']}  |  Matrix {matrix}  |  {piece['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Inference debug panel ─────────────────────────────────────────────────
    if SAGEMAKER_ENDPOINT:
        st.markdown("#### Inference debug — SageMaker endpoint call")
        predictor = load_predictor()
        t0 = time.time()
        result = predictor.predict(
            die_matrix=matrix,
            lifetime_2nd_strike_s=float(piece["lifetime_2nd_strike_s"]),
            oee_cycle_time_s=float(piece["oee_cycle_time_s"]) if pd.notna(piece["oee_cycle_time_s"]) else None,
        )
        debug = result.get("_debug", {})

        d1, d2, d3 = st.columns(3)
        d1.metric("Endpoint", debug.get("endpoint", SAGEMAKER_ENDPOINT))
        d2.metric("Predicted bath time", f"{result.get('predicted_bath_time_s', '—')}s")
        d3.metric("Round-trip latency", f"{debug.get('latency_ms', '—')} ms")

        col_pay, col_resp = st.columns(2)
        with col_pay:
            st.markdown("**Input payload (CSV)**")
            st.code(debug.get("payload", ""), language="text")
        with col_resp:
            st.markdown("**Raw endpoint response**")
            st.code(debug.get("raw_response", ""), language="text")

    # ── Cumulative times vs reference ─────────────────────────────────────────
    st.markdown("#### Cumulative travel times vs matrix reference")
    cum_rows = []
    for col, label in zip(CUMULATIVE_COLS, CUMULATIVE_LABELS):
        actual = piece[col]
        ref_val = matrix_ref[col] if matrix_ref is not None else None
        deviation = (actual - ref_val) if (actual is not None and ref_val is not None) else None
        cum_rows.append({
            "Stage": label,
            "Actual (s)": round(actual, 2) if pd.notna(actual) else "—",
            "Reference (s)": round(ref_val, 2) if ref_val is not None else "—",
            "Deviation (s)": round(deviation, 2) if deviation is not None else "—",
        })
    st.dataframe(pd.DataFrame(cum_rows), use_container_width=True, hide_index=True)

    # ── Partial times vs reference ────────────────────────────────────────────
    st.markdown("#### Partial times vs matrix reference")
    partial_rows = []
    for col, label in zip(PARTIAL_COLS, PARTIAL_LABELS):
        actual = piece[col]
        ref_val = matrix_ref[col] if matrix_ref is not None else None
        deviation = (actual - ref_val) if (pd.notna(actual) and ref_val is not None) else None
        if deviation is not None:
            status = "🔴 SLOW" if deviation > 2 else "🟢 OK"
        else:
            status = "—"
        partial_rows.append({
            "Segment": label,
            "Actual (s)": round(actual, 2) if pd.notna(actual) else "—",
            "Reference (s)": round(ref_val, 2) if ref_val is not None else "—",
            "Deviation (s)": round(deviation, 2) if deviation is not None else "—",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(partial_rows), use_container_width=True, hide_index=True)

    # ── Bar chart: actual vs reference ────────────────────────────────────────
    st.markdown("#### Process synoptic — actual vs reference partial times")
    chart_data = []
    for col, label in zip(PARTIAL_COLS, PARTIAL_LABELS):
        actual = piece[col]
        ref_val = matrix_ref[col] if matrix_ref is not None else None
        if pd.notna(actual):
            chart_data.append({"Segment": label, "Time (s)": round(actual, 2), "Type": "Actual"})
        if ref_val is not None:
            chart_data.append({"Segment": label, "Time (s)": round(ref_val, 2), "Type": "Reference"})

    if chart_data:
        chart = alt.Chart(pd.DataFrame(chart_data)).mark_bar(opacity=0.8).encode(
            x=alt.X("Segment:N", sort=PARTIAL_LABELS, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Time (s):Q"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Actual", "Reference"],
                range=["#e45756", "#4c78a8"]
            )),
            xOffset="Type:N",
            tooltip=["Segment:N", "Type:N", "Time (s):Q"],
        ).properties(width=600, height=300)
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("Select a piece from the table above to see its per-stage timing detail.")
