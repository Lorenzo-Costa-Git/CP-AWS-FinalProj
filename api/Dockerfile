# Per §2.1: packages the FastAPI app and reference_times.json into one image.
# reference_times.json is bundled here so it is loaded once at startup —
# not fetched from S3 or read on every request.

FROM python:3.13-slim

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install only runtime dependencies (pandas/pyarrow are compute-only, not needed here)
RUN uv pip install --system fastapi uvicorn

# Copy application source
COPY src/ ./src/

# Bundle reference_times.json inside the image (§2.1: loaded once at startup)
COPY reference_times.json .

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
