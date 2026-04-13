FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (no dev groups, no editable install yet)
RUN uv sync --frozen --no-dev --no-install-project

# Copy only what the app needs
# models/ is intentionally excluded — inference is done via SageMaker endpoint
COPY src/ ./src/
COPY app/ ./app/
COPY data/gold/ ./data/gold/

# Install the project package
RUN uv sync --frozen --no-dev

EXPOSE 8501

# SAGEMAKER_ENDPOINT_NAME and AWS_DEFAULT_REGION are injected at runtime
# by ECS task environment variables.
CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
