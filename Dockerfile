FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies (no dev extras)
RUN uv sync --no-dev

# Copy application code
COPY app.py ./
COPY pages/ ./pages/
COPY ui/ ./ui/
COPY config/ ./config/
COPY data/fixtures/ ./data/fixtures/
COPY .streamlit/ ./.streamlit/
COPY DISCLAIMER.md ./

# Generate fixtures if not present
RUN uv run python data/fixtures/generate.py 2>/dev/null || true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/?health=1 || exit 1

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501"]
