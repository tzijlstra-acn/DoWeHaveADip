FROM python:3.12-slim

WORKDIR /app

# Install uv at a pinned version for reproducible builds
RUN pip install uv==0.5.4

# Copy lockfile and project metadata first for layer caching
COPY uv.lock pyproject.toml ./
COPY src/ ./src/

# Install dependencies from the locked lockfile (no dev extras)
RUN uv sync --frozen --no-dev

# Copy application code
COPY app.py ./
COPY pages/ ./pages/
COPY ui/ ./ui/
COPY config/ ./config/
COPY data/episodes.yaml ./data/
COPY .streamlit/ ./.streamlit/
COPY DISCLAIMER.md ./

EXPOSE 8501

# Use the native Streamlit health endpoint (no curl required)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501"]
