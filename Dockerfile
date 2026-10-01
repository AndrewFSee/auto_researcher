# Auto Researcher: Streamlit dashboard + CLI tools
#
#   docker build -t auto-researcher .
#   docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data:/app/data" auto-researcher
#
# Extras are selected with a build arg, e.g. --build-arg EXTRAS=dashboard,llm,nlp

# -----------------------------------------------------------------------------
# Build stage: install the package into a virtualenv
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ARG EXTRAS=dashboard,llm
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[${EXTRAS}]"

# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp is needed by LightGBM/XGBoost at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r researcher && useradd -r -g researcher -m researcher

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
# app.py and the pipeline scripts resolve paths relative to /app
COPY app.py ./
COPY .streamlit/ .streamlit/
COPY scripts/ scripts/
COPY src/ src/
RUN mkdir -p data logs && chown -R researcher:researcher /app

USER researcher
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
