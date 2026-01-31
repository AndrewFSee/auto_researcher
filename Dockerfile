# Auto Researcher
# Multi-stage build for production deployment

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.lock .
RUN pip install --no-cache-dir --user -r requirements.lock

# Copy source code
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir --user -e .


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

# Security: Run as non-root user
RUN groupadd -r researcher && useradd -r -g researcher researcher

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/researcher/.local

# Copy application code
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/

# Copy additional files
COPY scripts/ scripts/
COPY configs/ configs/

# Set up directories
RUN mkdir -p /app/data /app/logs /app/cache \
    && chown -R researcher:researcher /app

# Environment variables
ENV PATH=/home/researcher/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # Application settings
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data \
    CACHE_DIR=/app/cache

# Switch to non-root user
USER researcher

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import auto_researcher; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "auto_researcher"]


# =============================================================================
# Stage 3: Development
# =============================================================================
FROM runtime as development

USER root

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    ipython \
    jupyter

# Install additional tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER researcher

# Override for dev
CMD ["python", "-m", "pytest", "tests/", "-v"]
