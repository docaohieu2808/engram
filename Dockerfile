# ---- Builder stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only what's needed for installing dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package and all runtime dependencies into a prefix
RUN pip install --no-cache-dir --prefix=/install .


# ---- Runtime stage ----
FROM python:3.11-slim

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ /app/src/

WORKDIR /app

# Default data directory (overridable via env / volume)
ENV ENGRAM_DATA_DIR=/data

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

CMD ["engram", "serve", "--host", "0.0.0.0"]
