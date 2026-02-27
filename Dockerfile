# ---- Builder stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install into isolated venv for clean copy
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir . \
    # ---- Prune heavy packages unused at runtime ----
    # onnxruntime: chromadb default embedder — we use Gemini API instead
    && /opt/venv/bin/pip uninstall -y onnxruntime \
    # sympy: transitive dep via onnxruntime, not needed
    && /opt/venv/bin/pip uninstall -y sympy mpmath \
    # kubernetes: litellm optional dep, engram doesn't use k8s routing
    && /opt/venv/bin/pip uninstall -y kubernetes \
    # hf_xet/huggingface-hub: HF download machinery, not used (we use Gemini API)
    # Keep tokenizers — required by litellm for token counting
    && /opt/venv/bin/pip uninstall -y hf_xet huggingface-hub hf-transfer \
    # pip/setuptools: not needed in runtime image
    && /opt/venv/bin/pip uninstall -y pip setuptools \
    # Strip debug symbols from .so files (~30-50% size reduction on native libs)
    && find /opt/venv -name "*.so" -exec strip --strip-debug {} + 2>/dev/null; \
       find /opt/venv -name "*.so.*" -exec strip --strip-debug {} + 2>/dev/null; \
    # Remove .pyc, __pycache__, tests, dist-info docs
       find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
       find /opt/venv -type d -name tests -exec rm -rf {} + 2>/dev/null; \
       find /opt/venv -type d -name test -exec rm -rf {} + 2>/dev/null; \
       find /opt/venv -name "*.pyc" -delete 2>/dev/null; \
       find /opt/venv -name "*.pyo" -delete 2>/dev/null; \
       find /opt/venv -name "*.dist-info" -type d -exec sh -c 'rm -rf "$1"/RECORD "$1"/LICENSE* "$1"/NOTICE*' _ {} \; 2>/dev/null; \
       true


# ---- Runtime stage (slim, not distroless — needs C extensions) ----
FROM python:3.11-slim

# curl for HEALTHCHECK only
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY src/ /app/src/

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENGRAM_DATA_DIR=/data \
    # 0.0.0.0 inside container is safe — Docker port mapping handles isolation
    ENGRAM_ALLOW_INSECURE=1

# Non-root user with home dir for ~/.engram config
RUN useradd -r -u 1000 -m -d /home/engram -s /sbin/nologin engram \
    && mkdir -p /data /home/engram/.engram \
    && chown -R engram:engram /data /home/engram/.engram

USER engram

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

CMD ["engram", "serve", "--host", "0.0.0.0"]
