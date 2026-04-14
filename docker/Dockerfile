# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix for copying
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Network Intrusion Detection Service" \
      org.opencontainers.image.description="Ensemble ML API (LSTM + RF + XGBoost)" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Model & artefact locations
    ARTIFACT_DIR=/app/artifacts \
    # API config
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=INFO \
    # Kafka config (override at runtime)
    KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
    KAFKA_TOPIC=network_traffic \
    KAFKA_ALERT_TOPIC=intrusion_alerts \
    KAFKA_GROUP_ID=ids_consumer_group \
    ANOMALY_THRESHOLD=0.5 \
    MAX_BATCH_SIZE=256

# Runtime system deps (libgomp for XGBoost / LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY app.py \
     detect_anomaly.py \
     preprocess.py \
     sequence_builder.py \
     shap_explain.py \
     kafka_consumer.py \
     kafka_producer.py \
     train_autoencoder.py \
     ./

# Create artefacts directory (populated at runtime via volume or training job)
RUN mkdir -p /app/artifacts

# Non-root user for security
RUN addgroup --system ids && adduser --system --ingroup ids ids
RUN chown -R ids:ids /app
USER ids

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

# Default: run the API server
# Override CMD to run the consumer: ["python", "kafka_consumer.py"]
CMD ["sh", "-c", \
     "uvicorn app:app --host $HOST --port $PORT --workers $WORKERS --log-level ${LOG_LEVEL,,}"]