"""
app.py
------
FastAPI service for real-time network intrusion detection.

Endpoints
---------
GET  /health           → liveness + model-load status
POST /predict          → ensemble anomaly prediction
POST /predict/batch    → batch prediction (up to 256 records)
POST /explain          → SHAP explanation for a single record
GET  /metrics          → basic request counters (Prometheus-style text)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from detect_anomaly import DetectionResult, detector
from shap_explain import explain_one

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Simple in-memory counters (replace with Prometheus client in production) ──
_COUNTERS: Dict[str, int] = {
    "requests_total": 0,
    "anomalies_detected": 0,
    "errors_total": 0,
}


# ── Lifespan: load models once at startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models …")
    detector.load()
    logger.info("Models ready. Service is live.")
    yield
    logger.info("Shutting down. Unloading models.")
    detector.unload()


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Network Intrusion Detection API",
    description=(
        "Ensemble ML service (LSTM + Random Forest + XGBoost) "
        "for real-time anomaly detection on UNSW-NB15 network traffic features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "256"))


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class NetworkRecord(BaseModel):
    """
    Raw network-traffic feature record.
    All fields match UNSW-NB15 column names (lowercase).
    Additional fields are passed through and ignored gracefully.
    """

    model_config = {"extra": "allow"}

    # Required top-level identifiers kept optional to support partial records
    dur: Optional[float] = None
    proto: Optional[str] = None
    service: Optional[str] = None
    state: Optional[str] = None


class PredictRequest(BaseModel):
    record: NetworkRecord = Field(..., description="Single network-traffic record")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    records: List[NetworkRecord] = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    is_anomaly: bool
    ensemble_label: int
    confidence: float
    model_scores: Dict[str, float]
    model_labels: Dict[str, int]
    latency_ms: float


class ExplainRequest(BaseModel):
    record: NetworkRecord


# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def count_requests(request: Request, call_next):
    _COUNTERS["requests_total"] += 1
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _COUNTERS["errors_total"] += 1
        raise
    elapsed_ms = (time.perf_counter() - start) * 1_000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "models_loaded": detector._loaded,
        "version": app.version,
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics():
    lines = [f'ids_{k} {v}' for k, v in _COUNTERS.items()]
    return "\n".join(lines) + "\n"


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(body: PredictRequest):
    t0 = time.perf_counter()
    try:
        record_dict = body.record.model_dump()
        results: List[DetectionResult] = detector.predict(
            [record_dict], threshold=body.threshold
        )
        result = results[0]
    except Exception as exc:
        _COUNTERS["errors_total"] += 1
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    if result.is_anomaly:
        _COUNTERS["anomalies_detected"] += 1

    latency_ms = (time.perf_counter() - t0) * 1_000
    return PredictionResponse(
        is_anomaly=result.is_anomaly,
        ensemble_label=result.ensemble_label,
        confidence=result.confidence,
        model_scores=result.model_scores,
        model_labels=result.model_labels,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", tags=["inference"])
def predict_batch(body: BatchPredictRequest):
    if len(body.records) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size {len(body.records)} exceeds limit {MAX_BATCH_SIZE}.",
        )

    t0 = time.perf_counter()
    try:
        records_dicts = [r.model_dump() for r in body.records]
        results: List[DetectionResult] = detector.predict(
            records_dicts, threshold=body.threshold
        )
    except Exception as exc:
        _COUNTERS["errors_total"] += 1
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1_000, 2)
    anomaly_count = sum(r.is_anomaly for r in results)
    _COUNTERS["anomalies_detected"] += anomaly_count

    return {
        "total": len(results),
        "anomaly_count": anomaly_count,
        "latency_ms": latency_ms,
        "predictions": [
            {
                "is_anomaly": r.is_anomaly,
                "ensemble_label": r.ensemble_label,
                "confidence": r.confidence,
                "model_scores": r.model_scores,
            }
            for r in results
        ],
    }


@app.post("/explain", tags=["explainability"])
def explain(body: ExplainRequest):
    try:
        explanation = explain_one(body.record.model_dump())
    except Exception as exc:
        _COUNTERS["errors_total"] += 1
        logger.exception("Explanation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return explanation


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=int(os.getenv("WORKERS", "1")),
    )