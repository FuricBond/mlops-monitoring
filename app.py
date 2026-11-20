"""
MLOps Assignment 8: FastAPI Model Prediction Service
Exposes REST endpoints for model inference, health verification, and Prometheus monitoring.
Instrumented with operational and ML-specific monitoring (confidence and data drift).
"""

from collections import deque
from contextlib import asynccontextmanager
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("iris-prediction-service")

# Resolve model and metadata file paths relative to app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
STATS_PATH = BASE_DIR / "model" / "training_stats.json"

# ==============================================================================
# Prometheus Operational Metrics Definition
# ==============================================================================
PREDICTION_REQUESTS = Counter(
    "ml_prediction_requests_total",
    "Total prediction requests received by the service",
)

PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Total failed prediction requests (including validation and inference failures)",
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Distribution of prediction request latency in seconds",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total successfully generated model predictions",
    ["predicted_class"],
)

# Pre-initialize Prometheus counters for known Iris species to ensure 0.0 values appear at startup
for species_name in ["setosa", "versicolor", "virginica"]:
    PREDICTIONS_TOTAL.labels(predicted_class=species_name)

# ==============================================================================
# Prometheus ML Model Performance and Drift Metrics Definition
# ==============================================================================
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Distribution of prediction confidence (maximum class probability) for successful predictions",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
)

FEATURE_DRIFT_ZSCORE = Gauge(
    "model_feature_drift_zscore",
    "Real-time feature drift z-score comparing rolling inference window to training baseline (simplified mean-shift)",
    ["feature"],
)


# ==============================================================================
# Data Drift Detection Mechanism
# ==============================================================================
class FeatureDriftMonitor:
    """
    Thread-safe rolling window feature drift monitor.
    Uses a simplified mean-shift monitoring approach based on the standard error
    of the mean (Z-score test), comparing recent inference observations to the
    training distribution parameters recorded in training_stats.json.

    Formula:
        z = (recent_mean - training_mean) / (training_std / sqrt(n))

    Note on methodology:
    This is a simplified mean-shift data drift monitoring approach. It measures
    whether the empirical mean of recent incoming features deviates significantly
    from the reference training distribution mean (covariate shift). It provides
    lightweight, real-time alerting suitable for streaming metrics without
    retraining or modifying the underlying model.
    """

    def __init__(self, window_size: int = 100, min_observations: int = 20) -> None:
        self.window_size = window_size
        self.min_observations = min_observations
        self.lock = threading.Lock()
        self.window: deque[List[float]] = deque(maxlen=window_size)

    def record_and_compute_drift(
        self,
        features: List[float],
        feature_names: List[str],
        training_stats: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Records a valid prediction feature vector in a thread-safe manner
        and calculates feature drift z-scores once at least min_observations are collected.
        Returns a dictionary mapping feature names to their z-scores (empty if n < min_observations).
        """
        with self.lock:
            self.window.append(features)
            n = len(self.window)
            if n < self.min_observations:
                return {}

            # Vectorized calculation over current snapshot of the rolling window
            window_data = np.array(self.window, dtype=float)
            recent_means = np.mean(window_data, axis=0)

            z_scores: Dict[str, float] = {}
            mean_dict = training_stats.get("mean", {})
            std_dict = training_stats.get("std", {})

            for idx, feat_name in enumerate(feature_names):
                train_mean = mean_dict.get(feat_name)
                train_std = std_dict.get(feat_name)

                # Safely handle missing, non-positive, or near-zero standard deviations
                if train_mean is None or train_std is None or train_std <= 1e-9:
                    z = 0.0
                else:
                    se = train_std / math.sqrt(n)
                    z = float((recent_means[idx] - train_mean) / se)

                z_scores[feat_name] = round(z, 4)

            return z_scores


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads the model artifact and metadata
    once at application startup and attaches them to app.state.
    """
    logger.info("Initializing application and loading model artifacts...")

    # Validate file existence
    if not MODEL_PATH.exists():
        error_msg = f"Model artifact not found at {MODEL_PATH}. Please train the model first using train.py."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not STATS_PATH.exists():
        error_msg = f"Training statistics file not found at {STATS_PATH}. Please run train.py to generate metadata."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Load model and training statistics
    try:
        model = joblib.load(MODEL_PATH)
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as exc:
        error_msg = f"Failed to load or parse model artifacts from {MODEL_PATH} and {STATS_PATH}: {exc}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from exc

    # Validate model capabilities
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        error_msg = "Loaded model does not implement required predict() and predict_proba() methods."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Store in app.state for request handlers
    app.state.model = model
    app.state.training_stats = stats
    app.state.model_version = stats.get("model_version", "1.0.0")
    app.state.feature_names = stats.get("feature_names", [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ])
    app.state.target_class_names = stats.get("target_class_names", [
        "setosa",
        "versicolor",
        "virginica",
    ])

    # Initialize thread-safe feature drift monitor
    app.state.drift_monitor = FeatureDriftMonitor(window_size=100, min_observations=20)

    # Pre-initialize drift gauges for known features
    for feat_name in app.state.feature_names:
        FEATURE_DRIFT_ZSCORE.labels(feature=feat_name).set(0.0)

    logger.info(
        "Model successfully loaded. Algorithm: %s, Model Version: %s, Target Classes: %s",
        model.__class__.__name__,
        app.state.model_version,
        app.state.target_class_names,
    )

    yield

    logger.info("Shutting down Iris Prediction Service...")


# Initialize FastAPI application
app = FastAPI(
    title="Iris Flower Classification Prediction Service",
    description=(
        "FastAPI prediction service serving a scikit-learn Random Forest model "
        "trained on the Iris dataset. Instrumented with Prometheus metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_prediction_metrics(request: Request, call_next):
    """
    HTTP middleware that measures request volume, errors, and end-to-end latency
    specifically for the POST /predict endpoint without interfering with other routes.
    """
    if request.method == "POST" and request.url.path.rstrip("/") == "/predict":
        PREDICTION_REQUESTS.inc()
        start_time = time.perf_counter()
        is_error = False
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                is_error = True
            return response
        except Exception:
            is_error = True
            raise
        finally:
            latency = time.perf_counter() - start_time
            PREDICTION_LATENCY.observe(latency)
            if is_error:
                PREDICTION_ERRORS.inc()

    return await call_next(request)


class IrisPredictionRequest(BaseModel):
    """
    Pydantic schema for Iris flower measurements.
    Enforces positive numerical values up to 20 cm.
    """
    sepal_length: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Sepal length in centimeters (must be > 0 and <= 20)",
        examples=[5.1],
    )
    sepal_width: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Sepal width in centimeters (must be > 0 and <= 20)",
        examples=[3.5],
    )
    petal_length: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Petal length in centimeters (must be > 0 and <= 20)",
        examples=[1.4],
    )
    petal_width: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Petal width in centimeters (must be > 0 and <= 20)",
        examples=[0.2],
    )


class IrisPredictionResponse(BaseModel):
    """
    Pydantic schema for prediction response.
    """
    prediction: str = Field(description="Predicted Iris species name")
    predicted_species: str = Field(description="Predicted Iris species name")
    predicted_class: int = Field(description="Predicted integer class index (0, 1, or 2)")
    confidence: float = Field(description="Confidence probability associated with the predicted class")
    class_probabilities: Dict[str, float] = Field(description="Predicted probabilities across all classes")
    model_version: str = Field(description="Model artifact version")


@app.get(
    "/",
    tags=["General"],
    summary="Service Root",
    description="Returns a welcome message and directory of available service endpoints.",
)
async def root() -> Dict[str, Any]:
    return {
        "message": "Welcome to the Iris Flower Classification Prediction Service",
        "endpoints": {
            "GET /": "Service root and endpoint directory",
            "GET /health": "Health and model readiness check",
            "POST /predict": "Submit Iris measurements for species classification",
            "GET /metrics": "Prometheus metrics exposition endpoint",
            "GET /docs": "Interactive Swagger API documentation",
            "GET /redoc": "Interactive ReDoc API documentation",
        },
        "model_version": getattr(app.state, "model_version", "unknown"),
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Verifies the operational status of the service and confirms the ML model is loaded.",
)
async def health() -> Dict[str, Any]:
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    if not model_loaded:
        logger.error("Health check failed: model is not loaded in application state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Unavailable: Machine learning model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": getattr(app.state, "model_version", "1.0.0"),
    }


@app.post(
    "/predict",
    response_model=IrisPredictionResponse,
    tags=["Inference"],
    summary="Predict Iris Species",
    description=(
        "Accepts four Iris physical measurements (sepal_length, sepal_width, "
        "petal_length, petal_width) and returns the predicted species, class index, "
        "confidence, class probabilities, and model version."
    ),
)
async def predict(payload: IrisPredictionRequest) -> IrisPredictionResponse:
    # Verify model readiness
    if not hasattr(app.state, "model") or app.state.model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded.",
        )

    try:
        # Preserve exact feature ordering used during model training:
        # 1. sepal length (cm), 2. sepal width (cm), 3. petal length (cm), 4. petal width (cm)
        features = np.array(
            [[
                payload.sepal_length,
                payload.sepal_width,
                payload.petal_length,
                payload.petal_width,
            ]],
            dtype=float,
        )

        model = app.state.model
        raw_pred = model.predict(features)
        raw_proba = model.predict_proba(features)

        pred_class = int(raw_pred[0])
        target_classes = getattr(app.state, "target_class_names", ["setosa", "versicolor", "virginica"])

        if 0 <= pred_class < len(target_classes):
            predicted_species = target_classes[pred_class]
        else:
            predicted_species = f"class_{pred_class}"

        probabilities = {
            target_classes[idx] if idx < len(target_classes) else f"class_{idx}": float(prob)
            for idx, prob in enumerate(raw_proba[0])
        }
        confidence = float(raw_proba[0][pred_class])

        # Increment successful prediction counter with species label
        PREDICTIONS_TOTAL.labels(predicted_class=predicted_species).inc()

        # Record prediction confidence for successful predictions
        PREDICTION_CONFIDENCE.observe(confidence)

        # Update real-time feature drift z-scores
        drift_monitor: Optional[FeatureDriftMonitor] = getattr(app.state, "drift_monitor", None)
        if drift_monitor is not None:
            feature_vector = [
                payload.sepal_length,
                payload.sepal_width,
                payload.petal_length,
                payload.petal_width,
            ]
            drift_scores = drift_monitor.record_and_compute_drift(
                features=feature_vector,
                feature_names=app.state.feature_names,
                training_stats=app.state.training_stats,
            )
            for feat_name, z_val in drift_scores.items():
                FEATURE_DRIFT_ZSCORE.labels(feature=feat_name).set(z_val)

        return IrisPredictionResponse(
            prediction=predicted_species,
            predicted_species=predicted_species,
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            class_probabilities={k: round(v, 4) for k, v in probabilities.items()},
            model_version=getattr(app.state, "model_version", "1.0.0"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error occurred during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing prediction request.",
        ) from None


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus Metrics",
    description="Exposes application and model serving metrics for Prometheus scraping.",
)
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
