"""
MLOps Assignment 8: FastAPI Model Prediction Service
Exposes REST endpoints for model inference and health verification.
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

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
        "trained on the Iris dataset. Part of MLOps Assignment 8 monitoring framework."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


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
