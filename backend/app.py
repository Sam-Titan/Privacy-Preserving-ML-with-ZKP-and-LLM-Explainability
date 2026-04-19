"""
backend/app.py
──────────────
FastAPI REST server exposing the privacy-preserving prediction pipeline.

Endpoints
─────────
  GET  /           — health check
  POST /predict    — full pipeline (inference + ZKP + explanation)
  GET  /health     — JSON health status

Run: python -m backend.app
  or: uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_HOST, API_PORT, BASE_DIR
from backend.pipeline import run_pipeline


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Privacy-Preserving Heart Disease Prediction API",
    description=(
        "Combines ML inference + Zero-Knowledge Proofs + LLaMA 3 explanations. "
        "Patient data never leaves the private computation layer."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ── Request / Response schemas ────────────────────────────────────────────────

class PatientInput(BaseModel):
    age:      float = Field(..., ge=0,  le=120, description="Age in years")
    sex:      int   = Field(..., ge=0,  le=1,   description="0=Female, 1=Male")
    cp:       int   = Field(..., ge=0,  le=3,
                            description="0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
    trestbps: float = Field(..., ge=50, le=250, description="Resting BP (mm Hg)")
    chol:     float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,  le=1,   description="Fasting BS >120 mg/dl: 0=No, 1=Yes")
    restecg:  int   = Field(..., ge=0,  le=2,   description="0=Normal, 1=ST-T abnormality, 2=LV hypertrophy")
    thalach:  float = Field(..., ge=50, le=250, description="Max heart rate achieved (bpm)")
    exang:    int   = Field(..., ge=0,  le=1,   description="Exercise-induced angina: 0=No, 1=Yes")
    oldpeak:  float = Field(..., ge=0,  le=10,  description="ST depression")
    slope:    int   = Field(..., ge=0,  le=2,   description="0=Upsloping, 1=Flat, 2=Downsloping")
    ca:       int   = Field(..., ge=0,  le=3,   description="Number of major vessels (0-3)")
    thal:     int   = Field(..., ge=0,  le=2,   description="0=Normal, 1=Fixed Defect, 2=Reversible Defect")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Privacy-Preserving Heart Disease Prediction API v1.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "HeartDiseaseNet", "llm": "llama3 (Ollama)"}


@app.post("/predict")
async def predict(patient: PatientInput):
    """
    Run the full privacy-preserving pipeline for a patient input.

    The API server never logs or stores the raw feature values.
    Only the final prediction, proof metadata, and explanation are returned.
    """
    try:
        raw_input = patient.model_dump()
        result    = await run_pipeline(raw_input)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found. Run the setup steps first. ({exc})",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )