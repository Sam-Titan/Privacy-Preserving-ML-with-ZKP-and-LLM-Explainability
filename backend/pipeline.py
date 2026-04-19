"""
backend/pipeline.py
───────────────────
Orchestrates the full privacy-preserving inference pipeline:

  raw_input (dict)
      │
      ▼
  [1] Preprocess  ─── StandardScaler ──► scaled_input  (kept private)
      │
      ▼
  [2] ML Inference ─► prediction_prob                  (private computation)
      │
      ▼
  [3] ZKP Prove   ─── EZKL ──────────► proof.json      (proves output is genuine)
      │
      ▼
  [4] ZKP Verify  ─── EZKL ──────────► is_valid        (any verifier can do this)
      │
      ▼
  [5] LLM Explain ─── LLaMA 3 ───────► explanation     (sees only prediction)
      │
      ▼
  Final JSON response
"""
import asyncio
import json
import os
import sys
from typing import Any

import numpy as np
import torch

# --- WINDOWS RUST PANIC FIX ---
# Inject Linux-style environment variables so Rust doesn't panic on Windows
os.environ["HOME"] = os.environ.get("USERPROFILE", "C:\\")
os.environ["XDG_DATA_HOME"] = os.environ.get("APPDATA", "C:\\")
os.environ["XDG_CONFIG_HOME"] = os.environ.get("LOCALAPPDATA", "C:\\")
os.environ["RUST_BACKTRACE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR
from models.model import HeartDiseaseNet
from models.preprocess import load_scaler, load_feature_cols
from zkp.prove import generate_proof
from zkp.verify import verify_proof
from explainer.llm_explainer import get_llm_explanation


# ── Module-level cache so we don't reload on every request ────────────────────
_model_cache: dict[str, Any] = {}


def _load_artifacts():
    if not _model_cache:
        feature_cols = load_feature_cols()
        scaler       = load_scaler()

        model = HeartDiseaseNet(input_size=len(feature_cols))
        model.load_state_dict(
            torch.load(os.path.join(MODELS_DIR, "heart_model.pt"),
                       map_location="cpu")
        )
        model.eval()

        _model_cache["model"]        = model
        _model_cache["scaler"]       = scaler
        _model_cache["feature_cols"] = feature_cols

    return (
        _model_cache["model"],
        _model_cache["scaler"],
        _model_cache["feature_cols"],
    )


async def run_pipeline(raw_input: dict) -> dict:
    """
    Parameters
    ----------
    raw_input : dict
        Unscaled patient features, keyed by column name.
        Example: {"age": 63, "sex": 1, "cp": 3, ...}

    Returns
    -------
    dict with keys:
        prediction : {label, probability, confidence_pct}
        zkp        : {proof_valid, proof_size_kb, proof_time_s,
                      witness_time_s, verify_time_s, message}
        explanation: str
    """
    model, scaler, feature_cols = _load_artifacts()
    
    torch.set_num_threads(1)

    # ── Step 1: Build feature vector ──────────────────────────────────────────
    raw_vector = np.array(
        [[raw_input[col] for col in feature_cols]], dtype=np.float32
    )
    scaled_vector = scaler.transform(raw_vector)   # shape: (1, n_features)

    # ── Step 2: Private ML inference ─────────────────────────────────────────
    with torch.no_grad():
        prob = model(torch.FloatTensor(scaled_vector)).item()

    label      = "Heart Disease Detected" if prob >= 0.5 else "No Heart Disease Detected"
    confidence = (prob if prob >= 0.5 else 1.0 - prob) * 100

    print(f"\n[Pipeline] Prediction: {label}  ({prob:.4f})", flush=True)

    # ── Step 3: Generate ZKP ──────────────────────────────────────────────────
    proof_info = await asyncio.to_thread(generate_proof, scaled_vector[0].tolist())

    # ── Step 4: Verify ZKP ───────────────────────────────────────────────────
    verify_info = await asyncio.to_thread(verify_proof)

    # ── Step 5: LLM explanation ───────────────────────────────────────────────
    # LLM receives only the prediction + human-readable feature labels.
    # It never sees the raw scaled_vector used in the ZKP.
    print("[Pipeline] Requesting LLM explanation …", flush=True)
    
    # Run the blocking Groq network request in a background thread so it doesn't freeze FastAPI
    explanation = await asyncio.to_thread(
        get_llm_explanation,
        prediction_prob=prob,
        raw_input=raw_input,
        proof_valid=verify_info["is_valid"]
    )
    
    print("   ✓ Explanation generated successfully!", flush=True)

    return {
        "prediction": {
            "label":          label,
            "probability":    round(prob, 4),
            "confidence_pct": round(confidence, 1),
        },
        "zkp": {
            "proof_valid":    verify_info["is_valid"],
            "proof_size_kb":  proof_info["proof_size_kb"],
            "proof_time_s":   proof_info["proof_time_s"],
            "witness_time_s": proof_info["witness_time_s"],
            "verify_time_s":  verify_info["verify_time_s"],
            "message":        verify_info["message"],
        },
        "explanation": explanation,
    }


# ── CLI quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    result = asyncio.run(run_pipeline(sample))
    print(json.dumps({k: v for k, v in result.items() if k != "explanation"},
                     indent=2))
    print("\n── EXPLANATION ──────────────────────────────\n")
    print(result["explanation"])