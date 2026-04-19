"""
explainer/llm_explainer.py
──────────────────────────
Generates a compassionate, plain-English explanation of a heart disease
prediction using Groq's cloud inference API.

Privacy guarantee
─────────────────
The LLM sees ONLY:
  • The final prediction (label + probability)
  • Human-readable summaries of feature values  (no raw scaled numbers)
  • Whether the ZKP was valid

It never receives the raw or scaled input array used for ZKP generation.

Prerequisites
─────────────
1. Get an API key from https://console.groq.com
2. Add GROQ_API_KEY and GROQ_MODEL to your config.py
"""
import os
import sys
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Swapped Ollama imports for Groq imports
try:
    from config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL = "llama-3.1-8b-instant"  # <-- UPDATE THIS FALLBACK
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 1024

# ── Human-readable label mappings ─────────────────────────────────────────────

_SEX_LABELS     = {0: "Female", 1: "Male"}
_CP_LABELS      = {0: "Typical Angina", 1: "Atypical Angina",
                   2: "Non-anginal Pain", 3: "Asymptomatic"}
_RESTECG_LABELS = {0: "Normal", 1: "ST-T Wave Abnormality",
                   2: "Left Ventricular Hypertrophy"}
_EXANG_LABELS   = {0: "No", 1: "Yes"}
_SLOPE_LABELS   = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
_THAL_LABELS    = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}
_FBS_LABELS     = {0: "No (≤120 mg/dl)", 1: "Yes (>120 mg/dl)"}

_FEATURE_DISPLAY = {
    "age":      ("Age",                          lambda v: f"{int(v)} years"),
    "sex":      ("Sex",                          lambda v: _SEX_LABELS.get(int(v), str(v))),
    "cp":       ("Chest Pain Type",              lambda v: _CP_LABELS.get(int(v), str(v))),
    "trestbps": ("Resting Blood Pressure",       lambda v: f"{v:.0f} mm Hg"),
    "chol":     ("Serum Cholesterol",            lambda v: f"{v:.0f} mg/dl"),
    "fbs":      ("Fasting Blood Sugar >120 mg/dl", lambda v: _FBS_LABELS.get(int(v), str(v))),
    "restecg":  ("Resting ECG",                  lambda v: _RESTECG_LABELS.get(int(v), str(v))),
    "thalach":  ("Maximum Heart Rate",           lambda v: f"{v:.0f} bpm"),
    "exang":    ("Exercise-Induced Angina",      lambda v: _EXANG_LABELS.get(int(v), str(v))),
    "oldpeak":  ("ST Depression (Oldpeak)",      lambda v: f"{v:.1f}"),
    "slope":    ("Slope of Peak ST Segment",     lambda v: _SLOPE_LABELS.get(int(v), str(v))),
    "ca":       ("Major Vessels (Fluoroscopy)",  lambda v: str(int(v))),
    "thal":     ("Thalassemia",                  lambda v: _THAL_LABELS.get(int(v), str(v))),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_patient_summary(raw_input: dict) -> str:
    lines = []
    for key, val in raw_input.items():
        if key in _FEATURE_DISPLAY:
            label, formatter = _FEATURE_DISPLAY[key]
            try:
                display_val = formatter(val)
            except Exception:
                display_val = str(val)
            lines.append(f"  • {label}: {display_val}")
    return "\n".join(lines)


def _build_prompt(prediction_prob: float,
                  raw_input: dict,
                  proof_valid: bool) -> str:
    label      = "Heart Disease Detected" if prediction_prob >= 0.5 else "No Heart Disease Detected"
    confidence = (prediction_prob if prediction_prob >= 0.5 else 1.0 - prediction_prob) * 100
    proof_str  = (
        "cryptographically verified via Zero-Knowledge Proof — meaning the "
        "computation was proven correct without exposing the patient's raw data"
        if proof_valid else
        "NOT verified (proof was unavailable or failed)"
    )
    patient_summary = _build_patient_summary(raw_input)

    return f"""You are a compassionate, knowledgeable medical AI assistant helping a patient understand their heart disease screening result.

══════════════════════════════════════════════════
SCREENING RESULT  ({proof_str})
══════════════════════════════════════════════════
  Prediction   : {label}
  Confidence   : {confidence:.1f}%

PATIENT PROFILE
══════════════════════════════════════════════════
{patient_summary}

══════════════════════════════════════════════════

Please respond with EXACTLY these four sections:

1. WHAT THIS RESULT MEANS
   Explain in simple, non-alarming language what "{label}" means for this patient.

2. KEY CONTRIBUTING FACTORS
   Based on established cardiology knowledge (NOT data the model learned),
   identify 2-4 patient features above that most likely influenced this result,
   and briefly explain why each is clinically relevant.

3. RECOMMENDED NEXT STEPS
   Suggest 3-4 concrete, actionable steps the patient should discuss with their doctor.

4. ABOUT THE PRIVACY GUARANTEE
   In 1-2 sentences, explain in plain language what "Zero-Knowledge Proof verification"
   means — why it means the patient's data was never exposed.

Tone: compassionate, clear, reassuring but honest. Avoid heavy medical jargon.
Do NOT make a definitive clinical diagnosis. This is a screening aid only."""


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm_explanation(
    prediction_prob: float,
    raw_input: dict,
    proof_valid: bool,
    model: Optional[str] = None,
) -> str:
    """
    Call the Groq API and return a plain-text explanation.

    Parameters
    ----------
    prediction_prob : float   Value in [0, 1] from the ML model.
    raw_input       : dict    Original (unscaled) feature dictionary.
    proof_valid     : bool    Whether the ZKP verification passed.
    model           : str     Override the Groq model name (optional).

    Returns
    -------
    str  — formatted explanation from the LLM
    """
    prompt      = _build_prompt(prediction_prob, raw_input, proof_valid)
    model_name  = model or GROQ_MODEL

    # Groq uses OpenAI-compatible payload structure
    payload = {
        "model":  model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "top_p": 0.9,
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Groq returns standard OpenAI-style message format
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        return "⚠️  LLM Offline: Could not reach the Groq API. Please check your internet connection."
    except requests.exceptions.HTTPError as exc:
        return f"⚠️  LLM API Error: {exc.response.text}"
    except requests.exceptions.Timeout:
        return "⚠️  LLM Timeout: Groq took too long to respond. Try again."
    except Exception as exc:
        return f"⚠️  LLM Error: {exc}"


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    result = get_llm_explanation(0.87, sample, proof_valid=True)
    print(result)