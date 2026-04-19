# 🫀 Privacy-Preserving Heart Disease Prediction API

An end-to-end Privacy-Preserving Machine Learning (PPML) pipeline that performs neural network inference for heart disease prediction without exposing sensitive patient data. 

This project leverages **Zero-Knowledge Proofs (ZK-SNARKs)** via EZKL to cryptographically guarantee computation integrity, and the **Groq API (LLaMA 3.3 70B)** to provide explainable AI insights based strictly on verified outputs.

---

## 🏗️ System Architecture & Workflow

1. **Client-Side Preprocessing (`models/preprocess.py`)**: Patient features are mapped and scaled locally using a fitted `StandardScaler`.
2. **Zero-Knowledge Inference**: The input is passed through a compiled ZK-circuit (`outputs/zkp/model.compiled`) of the PyTorch neural network.
3. **Cryptographic Proving (`backend/pipeline.py`)**: A ZK-SNARK proof is generated, proving the model executed correctly on private inputs.
4. **Verification**: A verifier checks the proof (`proof.json`) against the verification key (`vk.key`) in milliseconds.
5. **LLM Explanation**: Once verified, the prediction probability is sent to LLaMA 3.3 via Groq to generate a clinical explanation.

---

## 📊 Empirical Benchmarks

The system was evaluated using a 13-feature UCI Heart Disease dataset across 920 samples. Below are the empirical results generated from our benchmarking suite.

### 1. Model Performance (`models.train`)
* **Architecture:** Multi-Layer Perceptron (Hidden Layers: 32, 16)
* **Dataset Shape:** 920 samples, 13 features (Positive rate: 55.33%)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 83.0% |
| **F1 Score** | 85.31% |
| **FP32 Latency (Plain Inference)** | ~0.35 ms |

### 2. ZK-SNARK Cryptographic Overhead (`benchmark.py`)
Tested with `logrows=15` over the BN254 elliptic curve. Generating the Zero-Knowledge Proof incurs a significant computational overhead to guarantee absolute privacy, but verification remains highly asymmetrical and efficient.

| Cryptographic Phase | Average Time / Metric |
| :--- | :--- |
| **Witness Generation** | 0.2759 s |
| **Proof Generation** | 2.4702 s (±0.1820) |
| **Proof Verification** | 0.2921 s |
| **Total ZKP Pipeline Time** | 3.0383 s |
| **Average Proof Size** | 20.03 KB |
| **Privacy Overhead Ratio** | 7,965x slower vs plain inference |
| **Cryptographic Validity** | 100.0% Valid |

**Conclusion:** The asymmetrical nature of ZK-SNARKs allows proofs that take ~2.5 seconds to generate on the client side to be verified centrally in just ~0.29 seconds. Furthermore, the 20 KB proof size makes this architecture highly bandwidth-efficient for edge computing.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Rust (Cargo) & EZKL library
* PyTorch, Scikit-Learn, FastAPI, Uvicorn

### 1. Configuration
Create a `config.py` file in the root directory (this is git-ignored for security). 
```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "outputs")
ZKP_DIR = os.path.join(BASE_DIR, "outputs", "zkp")

# Groq LLM Settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1024

# 1. Train the model and fit the StandardScaler
python -m models.train

# 2. Export the trained model to ONNX (opset 11)
python -m models.export_onnx

# 3. Generate settings, calibrate quantization, and compile circuit
python -m zkp.setup

# 4. Generate the Structured Reference String (SRS) offline
python make_srs.py

# 5. Generate Proving (PK) and Verification (VK) keys
python finish.py

python -m backend.app

python benchmark.py --samples 10