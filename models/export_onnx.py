"""
models/export_onnx.py
─────────────────────
Exports the trained HeartDiseaseNet to ONNX format, which EZKL
requires to compile the ZK circuit.

Run: python -m models.export_onnx

Prerequisites: models/train.py must have been run first.
"""
import os
import sys
import json

import torch
import onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, ONNX_MODEL_PATH
from models.model import HeartDiseaseNet
from models.preprocess import load_feature_cols


def export():
    os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)

    # ── Load trained weights ──────────────────────────────────────────────────
    feature_cols = load_feature_cols()
    input_size   = len(feature_cols)

    model = HeartDiseaseNet(input_size=input_size)
    state = torch.load(os.path.join(MODELS_DIR, "heart_model.pt"),
                       map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ── Dummy input — batch size 1 ────────────────────────────────────────────
    dummy = torch.randn(1, input_size, requires_grad=False)

    # ── Export ────────────────────────────────────────────────────────────────
    torch.onnx.export(
        model,
        dummy,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,           # EZKL works best with opset 11
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False   # 🔥 ADD THIS LINE (CRITICAL FIX)
    )

    # ── Verify ONNX model is well-formed ─────────────────────────────────────
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)

    size_kb = os.path.getsize(ONNX_MODEL_PATH) / 1024
    print(f"✓  ONNX model exported → {ONNX_MODEL_PATH}  ({size_kb:.1f} KB)")
    print(f"   Input  shape : (1, {input_size})")
    print(f"   Output shape : (1, 1)")
    return ONNX_MODEL_PATH


if __name__ == "__main__":
    export()