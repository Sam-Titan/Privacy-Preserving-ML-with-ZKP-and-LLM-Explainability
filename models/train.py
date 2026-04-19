"""
models/train.py
───────────────
Train HeartDiseaseNet on the UCI Heart Disease dataset.
Run: python -m models.train

Outputs written to outputs/
  • heart_model.pt      — PyTorch state dict
  • scaler.pkl          — fitted StandardScaler
  • feature_cols.json   — ordered list of feature names
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE2
from models.model import HeartDiseaseNet
from models.preprocess import (
    load_and_preprocess, fit_scaler, save_feature_cols
)


# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS      = 150
BATCH_SIZE  = 32
LR          = 1e-3
RANDOM_SEED = 42


def train(data_path: str = None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[1/5] Loading & preprocessing data …")
    X, y, feature_cols = load_and_preprocess(data_path)
    print(f"      Dataset shape: {X.shape}  |  Positive rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = fit_scaler(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    save_feature_cols(feature_cols)
    print(f"      Features ({len(feature_cols)}): {feature_cols}")

    # ── Tensors & DataLoader ──────────────────────────────────────────────────
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    Xtest_t = torch.FloatTensor(X_test)
    ytest_t  = torch.FloatTensor(y_test).unsqueeze(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/5] Building model …")
    model = HeartDiseaseNet(
        input_size=len(feature_cols),
        hidden1=MODEL_HIDDEN_SIZE,
        hidden2=MODEL_HIDDEN_SIZE2,
    )
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # ── Training loop ──────────────────────────────────────────────────────────
    print("[3/5] Training …")
    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── Validation every 25 epochs ────────────────────────────────────────
        if epoch % 25 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                probs = model(Xtest_t).numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds, zero_division=0)
            print(f"  Epoch {epoch:>3}/{EPOCHS} | "
                  f"Loss: {epoch_loss / len(loader):.4f} | "
                  f"Acc: {acc:.4f} | F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1   = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("[4/5] Final evaluation on test set …")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(Xtest_t).numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    print(classification_report(y_test, preds,
                                target_names=["No Disease", "Heart Disease"]))

    # ── Persist ───────────────────────────────────────────────────────────────
    print("[5/5] Saving artifacts …")
    model_path = os.path.join(MODELS_DIR, "heart_model.pt")
    torch.save(best_state, model_path)
    print(f"  ✓  Model    → {model_path}")
    print(f"  ✓  Scaler   → outputs/scaler.pkl")
    print(f"  ✓  Features → outputs/feature_cols.json")
    print(f"\nBest F1 on test: {best_f1:.4f}")

    return model, feature_cols


if __name__ == "__main__":
    train()