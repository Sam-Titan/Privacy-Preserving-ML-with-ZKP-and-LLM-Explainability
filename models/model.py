"""
models/model.py
───────────────
A deliberately small, ZKP-friendly neural network for binary classification.

Design decisions for ZKP compatibility:
  • Shallow (2 hidden layers only) — deeper = more expensive ZK circuit.
  • ReLU activations — EZKL approximates these natively within its
    quantization pipeline.  If you switch to Circom manually you would
    replace them with degree-2/3 polynomial approximations.
  • Sigmoid at the output — produces a probability in [0,1].
  • No BatchNorm / Dropout — these add non-arithmetic ops that are
    difficult to arithmetize for SNARKs.
"""
import torch
import torch.nn as nn


class HeartDiseaseNet(nn.Module):
    """
    Input  : (batch, input_size)   — scaled float features
    Output : (batch, 1)            — P(heart disease) in [0, 1]
    """

    def __init__(self, input_size: int = 13,
                 hidden1: int = 32,
                 hidden2: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)