"""hello-xor: the PyTorch model (kind A). A 2-2-1 MLP with a tanh hidden layer: the smallest network that can learn XOR.

Submodule names become layer ids and tensor names in spec.json (fc1.weight, fc1.bias, fc2.weight, fc2.bias).
Train it with train.py; export it with export.py (or `npm run atlas -- export hello-xor` from site/).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# The XOR truth table: the four inputs and the target for each.
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


class XorNet(nn.Module):
    """x (2 numbers) -> fc1 (2 hidden units) -> tanh -> fc2 (1 output score). Read the score as "how XOR is this point"."""

    def __init__(self, hidden: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)  # the hidden layer
        self.fc2 = nn.Linear(hidden, 1)  # the output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


if __name__ == "__main__":
    torch.manual_seed(0)
    m = XorNet()
    print(m, sum(p.numel() for p in m.parameters()), "params")
