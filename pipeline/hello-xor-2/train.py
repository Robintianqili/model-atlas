"""hello-xor-2: train XorNet (sigmoid output) on the four XOR points (full batch, binary cross-entropy, Adam lr 0.05).

    python pipeline/hello-xor-2/train.py [--seed 0] [--steps 1500]

CPU, about a second. Writes pipeline/hello-xor-2/model_final.pt and train_metrics.json, and prints the final loss and
the four outputs. The loss is computed from the logits (binary_cross_entropy_with_logits, the stable form of
binary cross-entropy on sigmoid(logit)); the model's forward() returns the sigmoid itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import X, Y, XorNet  # noqa: E402

LR = 0.05


def train(seed: int = 0, steps: int = 1500, lr: float = LR) -> XorNet:
    torch.manual_seed(seed)
    m = XorNet()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m.logits(X), Y)
        loss.backward()
        opt.step()
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    m = train(a.seed, a.steps)
    with torch.no_grad():
        out = m(X).squeeze(1)
        final = float(F.binary_cross_entropy_with_logits(m.logits(X), Y))
    torch.save(m.state_dict(), HERE / "model_final.pt")
    report = {"seed": a.seed, "steps": a.steps, "lr": LR, "optimizer": "Adam", "loss": "binary cross-entropy",
              "final_loss": final, "outputs": [round(float(v), 6) for v in out],
              "correct": int(((out > 0.5).float() == Y.squeeze(1)).sum())}
    with open(HERE / "train_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
