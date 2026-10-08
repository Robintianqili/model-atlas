"""hello-xor: train XorNet on the four XOR points (full batch, mean squared error, Adam lr 0.05, 1,500 steps).

    python pipeline/hello-xor/train.py [--seed 0] [--steps 1500]

CPU, about a second. Writes pipeline/hello-xor/model_final.pt (1 KB, committed like pipeline/tinycnn/model_final.pt)
and train_metrics.json, and prints the final loss and the four outputs. With torch seed 0 the loss ends below 1e-6.
Some seeds (2, for example) get stuck with one point wrong; the lesson's limits step shows the same thing live.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import X, Y, XorNet  # noqa: E402


def train(seed: int = 0, steps: int = 1500, lr: float = 0.05) -> XorNet:
    torch.manual_seed(seed)
    m = XorNet()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((m(X) - Y) ** 2).mean()
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
        final = float(((m(X) - Y) ** 2).mean())
    torch.save(m.state_dict(), HERE / "model_final.pt")
    report = {"seed": a.seed, "steps": a.steps, "lr": 0.05, "optimizer": "Adam", "final_loss": final,
              "outputs": [round(float(v), 6) for v in out], "correct": int(((out > 0.5).float() == Y.squeeze(1)).sum())}
    with open(HERE / "train_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
