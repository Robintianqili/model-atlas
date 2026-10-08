"""vae-2d: which five classes? Train a 2-D VAE briefly on each candidate set and measure how well the 2-D map keeps
the classes apart (share of test drawings whose 10 nearest test drawings on the map, by mu, are mostly their class).

    npm run atlas -- py pipeline/vae-2d/screen.py [--epochs 25]

Prints one line per set; notes/vae-2d.md records the result and the choice (model.CLASSES).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import evaluate, fit, tensors  # noqa: E402

SETS = [
    ["star", "donut", "ladder", "lightning", "t-shirt"],
    ["star", "house", "donut", "ladder", "lightning"],
    ["star", "house", "sun", "ladder", "fish"],
    ["star", "moon", "ladder", "envelope", "donut"],
    ["star", "donut", "ladder", "umbrella", "eyeglasses"],
    ["star", "cloud", "ladder", "lightning", "cup"],
    ["star", "ladder", "umbrella", "eyeglasses", "t-shirt"],
    ["star", "donut", "ladder", "umbrella", "lightning"],
    ["star", "ladder", "umbrella", "eyeglasses", "house"],
    ["star", "donut", "umbrella", "eyeglasses", "lightning"],
]


@torch.no_grad()
def knn_agree(mu: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    d = torch.cdist(mu, mu)
    d.fill_diagonal_(float("inf"))
    nn = d.topk(k, largest=False).indices
    votes = torch.nn.functional.one_hot(y[nn], int(y.max()) + 1).sum(1)
    return float((votes.argmax(1) == y).float().mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--from", dest="start", type=int, default=0)
    a = ap.parse_args()
    for cls in SETS[a.start:]:
        m, _ = fit(cls, epochs=a.epochs, log=lambda s: None)
        X, y = tensors(cls, "test")
        mu, _ = m.encode(X)
        ev = evaluate(m, X, "cpu")
        print(f"{','.join(cls):48s} knn10 {knn_agree(mu, y):.3f}  rec {ev['rec_mu']:.1f}  kl {ev['kl']:.2f}", flush=True)
