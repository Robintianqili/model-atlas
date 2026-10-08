"""vae-2d: numbers measured on the test split (CPU, float32, TF32 off) for facts.json and the latent map (latent.json).

measure(model) -> (facts, latent, picks):
  facts   the test losses (reconstruction with z = mu + sigma * eps, seed 0, and with z = mu; KL), how well the 2-D map
          keeps the classes apart (10 nearest neighbours by mu), the typical sigma, the spread of the map, how grey the
          redrawings are compared with the drawings (the "blurry" limit), and the 16-D comparison model's losses
          (model_z16.pt, trained by train.py --zdim 16 the same way; not shipped).
  latent  MAP_PER_CLASS test drawings per class: mu (4 decimals) and class, plus the class centres (median mu).
  picks   per class, the test drawings used as presets: among the 50 nearest the class centre, the 6 with the lowest
          reconstruction loss (typical drawings the model redraws well).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CLASSES, VAE, losses  # noqa: E402
from train import tensors  # noqa: E402

MAP_PER_CLASS = 400
K = 10
SRC = "pipeline/vae-2d/stats.py (CPU float32, the 7,500 test drawings)"


@torch.no_grad()
def encode_all(m: VAE, X: torch.Tensor):
    mu, lv = m.encode(X)
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(mu.shape, generator=g)
    rec_s, kl = losses(m.decode_logits(mu + torch.exp(0.5 * lv) * eps), X, mu, lv)
    rec_mu, _ = losses(m.decode_logits(mu), X, mu, lv)
    out = m.decode(mu + torch.exp(0.5 * lv) * eps)
    return mu, lv, rec_s, rec_mu, kl, out


@torch.no_grad()
def knn_agree(mu: torch.Tensor, y: torch.Tensor, k: int = K) -> float:
    d = torch.cdist(mu, mu)
    d.fill_diagonal_(float("inf"))
    nn = d.topk(k, largest=False).indices
    votes = torch.nn.functional.one_hot(y[nn], len(CLASSES)).sum(1)
    return float((votes.argmax(1) == y).float().mean())


def grey_share(x: torch.Tensor) -> float:
    """share of pixels between 0.2 and 0.8 among the pixels above 0.05 (neither empty paper nor full ink)"""
    on = x > 0.05
    grey = (x > 0.2) & (x < 0.8)
    return float(grey.sum() / on.sum())


def measure(m: VAE):
    X, y = tensors(CLASSES, "test")
    mu, lv, rec_s, rec_mu, kl, out = encode_all(m, X)
    sigma = torch.exp(0.5 * lv)
    facts = {
        "testN": (len(X), "drawings"),
        "classesN": (len(CLASSES), "classes"),
        "testRec": (round(float(rec_s.mean()), 2), "nats"),
        "testRecMu": (round(float(rec_mu.mean()), 2), "nats"),
        "testKl": (round(float(kl.mean()), 3), "nats"),
        "testLoss": (round(float((rec_s + kl).mean()), 2), "nats"),
        "knnK": (K, "drawings"),
        "knnAgree": (round(knn_agree(mu, y), 4), "share"),
        "sigmaMedian": (round(float(sigma.median()), 4), "latent units"),
        "muStd": (round(float(mu.std(0).mean()), 3), "latent units"),
        "greyIn": (round(grey_share(X), 4), "share"),
        "greyOut": (round(grey_share(out), 4), "share"),
        "mapPerClass": (MAP_PER_CLASS, "drawings"),
    }
    z16 = HERE / "model_z16.pt"
    if z16.is_file():
        m16 = VAE(zdim=16)
        m16.load_state_dict(torch.load(z16, map_location="cpu"))
        m16.eval()
        _, _, r16, _, k16, o16 = encode_all(m16, X)
        facts["z16Dim"] = (16, "latent numbers")
        facts["z16Rec"] = (round(float(r16.mean()), 2), "nats")
        facts["z16Kl"] = (round(float(k16.mean()), 3), "nats")
        facts["z16GreyOut"] = (round(grey_share(o16), 4), "share")
    # the map: a fixed subset of test drawings per class
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.where(y.numpy() == k)[0], MAP_PER_CLASS, replace=False) for k in range(len(CLASSES))])
    centres = [[round(float(v), 4) for v in mu[y == k].median(0).values] for k in range(len(CLASSES))]
    latent = {"format": "vae-2d-latent/1", "classes": CLASSES, "note": "mu of test drawings (Quick, Draw!, CC BY 4.0); " + SRC,
              "x": [round(float(v), 4) for v in mu[idx, 0]], "y": [round(float(v), 4) for v in mu[idx, 1]],
              "c": [int(v) for v in y[idx]], "centres": centres}
    # presets: typical drawings that the model redraws well
    picks = []
    for k in range(len(CLASSES)):
        ids = torch.where(y == k)[0]
        d = (mu[ids] - torch.tensor(centres[k])).norm(dim=1)
        near = ids[d.argsort()[:50]]
        picks.append([int(i) for i in near[rec_s[near].argsort()[:6]]])
    return facts, latent, picks, (X, y)


if __name__ == "__main__":
    m = VAE()
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    f, lat, p, _ = measure(m.eval())
    print(json.dumps({k: v[0] for k, v in f.items()}, indent=1))
    print("picks", p)
