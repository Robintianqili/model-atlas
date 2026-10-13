"""vq-vae: numbers measured on the test split (CPU, float32, TF32 off) for facts.json and usage.json.

measure(model) -> (facts, usage, picks, (X, y)):
  facts   the test redrawing error (BCE per pixel and per drawing), how far patches sit from their nearest and second
          nearest code, how the codes are used (the blank code's share, codes used by fewer than 1 in 100 patches,
          distinct codes per drawing), where the error sits (pixels on the edge of a 7x7 block versus inside), and the
          comparison models of the limits step (train.py --codes 128 with this recipe, and --plain --codes 128, the
          textbook codebook loss from a small uniform start; neither is shipped).
  usage   how many of the 120,000 test patches snapped to each code.
  picks   per class, the 6 test drawings used as presets: the lowest redrawing error among the first 200 of the class.

    npm run atlas -- py pipeline/vq-vae/stats.py      # prints the facts (export.py calls measure())
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CLASSES, P, VQVAE  # noqa: E402
from train import tensors  # noqa: E402

SRC = "pipeline/vq-vae/stats.py (CPU float32, the 7,500 test drawings)"
EDGE = torch.tensor([(r % P in (0, P - 1)) or (c % P in (0, P - 1)) for r in range(28) for c in range(28)]).reshape(28, 28)


def load(name: str, k: int) -> VQVAE:
    m = VQVAE(k=k)
    m.load_state_dict(torch.load(HERE / name, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def run(m: VQVAE, X: torch.Tensor):
    ze = m.encode(X)
    zq, idx, dist = m.quantize(ze)
    logits = m.decode_logits(zq)
    px = F.binary_cross_entropy_with_logits(logits, X, reduction="none")[:, 0]   # [N, 28, 28]
    return idx, dist, px, torch.sigmoid(logits)[:, 0]


def measure(m: VQVAE):
    X, y = tensors(CLASSES, "test")
    idx, dist, px, out = run(m, X)
    K = m.codebook.shape[0]
    use = torch.bincount(idx.flatten(), minlength=K)
    share = use.double() / use.sum()
    blank = int(m.quantize(m.encode(torch.zeros(1, 1, 28, 28)))[1].flatten()[0])
    d = dist.permute(0, 2, 3, 1).reshape(-1, K).sort(1).values
    distinct = torch.tensor([len(set(i.flatten().tolist())) for i in idx], dtype=torch.float64)
    err_edge = float(px[:, EDGE].sum() / px.sum())
    facts = {
        "testN": (len(X), "drawings"),
        "classesN": (len(CLASSES), "classes"),
        "testPatches": (int(use.sum()), "patches"),
        "testRecPx": (round(float(px.mean()), 4), "nats per pixel"),
        "testRec": (round(float(px.sum((1, 2)).mean()), 1), "nats per drawing"),
        "nearMedian": (round(float(d[:, 0].median()), 3), "squared distance"),
        "secondMedian": (round(float(d[:, 1].median()), 3), "squared distance"),
        "blankShare": (round(float(share[blank]), 4), "share of test patches"),
        "rareN": (int((share < 0.01).sum()), "codes used by fewer than 1 in 100 test patches"),
        "rareMin": (round(float(share.min()), 4), "share of test patches (the rarest code)"),
        "usedN": (int((use > 0).sum()), "codes used at least once"),
        "distinctMedian": (float(distinct.median()), "distinct codes per drawing"),
        "edgePx": (round(float(EDGE.float().mean()), 4), "share of pixels on the edge of a 7x7 block"),
        "edgeErr": (round(err_edge, 4), "share of the redrawing error on those pixels"),
    }
    for name, k, tag in (("model_k128.pt", 128, "k128"), ("model_plain_k128.pt", 128, "plainK128")):
        if (HERE / name).is_file():
            mk = load(name, k)
            ik, _, pk, _ = run(mk, X)
            uk = torch.bincount(ik.flatten(), minlength=k)
            facts[f"{tag}Used"] = (int((uk > 0).sum()), "codes used at least once on the test drawings")
            facts[f"{tag}RecPx"] = (round(float(pk.mean()), 4), "nats per pixel")
    if "k128Used" in facts:
        facts["k128Codes"] = (128, "codes")
    usage = {"counts": use.tolist(), "patches": int(use.sum()), "blank": blank,
             "note": "test patches (7,500 drawings x 16) that snapped to each code; pipeline/vq-vae/stats.py"}
    rec = px.sum((1, 2))
    picks = []
    for k in range(len(CLASSES)):
        ids = torch.nonzero(y == k).flatten()[:200]
        picks.append(ids[rec[ids].argsort()[:6]].tolist())
    return facts, usage, picks, (X, y)


if __name__ == "__main__":
    f, u, p, _ = measure(load("model_final.pt", 32))
    print(json.dumps({k: v[0] for k, v in f.items()}, indent=1))
    print(json.dumps({k: v for k, v in u.items() if k != "note"}))
