"""tinyclip: measure the trained model on the 2,000 test pairs (CPU float32, TF32 off) -> the facts the lesson narrates.

    npm run atlas -- py pipeline/tinyclip/stats.py [--ckpt pipeline/tinyclip/model_final.pt]

- batches of 6 test pairs (as the page's matrix; consecutive, 333 batches): how often each drawing's own caption gets the
  highest score in its row, how often each caption's own drawing wins its column, and the mean InfoNCE loss;
- zero-shot questions on the one-shape test drawings, using only captions (no training on these questions):
  shape ("a circle" ... "a star"), size ("a small S" vs "a big S", S the true shape) and place ("a S on the left", ...);
- two-shape test drawings: the true "a A above a B" against the swapped "a B above a A" (bag of words: always a tie).
export.py imports measure() and writes the numbers to facts.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import PLACES, SHAPES, TinyCLIP, n_params, tokenize  # noqa: E402
from train import load_pairs  # noqa: E402

PHRASE = {"left": "on the left", "right": "on the right", "top": "at the top", "bottom": "at the bottom", "middle": "in the middle"}


def ids_of(caps: list[str]) -> torch.Tensor:
    rows = [tokenize(c) for c in caps]
    L = max(len(r) for r in rows)
    return torch.tensor([r + [0] * (L - len(r)) for r in rows])


@torch.no_grad()
def measure(m: TinyCLIP) -> dict:
    X, I, S, caps, hashes, _ = load_pairs()
    te = slice(22000, 24000)
    X, I, S = X[te], I[te], S[te]
    a, b = m.encode_image(X), m.encode_text(I)
    s = float(m.scale())
    rows = cols = n6 = 0
    loss = 0.0
    for k in range(0, len(X) - 5, 6):
        L = s * a[k:k + 6] @ b[k:k + 6].t()
        y = torch.arange(6)
        rows += int((L.argmax(1) == y).sum()); cols += int((L.argmax(0) == y).sum()); n6 += 6
        loss += float((F.cross_entropy(L, y) + F.cross_entropy(L.t(), y)) / 2)
    nb = n6 // 6
    one = S[:, 0] == 0
    a1, S1 = a[one], S[one]
    shape_txt = m.encode_text(ids_of([f"a {x}" for x in SHAPES]))
    shape_acc = float(((a1 @ shape_txt.t()).argmax(1) == S1[:, 1]).float().mean())
    size_ok = place_ok = 0
    for i in range(len(a1)):
        sh = SHAPES[int(S1[i, 1])]
        z = m.encode_text(ids_of([f"a small {sh}", f"a big {sh}"]))
        size_ok += int(int((z @ a1[i]).argmax()) == int(S1[i, 2]))
        p = m.encode_text(ids_of([f"a {sh} {PHRASE[q]}" for q in PLACES]))
        place_ok += int(int((p @ a1[i]).argmax()) == int(S1[i, 3]))
    two = S[:, 0] == 1
    a2, S2 = a[two], S[two]
    ties, maxgap = 0, 0.0
    for i in range(len(a2)):
        A, B, rel = SHAPES[int(S2[i, 1])], SHAPES[int(S2[i, 2])], int(S2[i, 3])
        w = "above" if rel == 0 else "left of"
        z = m.encode_text(ids_of([f"a {A} {w} a {B}", f"a {B} {w} a {A}"])) @ a2[i] * s
        gap = float((z[0] - z[1]).abs())
        maxgap = max(maxgap, gap)
        ties += int(gap <= 1e-5)
    return {
        "params": n_params(m), "scale": s, "testN": len(X), "batches6": nb,
        "rowAcc6": rows / n6, "colAcc6": cols / n6, "loss6": loss / nb,
        "oneN": int(one.sum()), "shapeAcc": shape_acc, "sizeAcc": size_ok / len(a1), "placeAcc": place_ok / len(a1),
        "twoN": int(two.sum()), "orderTies": ties, "orderMaxGap": maxgap, "chunkHash0": hashes[0],
    }


if __name__ == "__main__":
    ckpt = Path(sys.argv[sys.argv.index("--ckpt") + 1]) if "--ckpt" in sys.argv else HERE / "model_final.pt"
    m = TinyCLIP()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    print(json.dumps(measure(m), indent=1))
