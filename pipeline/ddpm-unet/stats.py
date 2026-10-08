"""ddpm-unet: measured numbers for the lesson (stats.json -> facts.json via export.py).

    npm run atlas -- py pipeline/ddpm-unet/stats.py [--n 1000]

Samples n drawings from model_final.pt (CPU float32, TF32 off; torch seed 0) with all T steps and, from the same starting
noise, with 10 steps (stride 10, respaced as the page's `few` source does), maps them back to ink, (x + 1) / 2 clamped to
0..1 (as the page draws them),
and measures:
  judged*     the share TinyCNN (pipeline/tinycnn/model_final.pt, the site's 10-class drawing classifier) calls one of the
              DDPM's classes, for the samples and for the real test drawings of those classes
  grey*       the share of inked pixels (> 0.05) that are grey (between 0.2 and 0.8): soft, blurry strokes score high
  ink*        the share of pixels with more than half ink
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tinycnn"))
from model import CH, CLASSES, T, UNet, sample  # noqa: E402
import common as tc  # noqa: E402
from classes import BASE  # noqa: E402


def grey(x: torch.Tensor) -> float:
    on = (x > 0.05).sum().item()
    return ((x > 0.2) & (x < 0.8)).sum().item() / on if on else float("nan")


@torch.no_grad()
def main(n: int) -> dict:
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    u = UNet(CH)
    u.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    u.eval()
    judge = tc.TinyVGG()
    judge.load_state_dict(torch.load(HERE.parent / "tinycnn" / "model_final.pt", map_location="cpu"))
    judge.eval()
    ids = [BASE.index(c) for c in CLASSES]
    inside = lambda x: float(torch.isin(judge(x).argmax(1), torch.tensor(ids)).float().mean())  # noqa: E731
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 1, 28, 28, generator=g)
    z = [torch.randn(n, 1, 28, 28, generator=g) for _ in range(T)]
    full = ((sample(u, x, z, list(range(T, 0, -1))) + 1) / 2).clamp(0, 1)
    few = ((sample(u, x, z, list(range(T, 0, -10))) + 1) / 2).clamp(0, 1)
    X, _, _ = ak.data.quickdraw(CLASSES, "test")
    real = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255
    top = judge(full).argmax(1)
    out = {"judgedN": n, "judgedIn": inside(full), "judgedInFew": inside(few), "judgedReal": inside(real),
           "greyFull": grey(full), "greyFew": grey(few), "greyReal": grey(real),
           "inkFull": float((full > 0.5).float().mean()), "inkFew": float((few > 0.5).float().mean()), "inkReal": float((real > 0.5).float().mean())}
    for c, i in zip(CLASSES, ids):
        out[f"judged_{c.replace(' ', '_')}"] = float((top == i).float().mean())
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1000)
    s = main(ap.parse_args().n)
    (HERE / "stats.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(s))
