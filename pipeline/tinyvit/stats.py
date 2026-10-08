"""tinyvit: the measured numbers behind the lesson (facts.json), on the CPU in float32 over TinyCNN's 15,000 test drawings.

    npm run atlas -- py pipeline/tinyvit/stats.py          # prints them; export.py writes them into facts.json

- testAcc, testN: TinyViT's test accuracy (model_final.pt).
- cnnTestAcc, cnnParams: TinyCNN (pipeline/tinycnn/model_final.pt, its TinyVGG) on the same drawings.
- agree: the share of test drawings on which both models give the same top answer; bothWrong: neither is right.
- maxTopP: TinyViT's largest top-class probability on any test drawing (label smoothing keeps it near 0.91).
- Where the [CLS] row looks: a patch "has ink" when its brightest pixel is >= INK (0.5). For every block, the share of
  the [CLS] row's attention weight that lands on patches with ink (averaged over drawings; per head and over heads),
  next to the share of patches that have ink. HEAD is the block-1 head the lesson zooms into: the one whose [CLS]
  row leans most toward ink per patch (weight per inked patch / weight per empty patch).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CLASSES, GRID, PATCH, TinyViT, n_params  # noqa: E402

INK = 0.5


def tinycnn():
    p = ak.root_path("pipeline", "tinycnn", "common.py")
    spec = importlib.util.spec_from_file_location("tinyvit_tinycnn_common", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.TinyVGG()
    m.load_state_dict(torch.load(ak.root_path("pipeline", "tinycnn", "model_final.pt"), map_location="cpu"))
    return m.eval()


def inked_patches(X: torch.Tensor) -> torch.Tensor:
    """[N, 1, 28, 28] -> [N, 16] (1.0 where the patch's brightest pixel is >= INK), patches row-major."""
    p = X.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH).reshape(len(X), GRID * GRID, PATCH * PATCH)
    return (p.amax(-1) >= INK).float()


@torch.no_grad()
def measure(vit: TinyViT) -> dict:
    """{fact key: (value, unit)}"""
    X, y, _ = ak.data.quickdraw(CLASSES, "test")
    X = torch.tensor(X).float().unsqueeze(1) / 255
    y = torch.tensor(y)
    vit = vit.cpu().eval()
    keep: list = []
    probs = torch.cat([vit(X[i:i + 1500], keep).softmax(1) for i in range(0, len(X), 1500)])
    pv = probs.argmax(1)
    cnn = tinycnn()
    pc = torch.cat([cnn(X[i:i + 1500]).argmax(1) for i in range(0, len(X), 1500)])
    nb = len(vit.blocks)
    attn = [torch.cat(keep[b::nb]) for b in range(nb)]          # per block: [N, H, 17, 17]
    ink = inked_patches(X)                                        # [N, 16]
    out = {
        "testAcc": (round(float((pv == y).float().mean()), 4), "share"),
        "testN": (len(y), "drawings"),
        "params": (n_params(vit), "count"),
        "cnnTestAcc": (round(float((pc == y).float().mean()), 4), "share"),
        "cnnParams": (n_params(cnn), "count"),
        "agree": (round(float((pv == pc).float().mean()), 4), "share"),
        "bothWrong": (round(float(((pv != y) & (pc != y)).float().mean()), 4), "share"),
        "maxTopP": (round(float(probs.max(1).values.max()), 4), "share"),
        "inkThreshold": (INK, "pixel value"),
        "inkedShare": (round(float(ink.mean()), 4), "share"),
    }
    lean = []
    for b, w in enumerate(attn):
        cls = w[:, :, 0, 1:]                                      # [N, H, 16]: the [CLS] row over the 16 patches
        on = (cls * ink[:, None]).sum(-1)                         # [N, H]
        n_ink = ink.sum(-1, keepdim=True)
        per_ink = on / n_ink.clamp_min(1)
        per_empty = (cls.sum(-1) - on) / (16 - n_ink).clamp_min(1)
        ok = (n_ink[:, 0] > 0) & (n_ink[:, 0] < 16)               # drawings with both kinds of patch
        if b == 0:
            lean = (per_ink[ok] / per_empty[ok]).mean(0)          # [H]
        out[f"b{b + 1}InkShare"] = (round(float(on.mean()), 4), "share")
        out[f"b{b + 1}SelfShare"] = (round(float(w[:, :, 0, 0].mean()), 4), "share")
        for h in range(w.shape[1]):
            out[f"b{b + 1}h{h}InkShare"] = (round(float(on[:, h].mean()), 4), "share")
    head = int(torch.argmax(lean))
    out["head"] = (head, "head index (from 0)")
    out["headLean"] = (round(float(lean[head]), 3), "ratio")
    return out


if __name__ == "__main__":
    m = TinyViT()
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    print(json.dumps({k: v for k, (v, _) in measure(m).items()}, indent=1))
