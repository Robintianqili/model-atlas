"""ddpm-unet: judge a checkpoint by its samples (not shipped; used to choose the classes and the width).

    npm run atlas -- py pipeline/ddpm-unet/screen.py model_x.pt --classes star,umbrella,house [--ch 16,32,64] [--n 256]

Samples n drawings (100 steps, torch seed 0, CPU float32), asks TinyCNN (pipeline/tinycnn/model_final.pt, its 10
classes) what each one is, prints the share it calls one of the trained classes and the share per class, and writes a
PNG grid of the first 64 samples to screen/<stem>.png (white ink on black, 2x).
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
import struct
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tinycnn"))
from model import CH, T, UNet, sample  # noqa: E402
import common as tc  # noqa: E402
from classes import BASE  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def png(path: Path, img: np.ndarray) -> None:
    """write a grey uint8 image as PNG (no PIL needed)."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def grid(x: np.ndarray, cols=16, scale=2) -> np.ndarray:
    n = len(x)
    rows = (n + cols - 1) // cols
    g = np.full((rows * 30, cols * 30), 60, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        g[r * 30 + 1:r * 30 + 29, c * 30 + 1:c * 30 + 29] = (np.clip(x[i], 0, 1) * 255).round().astype(np.uint8)
    return g.repeat(scale, 0).repeat(scale, 1)


def judge() -> torch.nn.Module:
    m = tc.TinyVGG()
    m.load_state_dict(torch.load(HERE.parent / "tinycnn" / "model_final.pt", map_location="cpu"))
    return m.eval()


@torch.no_grad()
def run(ckpt: str, classes, ch, n=256, seed=0, steps=None, pm1=False):
    u = UNet(ch)
    u.load_state_dict(torch.load(HERE / ckpt, map_location="cpu"))
    u.eval()
    g = torch.Generator().manual_seed(seed)
    steps = steps or list(range(T, 0, -1))
    x = torch.randn(n, 1, 28, 28, generator=g)
    z = [torch.randn(n, 1, 28, 28, generator=g) for _ in steps]
    out = sample(u, x, z, steps)
    out = ((out + 1) / 2 if pm1 else out).clamp(0, 1)
    p = judge()(out).softmax(1)
    top = p.argmax(1)
    share = {c: float((top == BASE.index(c)).float().mean()) for c in classes if c in BASE}
    return out[:, 0].numpy(), share


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--ch", default=",".join(map(str, CH)))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=T)
    ap.add_argument("--pm1", action="store_true", help="the model was trained on -1..1 pixels")
    a = ap.parse_args()
    ch = tuple(int(c) for c in a.ch.split(","))
    steps = [int(round(v)) for v in np.linspace(T, 1, a.steps)] if a.steps != T else None
    x, share = run(a.ckpt, a.classes.split(","), ch, a.n, steps=steps, pm1=a.pm1)
    (HERE / "screen").mkdir(exist_ok=True)
    stem = Path(a.ckpt).stem + ("" if a.steps == T else f"_k{a.steps}")
    png(HERE / "screen" / f"{stem}.png", grid(x[:64]))
    print(json.dumps({"ckpt": a.ckpt, "steps": a.steps, "share": share, "inClasses": round(sum(share.values()), 3)}))
