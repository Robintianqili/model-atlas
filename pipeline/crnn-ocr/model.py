"""crnn-ocr: the PyTorch model (kind A) and the data reader shared by train, export, stats and reference.

A tiny CRNN (Shi, Bai and Yao 2015) read with CTC (Graves et al. 2006): a strip of 16 x 64 ink values -> two 3x3
convolutions (8 and 16 channels, ReLU, one 2x2 max pool) -> 32 columns of 16 x 8 = 128 features -> a bidirectional
LSTM written as two one-layer LSTMs (`fwd` reads the columns left to right, `bwd` right to left: flip, LSTM, flip back;
the same arithmetic as nn.LSTM(bidirectional=True)) -> 64 numbers per column -> 11 scores per column (class 0 is the CTC
blank, class d + 1 is digit d). Submodule names become layer ids and tensor names (conv1.weight, fwd.weight_ih_l0, ...).

    npm run atlas -- py pipeline/crnn-ocr/model.py        (smoke test: shapes and the parameter count)
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CRNN_OCR_DATA", HERE / "runs" / "data"))
SYMBOLS = ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def columns(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [W, B, C*H]: column w holds every channel's values in that column (feature c*H + h)."""
    b, c, h, w = x.shape
    return x.permute(3, 0, 1, 2).reshape(w, b, c * h)


class CRNN(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, hidden: int = 32, classes: int = len(SYMBOLS), height: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fwd = nn.LSTM(c2 * height // 2, hidden)
        self.bwd = nn.LSTM(c2 * height // 2, hidden)
        self.head = nn.Linear(2 * hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 1, 16, 64] -> scores [32, B, 11] (time first, as CTC wants it)."""
        f = F.max_pool2d(F.relu(self.conv1(x)), 2)
        cols = columns(F.relu(self.conv2(f)))
        hf, _ = self.fwd(cols)
        hb, _ = self.bwd(torch.flip(cols, [0]))
        return self.head(torch.cat([hf, torch.flip(hb, [0])], dim=-1))


def greedy(scores: torch.Tensor) -> list[str]:
    """CTC greedy decoding of [T, B, 11]: the top class per column, repeats merged, blanks dropped."""
    out = []
    for path in scores.argmax(-1).T.tolist():
        s, prev = "", 0
        for k in path:
            if k != prev and k != 0:
                s += SYMBOLS[k]
            prev = k
        out.append(s)
    return out


def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


def load(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """(images [n, 1, 16, 64] float32 = byte / 16, labels [n, 8] int64, lengths [n] int64, texts) from make_data.ts."""
    mt = meta()
    m, h, w, ml = mt["sets"][name], mt["h"], mt["w"], mt["maxl"]
    raw = (DATA / f"{name}.bin").read_bytes()
    a, b = m["n"] * h * w, m["n"] * ml
    parts = [raw[:a], raw[a:a + b], raw[a + b:a + b + m["n"]]]
    if [zlib.crc32(p) for p in parts] != m["crc"]:
        raise SystemExit(f"{name}: CRC-32 of the bytes read differs from the TS record (re-run make_data.ts)")
    x = torch.from_numpy(np.frombuffer(parts[0], dtype=np.uint8).reshape(m["n"], 1, h, w).astype(np.float32) / 16)
    y = torch.from_numpy(np.frombuffer(parts[1], dtype=np.uint8).reshape(m["n"], ml).astype(np.int64))
    n = torch.from_numpy(np.frombuffer(parts[2], dtype=np.uint8).astype(np.int64))
    texts = ["".join(SYMBOLS[k] for k in row[:k_]) for row, k_ in zip(y.tolist(), n.tolist())]
    return x, y, n, texts


def cases() -> list[dict]:
    """[{id, label, edge, text, ink: [1, 16, 64] float32}] (the page's presets and edge fixtures)."""
    out = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))["cases"]
    for c in out:
        c["ink"] = torch.tensor(c["ink"], dtype=torch.float32).reshape(1, 16, 64)
    return out


if __name__ == "__main__":
    m = CRNN()
    print({"params": sum(p.numel() for p in m.parameters()), "names": list(m.state_dict())})
    print(m(torch.zeros(2, 1, 16, 64)).shape)
