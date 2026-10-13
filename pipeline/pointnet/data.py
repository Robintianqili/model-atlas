"""pointnet: read the sets make_data.ts wrote (runs/data/<set>.bin, meta.json, cases.json) and check their CRC-32.

Each set is clouds as float32 [count, PAD, 2] (normalized by the page's shapes.ts, repeated cyclically up to PAD points),
classes as uint8 [count] and point counts as uint16 [count]. The CRC-32 of every part must equal what TS recorded.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent / "runs" / "data"


def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


def load(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(clouds [count, PAD, 2] float32, classes [count] int64, counts [count] int64)"""
    mt = meta()
    m, pad = mt["sets"][name], mt["pad"]
    raw = (DATA / f"{name}.bin").read_bytes()
    a, b = m["n"] * pad * 2 * 4, m["n"]
    parts = [raw[:a], raw[a:a + b], raw[a + b:a + b + 2 * b]]
    if [zlib.crc32(p) for p in parts] != m["crc"]:
        raise SystemExit(f"{name}: CRC-32 of the bytes read differs from the TS record (re-run make_data.ts)")
    x = np.frombuffer(parts[0], dtype="<f4").reshape(m["n"], pad, 2).copy()
    y = np.frombuffer(parts[1], dtype=np.uint8).astype(np.int64)
    n = np.frombuffer(parts[2], dtype="<u2").astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(n)


def cases() -> list[dict]:
    """[{id, label, edge, xy: [n, 2] float32 tensor}] (the page's presets and edge fixtures, the largest cloud)"""
    out = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))["cases"]
    for c in out:
        c["xy"] = torch.tensor(c["xy"], dtype=torch.float32).reshape(-1, 2)
    return out
