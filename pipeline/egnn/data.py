"""egnn: read the sets make_data.ts wrote (<set>.bin, meta.json, cases.json) and check their CRC-32.

The folder: EGNN_DATA if set, else pipeline/egnn/runs/data. Each set holds positions float32 [count, PAD, 2], kinds uint8
[count, PAD], atom counts uint8 [count], toy energies float32 [count] and forces float32 [count, PAD, 2].
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch

DATA = Path(os.environ.get("EGNN_DATA") or Path(__file__).resolve().parent / "runs" / "data")


def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


def _f32(b: bytes, *shape: int) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype="<f4").reshape(*shape).copy())


def _u8(b: bytes, *shape: int) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(b, dtype=np.uint8).reshape(*shape).astype(np.int64))


def load(name: str) -> dict:
    """{pos [count, PAD, 2], kind [count, PAD] int64, n [count] int64, E [count], F [count, PAD, 2]}"""
    mt = meta()
    c, pad = mt["sets"][name]["n"], mt["pad"]
    raw = (DATA / f"{name}.bin").read_bytes()
    parts, at = [], 0
    for s in (c * pad * 2 * 4, c * pad, c, c * 4, c * pad * 2 * 4):
        parts.append(raw[at:at + s])
        at += s
    if [zlib.crc32(p) for p in parts] != mt["sets"][name]["crc"]:
        raise SystemExit(f"{name}: CRC-32 of the bytes read differs from the TS record (re-run make_data.ts)")
    return {"pos": _f32(parts[0], c, pad, 2), "kind": _u8(parts[1], c, pad), "n": _u8(parts[2], c), "E": _f32(parts[3], c),
            "F": _f32(parts[4], c, pad, 2)}


def cases() -> list[dict]:
    """[{id, label, edge, pos [n, 2] float32, kind [n] int64, E, F [n, 2]}]"""
    out = json.loads((DATA / "cases.json").read_text(encoding="utf-8"))["cases"]
    for c in out:
        c["pos"] = torch.tensor(c["pos"], dtype=torch.float32).reshape(-1, 2)
        c["kind"] = torch.tensor(c["kind"], dtype=torch.int64)
        c["F"] = torch.tensor(c["F"], dtype=torch.float32).reshape(-1, 2)
    return out


def by_size(d: dict) -> dict[int, dict]:
    """the set split by atom count n: {n: {pos [c, n, 2], kind one-hot [c, n, 3], E [c], F [c, n, 2]}}"""
    out = {}
    for n in sorted(set(d["n"].tolist())):
        idx = (d["n"] == n).nonzero().squeeze(1)
        out[n] = {"pos": d["pos"][idx, :n], "kind": torch.nn.functional.one_hot(d["kind"][idx, :n], 3).float(),
                  "E": d["E"][idx], "F": d["F"][idx, :n]}
    return out
