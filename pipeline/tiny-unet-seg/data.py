"""tiny-unet-seg: read the sets make_data.ts wrote (runs/data/<set>.bin, meta.json) and check their CRC-32.

Each set is n pictures as float32 [n, 32, 32] followed by their true classes as uint8 [n, 32, 32], made by the page's own
scenes.ts render(). The CRC-32 of both parts must equal what TS recorded, so training uses exactly the page's bytes.
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


def load(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(pictures [n, 1, 32, 32] float32, classes [n, 32, 32] int64); raises if the bytes differ from what TS wrote."""
    m = meta()["sets"][name]
    n, s = m["n"], meta()["size"]
    raw = (DATA / f"{name}.bin").read_bytes()
    ib, lb = raw[: n * s * s * 4], raw[n * s * s * 4:]
    if zlib.crc32(ib) != m["crcImages"] or zlib.crc32(lb) != m["crcLabels"]:
        raise SystemExit(f"{name}: CRC-32 of the bytes read differs from the TS record (re-run make_data.ts)")
    x = np.frombuffer(ib, dtype="<f4").reshape(n, 1, s, s).copy()
    y = np.frombuffer(lb, dtype=np.uint8).reshape(n, s, s).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def case_ids() -> list[str]:
    return meta()["sets"]["cases"]["ids"]
