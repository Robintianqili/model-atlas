"""ak.data.quickdraw(classes, split): Quick, Draw! 28x28 bitmaps with TinyCNN's split.

Wraps pipeline/tinycnn/common.py (load_split: the first 15,000 drawings per class, split 12,000 / 1,500 / 1,500 with
seed 20261005), so every model trained on drawings uses the same split and the same pixel/255 scaling as TinyCNN.
Data: <ATLAS_ROOT>/pipeline/data/quickdraw/<class>.npy (CC BY 4.0, Quick, Draw! dataset by Google Creative Lab).
No download: a missing class file raises AK002 with the path.
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Sequence

from ..errors import fail
from ..paths import root_path

_common = None


def _load_common():
    global _common
    if _common is None:
        p = root_path("pipeline", "tinycnn", "common.py")
        if not p.is_file():
            fail("AK002", f"{p.as_posix()} not found", fix="ak.data.quickdraw wraps TinyCNN's loaders; run from the main repo")
        spec = importlib.util.spec_from_file_location("atlas_kit_tinycnn_common", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _common = mod
    return _common


def quickdraw(classes: Sequence[str], split: str = "train"):
    """(X uint8 [N, 28, 28], y int64 [N], idx_in_file [N]) for split 'train' | 'val' | 'test'.
    Labels are positions in `classes`. Model input = X / 255 (white strokes on black)."""
    if split not in ("train", "val", "test"):
        fail("AK301", f"split must be 'train', 'val' or 'test', not {split!r}")
    common = _load_common()
    folder = root_path("pipeline", "data", "quickdraw")
    missing = [c for c in classes if not (folder / (c.replace(" ", "_") + ".npy")).is_file()]
    if missing:
        fail("AK002", f"Quick, Draw! bitmaps missing for {missing} in {folder.as_posix()}",
             likely="this class was never downloaded (TinyCNN uses 12 classes)",
             fix="pick from the downloaded classes, or ask the user before downloading new data")
    return common.load_split(list(classes), split)


def available() -> list[str]:
    """Class names with a downloaded bitmap file."""
    folder = root_path("pipeline", "data", "quickdraw")
    return sorted(p.stem.replace("_", " ") for p in folder.glob("*.npy")) if folder.is_dir() else []
