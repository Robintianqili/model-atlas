"""atlas-ref-1 reference files (design §4.2; CONTRACTS "DevKit additions").

    { format: 'atlas-ref-1', model, version, against: 'pytorch-fp32-cpu', note,
      cases: [{ id, label, edge, inputs: { <name>: { shape, dtype, b64 } }, order,
                tensors: { <id or id#sub>: { shape, dtype: 'float32', b64 } } }] }

`b64` is base64 of little-endian values (float32 for tensors; 'int32' or 'float32' for inputs). -Infinity is stored
as is. Pages never load references; site/src/lib/server/assets.ts leaves them out of the measured download size.
"""
from __future__ import annotations

import base64
from typing import Any

import numpy as np

FORMAT = "atlas-ref-1"
AGAINST = "pytorch-fp32-cpu"
_DT = {"float32": "<f4", "int32": "<i4", "float64": "<f8"}


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor (no torch import needed here)
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def b64_tensor(x: Any, dtype: str = "float32") -> dict:
    """{shape, dtype, b64} for a tensor or array ('float32' or 'int32'; ints are range-checked)."""
    a = _np(x)
    if dtype == "int32":
        if a.size and (a.min() < -2**31 or a.max() >= 2**31):
            raise ValueError("value does not fit int32")
        a = a.astype("<i4")
    else:
        a = a.astype(_DT[dtype])
    return {"shape": [int(d) for d in a.shape], "dtype": dtype, "b64": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")}


def decode_tensor(e: dict) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(e["b64"]), dtype=_DT[e.get("dtype", "float32")])
    return a.reshape(e["shape"]) if e.get("shape") is not None else a


def reference_doc(model: str, version: str, cases: list[dict], note: str = "") -> dict:
    """cases: [{id, label, edge, inputs: {name: (array, 'float32'|'int32')}, order: [...], tensors: {id: array}}]."""
    out = []
    for c in cases:
        out.append({
            "id": c["id"], "label": c["label"], "edge": bool(c.get("edge", False)),
            "inputs": {k: b64_tensor(v, dt) for k, (v, dt) in c["inputs"].items()},
            "order": list(c["order"]),
            "tensors": {k: b64_tensor(c["tensors"][k]) for k in c["order"]},
        })
    return {"format": FORMAT, "model": model, "version": version, "against": AGAINST,
            "note": note or "PyTorch fp32 outputs on the CPU (TF32 off) for every layer id and sub-step, written by atlas_kit.export_model.",
            "cases": out}


def read_reference(doc: dict) -> list[dict]:
    """Decode an atlas-ref-1 document into [{id, label, edge, inputs: {name: array}, order, tensors: {id: array}}]."""
    if doc.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} document (format={doc.get('format')!r})")
    return [{"id": c["id"], "label": c.get("label"), "edge": bool(c.get("edge")),
             "inputs": {k: decode_tensor(v) for k, v in c.get("inputs", {}).items()},
             "order": list(c["order"]),
             "tensors": {k: decode_tensor(v) for k, v in c["tensors"].items()}} for c in doc["cases"]]
