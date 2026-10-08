"""Numerics policy (CONTRACTS "DevKit additions" / Numerics): references and goldens are computed on the CPU
with TF32 off. Importing atlas_kit calls enforce(). Exporting a CUDA tensor or module, or exporting with TF32
switched back on, is refused with AK102.

Training may use the GPU; move the model back with model.cpu() before export.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .errors import fail

try:  # torch is optional for kind-C references (ak.Rng and ak.TapeWriter work without it)
    import torch
except ImportError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]


def enforce() -> None:
    """Switch TF32 off for cuDNN and CUDA matmuls. Safe to call many times."""
    if torch is None:
        return
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def tf32_state() -> dict:
    if torch is None:
        return {"torch": False}
    return {"cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}


def check_tf32(where: Optional[str] = None) -> None:
    """AK102 when something switched TF32 back on after atlas_kit was imported."""
    if torch is None:
        return
    on = [k for k, v in tf32_state().items() if v]
    if on:
        fail("AK102", f"export: TF32 is on ({', '.join(on)})", where=where,
             likely="a training helper set torch.backends.*.allow_tf32 = True in this process",
             fix="export in a fresh process, or call atlas_kit.numerics.enforce() before export_model(...)")


def _device_of(x: Any) -> Optional[str]:
    if torch is None:
        return None
    if isinstance(x, torch.Tensor):
        return str(x.device)
    return None


def check_cpu(x: Any, name: str, where: Optional[str] = None) -> None:
    """AK102 when x (a tensor, a module, or a list/tuple/dict of them) holds anything that is not on the CPU."""
    if torch is None:
        return
    if isinstance(x, torch.nn.Module):
        for pname, p in list(x.named_parameters()) + list(x.named_buffers()):
            if p.device.type != "cpu":
                fail("AK102", f"export: module {name!r} has {pname!r} on {p.device}", where=where,
                     likely="the model was trained on the GPU and not moved back",
                     fix="call model.cpu() before describing or exporting it (references are computed on the CPU)")
        return
    if isinstance(x, torch.Tensor):
        if x.device.type != "cpu":
            fail("AK102", f"export: tensor {name!r} is on {x.device}", where=where,
                 likely="an input or weight was created on (or moved to) the GPU",
                 fix="call .cpu() on it before passing it to the SpecBuilder (references are computed on the CPU)")
        return
    if isinstance(x, dict):
        for k, v in x.items():
            check_cpu(v, f"{name}[{k!r}]", where)
        return
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            check_cpu(v, f"{name}[{i}]", where)


def check_all_cpu(items: Iterable[tuple[str, Any]], where: Optional[str] = None) -> None:
    for name, x in items:
        check_cpu(x, name, where)


enforce()
