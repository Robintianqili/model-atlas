"""File writing for every SDK output: UTF-8 without BOM and LF line endings, also on Windows (where Python's text
mode would write CRLF). JSON never contains NaN or Infinity (JSON.parse in the browser rejects them)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Optional

from .errors import fail


def _find_nonfinite(o: Any, path: str = "$") -> Optional[str]:
    if isinstance(o, float) and not math.isfinite(o):
        return path
    if isinstance(o, dict):
        for k, v in o.items():
            r = _find_nonfinite(v, f"{path}.{k}")
            if r:
                return r
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            r = _find_nonfinite(v, f"{path}[{i}]")
            if r:
                return r
    return None


def json_bytes(obj: Any, *, indent: Optional[int] = None, where: str = "") -> bytes:
    """Serialize to UTF-8 JSON bytes (LF only, no BOM, no NaN/Infinity: AK301 names the first bad path)."""
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False,
                          separators=(",", ":") if indent is None else (",", ": "))
    except ValueError:
        bad = _find_nonfinite(obj)
        fail("AK301", f"non-finite number at {bad} cannot be written as JSON", where=where or None,
             fix="replace NaN/Infinity with null or a finite value (tensors go into b64, where -Infinity is fine)")
    return (text + "\n").encode("utf-8")


def file_safe(id: str) -> str:
    """A file-name-safe form of an id ('train:fit' -> 'train-fit'); the same rule as fileSafe() in scripts/atlas/lib/io.ts.
    ':' would create an NTFS alternate data stream on Windows."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "-", id)


def write_bytes(path: os.PathLike | str, data: bytes) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:  # binary mode: no newline translation
        f.write(data)
    os.replace(tmp, p)
    return len(data)


def write_json(path: os.PathLike | str, obj: Any, *, indent: Optional[int] = None) -> int:
    return write_bytes(path, json_bytes(obj, indent=indent, where=str(path)))


def write_text(path: os.PathLike | str, text: str) -> int:
    return write_bytes(path, text.replace("\r\n", "\n").encode("utf-8"))


def read_json(path: os.PathLike | str) -> Any:
    """Read JSON, tolerating a UTF-8 BOM."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_arg(s: str | None, default: Any = None) -> Any:
    """A JSON command-line argument: inline JSON, or '@<file>' naming a JSON file (DevKit K1: `atlas parity` passes a long
    process input to reference.py as --input @<file>, because the Windows command line holds about 32K characters)."""
    if s is None or s == "":
        return default
    if s.startswith("@"):
        return read_json(s[1:])
    return json.loads(s)
