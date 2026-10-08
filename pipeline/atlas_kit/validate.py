"""validate(dir): structural checks of a model folder (offsets, lengths, ids, budgets). Returns a list of problems,
each starting with its AK code; [] means the folder passed. The full schema check is TS (`atlas check`, AK301).

Works on kit (spec 1.1) folders and on the two legacy v1 folders (tinycnn, tinygpt). Standard library only.
"""
from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path
from typing import Any

FILE_MB = 5
TOTAL_MB = 15
_SUB = re.compile(r"^(.+)#([^#]+)$")


def _prod(shape: list) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _load(p: Path, problems: list[str]) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        problems.append(f"AK301 {p.name}: cannot parse JSON ({e})")
        return None


def validate(dir: str | Path, *, file_mb: float = FILE_MB, total_mb: float = TOTAL_MB) -> list[str]:
    d = Path(dir)
    out: list[str] = []
    if not (d / "spec.json").is_file():
        return [f"AK301 {d.as_posix()}: no spec.json"]
    spec = _load(d / "spec.json", out)
    if not isinstance(spec, dict):
        return out
    raw = (d / "spec.json").read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        out.append("AK301 spec.json: starts with a UTF-8 BOM")
    if b"\r\n" in raw:
        out.append("AK301 spec.json: CRLF line endings")

    for k in ("slug", "name", "version", "params", "precision", "tensors", "layers", "parity"):
        if k not in spec:
            out.append(f"AK301 spec.json: missing key {k!r}")
    if out and any("missing key" in p for p in out):
        return out
    if d.parent.name == "models" and spec["slug"] != d.name:  # static/models/<slug>
        out.append(f"AK301 spec.slug {spec['slug']!r} differs from the folder name {d.name!r}")
    if not isinstance(spec["name"], dict) or not spec["name"].get("en"):
        out.append("AK301 spec.name needs {en, zh}")

    # ---- tensors and weights.bin
    wfile = (spec.get("files") or {}).get("weights", "weights.bin")
    wpath = d / wfile
    nfloat = wpath.stat().st_size // 4 if wpath.is_file() else None
    if spec["tensors"] and nfloat is None:
        out.append(f"AK301 {wfile} is missing")
    if wpath.is_file() and wpath.stat().st_size % 4:
        out.append(f"AK301 {wfile}: {wpath.stat().st_size} bytes is not a multiple of 4")
    names: set[str] = set()
    spans = []
    for t in spec["tensors"]:
        nm = t.get("name")
        if nm in names:
            out.append(f"AK301 tensor {nm!r} listed twice")
        names.add(nm)
        if _prod(t.get("shape", [])) != t.get("length"):
            out.append(f"AK301 tensor {nm!r}: length {t.get('length')} != prod(shape {t.get('shape')})")
        if not isinstance(t.get("offset"), int) or t["offset"] < 0:
            out.append(f"AK301 tensor {nm!r}: bad offset {t.get('offset')!r}")
            continue
        if nfloat is not None and t["offset"] + t["length"] > nfloat:
            out.append(f"AK301 tensor {nm!r}: offset {t['offset']} + length {t['length']} > {nfloat} floats in {wfile}")
        spans.append((t["offset"], t["offset"] + t["length"], nm))
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            out.append(f"AK301 tensors {an!r} and {bn!r} overlap in {wfile}")
    used = sum(b - a for a, b, _ in spans)
    if nfloat is not None and spans and used != nfloat:
        out.append(f"AK301 {wfile} holds {nfloat} floats; the tensors cover {used}")
    if isinstance(spec["params"], int) and spans and spec["params"] != used and spec.get("atlas"):
        out.append(f"AK301 spec.params {spec['params']} != {used} floats in the tensor table")

    # ---- inputs, layers, outputs
    inputs = {i["name"] for i in spec.get("inputs") or []}
    if not inputs and "input" in spec:  # v1: one implicit input
        inputs = {"input"}
    for nm, dd in (spec.get("dims") or {}).items():
        if not (isinstance(dd.get("min"), int) and isinstance(dd.get("max"), int) and 1 <= dd["min"] <= dd["max"]):
            out.append(f"AK301 dims.{nm}: need integers 1 <= min <= max")
    dimnames = set((spec.get("dims") or {}).keys()) | {s for i in spec.get("inputs") or [] for s in i.get("shape", []) if isinstance(s, str)}
    dimnames |= {s for s in ((spec.get("input") or {}).get("shape") or []) if isinstance(s, str)}  # v1 (TinyGPT: ['T'])
    ids: dict[str, dict] = {}
    for i, lay in enumerate(spec["layers"]):
        lid = lay.get("id")
        if not isinstance(lid, str) or not lid:
            out.append(f"AK301 layers[{i}]: missing id")
            continue
        if "#" in lid:
            out.append(f"AK301 layer {lid!r}: '#' is the sub-step separator and never appears in a layer id")
        if lid in ids:
            out.append(f"AK301 layer id {lid!r} used twice")
        for k in ("op", "label", "in", "outShape"):
            if k not in lay:
                out.append(f"AK301 layer {lid!r}: missing {k!r}")
        for src in lay.get("in", []):
            if src not in ids and src not in inputs:
                out.append(f"AK301 layer {lid!r}: input {src!r} is neither an earlier layer nor a declared input")
        for dim in lay.get("outShape", []):
            if isinstance(dim, str) and dim not in dimnames:
                out.append(f"AK301 layer {lid!r}: outShape uses undeclared dim {dim!r}")
        for slot, tname in (lay.get("w") or {}).items():
            if tname not in names:
                out.append(f"AK301 layer {lid!r}: w.{slot} -> {tname!r} is not in the tensor table")
        for slot, v in (lay.get("params") or {}).items():
            if isinstance(v, str) and slot in ("weight", "bias") and v not in names:  # v1 string-valued weights
                out.append(f"AK301 layer {lid!r}: params.{slot} -> {v!r} is not in the tensor table")
        ids[lid] = lay
    for o in spec.get("outputs") or []:
        if o not in ids:
            out.append(f"AK301 outputs: {o!r} is not a layer id")
    p = spec["parity"]
    if not isinstance(p, dict) or "against" not in p:
        out.append("AK301 parity: needs at least {against, max_abs_diff, checked}")

    # ---- reference.json (atlas-ref-1 only; legacy layouts are read by graph/refs.ts)
    rfile = (spec.get("files") or {}).get("reference", "reference.json")
    rpath = d / rfile
    if rpath.is_file():
        ref = _load(rpath, out)
        if isinstance(ref, dict) and ref.get("format") == "atlas-ref-1":
            for c in ref.get("cases", []):
                cid = c.get("id")
                for nm, e in list(c.get("inputs", {}).items()) + list(c.get("tensors", {}).items()):
                    try:
                        nbytes = len(base64.b64decode(e["b64"], validate=True))
                    except Exception:  # noqa: BLE001
                        out.append(f"AK301 reference case {cid!r}: {nm!r} has invalid base64")
                        continue
                    if nbytes != _prod(e["shape"]) * (8 if e.get("dtype") == "float64" else 4):
                        out.append(f"AK301 reference case {cid!r}: {nm!r} has {nbytes} bytes for shape {e['shape']}")
                for tid in c.get("order", []):
                    m = _SUB.match(tid)
                    lid, sub = (m.group(1), m.group(2)) if m else (tid, None)
                    if lid not in ids:
                        out.append(f"AK301 reference case {cid!r}: {tid!r} is not a layer id")
                    elif sub is not None and sub not in (ids[lid].get("sub") or []):
                        out.append(f"AK301 reference case {cid!r}: {tid!r}: layer {lid!r} declares no sub-step {sub!r}")
                    if tid not in c.get("tensors", {}):
                        out.append(f"AK301 reference case {cid!r}: {tid!r} is in order but has no tensor")

    # ---- facts.json
    fpath = d / "facts.json"
    if fpath.is_file():
        facts = _load(fpath, out)
        if isinstance(facts, dict):
            if facts.get("format") != "atlas-facts/1":
                out.append("AK301 facts.json: format must be 'atlas-facts/1'")
            for k, e in (facts.get("facts") or {}).items():
                if not isinstance(e, dict) or "value" not in e or "source" not in e or "measured" not in e:
                    out.append(f"AK301 facts.json: {k!r} needs value, source and measured")
                elif isinstance(e["value"], float) and not math.isfinite(e["value"]):
                    out.append(f"AK301 facts.json: {k!r} is not finite")

    # ---- tapes
    for tj in sorted(d.glob("tapes/*.tape.json")):
        h = _load(tj, out)
        if not isinstance(h, dict):
            continue
        if h.get("format") != "atlas-tape/1":
            out.append(f"AK301 {tj.relative_to(d).as_posix()}: format must be 'atlas-tape/1'")
        b = tj.with_name(tj.name[: -len(".tape.json")] + ".tape.bin")
        want = (h.get("bin") or {}).get("bytes")
        if want is not None and (not b.is_file() or b.stat().st_size != want):
            out.append(f"AK301 {tj.relative_to(d).as_posix()}: {b.name} should hold {want} bytes")
        if tj.stat().st_size + (b.stat().st_size if b.is_file() else 0) > 2e6:
            out.append(f"AK303 {tj.relative_to(d).as_posix()}: a phone tape is over 2 MB")

    # ---- budgets (AK303)
    total = 0
    for f in d.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
            if f.stat().st_size > file_mb * 1e6:
                out.append(f"AK303 {f.relative_to(d).as_posix()}: {f.stat().st_size / 1e6:.2f} MB > {file_mb:g} MB per file")
    if total > total_mb * 1e6:
        out.append(f"AK303 {d.name}: {total / 1e6:.2f} MB > {total_mb:g} MB per model")
    return out


if __name__ == "__main__":  # python -m atlas_kit.validate <dir> [...]
    import sys
    bad = 0
    for arg in sys.argv[1:] or ["."]:
        probs = validate(arg)
        print(f"{arg}: {'OK' if not probs else f'{len(probs)} problem(s)'}")
        for pr in probs:
            print("  " + pr)
        bad += bool(probs)
    sys.exit(1 if bad else 0)
