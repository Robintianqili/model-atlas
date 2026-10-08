"""export_model(): describe() once per case -> spec.json (1.1, parity null), weights.bin, reference.json
(atlas-ref-1), facts.json, presets.json (design §4.11).

    report = ak.export_model(describe, cases, slug="demo", name=("Demo MLP", "演示 MLP"),
                             module=model, module_input=lambda c: c.input.unsqueeze(0))

CPU only (AK102 for a CUDA tensor or module, or TF32 switched back on). The first describe() call fixes the graph
and the weights; later calls must give the same graph (AK101). With `module`, every case proves that the recorded
outputs equal module(x) (AK103). Budgets: each file <= file_mb, the model folder <= total_mb (AK105); nothing is
written when a budget fails. Files are UTF-8 without BOM, LF line endings. The CLI then runs `atlas parity`.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from . import __version__
from . import ops as _ops
from ._io import json_bytes, write_bytes
from .errors import fail
from .numerics import check_cpu, check_tf32
from .refs import reference_doc
from .spec import Case, SpecBuilder, _Session, i18n

_RESERVED = {"slug", "name", "version", "params", "precision", "input", "tensors", "layers", "training", "parity",
             "atlas", "inputs", "outputs", "dims", "files"}


@dataclass
class ExportReport:
    out: Path
    slug: str
    params: int
    layers: int
    cases: int
    files: dict[str, int]                 # file name -> bytes written
    total_bytes: int                      # the whole model folder after writing
    module_max_abs_diff: Optional[float]  # max |describe() - module(x)| over cases (None without module)
    dims: dict[str, list[int]] = field(default_factory=dict)  # symbolic dim -> values seen across cases
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"atlas export {self.slug}: {self.layers} layers, {self.params:,} params, {self.cases} cases -> {self.out.as_posix()}"]
        for k, v in self.files.items():
            lines.append(f"  {k:<16} {v / 1024:9.1f} KB")
        lines.append(f"  {'(model folder)':<16} {self.total_bytes / 1024:9.1f} KB")
        if self.module_max_abs_diff is not None:
            lines.append(f"  describe() vs module(x): max |diff| {self.module_max_abs_diff:.3g}")
        for d, vals in self.dims.items():
            lines.append(f"  dim {d}: cases cover {sorted(set(vals))}")
        lines += [f"  note: {w}" for w in self.warnings]
        return "\n".join(lines)


def _caller_file() -> Optional[str]:
    here = Path(__file__).resolve().parent
    for fr in inspect.stack()[2:]:
        p = Path(fr.filename).resolve()
        if here != p.parent:
            try:
                from .paths import rel
                return rel(p)
            except Exception:  # noqa: BLE001
                return p.as_posix()
    return None


def _flatten_outputs(y: Any) -> list[torch.Tensor]:
    if isinstance(y, torch.Tensor):
        return [y]
    if isinstance(y, dict):
        return [v for v in y.values() if isinstance(v, torch.Tensor)]
    if isinstance(y, (list, tuple)):
        return [v for v in y if isinstance(v, torch.Tensor)]
    fail("AK103", f"module(x) returned {type(y).__name__}; expected a tensor, a tuple of tensors or a dict")


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().cpu()
    b = b.detach().double().cpu()
    if a.shape != b.shape and a.numel() == b.numel():
        a = a.reshape(b.shape)  # a leading batch dim of 1
    if a.shape != b.shape:
        return math.inf
    both_inf = torch.isinf(a) & torch.isinf(b) & (torch.sign(a) == torch.sign(b))
    d = torch.where(both_inf, torch.zeros_like(a), (a - b).abs())
    if torch.isnan(d).any():
        return math.nan
    return float(d.max()) if d.numel() else 0.0


def _resolve_dims(session: _Session, records: list[dict]) -> tuple[dict[str, list[int]], Callable[[str, list[int]], list]]:
    """Per-case values of every declared symbolic dim, and a function that turns per-case concrete shapes into a
    spec outShape (numbers, or dim names where an axis follows a declared dim across cases)."""
    dimvals: dict[str, list[int]] = {}
    for rec in records:
        seen: dict[str, int] = {}
        for decl in rec["inputs"]:
            for d, n in zip(decl["shape"], rec["input_shapes"][decl["name"]]):
                if isinstance(d, str):
                    if d in seen and seen[d] != n:
                        fail("AK101", f"dim {d!r} is {seen[d]} and {n} in the same case", where=f"case {rec['case'].id!r}")
                    seen[d] = n
        for d, n in seen.items():
            dimvals.setdefault(d, []).append(n)
    for d, vals in dimvals.items():
        if len(vals) != len(records):
            fail("AK101", f"dim {d!r} is declared in some cases only")
        if len(set(vals)) == 1 and len(records) > 1:
            session.warn(f"dim {d!r} has the same value ({vals[0]}) in every case: axes of that size stay numbers. "
                         f"Add a case with a different {d} (parity also wants the dim extremes)")
        elif len(records) == 1:
            session.warn(f"only one case: dim {d!r} cannot be traced through the graph, so outShapes stay numbers")

    def out_shape(where: str, per_case: list[list[int]]) -> list:
        ranks = {len(s) for s in per_case}
        if len(ranks) != 1:
            fail("AK101", f"{where}: the number of axes changes between cases ({sorted(ranks)})")
        shape: list = []
        for ax in range(ranks.pop()):
            v = [s[ax] for s in per_case]
            if len(set(v)) == 1:
                shape.append(v[0])
                continue
            hits = [d for d, vals in dimvals.items() if vals == v]
            if not hits:
                fail("AK101", f"{where}: axis {ax} changes between cases ({v}) but no declared dim does",
                     likely="the axis is a function of a dim (T*D after a flatten, T-1, ...) or the graph depends on the input",
                     fix="keep the symbolic dim on its own axis, or declare the input shape with that dim")
            if len(hits) > 1:
                session.warn(f"{where}: axis {ax} matches dims {hits}; using {hits[0]!r}")
            shape.append(hits[0])
        return shape

    return dimvals, out_shape


def export_model(describe: Callable[[SpecBuilder, Case], None], cases: Sequence[Case], *, slug: str,
                 name: tuple[str, str], kinds: Sequence[str] = ("trace",), out: Optional[str] = None,
                 module: Optional[torch.nn.Module] = None, module_input: Optional[Callable[[Case], Any]] = None,
                 extras: Optional[dict] = None, training: Optional[dict] = None, facts: Optional[dict] = None,
                 presets: Optional[dict] = None, bar: float = 1e-4, file_mb: float = 5, total_mb: float = 15,
                 version: Optional[str] = None, note: str = "") -> ExportReport:
    from .paths import atlas_site, rel
    check_tf32(where="export_model")  # importing atlas_kit switched TF32 off; refuse if something turned it back on
    if not cases:
        fail("AK301", "export_model needs at least one Case")
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        fail("AK301", f"case ids must be unique: {ids}")
    if module is not None:
        check_cpu(module, type(module).__name__, "export_model(module=...)")
        module.eval()
    out_dir = Path(out) if out else atlas_site() / "static" / "models" / slug
    script = _caller_file()
    session = _Session(site=None)

    # ---------------------------------------------------------------- describe every case
    records: list[dict] = []
    module_diff: Optional[float] = None
    with torch.no_grad():
        for i, case in enumerate(cases):
            session.case_index = i
            b = SpecBuilder(session, case)
            describe(b, case)
            rec = b._finish()
            rec["case"] = case
            if i == 0:
                session.layers, session.inputs, session.outputs = rec["layers"], rec["inputs"], rec["outputs"]
            else:
                _same_graph(records[0], rec)
            records.append(rec)
            if module is not None and module_input is not None:
                x = module_input(case)
                check_cpu(x, "module_input(case)", f"case {case.id!r}")
                ys = _flatten_outputs(module(x))
                outs = rec["outputs"]
                if len(ys) != len(outs):
                    fail("AK103", f"module(x) gave {len(ys)} output(s); describe() declared {len(outs)} ({outs})",
                         where=f"case {case.id!r}", fix="call b.output(...) with the nodes module(x) returns, in order")
                for y, oid in zip(ys, outs):
                    d = _max_diff(y, rec["values"][oid])
                    module_diff = d if module_diff is None or not (d <= module_diff) else module_diff
                    tol = min(bar, 1e-5)
                    if not d <= tol:
                        fail("AK103", f"describe() output {oid!r} differs from module(x) by {d:.3g} (> {tol:g})",
                             where=f"case {case.id!r}", likely="describe() skips or reorders a step the module runs",
                             fix="compare the layers of describe() with the module's forward() one by one")

    # ---------------------------------------------------------------- symbolic dims and layer table
    dimvals, out_shape = _resolve_dims(session, records)
    first = records[0]
    layers: list[dict] = []
    for li, lay in enumerate(first["layers"]):
        where = f"layer {lay.id!r}"
        shape = lay.meta.get("_outShape") or out_shape(where, [r["layers"][li].shape for r in records])
        d: dict = {"id": lay.id, "op": lay.op, "label": lay.meta["label"], "in": lay.ins, "outShape": shape}
        if lay.params:
            d["params"] = lay.params
        if lay.w:
            d["w"] = lay.w
        for k in ("desc", "axes", "code", "ui", "teach"):
            if k in lay.meta:
                d[k] = lay.meta[k]
        if lay.sub:
            d["sub"] = lay.sub
            for sub in lay.sub:
                out_shape(f"{where} sub-step {sub!r}", [r["layers"][li].sub_shapes[sub] for r in records])
        if not lay.meta["label"].get("zh"):
            session.warn(f"{where}: label has no zh (lint AK406 will ask for it)")
        layers.append(d)

    # ---------------------------------------------------------------- weights.bin
    tensors, chunks, off = [], [], 0
    for tname, t in session.tensors.items():
        a = t.numpy().astype("<f4")
        tensors.append({"name": tname, "shape": [int(s) for s in a.shape], "offset": off, "length": int(a.size)})
        chunks.append(a.ravel())
        off += int(a.size)
    weights = np.concatenate(chunks).astype("<f4").tobytes() if chunks else b""

    # ---------------------------------------------------------------- spec.json
    dims: dict = {}
    for dname, vals in dimvals.items():
        decl = session.dims.get(dname)
        if decl is None:
            session.warn(f"dim {dname!r} has no declared min/max/label: using the case range {min(vals)}..{max(vals)}; "
                         f"declare dims={{'{dname}': (min, max, (en, zh))}} on b.input(...)")
            decl = {"min": min(vals), "max": max(vals), "label": {"en": dname, "zh": dname}}
        if not decl["min"] <= min(vals) <= max(vals) <= decl["max"]:
            fail("AK101", f"dim {dname!r}: cases use {sorted(set(vals))}, outside the declared {decl['min']}..{decl['max']}")
        dims[dname] = decl
    today = version or _dt.date.today().isoformat()
    name_i = i18n(name, "export_model(name=...)", "name")
    spec: dict = {"slug": slug, "name": name_i, "version": today, "params": int(off), "precision": "fp32"}
    if len(session.inputs) == 1:
        i0 = session.inputs[0]
        spec["input"] = {"kind": i0["kind"], "shape": i0["shape"]}
    spec["tensors"] = tensors
    spec["layers"] = layers
    if training is not None:
        spec["training"] = training
    spec["parity"] = {"against": "pytorch-fp32-cpu", "max_abs_diff": None, "checked": None, "bar": bar, "scope": "forward"}
    spec["atlas"] = {"spec": "1.1", "kinds": list(kinds), "exporter": f"atlas_kit {__version__}" + (f" ({script})" if script else "")}
    spec["inputs"] = session.inputs
    spec["outputs"] = session.outputs
    if dims:
        spec["dims"] = dims
    files = {"weights": "weights.bin", "reference": "reference.json"}
    if facts:
        files["facts"] = "facts.json"
    if presets is not None:
        files["presets"] = "presets.json"
    spec["files"] = files
    for k, v in (extras or {}).items():
        if k in _RESERVED:
            fail("AK301", f"extras key {k!r} collides with a spec key", fix="rename it (extras are model-specific keys such as classes, vocabSize)")
        spec[k] = v

    # ---------------------------------------------------------------- reference.json, facts.json, presets.json
    ref = reference_doc(slug, today, [{
        "id": r["case"].id, "label": r["case"].label, "edge": r["case"].edge,
        "inputs": {k: (v, "int32" if not v.dtype.is_floating_point else "float32") for k, v in r["input_values"].items()},
        "order": r["order"], "tensors": r["values"]} for r in records], note=note)
    blobs: dict[str, bytes] = {
        "spec.json": json_bytes(spec, indent=1, where="spec.json"),
        "weights.bin": weights,
        "reference.json": json_bytes(ref, where="reference.json"),
    }
    if facts:
        blobs["facts.json"] = json_bytes(_facts_doc(facts, today, script), indent=1, where="facts.json")
    if presets is not None:
        blobs["presets.json"] = json_bytes(presets, where="presets.json")

    # ---------------------------------------------------------------- budgets (AK105) before anything is written
    for fname, data in blobs.items():
        if len(data) > file_mb * 1e6:
            fail("AK105", f"{fname} would be {len(data) / 1e6:.2f} MB (> {file_mb:g} MB per file)", where=out_dir.as_posix(),
                 fix="fewer or shorter reference cases, fewer presets, or a smaller model")
    others = 0
    if out_dir.is_dir():
        for p in out_dir.rglob("*"):
            if p.is_file() and p.relative_to(out_dir).as_posix() not in blobs:
                others += p.stat().st_size
    total = others + sum(len(b) for b in blobs.values())
    if total > total_mb * 1e6:
        fail("AK105", f"the model folder would be {total / 1e6:.2f} MB (> {total_mb:g} MB)", where=out_dir.as_posix(),
             fix="shrink the files above, or remove stale files from the folder")

    written = {fname: write_bytes(out_dir / fname, data) for fname, data in blobs.items()}
    return ExportReport(out=out_dir, slug=slug, params=int(off), layers=len(layers), cases=len(records), files=written,
                        total_bytes=total, module_max_abs_diff=module_diff, dims=dimvals,
                        warnings=session.warnings + list(_ops.NOTES))


def _same_graph(a: dict, b: dict) -> None:
    """AK101 when case b's graph differs from case a's (ids, ops, inputs, params, weight names, sub-steps)."""
    cid = b["case"].id
    la, lb = a["layers"], b["layers"]
    if [x.id for x in la] != [x.id for x in lb]:
        fail("AK101", f"case {cid!r} records layers {[x.id for x in lb]}; the first case recorded {[x.id for x in la]}",
             likely="describe() branches on the input", fix="record the same layers for every case")
    for x, y in zip(la, lb):
        for attr in ("op", "ins", "params", "weights", "sub"):
            if getattr(x, attr) != getattr(y, attr):
                fail("AK101", f"case {cid!r}, layer {x.id!r}: {attr} {getattr(y, attr)!r} differs from the first case's {getattr(x, attr)!r}")
    if [d["name"] for d in a["inputs"]] != [d["name"] for d in b["inputs"]]:
        fail("AK101", f"case {cid!r} declares different inputs")
    for da, db in zip(a["inputs"], b["inputs"]):
        if da != db:
            fail("AK101", f"case {cid!r}: input {db['name']!r} is declared as {db} (first case: {da})")
    if a["outputs"] != b["outputs"]:
        fail("AK101", f"case {cid!r}: outputs {b['outputs']} differ from the first case's {a['outputs']}")


def _facts_doc(facts: dict, measured: str, script: Optional[str]) -> dict:
    """atlas-facts/1: {format, facts: {key: {value, unit?, source, measured}}}. A fact is a value or a dict."""
    out = {}
    for k, v in facts.items():
        e = dict(v) if isinstance(v, dict) else {"value": v}
        if "value" not in e:
            fail("AK301", f"fact {k!r} has no value")
        if isinstance(e["value"], (np.generic,)):
            e["value"] = e["value"].item()
        if isinstance(e["value"], float) and not math.isfinite(e["value"]):
            fail("AK301", f"fact {k!r} is {e['value']}", fix="facts are measured numbers; leave a broken one out")
        e.setdefault("source", script or "export_model")
        e.setdefault("measured", measured)
        out[k] = e
    return {"format": "atlas-facts/1", "facts": out}
