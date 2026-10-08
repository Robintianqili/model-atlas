"""ak.TapeWriter: atlas-tape/1 files written from Python (kind-C references, design §4.11).

Layout (the same as the TS writer in site/scripts/atlas/lib/tapefile.ts; W1-C's saveTape/loadTape must agree):

  <stem>.tape.json   TapeHeader (types/process.ts) plus two keys:
      { format: 'atlas-tape/1', process, version, seed, opt, provenance, schema (SchemaJson), phases,
        keyEvery, bin: { file: '<name>.tape.bin', bytes }, frames: [FrameJson, ...] }
  <stem>.tape.bin    the tensor bytes, little-endian, each tensor starting at a multiple of 8 bytes (zero padding)

  FrameJson = { phase, step?, sub?, key?: true, state, changes?, focus?, metrics?, caption?, draws? }
    - key: true on keyframes (frame index % keyEvery == 0). A keyframe's state holds the full current value of
      every field with track: 'changes'. Other frames leave those fields out and carry their edits in `changes`.
    - fields with the default track ('snapshot') are stored in the frames that have them.
    - derived fields are never stored.
    - a non-key frame may also store a tracked field's full value: the process replaced the field as a whole (it
      gave the field without any change for it, and the value differs from the running one).
  Bin order (so TS and Python writers produce identical bytes): per frame, the untracked state fields in the order
  given, then the tracked fields in schema order, then changes, focus, metrics and draws.
  Values inside state, metrics, changes and draws:
    - a tensor (numpy array or torch tensor) -> { "$t": "f32" | "f64" | "i32", "shape": [...], "o": byteOffset, "n": count }
    - a non-finite number -> { "$f": "NaN" | "Infinity" | "-Infinity" }   (JSON has no NaN)
    - dicts and lists recurse; numbers, strings, bools and null are plain JSON.

    w = ak.TapeWriter("q", schema, seed=7, opt={"alpha": 0.5}, phases=PHASES)
    w.frame("update", {"Q": q_table}, step=3, changes=[{"field": "Q", "index": [s, a], "from": old, "to": new}],
            metrics={"r": 1.0})
    w.write(ak.site_path(".atlas", "gridworld", "q.ref"))   # -> q.ref.tape.json + q.ref.tape.bin
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ._io import json_bytes, write_bytes
from .errors import fail

FORMAT = "atlas-tape/1"
FIELD_KINDS = ("tensor", "scalar", "int", "index", "text", "json")
PHASE_KINDS = ("input", "forward", "loss", "backward", "update", "observe", "act", "reward", "message", "aggregate",
               "sample", "merge", "output", "other")
_FRAME_KEYS = ("phase", "step", "sub", "key", "state", "changes", "focus", "metrics", "caption", "draws")


def _is_tensor(x: Any) -> bool:
    return isinstance(x, np.ndarray) or (hasattr(x, "detach") and hasattr(x, "numpy"))


def _as_np(x: Any) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
    return x


class _Blob:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.size = 0

    def add(self, a: np.ndarray, where: str) -> dict:
        if a.dtype == np.float32:
            code, b = "f32", a.astype("<f4")
        elif a.dtype == np.float64:
            code, b = "f64", a.astype("<f8")
        elif a.dtype.kind in "iub":
            if a.size and (a.min() < -2**31 or a.max() >= 2**31):
                fail("AK301", f"tensor at {where} does not fit int32")
            code, b = "i32", a.astype("<i4")
        else:
            fail("AK301", f"tensor at {where} has dtype {a.dtype}; tapes store float32, float64 and int32")
        pad = (-self.size) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        data = np.ascontiguousarray(b).tobytes()
        ref = {"$t": code, "shape": [int(d) for d in a.shape], "o": self.size, "n": int(a.size)}
        self.parts.append(data)
        self.size += len(data)
        return ref

    def bytes(self) -> bytes:
        return b"".join(self.parts)


def encode(x: Any, blob: _Blob, where: str) -> Any:
    """A Python value -> its FrameJson form (tensors go into the blob)."""
    if _is_tensor(x):
        return blob.add(_as_np(x), where)
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return {"$f": "NaN"}
        if math.isinf(x):
            return {"$f": "Infinity" if x > 0 else "-Infinity"}
        return x + 0.0  # -0.0 -> 0.0: JSON.stringify writes -0 as 0, so both writers give the same text
    if isinstance(x, dict):
        return {str(k): encode(v, blob, f"{where}.{k}") for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v, blob, f"{where}[{i}]") for i, v in enumerate(x)]
    fail("AK301", f"cannot store a {type(x).__name__} in a tape (at {where})")


def _copy(v: Any) -> Any:
    if _is_tensor(v):
        return np.array(_as_np(v), copy=True)
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy(x) for x in v]
    return v


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and             bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def _i18n(v: Any) -> Any:
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return {"en": v[0], "zh": v[1]}
    if isinstance(v, str):
        return {"en": v}
    return v


class TapeWriter:
    """Collect frames, then write(stem). See the module docstring for the layout."""

    def __init__(self, process: str, schema: dict, seed: int, opt: dict, key_every: int = 64, *,
                 phases: Optional[dict] = None, version: str = "1", provenance: str = "precomputed"):
        if key_every < 1:
            fail("AK301", "key_every must be >= 1")
        if provenance not in ("live", "precomputed", "illustrative"):
            fail("AK301", f"provenance must be live, precomputed or illustrative, not {provenance!r}")
        self.process = process
        self.seed = int(seed)
        self.opt = dict(opt)
        self.key_every = int(key_every)
        self.version = str(version)
        self.provenance = provenance
        self.schema = self._schema(schema)
        self.phases = self._phases(phases or {})
        self.frames: list[dict] = []
        self._tracked = [k for k, f in self.schema.items() if f.get("track") == "changes"]  # schema order
        self._current: dict[str, Any] = {}   # running full value of every tracked field
        self._blob = _Blob()

    @staticmethod
    def _schema(schema: dict) -> dict:
        out = {}
        for k, f in schema.items():
            f = dict(f)
            if "derive" in f:
                fail("AK301", f"schema field {k!r}: 'derive' functions live in the TS ProcessDef; write derived: True")
            if f.get("kind") not in FIELD_KINDS:
                fail("AK301", f"schema field {k!r}: kind must be one of {FIELD_KINDS}")
            f["label"] = _i18n(f.get("label", k))
            out[k] = f
        return out

    @staticmethod
    def _phases(phases: dict) -> dict:
        out = {}
        for k, p in phases.items():
            p = dict(p)
            if p.get("kind") not in PHASE_KINDS:
                fail("AK301", f"phase {k!r}: kind must be one of {PHASE_KINDS}")
            p["label"] = _i18n(p.get("label", k))
            out[k] = {kk: p[kk] for kk in ("label", "kind", "tex") if kk in p}
        return out

    def frame(self, phase: str, state: dict, step: Optional[int] = None, sub: Optional[int] = None,
              changes: Optional[list] = None, metrics: Optional[dict] = None, draws: Optional[list] = None, *,
              focus: Optional[list] = None, caption: Any = None) -> None:
        i = len(self.frames)
        where = f"frame {i} ({phase})"
        if self.phases and phase not in self.phases:
            fail("AK301", f"{where}: phase {phase!r} is not declared", fix=f"declared phases: {', '.join(self.phases)}")
        for k in state:
            if k not in self.schema:
                fail("AK301", f"{where}: state field {k!r} is not in the schema")
            if self.schema[k].get("derived"):
                fail("AK301", f"{where}: {k!r} is a derived field; derived fields are never stored")
        for c in changes or []:
            if c.get("field") not in self._tracked:
                fail("AK301", f"{where}: change to {c.get('field')!r}, which is not a track: 'changes' field")
        # Tracked fields: the running value is the previous one with this frame's changes applied. A value given in
        # `state` must equal that (unreported mutations fail, as the TS keyEvery check does), unless the frame has no
        # changes for the field, which makes it a replacement (stored in this frame).
        store: set[str] = set()
        for k in self._tracked:
            prev = self._current.get(k)
            ch = [c for c in changes or [] if c.get("field") == k]
            new = None
            if prev is not None:
                new = _copy(prev)
                for c in ch:
                    new = self._apply(new, c, where)
            elif ch:
                fail("AK301", f"{where}: change to {k!r} before its first full value",
                     fix="pass the field's full value in state on the first frame")
            if k in state:
                given = _copy(state[k])
                if prev is not None and ch and not _equal(new, given):
                    fail("AK301", f"{where}: the changes to {k!r} do not explain its new value",
                         likely="the process changed the field without reporting every edit in `changes`")
                if prev is None or (not ch and not _equal(prev, given)):
                    store.add(k)
                self._current[k] = given
            elif new is not None:
                self._current[k] = new
        key = i % self.key_every == 0
        st: dict = {}
        for k, v in state.items():
            if k in self._tracked:
                continue
            st[k] = encode(v, self._blob, f"{where}.state.{k}")
        for k in self._tracked:
            if key or k in store:
                if k not in self._current:
                    # K1: a tracked field may start later (the TS TapeStore keeps it absent until its first full
                    # value, and saveTape leaves it out of earlier keyframes): nothing to store yet
                    continue
                st[k] = encode(self._current[k], self._blob, f"{where}.state.{k}")
        fr: dict = {"phase": phase}
        if step is not None:
            fr["step"] = int(step)
        if sub is not None:
            fr["sub"] = int(sub)
        if key:
            fr["key"] = True
        fr["state"] = st
        if changes:
            fr["changes"] = encode([{kk: _i18n(vv) if kk == "note" else vv for kk, vv in c.items()} for c in changes],
                                   self._blob, f"{where}.changes")
        if focus:
            fr["focus"] = encode(focus, self._blob, f"{where}.focus")
        if metrics:
            fr["metrics"] = encode(dict(metrics), self._blob, f"{where}.metrics")
        if caption is not None:
            fr["caption"] = _i18n(caption)
        if draws:  # an empty list is omitted, as the TS TapeStore does
            fr["draws"] = encode([float(d) for d in draws], self._blob, f"{where}.draws")
        self.frames.append({k: fr[k] for k in _FRAME_KEYS if k in fr})

    @staticmethod
    def _apply(cur: Any, c: dict, where: str) -> Any:
        """Apply one change {field, key?, index?, to} to a copy of a tracked value; returns the new value."""
        if "to" not in c:
            fail("AK301", f"{where}: a change to {c['field']!r} needs 'to'")
        if c.get("key") is not None:
            if not isinstance(cur, dict):
                fail("AK301", f"{where}: change with key {c['key']!r} on a field that is not a record")
            if c.get("index") is None:
                cur[c["key"]] = c["to"]
                return cur
            cur[c["key"]][tuple(c["index"])] = c["to"]
            return cur
        if isinstance(cur, np.ndarray):
            if c.get("index") is None:
                fail("AK301", f"{where}: a change to tensor field {c['field']!r} needs an index")
            cur[tuple(c["index"])] = c["to"]
            return cur
        return c["to"]  # a scalar field

    def header(self, bin_name: str) -> dict:
        return {"format": FORMAT, "process": self.process, "version": self.version, "seed": self.seed, "opt": self.opt,
                "provenance": self.provenance, "schema": self.schema, "phases": self.phases, "keyEvery": self.key_every,
                "bin": {"file": bin_name, "bytes": self._blob.size}, "frames": self.frames}

    def write(self, stem: str | Path) -> tuple[Path, Path]:
        """Write <stem>.tape.json and <stem>.tape.bin (UTF-8 without BOM, LF). Returns both paths.
        Build stems from ids with file_safe(): 'train:fit' -> 'train-fit' (':' is an NTFS stream separator)."""
        stem = Path(stem)
        if ":" in stem.name:
            fail("AK301", f"tape file name {stem.name!r} contains ':'", fix="use atlas_kit._io.file_safe(process_id) for the stem")
        jpath = stem.with_name(stem.name + ".tape.json")
        bpath = stem.with_name(stem.name + ".tape.bin")
        write_bytes(bpath, self._blob.bytes())
        write_bytes(jpath, json_bytes(self.header(bpath.name), where=str(jpath)))
        return jpath, bpath


def read_tape(stem: str | Path) -> tuple[dict, bytes]:
    """(header, blob) of a written tape; decode values with decode()."""
    stem = Path(stem)
    from ._io import read_json
    h = read_json(stem.with_name(stem.name + ".tape.json"))
    if h.get("format") != FORMAT:
        fail("AK301", f"{stem}: not an {FORMAT} tape")
    return h, stem.with_name(stem.name + ".tape.bin").read_bytes()


def decode(x: Any, blob: bytes) -> Any:
    """FrameJson value -> Python (tensors become numpy arrays)."""
    if isinstance(x, dict):
        if "$t" in x:
            dt = {"f32": "<f4", "f64": "<f8", "i32": "<i4"}[x["$t"]]
            return np.frombuffer(blob, dtype=dt, count=x["n"], offset=x["o"]).reshape(x["shape"])
        if "$f" in x and len(x) == 1:
            return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[x["$f"]]
        return {k: decode(v, blob) for k, v in x.items()}
    if isinstance(x, list):
        return [decode(v, blob) for v in x]
    return x
