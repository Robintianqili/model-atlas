"""SpecBuilder: write the graph once in Python; every call computes the PyTorch value for the current case and
records the layer (design §4.11).

    def describe(b: ak.SpecBuilder, case: ak.Case):
        x = b.input("input", case.input, adapter="points2d", kind="points2d", axes="f")
        h = b.relu("relu1", b.linear("fc1", x, model.fc1, label=("Hidden layer", "隐藏层")))
        b.output(b.linear("fc2", h, model.fc2))

Values carry no batch dimension (CONTRACTS trace format: images [C,H,W], sequences [T,D]).
export_model() calls describe() once per case: the first call fixes the graph and the weights, later calls must
give the same graph (AK101). Every tensor handed to the builder must be on the CPU (AK102).

**meta on every call: label=(en, zh), desc=(en, zh), ui={...}, teach={...}, code=("pipeline/<slug>/model.py", line),
axes="c,y,x". Layer ids may contain dots but never '#'; sub-steps are recorded as "<id>#<sub>".
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch

from . import ops as _ops
from .errors import fail
from .numerics import check_cpu

AXES = ("c", "y", "x", "t", "d", "h", "tq", "tk", "n", "f", "class", "vocab", "p", "xy", "k", "nk", "b")


@dataclass
class Case:
    """One reference input. `input` is whatever describe() expects; `edge` marks edge cases (blank, T = 1, ...)."""
    id: str
    label: str
    input: Any
    edge: bool = False


class Node:
    """A recorded value: a layer output or a declared input. `value` is a CPU float32 (or int) tensor, no batch dim."""
    __slots__ = ("id", "value")

    def __init__(self, id: str, value: torch.Tensor):
        self.id = id
        self.value = value

    @property
    def shape(self) -> list[int]:
        return list(self.value.shape)

    def __repr__(self) -> str:
        return f"Node({self.id!r}, shape={self.shape})"


def parse_axes(axes: Union[str, Sequence[str], None], where: str) -> Optional[list[str]]:
    """'c,y,x' | 'c y x' | 'cyx' | 'class' | ['t', 'd'] -> a list of Axis names (AK301 on an unknown axis)."""
    if axes is None:
        return None
    if isinstance(axes, str):
        s = axes.strip()
        if s in AXES:
            out = [s]
        elif "," in s or " " in s:
            out = [a for a in s.replace(",", " ").split() if a]
        else:
            out = list(s)
    else:
        out = list(axes)
    bad = [a for a in out if a not in AXES]
    if bad:
        fail("AK301", f"unknown axis name(s) {bad}", where=where, fix=f"use names from: {', '.join(AXES)}")
    return out


def i18n(v: Any, where: str, what: str) -> Optional[dict]:
    if v is None:
        return None
    if isinstance(v, dict) and "en" in v:
        return {"en": str(v["en"]), **({"zh": str(v["zh"])} if v.get("zh") else {})}
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return {"en": str(v[0]), "zh": str(v[1])}
    if isinstance(v, str):
        return {"en": v}
    fail("AK301", f"{what} must be (en, zh) or {{'en': ..., 'zh': ...}}, got {v!r}", where=where)


def _tensor_key(t: torch.Tensor) -> tuple:
    """Identity of a tensor's memory: tied parameters (the same Parameter, or views of it) share this key."""
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _caller(depth: int = 2) -> str:
    """'pipeline/<slug>/export.py:41' for messages (the first frame outside atlas_kit)."""
    here = Path(__file__).resolve().parent
    for fr in inspect.stack()[depth:]:
        p = Path(fr.filename).resolve()
        if here not in p.parents and p.parent != here:
            try:
                from .paths import rel
                return f"{rel(p)}:{fr.lineno}"
            except Exception:  # noqa: BLE001 - messages must never fail
                return f"{p.as_posix()}:{fr.lineno}"
    return "?"


@dataclass
class _Layer:
    id: str
    op: str
    ins: list[str]
    shape: list[int]                       # concrete outShape for this case
    params: dict = field(default_factory=dict)
    w: dict = field(default_factory=dict)  # slot -> tensor name (only non-default names are written)
    weights: dict = field(default_factory=dict)  # slot -> tensor name (all slots)
    sub: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    sub_shapes: dict = field(default_factory=dict)


class _Session:
    """State shared by the SpecBuilders of one export: the graph of case 0, the weight table, per-case traces."""

    def __init__(self, site: Optional[Path] = None):
        self.site = site
        self.case_index = 0
        self.layers: list[_Layer] = []            # case 0
        self.inputs: list[dict] = []              # InputDecl rows (case 0)
        self.dims: dict[str, dict] = {}           # declared symbolic dims -> {min, max, label}
        self.tensors: dict[str, torch.Tensor] = {}  # name -> f32 CPU tensor (case 0), insertion order = weights.bin order
        self.tensor_keys: dict[tuple, str] = {}
        self.outputs: Optional[list[str]] = None
        self.cases: list[dict] = []               # per case: {layers, inputs, input_shapes, order, values}
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


class SpecBuilder:
    """See the module docstring. Created by export_model(); one builder per case."""

    def __init__(self, session: _Session, case: Case):
        self._s = session
        self.case = case
        self._first = session.case_index == 0
        self._layers: list[_Layer] = []
        self._ids: set[str] = set()
        self._inputs: dict[str, torch.Tensor] = {}
        self._input_decl: list[dict] = []
        self._input_shapes: dict[str, list[int]] = {}
        self._values: dict[str, torch.Tensor] = {}
        self._outputs: Optional[list[str]] = None

    # ------------------------------------------------------------------ inputs and outputs
    def input(self, name: str, value: Any, *, adapter: str, kind: str, axes: Union[str, Sequence[str]],
              shape: Optional[list] = None, dims: Optional[dict] = None, dtype: Optional[str] = None,
              layer: Optional[bool] = None, **meta) -> Node:
        """Declare an input. `shape` may hold symbolic dims ('T'); `dims={'T': (1, 64, ('tokens', '词元'))}`.
        With layer=True (default for float inputs) an `input` layer with the same id shows it as layer 0."""
        where = f"input {name!r} ({_caller()})"
        self._check_id(name, where)
        t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        check_cpu(t, name, where)
        is_int = not t.dtype.is_floating_point
        dt = dtype or ("i32" if is_int else "f32")
        if dt not in ("f32", "i32"):
            fail("AK301", f"input dtype must be 'f32' or 'i32', got {dt!r}", where=where)
        t = t.detach().to(torch.int64 if dt == "i32" else torch.float32).contiguous()
        ax = parse_axes(axes, where)
        decl_shape = list(shape) if shape is not None else list(t.shape)
        if len(decl_shape) != t.dim():
            fail("AK301", f"input {name!r}: shape {decl_shape} has {len(decl_shape)} axes, the value has {t.dim()}", where=where)
        if ax is not None and len(ax) != t.dim():
            fail("AK301", f"input {name!r}: axes {ax} has {len(ax)} names, the value has {t.dim()} axes", where=where)
        for d, n in zip(decl_shape, t.shape):
            if isinstance(d, int) and d != n:
                fail("AK101", f"input {name!r}: declared size {d} but this case has {n}", where=where,
                     fix="declare the varying axis with a symbolic dim, e.g. shape=['T'] (and dims={'T': (min, max, (en, zh))})")
        for k, v in (dims or {}).items():
            self._s.dims.setdefault(k, self._dim_decl(k, v, where))
        decl = {"name": name, "adapter": adapter, "kind": kind, "shape": decl_shape, "dtype": dt}
        if ax is not None:
            decl["axes"] = ax
        self._input_decl.append(decl)
        self._inputs[name] = t
        self._input_shapes[name] = list(t.shape)
        node = Node(name, t)
        if layer is None:
            layer = dt == "f32"
        if layer:
            meta.setdefault("axes", ax)
            # the input layer shares the input's id (TinyCNN convention: layer 'input' reads input 'input')
            self._record("input", name, [name], t.float(), {}, {}, meta, where, out_shape=decl_shape)
        else:
            self._ids.add(name)  # reserve the name: no layer may reuse it
        return node

    def output(self, *nodes: Node) -> None:
        self._outputs = [n.id for n in nodes]

    # ------------------------------------------------------------------ CORE_OPS mirrors (all 20 ops of site/src/lib/atlas/ops)
    def linear(self, id: str, x: Node, module: Optional[torch.nn.Linear] = None, *, weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None, tied_to: Optional[str] = None, alias: Optional[str] = None,
               accumulate: Optional[str] = None, n_head: Optional[int] = None, **meta) -> Node:
        """y = x W^T + b. Weights from `module` (nn.Linear) or weight=/bias=. tied_to='tok_emb.weight' reuses a
        tensor already in the weight table (no copy in weights.bin). alias='dense' writes the op as 'dense'.
        n_head=4 writes the display hints nHead/dHead (a q/k/v projection whose outputs are 4 heads)."""
        where = f"layer {id!r} ({_caller()})"
        if module is not None:
            weight = module.weight if weight is None else weight
            bias = module.bias if bias is None and module.bias is not None else bias
        if tied_to is not None:
            if tied_to not in self._s.tensors:
                fail("AK301", f"tied_to={tied_to!r} is not in the weight table", where=where,
                     fix="tie only to a tensor an earlier layer registered, e.g. 'tok_emb.weight'")
            if weight is None:
                weight = self._s.tensors[tied_to]
        if weight is None:
            fail("AK301", "linear needs a module or weight=", where=where)
        params: dict = {}
        if accumulate is not None:
            params["accumulate"] = accumulate
        out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
        op = alias or "linear"
        if _ops.declared(op, "inFeatures", self._s.site):
            params["inFeatures"] = in_f
        if _ops.declared(op, "outFeatures", self._s.site):
            params["outFeatures"] = out_f
        if n_head is not None:
            if out_f % int(n_head):
                fail("AK301", f"n_head={n_head} does not divide the {out_f} outputs", where=where)
            params["nHead"], params["dHead"] = int(n_head), out_f // int(n_head)
        ws = {"weight": weight}
        if bias is not None:
            ws["bias"] = bias
        names = {"weight": tied_to} if tied_to else {}
        return self._apply(op, id, [x], params, ws, meta, where, names=names)

    def relu(self, id: str, x: Node, **meta) -> Node:
        return self._apply("relu", id, [x], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def tanh(self, id: str, x: Node, **meta) -> Node:
        return self._apply("tanh", id, [x], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def sigmoid(self, id: str, x: Node, **meta) -> Node:
        return self._apply("sigmoid", id, [x], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def silu(self, id: str, x: Node, **meta) -> Node:
        return self._apply("silu", id, [x], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def gelu(self, id: str, x: Node, approx: str = "none", *, alias: Optional[str] = None, **meta) -> Node:
        """GELU. approx='tanh' is the GPT-2 approximation (written as params {approx: 'tanh'}); alias='gelu_tanh'
        writes the op as 'gelu_tanh' instead (the alias carries approx='tanh')."""
        where = f"layer {id!r} ({_caller()})"
        if approx not in ("none", "tanh"):
            fail("AK106", f"gelu approx must be 'none' or 'tanh', got {approx!r}", where=where)
        if alias is not None:
            return self._apply(alias, id, [x], {}, {}, meta, where)
        return self._apply("gelu", id, [x], {"approx": "tanh"} if approx == "tanh" else {}, {}, meta, where)

    def conv2d(self, id: str, x: Node, module: Optional[torch.nn.Conv2d] = None, *, weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None, stride: Optional[int] = None, padding: Optional[int] = None,
               accumulate: Optional[str] = None, **meta) -> Node:
        """A square kernel over a [C, H, W] value. kernel, stride and padding are read from the nn.Conv2d (or given with
        weight=/bias=); inChannels/outChannels are written from the weight. Only square kernels, equal strides and
        equal integer padding (no dilation, no groups) are supported by the TS op."""
        where = f"layer {id!r} ({_caller()})"
        if module is not None:
            weight = module.weight if weight is None else weight
            bias = module.bias if bias is None and module.bias is not None else bias
            ks, st, pd = module.kernel_size, module.stride, module.padding
            if isinstance(pd, str):
                fail("AK106", f"conv2d padding={pd!r} is not supported", where=where, fix="use an integer padding")
            if ks[0] != ks[1] or st[0] != st[1] or pd[0] != pd[1]:
                fail("AK106", f"conv2d needs a square kernel and equal strides/padding, got kernel {ks}, stride {st}, padding {pd}", where=where)
            if module.dilation != (1, 1) or module.groups != 1 or module.padding_mode != "zeros":
                fail("AK106", "conv2d supports dilation 1, groups 1 and zero padding only", where=where,
                     fix="teach a local op (atlas new op <name> --local <slug> --like conv2d)")
            stride = st[0] if stride is None else stride
            padding = pd[0] if padding is None else padding
        if weight is None:
            fail("AK301", "conv2d needs a module or weight=", where=where)
        if weight.dim() != 4 or weight.shape[2] != weight.shape[3]:
            fail("AK106", f"conv2d weight must be [O, C, k, k], got {list(weight.shape)}", where=where)
        params: dict = {"kernel": int(weight.shape[2]), "stride": int(stride if stride is not None else 1),
                        "padding": int(padding if padding is not None else 0),
                        "inChannels": int(weight.shape[1]), "outChannels": int(weight.shape[0])}
        if accumulate is not None:
            params["accumulate"] = accumulate
        ws = {"weight": weight}
        if bias is not None:
            ws["bias"] = bias
        return self._apply("conv2d", id, [x], params, ws, meta, where)

    def maxpool2d(self, id: str, x: Node, kernel: int = 2, stride: Optional[int] = None, **meta) -> Node:
        """Max over kernel x kernel windows of a [C, H, W] value. Writes kernel and stride (stride defaults to kernel)."""
        return self._apply("maxpool2d", id, [x], {"kernel": int(kernel), "stride": int(stride if stride is not None else kernel)},
                           {}, meta, f"layer {id!r} ({_caller()})")

    def flatten(self, id: str, x: Node, **meta) -> Node:
        return self._apply("flatten", id, [x], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def concat(self, id: str, xs: Sequence[Node], axis: int = 0, **meta) -> Node:
        """torch.cat(xs, dim=axis) on values without a batch dim (axis 0 is the first axis of each value)."""
        return self._apply("concat", id, list(xs), {"axis": int(axis)}, {}, meta, f"layer {id!r} ({_caller()})")

    def add(self, id: str, a: Node, b: Node, **meta) -> Node:
        return self._apply("add", id, [a, b], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def softmax(self, id: str, x: Node, axis: int = -1, **meta) -> Node:
        """Softmax along `axis` (written only when it is not the default -1)."""
        return self._apply("softmax", id, [x], {} if axis == -1 else {"axis": int(axis)}, {}, meta, f"layer {id!r} ({_caller()})")

    def layernorm(self, id: str, x: Node, module: Optional[torch.nn.LayerNorm] = None, *, eps: Optional[float] = None,
                  weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, **meta) -> Node:
        """LayerNorm over the last axis (biased variance). eps is read from the module and always written."""
        where = f"layer {id!r} ({_caller()})"
        if module is not None:
            if len(module.normalized_shape) != 1:
                fail("AK106", f"layernorm normalizes the last axis only; normalized_shape is {tuple(module.normalized_shape)}", where=where)
            eps = module.eps if eps is None else eps
            if module.elementwise_affine:
                weight = module.weight if weight is None else weight
                bias = module.bias if bias is None and getattr(module, "bias", None) is not None else bias
        ws = {k: v for k, v in (("weight", weight), ("bias", bias)) if v is not None}
        return self._apply("layernorm", id, [x], {"eps": float(eps if eps is not None else 1e-5)}, ws, meta, where)

    def embedding(self, id: str, ids: Node, module: Optional[torch.nn.Embedding] = None, *,
                  weight: Optional[torch.Tensor] = None, **meta) -> Node:
        """Each integer id picks one row of the [V, D] table. `ids` is an i32 input node."""
        where = f"layer {id!r} ({_caller()})"
        weight = module.weight if module is not None and weight is None else weight
        if weight is None:
            fail("AK301", "embedding needs a module or weight=", where=where)
        return self._apply("embedding", id, [ids], {}, {"weight": weight}, meta, where)

    def position_embedding(self, id: str, ids: Node, module: Optional[torch.nn.Embedding] = None, *,
                           weight: Optional[torch.Tensor] = None, **meta) -> Node:
        """Rows 0..T-1 of a [P, D] position table, T = the length of `ids`."""
        where = f"layer {id!r} ({_caller()})"
        weight = module.weight if module is not None and weight is None else weight
        if weight is None:
            fail("AK301", "position_embedding needs a module or weight=", where=where)
        return self._apply("position_embedding", id, [ids], {}, {"weight": weight}, meta, where)

    def attention_scores(self, id: str, q: Node, k: Node, n_head: int, causal: bool = False,
                         scale: Optional[float] = None, **meta) -> Node:
        """Per-head q.k^T * scale -> [H, Tq, Tk] from q, k of shape [T, H*dH]. scale defaults to 1/sqrt(dH) (written
        only when given); causal=True fuses the mask (-Infinity above the diagonal)."""
        where = f"layer {id!r} ({_caller()})"
        width = int(q.value.shape[-1])
        if width % int(n_head):
            fail("AK301", f"n_head={n_head} does not divide the width {width}", where=where)
        params: dict = {"nHead": int(n_head), "dHead": width // int(n_head)}
        if causal:
            params["causal"] = True
        if scale is not None:
            params["scale"] = float(scale)
        return self._apply("attention_scores", id, [q, k], params, {}, meta, where)

    def causal_mask(self, id: str, s: Node, **meta) -> Node:
        return self._apply("causal_mask", id, [s], {}, {}, meta, f"layer {id!r} ({_caller()})")

    def attention_mix(self, id: str, w: Node, v: Node, n_head: int, **meta) -> Node:
        """Per head, weights [Tq, Tk] times values -> [Tq, H*dH]."""
        where = f"layer {id!r} ({_caller()})"
        width = int(v.value.shape[-1])
        if width % int(n_head):
            fail("AK301", f"n_head={n_head} does not divide the width {width}", where=where)
        return self._apply("attention_mix", id, [w, v], {"nHead": int(n_head), "dHead": width // int(n_head)}, {}, meta, where)

    def lstm(self, id: str, x: Node, module: torch.nn.LSTM, **meta) -> Node:
        """One unidirectional LSTM layer over x [T, In] -> h [T, H], with the gate sub-steps i f g o c h recorded as
        "<id>#<sub>". The reference loop is checked against module(x) (max |diff| <= 1e-5, else AK104); both biases are
        kept as nn.LSTM stores them."""
        where = f"layer {id!r} ({_caller()})"
        if module.num_layers != 1 or module.bidirectional or getattr(module, "proj_size", 0):
            fail("AK104", "lstm mirrors one unidirectional layer without projections", where=where,
                 fix="describe each layer with its own b.lstm(...) call (nn.LSTM(num_layers=1))")
        if module.batch_first:
            fail("AK104", "lstm: use batch_first=False (values carry no batch dim)", where=where)
        H = int(module.hidden_size)
        ws = {"weight_ih": module.weight_ih_l0, "weight_hh": module.weight_hh_l0}
        if module.bias:
            ws["bias_ih"], ws["bias_hh"] = module.bias_ih_l0, module.bias_hh_l0
        node = self._apply("lstm", id, [x], {"hidden": H, "inputSize": int(module.input_size)}, ws, meta, where)
        check_cpu(module, id, where)
        with torch.no_grad():
            y, _ = module(x.value.float())
        d = float((y.detach().float() - node.value).abs().max()) if y.numel() else 0.0
        if not d <= 1e-5:
            fail("AK104", f"the lstm gate reference differs from module(x) by {d:.3g} (> 1e-5)", where=where,
                 likely="the module is not a plain nn.LSTM (custom cell, dropout in train mode) or the gate order differs",
                 fix="call model.eval(); use nn.LSTM(num_layers=1, batch_first=False); or describe a custom cell as a local op")
        return node

    # ------------------------------------------------------------------ generic (local ops, and core ops without a helper yet)
    def op(self, type: str, id: str, inputs: Sequence[Node], *, params: Optional[dict] = None,
           weights: Optional[dict] = None, ref: Optional[Callable] = None, **meta) -> Node:
        """Record any op. `ref(xs, params, weights)` returns the output tensor, or (output, {sub: tensor}) to record
        sub-steps "<id>#<sub>". Without `ref` the Python mirror's reference is used (core ops only).
        Local ops: put the reference in pipeline/<slug>/ops_<op>.py; `atlas check` validates params against the TS op."""
        return self._apply(type, id, list(inputs), dict(params or {}), dict(weights or {}), meta,
                           f"layer {id!r} ({_caller()})", ref=ref)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _dim_decl(name: str, v: Any, where: str) -> dict:
        if isinstance(v, dict):
            lo, hi, label = v.get("min"), v.get("max"), v.get("label", name)
        else:
            lo, hi, label = (list(v) + [name])[:3]
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi):
            fail("AK301", f"dim {name!r}: need integers 1 <= min <= max, got {lo!r}, {hi!r}", where=where)
        return {"min": lo, "max": hi, "label": i18n(label, where, f"dim {name} label")}

    def _check_id(self, id: str, where: str) -> None:
        if not isinstance(id, str) or not id:
            fail("AK301", f"layer id must be a non-empty string, got {id!r}", where=where)
        if "#" in id:
            fail("AK301", f"layer id {id!r} contains '#'", where=where,
                 likely="'#' separates sub-steps ('L0.attn.scores#h2'); it never appears in a layer id",
                 fix="rename the layer (dots are fine: 'L0.attn.q')")
        if id in self._ids:
            fail("AK301", f"layer id {id!r} is used twice", where=where)

    def _apply(self, op: str, id: str, xs: list[Node], params: dict, weights: dict, meta: dict, where: str,
               names: Optional[dict] = None, ref: Optional[Callable] = None) -> Node:
        self._check_id(id, where)
        for x in xs:
            if not isinstance(x, Node):
                fail("AK301", f"inputs must be Nodes returned by the builder, got {type(x).__name__}", where=where)
        canon, alias_params, entry, mirror = _ops.resolve(op, self._s.site)
        p = {**alias_params, **params}
        _ops.check_attrs(op, params, where=where, site=self._s.site)
        # weights: CPU check, f32 copy for the table, identity for de-duplication of tied tensors
        wt: dict[str, torch.Tensor] = {}
        for slot, t in weights.items():
            if t is None:
                continue
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            check_cpu(t, f"{id}.{slot}", where)
            wt[slot] = t
        if ref is None:
            if mirror is None:
                fail("AK301", f"op {op!r} has no Python mirror", where=where,
                     fix="pass ref=lambda xs, params, weights: ... (and keep it in pipeline/<slug>/ops_<op>.py)")
            ref = mirror.reference
        with torch.no_grad():
            out = ref([x.value for x in xs], p, {k: v.detach().float() for k, v in wt.items()})
        subs: dict[str, torch.Tensor] = {}
        if isinstance(out, tuple):
            out, subs = out
        if not isinstance(out, torch.Tensor):
            fail("AK301", f"op {op!r}: the reference returned {type(out).__name__}, not a tensor", where=where)
        check_cpu(out, id, where)
        return self._record(op, id, [x.id for x in xs], out.detach().float(), params, wt, meta, where,
                            names=names or {}, subs=subs)

    def _record(self, op: str, id: str, ins: list[str], value: torch.Tensor, params: dict, weights: dict,
                meta: dict, where: str, names: Optional[dict] = None, subs: Optional[dict] = None,
                out_shape: Optional[list] = None) -> Node:
        s = self._s
        names = names or {}
        lay = _Layer(id=id, op=op, ins=ins, shape=list(value.shape), params=dict(params))
        for slot, t in weights.items():
            name = names.get(slot) or self._register(f"{id}.{slot}", t, where)
            lay.weights[slot] = name
            if name != f"{id}.{slot}":
                lay.w[slot] = name
        for sub, t in (subs or {}).items():
            if "#" in sub:
                fail("AK301", f"sub-step name {sub!r} contains '#'", where=where)
            lay.sub.append(sub)
            lay.sub_shapes[sub] = list(t.shape)
            self._values[f"{id}#{sub}"] = t.detach().float().contiguous()
        lay.meta = self._meta(op, id, meta, where)
        if out_shape is not None:
            lay.meta["_outShape"] = list(out_shape)
        self._layers.append(lay)
        self._ids.add(id)
        self._values[id] = value.contiguous()
        return Node(id, value)

    def _register(self, name: str, t: torch.Tensor, where: str) -> str:
        """Add a weight tensor (case 0) or check it against case 0 (later cases). Returns the tensor's table name."""
        s = self._s
        key = _tensor_key(t)
        if self._first:
            if key in s.tensor_keys:
                return s.tensor_keys[key]  # tied: the same memory is already in the table
            if name in s.tensors:
                fail("AK101", f"tensor name {name!r} registered twice", where=where)
            s.tensors[name] = t.detach().to(torch.float32).contiguous().clone()
            s.tensor_keys[key] = name
            return name
        known = s.tensor_keys.get(key)
        if known is not None:
            return known
        if name not in s.tensors or not torch.equal(s.tensors[name], t.detach().to(torch.float32)):
            fail("AK101", f"weights of {name!r} differ from the first case", where=where,
                 likely="describe() builds or changes weights per case", fix="build the model once, outside describe()")
        return name

    def _meta(self, op: str, id: str, meta: dict, where: str) -> dict:
        m: dict = {}
        known = {"label", "desc", "ui", "teach", "code", "axes"}
        unknown = [k for k in meta if k not in known]
        if unknown:
            fail("AK301", f"unknown keyword(s) {unknown} on layer {id!r}", where=where,
                 fix=f"layer meta keys are: {', '.join(sorted(known))}")
        label = i18n(meta.get("label"), where, "label")
        if label is None:
            _, _, _, mirror = _ops.resolve(op, self._s.site)
            en, zh = getattr(mirror, "LABEL", (op, op)) if mirror else (op, op)
            label = {"en": f"{en} ({id})" if id not in (op, "input") else en, "zh": f"{zh}（{id}）" if id not in (op, "input") else zh}
        m["label"] = label
        if meta.get("desc") is not None:
            m["desc"] = i18n(meta["desc"], where, "desc")
        if meta.get("axes") is not None:
            m["axes"] = parse_axes(meta["axes"], where)
        for k in ("ui", "teach"):
            if meta.get(k) is not None:
                if not isinstance(meta[k], dict):
                    fail("AK301", f"{k}= must be a dict", where=where)
                m[k] = meta[k]
        if meta.get("code") is not None:
            c = meta["code"]
            if isinstance(c, dict):
                m["code"] = c
            else:
                c = list(c)
                m["code"] = {"file": str(c[0]), "line": int(c[1]), **({"end": int(c[2])} if len(c) > 2 else {})}
        return m

    def _finish(self) -> dict:
        """The record of this case (called by export_model after describe())."""
        if not self._layers:
            fail("AK301", f"case {self.case.id!r}: describe() recorded no layers")
        order: list[str] = []
        for lay in self._layers:
            order.append(lay.id)
            order.extend(f"{lay.id}#{sub}" for sub in lay.sub)
        outputs = self._outputs or [self._layers[-1].id]
        return {"layers": self._layers, "inputs": self._input_decl, "input_values": self._inputs,
                "input_shapes": self._input_shapes, "order": order, "values": self._values, "outputs": outputs}
