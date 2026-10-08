"""Re-describe the shipped TinyCNN with the SDK (design §8.2 W1-E2 acceptance 1; §10.2 "the cheapest proof that the
conv/pool/dense mirrors and the writer are right").

    python -m atlas_kit.tests.redescribe_tinycnn          # PYTHONPATH=pipeline; writes $ATLAS_SITE/.atlas/tmp/tinycnn/

The checkpoint pipeline/tinycnn/model_final.pt is described with SpecBuilder calls (one per layer, in the order of the
shipped spec), using the shipped reference inputs as cases. Output goes to $ATLAS_SITE/.atlas/tmp/tinycnn/ and NEVER to
static/. Then it compares with $ATLAS_SITE/static/models/tinycnn/: layer ids, ops, outShape, params and a byte-identical
weights.bin. Prints one JSON line `REDESCRIBE {...}` with the result; exit code 1 on any difference.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

import atlas_kit as ak
from atlas_kit.paths import atlas_root, atlas_site


def load_model() -> nn.Module:
    sys.path.insert(0, str(atlas_root() / "pipeline" / "tinycnn"))
    from common import TinyVGG  # noqa: E402  (the shipped training code; nothing is downloaded)
    sd = torch.load(atlas_root() / "pipeline" / "tinycnn" / "model_final.pt", map_location="cpu")
    ch = (sd["conv1_1.weight"].shape[0], sd["conv1_2.weight"].shape[0], sd["conv2_1.weight"].shape[0], sd["conv2_2.weight"].shape[0])
    m = TinyVGG(ch, sd["dense.weight"].shape[0])
    m.load_state_dict(sd)
    return m.eval()


class WithSoftmax(nn.Module):
    """module(x) for the AK103 proof: the shipped graph ends with a softmax layer."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)


def redescribe(out: Path) -> dict:
    site = atlas_site()
    shipped_dir = site / "static" / "models" / "tinycnn"
    shipped = json.loads((shipped_dir / "spec.json").read_text(encoding="utf-8"))
    lab = {l["id"]: l for l in shipped["layers"]}
    ref = json.loads((shipped_dir / "reference.json").read_text(encoding="utf-8"))
    m = load_model()

    def meta(id: str) -> dict:
        l = lab[id]
        d = {"label": (l["label"]["en"], l["label"]["zh"])}
        if "desc" in l:
            d["desc"] = (l["desc"]["en"], l["desc"]["zh"])
        return d

    def describe(b: ak.SpecBuilder, c: ak.Case) -> None:
        x = b.input("input", c.input, adapter="drawing28", kind="drawing", axes="c,y,x", **meta("input"))
        h = b.conv2d("conv1_1", x, m.conv1_1, **meta("conv1_1"))
        h = b.relu("relu1_1", h, **meta("relu1_1"))
        h = b.conv2d("conv1_2", h, m.conv1_2, **meta("conv1_2"))
        h = b.relu("relu1_2", h, **meta("relu1_2"))
        h = b.maxpool2d("pool1", h, 2, **meta("pool1"))
        h = b.conv2d("conv2_1", h, m.conv2_1, **meta("conv2_1"))
        h = b.relu("relu2_1", h, **meta("relu2_1"))
        h = b.conv2d("conv2_2", h, m.conv2_2, **meta("conv2_2"))
        h = b.relu("relu2_2", h, **meta("relu2_2"))
        h = b.maxpool2d("pool2", h, 2, **meta("pool2"))
        h = b.flatten("flatten", h, **meta("flatten"))
        h = b.linear("dense", h, m.dense, alias="dense", **meta("dense"))
        b.output(b.softmax("softmax", h, **meta("softmax")))

    cases = [ak.Case(f"ref{i}", r["label"], torch.tensor(r["input"], dtype=torch.float32).reshape(1, 28, 28))
             for i, r in enumerate(ref["inputs"])]
    rep = ak.export_model(describe, cases, slug="tinycnn", name=(shipped["name"]["en"], shipped["name"]["zh"]),
                          out=str(out), module=WithSoftmax(m), module_input=lambda c: c.input.unsqueeze(0),
                          extras={"classes": shipped["classes"]}, version=shipped["version"],
                          note="Re-description of the shipped TinyCNN by atlas_kit (W1-E2 acceptance 1); never shipped.")
    mine = json.loads((out / "spec.json").read_text(encoding="utf-8"))

    def key(l: dict) -> dict:
        return {"id": l["id"], "op": l["op"], "outShape": l["outShape"], "params": l.get("params", {})}

    a = [key(l) for l in mine["layers"]]
    b = [key(l) for l in shipped["layers"]]
    diffs = [f"{x} != {y}" for x, y in zip(a, b) if x != y]
    if len(a) != len(b):
        diffs.append(f"{len(a)} layers vs {len(b)} shipped")
    wa = (out / "weights.bin").read_bytes()
    wb = (shipped_dir / "weights.bin").read_bytes()
    tens_same = [(t["name"], t["shape"], t["offset"], t["length"]) for t in mine["tensors"]] == \
                [(t["name"], t["shape"], t["offset"], t["length"]) for t in shipped["tensors"]]
    other = {"in": [(l["id"], l["in"]) for l, s in zip(mine["layers"], shipped["layers"]) if l["in"] != s["in"]],
             "labels_equal": all(l["label"] == s["label"] for l, s in zip(mine["layers"], shipped["layers"]))}
    return {
        "out": out.as_posix(),
        "layers": len(a),
        "layer_diffs": diffs,
        "same_ids_ops_outShape_params": not diffs,
        "tensors_table_equal": tens_same,
        "weights_bytes": len(wa),
        "weights_sha256": hashlib.sha256(wa).hexdigest(),
        "shipped_weights_sha256": hashlib.sha256(wb).hexdigest(),
        "weights_byte_identical": wa == wb,
        "params": mine["params"],
        "module_max_abs_diff": rep.module_max_abs_diff,
        "other_differences": other,
        "warnings": rep.warnings,
    }


def main() -> int:
    out = atlas_site() / ".atlas" / "tmp" / "tinycnn"
    r = redescribe(out)
    print("REDESCRIBE " + json.dumps(r, ensure_ascii=False))
    return 0 if r["same_ids_ops_outShape_params"] and r["weights_byte_identical"] and r["tensors_table_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
