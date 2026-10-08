"""Python mirrors of the tinyvit local ops (site/src/lib/explainers/tinyvit/ops/tokens.ts).

    to_tokens     [C, H, W] -> [H*W, C]        x.flatten(1).transpose(0, 1)
    cls_token     [T, D] + c [D] -> [T+1, D]   torch.cat([c[None], x], 0)
    select_token  [T, D] -> [D]                x[index]

Used from export.py's describe() as b.op("<op>", id, [x], ..., ref=<module>.<op>). Each module-level block below
mirrors the TS OpDef metadata (OP, VERSION, CATEGORY, LABEL, ATTRS, WEIGHTS), as the core mirrors in atlas_kit/ops do.
"""
import torch

OPS = {
    "to_tokens": {"version": 1, "category": "shape", "label": ("Patches become rows", "图块变成行"), "attrs": {}, "weights": []},
    "cls_token": {"version": 1, "category": "embedding", "label": ("Put the [CLS] row in front", "在前面放上 [CLS] 行"),
                  "attrs": {}, "weights": [{"slot": "weight"}]},
    "select_token": {"version": 1, "category": "shape", "label": ("Read the [CLS] row", "读出 [CLS] 行"),
                     "attrs": {"index": {"type": "int", "default": 0}}, "weights": []},
}


def to_tokens(xs, params, weights):
    return xs[0].flatten(1).transpose(0, 1).contiguous()


def cls_token(xs, params, weights):
    return torch.cat([weights["weight"].reshape(1, -1), xs[0]], 0)


def select_token(xs, params, weights):
    return xs[0][int(params.get("index", 0))].contiguous()
