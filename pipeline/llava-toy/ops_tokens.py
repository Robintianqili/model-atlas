"""Python mirrors of the llava-toy local ops (site/src/lib/explainers/llava-toy/ops/tokens.ts; to_tokens and cls_token
are copied from pipeline/tinyvit/ops_tokens.py).

    to_tokens  [C, H, W] -> [H*W, C]        x.flatten(1).transpose(0, 1)
    cls_token  [T, D] + c [D] -> [T+1, D]   torch.cat([c[None], x], 0)
    drop_cls   [T, D] -> [T-1, D]           x[1:]
    merge_image  img [N, D], words [T, D] -> [T, D]: rows 0..N-1 from img (the <image> placeholders), the rest from words

Used from export.py's describe() as b.op("<op>", id, [x], ..., ref=<module>.<op>). Each entry of OPS mirrors the TS
OpDef metadata (version, category, label, attrs, weights), as the core mirrors in atlas_kit/ops do.
"""
import torch

OPS = {
    "to_tokens": {"version": 1, "category": "shape", "label": ("Patches become rows", "图块变成行"), "attrs": {}, "weights": []},
    "cls_token": {"version": 1, "category": "embedding", "label": ("Put the [CLS] row in front", "在前面放上 [CLS] 行"),
                  "attrs": {}, "weights": [{"slot": "weight"}]},
    "drop_cls": {"version": 1, "category": "shape", "label": ("Drop the [CLS] row", "去掉 [CLS] 行"), "attrs": {}, "weights": []},
    "merge_image": {"version": 1, "category": "merge", "label": ("Image tokens into the placeholders", "图像词元放进占位行"), "attrs": {}, "weights": []},
}


def to_tokens(xs, params, weights):
    return xs[0].flatten(1).transpose(0, 1).contiguous()


def cls_token(xs, params, weights):
    return torch.cat([weights["weight"].reshape(1, -1), xs[0]], 0)


def drop_cls(xs, params, weights):
    return xs[0][1:].contiguous()


def merge_image(xs, params, weights):
    img, words = xs
    return torch.cat([img, words[img.shape[0]:]], 0).contiguous()
