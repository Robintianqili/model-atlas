"""Python mirror of the crnn-ocr local op 'columns' (site/src/lib/explainers/crnn-ocr/ops/columns.ts).

[C, H, W] -> [W, C * H]: row w holds every channel's values in column w, feature c * H + h (model.py columns()).
Values are copied, never computed.
"""
OP = "columns"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Cut into columns", "切成一列一列")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    c, h, w = xs[0].shape
    return xs[0].permute(2, 0, 1).reshape(w, c * h).contiguous()
