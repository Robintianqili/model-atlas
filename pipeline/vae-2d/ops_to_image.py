"""Python mirror of the vae-2d local op 'to_image' (site/src/lib/explainers/vae-2d/ops/to_image.ts): [S*S] -> [1, S, S]."""

OP = "to_image"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Lay out as a picture", "排成一幅图")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    n = xs[0].numel()
    s = int(round(n ** 0.5))
    return xs[0].reshape(1, s, s).contiguous()
