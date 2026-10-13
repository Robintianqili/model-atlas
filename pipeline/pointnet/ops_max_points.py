"""Python mirror of op 'max_points' (site/src/lib/explainers/pointnet/ops/max_points.ts): [n, F] -> [F], the largest
value of each feature over all n points, as PointNet.forward does it (x.max(dim=-2).values).

Use it from describe(): from ops_max_points import reference as max_points; g = b.op("max_points", "pool", [h], ref=max_points)
"""
OP = "max_points"
VERSION = 1
CATEGORY = "pool"
LABEL = ("Max over the points", "对所有点取最大值")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return xs[0].max(dim=0).values
