"""pointnet: a tiny PointNet for 2-D point clouds (kind A): which of five shapes do loose points form?

Input: n points [n, 2], already centred on the middle of the box around them
and scaled so the farthest is at distance 1 (the page's
shapes.ts normalize(); make_data.ts wrote the training clouds with it). Every point goes through the same small MLP on
its own (2 -> 32 -> 32, ReLU after each layer), then one max over the points keeps, for each of the 32 features, the
largest value any point gave: the global feature. A second MLP (32 -> 32 -> 5) turns it into the five class scores.
Order and repeats of the points cannot change a max, so the answer depends only on the set of points.

Module names are the layer ids export.py writes (fc1, fc2, fc3, fc4), so the Code tab and the weight names line up.
The original PointNet (Qi et al., 2017) also learns small transform networks and uses batch normalization and far more
features (1,024); this toy keeps only the shared MLP, the max and the classifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["circle", "square", "triangle", "star", "ring"]
ZH = {"circle": "圆", "square": "正方形", "triangle": "三角形", "star": "五角星", "ring": "圆环"}
H, FEAT, HEAD = 32, 32, 32


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, H)          # shared by every point
        self.fc2 = nn.Linear(H, FEAT)       # shared by every point
        self.fc3 = nn.Linear(FEAT, HEAD)    # the classifier, on the global feature
        self.fc4 = nn.Linear(HEAD, len(CLASSES))

    def point_features(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., n, 2] -> [..., n, FEAT]: the shared MLP, each point on its own."""
        return F.relu(self.fc2(F.relu(self.fc1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, n, 2] -> class scores [B, 5]."""
        g = self.point_features(x).max(dim=-2).values   # the max over the points
        return self.fc4(F.relu(self.fc3(g)))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = PointNet()
    x = torch.randn(3, 50, 2)
    perm = torch.randperm(50)
    print(n_params(m), "params; shuffled: max |diff|", float((m(x) - m(x[:, perm])).abs().max()))
