"""Shared data + model code for the TinyCNN pipeline."""
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "quickdraw")
N_PER_CLASS = 15000
N_TRAIN, N_VAL, N_TEST = 12000, 1500, 1500
SPLIT_SEED = 20261005


def load_class(name):
    a = np.load(os.path.join(DATA, name.replace(" ", "_") + ".npy"))
    assert a.shape == (N_PER_CLASS, 784), a.shape
    return a


def split_indices():
    perm = np.random.default_rng(SPLIT_SEED).permutation(N_PER_CLASS)
    return perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_VAL], perm[N_TRAIN + N_VAL:]


def load_split(classes, which):
    """Return (X uint8 [N,28,28], y int64 [N], idx_in_file [N]) for split 'train'|'val'|'test'."""
    tr, va, te = split_indices()
    sel = {"train": tr, "val": va, "test": te}[which]
    xs, ys, ids = [], [], []
    for k, c in enumerate(classes):
        a = load_class(c)
        xs.append(a[sel]); ys.append(np.full(len(sel), k, np.int64)); ids.append(sel)
    return np.concatenate(xs).reshape(-1, 28, 28), np.concatenate(ys), np.concatenate(ids)


class TinyVGG(nn.Module):
    """conv3x3(no pad) -> ReLU -> conv3x3 -> ReLU -> maxpool2 -> conv3x3 -> ReLU -> conv3x3 -> ReLU -> maxpool2 -> flatten -> dense."""

    def __init__(self, ch=(8, 8, 16, 16), n_classes=10):
        super().__init__()
        c1, c2, c3, c4 = ch
        self.ch = ch
        self.conv1_1 = nn.Conv2d(1, c1, 3)
        self.conv1_2 = nn.Conv2d(c1, c2, 3)
        self.conv2_1 = nn.Conv2d(c2, c3, 3)
        self.conv2_2 = nn.Conv2d(c3, c4, 3)
        self.dense = nn.Linear(c4 * 4 * 4, n_classes)

    def features(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        return x.flatten(1)

    def forward(self, x):
        return self.dense(self.features(x))

    def trace(self, x):
        """Every layer output, keyed by contract layer id (x: [B,1,28,28])."""
        t = {"input": x}
        t["conv1_1"] = self.conv1_1(x); t["relu1_1"] = F.relu(t["conv1_1"])
        t["conv1_2"] = self.conv1_2(t["relu1_1"]); t["relu1_2"] = F.relu(t["conv1_2"])
        t["pool1"] = F.max_pool2d(t["relu1_2"], 2)
        t["conv2_1"] = self.conv2_1(t["pool1"]); t["relu2_1"] = F.relu(t["conv2_1"])
        t["conv2_2"] = self.conv2_2(t["relu2_1"]); t["relu2_2"] = F.relu(t["conv2_2"])
        t["pool2"] = F.max_pool2d(t["relu2_2"], 2)
        t["flatten"] = t["pool2"].flatten(1)
        t["dense"] = self.dense(t["flatten"])
        t["softmax"] = F.softmax(t["dense"], dim=1)
        return t


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def augment(x, gen, shift=0.12, scale=(0.8, 1.1), rot_deg=12.0, thick_p=0.6):
    """x: float [B,1,28,28] in [0,1] on GPU. Random affine + stroke-thickness jitter."""
    B = x.shape[0]
    dev = x.device
    s = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=gen)
    a = torch.empty(B, device=dev).uniform_(-rot_deg, rot_deg, generator=gen) * math.pi / 180
    tx = torch.empty(B, device=dev).uniform_(-shift, shift, generator=gen)
    ty = torch.empty(B, device=dev).uniform_(-shift, shift, generator=gen)
    # affine_grid maps output coords -> input coords; scale s>1 enlarges the drawing => use 1/s
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # thickness jitter: blend toward dilation (thicker) or erosion (thinner)
    u = torch.rand(B, 1, 1, 1, device=dev, generator=gen)
    amt = torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.7
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    thick = u < thick_p / 2
    thin = (u >= thick_p / 2) & (u < thick_p)
    x = torch.where(thick, x + amt * (dil - x), x)
    x = torch.where(thin, x + amt * 0.6 * (ero - x), x)
    # renormalise so the brightest pixel is 1 (as in the dataset and preprocessDrawing)
    mx = x.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return x / mx
