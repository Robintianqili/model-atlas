"""tiny-unet-seg: a tiny U-Net for per-pixel segmentation of 32x32 synthetic scenes (4 classes).

    down path   enc1: conv 3x3 (1->8) ReLU, conv 3x3 (8->8) ReLU        [8, 32, 32]   (skip 1)
                pool 2x2 -> enc2: conv (8->16) ReLU, conv (16->16) ReLU  [16, 16, 16]  (skip 2)
    bottleneck  pool 2x2 -> conv (16->32) ReLU, conv (32->32) ReLU      [32, 8, 8]
    up path     bilinear x2 -> concat [skip 2, up] -> conv (48->16) ReLU, conv (16->16) ReLU   [16, 16, 16]
                bilinear x2 -> concat [skip 1, up] -> conv (24->8) ReLU, conv (8->8) ReLU      [8, 32, 32]
    head        conv 1x1 (8->4): a score per class and pixel; softmax over the 4 classes of each pixel

skip=False builds the same network without the two concatenations (the up convs read 32 and 16 channels): the lesson's
"upsampling alone" model. Padding 1 keeps every map's size; upsampling is bilinear with align_corners=False.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["background", "circle", "square", "triangle"]
ZH = {"background": "背景", "circle": "圆", "square": "正方形", "triangle": "三角形"}
C1, C2, C3 = 8, 16, 32


def conv(i: int, o: int) -> nn.Conv2d:
    return nn.Conv2d(i, o, 3, padding=1)


class UNet(nn.Module):
    def __init__(self, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.enc1a, self.enc1b = conv(1, C1), conv(C1, C1)
        self.enc2a, self.enc2b = conv(C1, C2), conv(C2, C2)
        self.bota, self.botb = conv(C2, C3), conv(C3, C3)
        self.up2a, self.up2b = conv(C3 + (C2 if skip else 0), C2), conv(C2, C2)
        self.up1a, self.up1b = conv(C2 + (C1 if skip else 0), C1), conv(C1, C1)
        self.head = nn.Conv2d(C1, len(CLASSES), 1)

    @staticmethod
    def up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        e1 = F.relu(self.enc1b(F.relu(self.enc1a(x))))
        e2 = F.relu(self.enc2b(F.relu(self.enc2a(F.max_pool2d(e1, 2)))))
        b = F.relu(self.botb(F.relu(self.bota(F.max_pool2d(e2, 2)))))
        u2 = self.up(b)
        if self.skip:
            u2 = torch.cat([e2, u2], dim=1)
        d2 = F.relu(self.up2b(F.relu(self.up2a(u2))))
        u1 = self.up(d2)
        if self.skip:
            u1 = torch.cat([e1, u1], dim=1)
        d1 = F.relu(self.up1b(F.relu(self.up1a(u1))))
        return self.head(d1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for s in (True, False):
        m = UNet(skip=s)
        y = m(torch.zeros(2, 1, 32, 32))
        print(f"skip={s}: {n_params(m)} params, output {tuple(y.shape)}")
