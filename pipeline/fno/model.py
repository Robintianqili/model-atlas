"""fno: the PyTorch model (kind A): a tiny Fourier neural operator for the heat equation on a ring of 64 points.

    lift      each of the 64 starting heights -> WIDTH numbers (a pointwise linear layer, nn.Linear(1, WIDTH))
    layer f0  spectral path: FFT -> keep the lowest MODES modes -> multiply by learned complex weights -> inverse FFT;
              pointwise path: nn.Linear(WIDTH, WIDTH) at every point; the two are added, then GELU
    layer f1  the same, without the GELU (as in Li et al.'s FNO, the last Fourier layer has no activation)
    proj      WIDTH numbers per point -> one height (nn.Linear(WIDTH, 1)): the predicted curve after STEPS solver steps

The complex weights are kept as two real tensors per layer, wr and wi [MODES, WIDTH, WIDTH] (real and imaginary parts).
`keep` (optional) cuts the kept modes further at run time: the page's slider. FFTs use norm="forward", so mode 0 is the
mean height (the same convention as heat.ts spectrum()). Submodule names become layer ids and tensor names in spec.json.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N, WIDTH, MODES = 64, 4, 16
FAMILIES = ["waves", "pulses", "bumps", "lines"]


class SpectralConv1d(nn.Module):
    def __init__(self, width: int = WIDTH, modes: int = MODES):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.wr = nn.Parameter(scale * torch.rand(modes, width, width))
        self.wi = nn.Parameter(scale * torch.rand(modes, width, width))

    def forward(self, v: torch.Tensor, keep: int | None = None) -> torch.Tensor:   # v [B, N, C]
        X = torch.fft.rfft(v, dim=1, norm="forward")[:, : self.modes]             # [B, MODES, C] complex
        if keep is not None:
            X = X * (torch.arange(self.modes, device=v.device) < keep)[None, :, None]
        Y = torch.einsum("bki,kio->bko", X, torch.complex(self.wr, self.wi))
        return torch.fft.irfft(Y, n=v.shape[1], dim=1, norm="forward")


class FNO1d(nn.Module):
    def __init__(self, width: int = WIDTH, modes: int = MODES):
        super().__init__()
        self.lift = nn.Linear(1, width)
        self.f0 = SpectralConv1d(width, modes)
        self.w0 = nn.Linear(width, width)
        self.f1 = SpectralConv1d(width, modes)
        self.w1 = nn.Linear(width, width)
        self.proj = nn.Linear(width, 1)

    def forward(self, u0: torch.Tensor, keep: int | None = None) -> torch.Tensor:   # u0 [B, N, 1] -> [B, N, 1]
        v = self.lift(u0)
        v = F.gelu(self.f0(v, keep) + self.w0(v))
        v = self.f1(v, keep) + self.w1(v)
        return self.proj(v)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = FNO1d()
    print(m, n_params(m), "params", m(torch.randn(2, N, 1)).shape)
