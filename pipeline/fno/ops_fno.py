"""Python mirrors of the fno local ops (site/src/lib/explainers/fno/ops/fourier.ts); the same arithmetic as model.py.

    rfft          v [n, c] -> [n/2 + 1, c, 2]: torch.fft.rfft(norm="forward") as (real, imaginary) pairs
    keep_modes    X [m, c, 2], keep [1] -> [modes, c, 2]: the lowest `modes` modes, those at or above keep set to 0
    spectral_mix  X [K, i, 2] -> [K, o, 2]: Y_k = X_k (wr + i wi)_k, per mode a complex matrix over the channels
    irfft         Y [K, c, 2] -> [n, c]: torch.fft.irfft(norm="forward", n)

Used from export.py's describe() as b.op("<op>", id, [...], ref=ops_fno.<op>). No sub-steps.
"""
import torch

OPS = {
    "rfft": {"version": 1, "category": "linear", "attrs": {}, "weights": []},
    "keep_modes": {"version": 1, "category": "shape", "attrs": {"modes": "int"}, "weights": []},
    "spectral_mix": {"version": 1, "category": "linear", "attrs": {}, "weights": [{"slot": "wr"}, {"slot": "wi"}]},
    "irfft": {"version": 1, "category": "linear", "attrs": {"n": "int"}, "weights": []},
}


def rfft(xs, params, weights):
    return torch.view_as_real(torch.fft.rfft(xs[0], dim=0, norm="forward")).contiguous()


def keep_modes(xs, params, weights):
    x, keep = xs
    K = params["modes"]
    cut = int(min(K, x.shape[0], max(0, round(float(keep[0])))))
    out = torch.zeros(K, x.shape[1], 2)
    out[:cut] = x[:cut]
    return out


def spectral_mix(xs, params, weights):
    X = torch.view_as_complex(xs[0].contiguous())
    Y = torch.einsum("ki,kio->ko", X, torch.complex(weights["wr"], weights["wi"]))
    return torch.view_as_real(Y).contiguous()


def irfft(xs, params, weights):
    return torch.fft.irfft(torch.view_as_complex(xs[0].contiguous()), n=params["n"], dim=0, norm="forward")
