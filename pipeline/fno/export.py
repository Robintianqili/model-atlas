"""fno: export the trained Fourier neural operator to site/static/models/fno/ (kind A).

    npm run atlas -- export fno          # runs this file, then parity --write and the lesson draft

describe() names every step of FNO1d.forward(u0, keep) (model.py) with a builder call: the 64 starting heights and the
number of kept modes (two inputs), the lift (1 -> 4 channels at every point), two Fourier layers (rfft, keep_modes,
spectral_mix, irfft: the local ops of ops_fno.py; plus the pointwise linear path, the sum and, in the first layer, GELU)
and the projection to one height per point. export_model() checks it against the module (AK103) and writes spec.json,
weights.bin, reference.json (test pairs of every family, cut to 1, 4 and 8 modes too, a flat line, a curve pinned at the
top and a single spike) and facts.json (measured in train.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ops_fno as ops  # noqa: E402
from model import FAMILIES, MODES, N, WIDTH, FNO1d, n_params  # noqa: E402
from train import load_sets  # noqa: E402

SRC = "pipeline/fno/export.py (PyTorch float32 on the CPU)"
model = FNO1d()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
LINES = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/fno/model.py", next(i + 1 for i, s in enumerate(LINES) if pattern in s))


def fourier(b: ak.SpecBuilder, i: int, v, keep, act: bool):
    f, w = getattr(model, f"f{i}"), getattr(model, f"w{i}")
    en, zh = f"Fourier layer {i + 1}", f"傅里叶层 {i + 1}"
    X = b.op("rfft", f"f{i}.fft", [v], ref=ops.rfft, axes="k,c,xy", label=(f"{en}: FFT", f"{zh}：FFT"), code=code("torch.fft.rfft"))
    X = b.op("keep_modes", f"f{i}.keep", [X, keep], params={"modes": MODES}, ref=ops.keep_modes, axes="k,c,xy",
             label=(f"{en}: keep the lowest {MODES} modes", f"{zh}：保留最低的 {MODES} 个模式"), code=code("[:, : self.modes]"))
    Y = b.op("spectral_mix", f"f{i}.mix", [X], weights={"wr": f.wr, "wi": f.wi}, ref=ops.spectral_mix, axes="k,c,xy",
             label=(f"{en}: complex weights", f"{zh}：复数权重"), code=code("torch.einsum"))
    s = b.op("irfft", f"f{i}.ifft", [Y], params={"n": N}, ref=ops.irfft, axes="x,c",
             label=(f"{en}: inverse FFT", f"{zh}：逆 FFT"), code=code("torch.fft.irfft"))
    p = b.linear(f"w{i}", v, w, axes="x,c", label=(f"{en}: pointwise path", f"{zh}：逐点通路"), code=code(f"self.w{i} ="))
    v = b.add(f"f{i}.sum", s, p, axes="x,c", label=(f"{en}: spectral + pointwise", f"{zh}：频域通路 + 逐点通路"),
              code=code(f"self.f{i}(v, keep) + self.w{i}(v)"))
    if act:
        v = b.gelu(f"f{i}.gelu", v, axes="x,c", label=(f"{en}: GELU", f"{zh}：GELU"), code=code("F.gelu("))
    return v


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    u0, keep = case.input
    x = b.input("input", u0, adapter="series", kind="series", axes="x,c", label=("Your starting curve: 64 heights", "你的起始曲线：64 个高度"),
                desc=("The temperature at 64 points around a ring.", "环上 64 个点的温度。"))
    k = b.input("keep", keep, adapter="series", kind="series", axes="k", label=("Modes kept (slider)", "保留的模式数（滑块）"),
                desc=("How many of the lowest modes each Fourier layer keeps.", "每个傅里叶层保留最低的几个模式。"))
    v = b.linear("lift", x, model.lift, axes="x,c", label=(f"Lift: {WIDTH} channels per point", f"升维：每个点 {WIDTH} 个通道"), code=code("self.lift ="))
    v = fourier(b, 0, v, k, True)
    v = fourier(b, 1, v, k, False)
    b.output(b.linear("proj", v, model.proj, axes="x,c", label=("Prediction: one height per point", "预测：每个点一个高度"), code=code("self.proj =")))


class WithKeep(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inp):
        u0, keep = inp
        return self.m(u0, int(keep.reshape(-1)[0]))


def kk(k: int) -> torch.Tensor:
    return torch.tensor([float(k)])


def cases(D) -> list[ak.Case]:
    U0, fam = D["test"][0], D["test"][2].numpy()
    out = []
    for j, name in enumerate(FAMILIES):
        i = int(np.where(fam == j)[0][0])
        out.append(ak.Case(f"test-{i}-{name}", f"test pair {i} ({name})", (U0[i].clone(), kk(MODES))))
    for k in (1, 4, 8):
        out.append(ak.Case(f"test-0-keep{k}", f"test pair 0 cut to {k} modes", (U0[0].clone(), kk(k)), edge=k == 1))
    out.append(ak.Case("flat", "a flat line", (torch.zeros(N, 1), kk(MODES)), edge=True))
    out.append(ak.Case("top", "pinned at the top of the box", (torch.full((N, 1), 1.5), kk(MODES)), edge=True))
    spike = torch.zeros(N, 1)
    spike[32, 0] = 1.5
    out.append(ak.Case("spike", "a single spike", (spike, kk(MODES)), edge=True))
    return out


def fact(v, unit, source=SRC):
    return {"value": v, "unit": unit, "source": source}


if __name__ == "__main__":
    D = load_sets()
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    T = "pipeline/fno/train.py"
    facts = {
        "params": fact(n_params(model), "numbers", "pipeline/fno/model.py"),
        "width": fact(WIDTH, "channels", "pipeline/fno/model.py"),
        "modes": fact(MODES, "modes", "pipeline/fno/model.py"),
        "trainN": fact(len(D["train"][0]), "pairs", "heat.ts via the TS process data (train.py read them)"),
        "testN": fact(len(D["test"][0]), "pairs", "heat.ts via the TS process data"),
        "trainSteps": fact(tm["steps"], "steps", T),
        "testRel": fact(tm["test_rel"], "mean relative error on the test pairs", T),
        "testMax": fact(tm["test_max"], "mean largest |error| per test curve", T),
        "test3xRel": fact(tm["test3x_rel"], "mean relative error on the test pairs made 3 times taller", T),
    }
    for k in (4, 8):
        facts[f"testMaxKeep{k}"] = fact(tm["test_max_by_keep"][str(k)], f"mean largest |error| per test curve, {k} modes kept", T)
    report = ak.export_model(
        describe, cases(D), slug="fno", name=("Fourier neural operator (1D)", "傅里叶神经算子（一维）"), kinds=("trace", "process"),
        module=WithKeep(model), module_input=lambda c: (c.input[0].unsqueeze(0), c.input[1]),
        facts=facts,
        training={"dataset": f"pairs made in TS (heat.ts): {len(D['train'][0]):,} random starting curves and the solver's curves 50 steps later",
                  "steps": tm["steps"], "batchSize": tm["batch"], "bestStep": tm["best_step"],
                  "optimizer": f"Adam lr {tm['lr']}, cosine decay to 0; loss = mean squared error over the 64 points",
                  "hardware": tm["hardware"],
                  "metrics": {"train_mse": float(f"{tm['train_mse']:.4g}"), "val_mse": float(f"{tm['val_mse']:.4g}"), "test_rel": round(tm["test_rel"], 5)}},
    )
    print(report)
