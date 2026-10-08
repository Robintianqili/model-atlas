"""ddpm-unet: export the trained noise predictor to site/static/models/ddpm-unet/ (kind A, design §4.11).

    npm run atlas -- export ddpm-unet          # runs this file, then parity --write and the lesson draft

describe() names every step of Noised.forward(x0, eps, t) (model.py) with a builder call: the three declared inputs
(the drawing, the noise eps and the timestep t: the page makes eps and t in its drawing28 adapter, here each case has its
own), the local op q_sample (ops_q_sample.py: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, abar as a weight), the local
op time_embed (ops_time_embed.py) and the U-Net: 1x1 convolutions for the time path, 3x3 convolutions with SiLU, max
pooling, the local op upsample (ops_upsample.py) and the skip concatenations. export_model() checks it against the module
(AK103) and writes spec.json, weights.bin, reference.json (a test drawing of each class at t = 10, 40, 70;
edge cases: a blank pad at t = T, one dot at t = 1), facts.json (train_metrics.json and stats.json) and presets.json (TinyCNN's format).
"""
from __future__ import annotations

import base64
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
import ops_q_sample  # noqa: E402
import ops_time_embed  # noqa: E402
import ops_upsample  # noqa: E402
from model import CH, CLASSES, D, T, ZH, Noised, UNet, schedule  # noqa: E402

unet = UNet(CH)
unet.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
noised = Noised(unet.cpu().eval()).eval()
LINES = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
EXTRA: dict[str, tuple[torch.Tensor, float]] = {}


def conv1x1(lin: torch.nn.Linear) -> torch.nn.Conv2d:
    """the time path's Linear layers as 1x1 convolutions on [E, 1, 1] maps (the same numbers): so the kit's conv2d
    computes them and its add broadcasts the [C, 1, 1] time bias over a [C, H, W] map."""
    c = torch.nn.Conv2d(lin.in_features, lin.out_features, 1)
    with torch.no_grad():
        c.weight.copy_(lin.weight[:, :, None, None])
        c.bias.copy_(lin.bias)
    return c.eval()


TIME = {name: conv1x1(getattr(noised.unet, name)) for name in ("tmlp", "down_t", "mid1_t", "up2_t")}  # made once: tensors keep their names


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/ddpm-unet/model.py", next(i + 1 for i, s in enumerate(LINES) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    eps_v, t_v = EXTRA[case.id]
    u = noised.unet
    x0 = b.input("input", case.input, adapter="drawing28", kind="drawing", axes="c,y,x",
                 label=("Your drawing x₀", "你的画 x₀"), desc=("28x28 numbers: 0 is paper, 1 is full ink.", "28x28 个数：0 是纸，1 是满墨。"))
    eps = b.input("eps", eps_v, adapter="drawing28", kind="noise", axes="c,y,x",
                  label=("Noise ε", "噪声 ε"), desc=("784 numbers from a standard bell curve N(0, 1).", "从标准钟形曲线 N(0, 1) 中抽出的 784 个数。"))
    t = b.input("t", torch.tensor([t_v], dtype=torch.float32), adapter="drawing28", kind="timestep", axes="k",
                label=("Step t", "步数 t"), desc=(f"Which of the {T} noise steps (1 to {T}).", f"{T} 个加噪步中的第几步（1 到 {T}）。"))
    xt = b.op("q_sample", "xt", [x0, eps, t], params={"abar": [float(v) for v in noised.abar]}, ref=ops_q_sample.reference, axes="c,y,x",
              label=("Noisy drawing x_t", "加噪的画 x_t"), code=code("xt = torch.sqrt(ab)"))
    te = b.op("time_embed", "temb", [t], params={"dim": D}, ref=ops_time_embed.reference, axes="c,y,x",
              label=("t as waves", "t 变成波"), code=code("te = F.silu(self.tmlp"))
    te = b.conv2d("tmlp.conv", te, TIME["tmlp"], label=("Time vector", "时间向量"), code=code("self.tmlp ="))
    te = b.silu("tmlp.silu", te, label=("Time vector, SiLU", "时间向量，SiLU"), code=code("te = F.silu(self.tmlp"))

    # layer ids: one prefix per block of the U (in, down, mid, up, top), without digits: the page's overview folds
    # prefixes like 'L1' into groups and calls them residual blocks, which these are not
    def conv(g: str, name: str, x, mod, label, line, tname: str | None = None):
        h = b.conv2d(f"{g}.{name}", x, mod, label=(f"{label[0]} · convolution", f"{label[1]} · 卷积"), code=code(line))
        if tname:
            tb = b.conv2d(f"{g}.time", te, TIME[tname], label=(f"{label[0]} · time bias", f"{label[1]} · 时间偏置"), code=code(f"self.{tname} ="))
            h = b.add(f"{g}.add", h, tb, label=(f"{label[0]} · + time bias", f"{label[1]} · 加上时间偏置"), code=code(line))
        return h

    def silu(g: str, name: str, h, label, line):
        return b.silu(f"{g}.{name}", h, label=(f"{label[0]} · SiLU", f"{label[1]} · SiLU"), code=code(line))

    L28, D14, M7, U14, U28 = ("In (28×28)", "输入（28×28）"), ("Down (14×14)", "下行（14×14）"), ("Middle (7×7)", "中间（7×7）"), ("Up (14×14)", "上行（14×14）"), ("Up (28×28)", "上行（28×28）")
    a = silu("in", "silu", conv("in", "conv", xt, u.inc, L28, "self.inc ="), L28, "a = F.silu(self.inc")
    h = b.maxpool2d("down.pool", a, label=(f"{D14[0]} · max pool", f"{D14[1]} · 最大池化"), code=code("F.max_pool2d(a, 2)"))
    bb = silu("down", "silu", conv("down", "conv", h, u.down, D14, "b = F.silu(add_time(self.down", "down_t"), D14, "b = F.silu(add_time(self.down")
    h = b.maxpool2d("mid.pool", bb, label=(f"{M7[0]} · max pool", f"{M7[1]} · 最大池化"), code=code("F.max_pool2d(b, 2)"))
    h = silu("mid", "silu1", conv("mid", "conv1", h, u.mid1, M7, "h = F.silu(add_time(self.mid1", "mid1_t"), M7, "h = F.silu(add_time(self.mid1")
    h = silu("mid", "silu2", conv("mid", "conv2", h, u.mid2, M7, "h = F.silu(self.mid2"), M7, "h = F.silu(self.mid2")
    up_line = 'F.interpolate(h, scale_factor=2, mode="nearest"), '
    h = b.op("upsample", "up.up", [h], params={"scale": 2}, ref=ops_upsample.reference, axes="c,y,x",
             label=(f"{U14[0]} · upsample", f"{U14[1]} · 上采样"), code=code(up_line + "b"))
    h = b.concat("up.cat", [h, bb], axis=0, axes="c,y,x", label=(f"{U14[0]} · skip: + the down maps", f"{U14[1]} · 跳跃连接：接上下行的图"), code=code(up_line + "b"))
    h = silu("up", "silu", conv("up", "conv", h, u.up2, U14, "h = F.silu(add_time(self.up2", "up2_t"), U14, "h = F.silu(add_time(self.up2")
    h = b.op("upsample", "top.up", [h], params={"scale": 2}, ref=ops_upsample.reference, axes="c,y,x",
             label=(f"{U28[0]} · upsample", f"{U28[1]} · 上采样"), code=code(up_line + "a"))
    h = b.concat("top.cat", [h, a], axis=0, axes="c,y,x", label=(f"{U28[0]} · skip: + the input maps", f"{U28[1]} · 跳跃连接：接上输入层的图"), code=code(up_line + "a"))
    h = silu("top", "silu", conv("top", "conv", h, u.up1, U28, "h = F.silu(self.up1"), U28, "h = F.silu(self.up1")
    b.output(b.conv2d("out", h, u.out, label=("Predicted noise ε̂", "预测的噪声 ε̂"), code=code("return self.out(h)")))


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def cases(X: np.ndarray, y: np.ndarray) -> list[ak.Case]:
    """one test drawing per class at t = 10, 40, 70; edge cases: a blank pad at t = T and one dot at t = 1 (the extremes
    of t). Every layer of every case goes into reference.json, about 0.85 MB a case, so five cases."""
    g = torch.Generator().manual_seed(1234)
    out = []
    for k, (name, tv) in enumerate(zip(CLASSES, (10, 40, 70))):
        j = int(np.where(y == k)[0][0])
        c = ak.Case(f"{name}-{j}-t{tv}", f"{name}, t = {tv}", torch.tensor(X[j], dtype=torch.float32).reshape(1, 28, 28) / 255)
        EXTRA[c.id] = (torch.randn(1, 28, 28, generator=g), float(tv))
        out.append(c)
    dot = torch.zeros(1, 28, 28)
    dot[0, 14, 14] = 1.0
    for cid, label, x0, tv in ((f"blank-t{T}", f"blank, t = {T}", torch.zeros(1, 28, 28), float(T)), ("dot-t1", "one dot, t = 1", dot, 1.0)):
        c = ak.Case(cid, label, x0, edge=True)
        EXTRA[c.id] = (torch.randn(1, 28, 28, generator=g), tv)
        out.append(c)
    return out


class Module(torch.nn.Module):
    """noised(x0, eps, t) for AK103: the case's eps and t."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.extra = None

    def forward(self, x0):
        eps, t = self.extra
        return self.m(x0, eps.unsqueeze(0), torch.tensor([t]))


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    stats = json.loads((HERE / "stats.json").read_text(encoding="utf-8")) if (HERE / "stats.json").is_file() else {}
    sch = schedule()
    src = "pipeline/ddpm-unet/stats.py (1,000 samples, PyTorch, CPU float32)"
    facts = {
        "params": {"value": sum(p.numel() for p in unet.parameters()), "unit": "numbers", "source": "pipeline/ddpm-unet/model.py"},
        "macs": {"value": tm["macs"], "unit": "multiply-adds per run", "source": "pipeline/ddpm-unet/model.py macs()"},
        "trainN": {"value": 12000 * len(CLASSES), "unit": "drawings", "source": "ak.data.quickdraw (TinyCNN split)"},
        "classesN": {"value": len(CLASSES), "unit": "kinds", "source": "pipeline/ddpm-unet/model.py"},
        "epochs": {"value": tm["epochs"], "unit": "epochs", "source": "pipeline/ddpm-unet/train.py"},
        "trainSteps": {"value": tm["steps"], "unit": "steps", "source": "pipeline/ddpm-unet/train.py"},
        "valLoss": {"value": round(tm["val_loss"], 5), "unit": "mean squared error", "source": "pipeline/ddpm-unet/train.py (validation split, fixed t and eps)"},
        "beta1": {"value": float(sch["beta"][0]), "unit": "", "source": "model.py schedule()"},
        "betaT": {"value": float(sch["beta"][-1]), "unit": "", "source": "model.py schedule()"},
        "abarT": {"value": float(sch["abar"][-1]), "unit": "", "source": "model.py schedule()"},
        **{k: {"value": v, "unit": "", "source": src} for k, v in stats.items()},
    }
    Xt, yt, it = ak.data.quickdraw(CLASSES, "test")
    presets = {"note": "Quick, Draw! test drawings (CC BY 4.0, Google Creative Lab); b64 = 784 uint8, white on black.",
               "classes": [{"id": c, "en": c, "zh": ZH[c],
                            "items": [{"b64": b64(Xt[j]), "fileIndex": int(it[j])} for j in np.where(yt == k)[0][:4]]}
                           for k, c in enumerate(CLASSES)]}
    mod = Module(noised)

    def module_input(c):
        mod.extra = EXTRA[c.id]
        return c.input.unsqueeze(0)

    report = ak.export_model(
        describe, cases(Xt, yt), slug="ddpm-unet", name=("Tiny DDPM with a U-Net", "带 U-Net 的小型 DDPM"), kinds=("trace",),
        module=mod, module_input=module_input,
        extras={"kinds": [{"id": c, "en": c, "zh": ZH[c]} for c in CLASSES], "T": T},
        facts=facts, presets=presets,
        training={"dataset": f"Quick, Draw! numpy bitmaps, {len(CLASSES)} classes ({', '.join(CLASSES)}), TinyCNN's split",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"],
                  "optimizer": f"Adam lr {tm['lr']}, cosine decay to 0; EMA {tm['ema']} of the weights; loss = mean squared error between the predicted and the real noise (t uniform in 1..{T})",
                  "hardware": tm["hardware"],
                  "metrics": {"val_noise_mse": round(tm["val_loss"], 5), "val_n": tm["val_n"]}},
    )
    print(report)
