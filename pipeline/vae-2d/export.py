"""vae-2d: export the trained 2-D VAE to site/static/models/vae-2d/ (kind A, design §4.11).

    npm run atlas -- export vae-2d          # runs this file, then parity --write and the lesson draft

describe() names every step of VAE.forward(x, eps) (model.py) with a builder call: two encoder layers, the two heads mu
and logvar, the noise eps (a second declared input: the page draws it in the drawing28 adapter, here each case has its
own), the local op reparam (ops_reparam.py: z = mu + exp(0.5 * logvar) * eps, sub-step sigma), the decoder and the local
op to_image (ops_to_image.py: 784 -> [1, 28, 28]) before the sigmoid. export_model() checks it against the module (AK103)
and writes spec.json, weights.bin, reference.json (2 test drawings per class, a large eps, and the blank and one-dot edge
cases), facts.json (stats.py), presets.json (TinyCNN's format: base64 uint8 28x28) and latent.json (the map: mu of
test drawings by class, and of each preset drawing).
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
import ops_reparam  # noqa: E402
import ops_to_image  # noqa: E402
from model import CLASSES, H1, H2, VAE, ZH  # noqa: E402
from stats import SRC, measure  # noqa: E402

HELD_OUT = ["house", "fish"]   # presets of classes the VAE never saw (the limits step)
model = VAE()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
LINES = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
EPS: dict[str, torch.Tensor] = {}


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/vae-2d/model.py", next(i + 1 for i, s in enumerate(LINES) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="drawing28", kind="drawing", axes="c,y,x",
                label=("Your drawing", "你的画"), desc=("28x28 numbers: 0 is empty, 1 is full ink.", "28x28 个数：0 是空白，1 是满墨。"))
    h = b.flatten("flat", x, label=("784 numbers in a row", "排成一行的 784 个数"), code=code("x.flatten(1)"))
    h = b.linear("enc1", h, model.enc1, label=(f"Encoder layer 1 ({H1})", f"编码器第 1 层（{H1}）"), code=code("self.enc1 ="))
    h = b.relu("enc1.relu", h, label=("Encoder ReLU 1", "编码器 ReLU 1"), code=code("F.relu(self.enc1"))
    h = b.linear("enc2", h, model.enc2, label=(f"Encoder layer 2 ({H2})", f"编码器第 2 层（{H2}）"), code=code("self.enc2 ="))
    h = b.relu("enc2.relu", h, label=("Encoder ReLU 2", "编码器 ReLU 2"), code=code("F.relu(self.enc2"))
    mu = b.linear("mu", h, model.mu, label=("μ: the centre (2 numbers)", "μ：中心（2 个数）"), code=code("self.mu ="),
                  desc=("Where the encoder puts your drawing on the 2-D map.", "编码器把你的画放在二维地图上的位置。"))
    lv = b.linear("logvar", h, model.logvar, label=("log σ²: the spread (2 numbers)", "log σ²：散布（2 个数）"), code=code("self.logvar ="),
                  desc=("How unsure the encoder is, as the log of the variance σ².", "编码器有多不确定，用方差 σ² 的对数表示。"))
    e = b.input("eps", EPS[case.id], adapter="drawing28", kind="noise", axes="f",
                label=("Noise ε (2 numbers)", "噪声 ε（2 个数）"), desc=("Drawn from a standard bell curve N(0, 1).", "从标准钟形曲线 N(0, 1) 中抽取。"))
    z = b.op("reparam", "z", [mu, lv, e], ref=ops_reparam.reference, axes="f",
             label=("z = μ + σ·ε: one point of the cloud", "z = μ + σ·ε：云中的一个点"), code=code("z = mu + torch.exp"))
    h = b.linear("dec1", z, model.dec1, label=(f"Decoder layer 1 ({H2})", f"解码器第 1 层（{H2}）"), code=code("self.dec1 ="))
    h = b.relu("dec1.relu", h, label=("Decoder ReLU 1", "解码器 ReLU 1"), code=code("F.relu(self.dec1"))
    h = b.linear("dec2", h, model.dec2, label=(f"Decoder layer 2 ({H1})", f"解码器第 2 层（{H1}）"), code=code("self.dec2 ="))
    h = b.relu("dec2.relu", h, label=("Decoder ReLU 2", "解码器 ReLU 2"), code=code("F.relu(self.dec2"))
    h = b.linear("dec3", h, model.dec3, label=("784 pixel scores", "784 个像素分数"), code=code("self.dec3 ="))
    h = b.op("to_image", "img", [h], ref=ops_to_image.reference, axes="c,y,x",
             label=("Scores as a 28×28 picture", "排成 28×28 的分数图"), code=code(".view(-1, 1, 28, 28)"))
    b.output(b.sigmoid("out", h, axes="c,y,x", label=("The redrawing", "重画出的画"), code=code("torch.sigmoid(self.decode_logits")))


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def cases(X: torch.Tensor, picks: list[list[int]]) -> list[ak.Case]:
    g = torch.Generator().manual_seed(1234)
    out = []
    for k, name in enumerate(CLASSES):
        for j in picks[k][:2]:
            c = ak.Case(f"{name}-{j}", name, X[j].reshape(1, 28, 28).clone())
            EPS[c.id] = torch.randn(2, generator=g)
            out.append(c)
    big = ak.Case(f"{CLASSES[0]}-{picks[0][0]}-big-eps", f"{CLASSES[0]}, large noise", X[picks[0][0]].reshape(1, 28, 28).clone(), edge=True)
    EPS[big.id] = torch.tensor([3.0, -3.0])
    dot = torch.zeros(1, 28, 28)
    dot[0, 14, 14] = 1.0
    blank = ak.Case("blank", "blank", torch.zeros(1, 28, 28), edge=True)
    one = ak.Case("dot", "one dot", dot, edge=True)
    EPS[blank.id] = torch.zeros(2)
    EPS[one.id] = torch.randn(2, generator=g)
    return out + [big, blank, one]


class Module(torch.nn.Module):
    """model(x, eps) for AK103: the same eps as the case."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.eps = None

    def forward(self, x):
        return self.m(x, self.eps)


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    facts_raw, latent, picks, (X, y) = measure(model)
    facts = {k: {"value": v, "unit": u, "source": SRC} for k, (v, u) in facts_raw.items()}
    for k in ("z16Dim", "z16Rec", "z16Kl", "z16GreyOut"):
        if k in facts:
            facts[k]["source"] = "pipeline/vae-2d/stats.py on model_z16.pt (train.py --zdim 16; not shipped)"
    facts["params"] = {"value": sum(p.numel() for p in model.parameters()), "unit": "numbers", "source": "pipeline/vae-2d/model.py"}
    facts["epochs"] = {"value": tm["epochs"], "unit": "epochs", "source": "pipeline/vae-2d/train.py"}
    facts["trainN"] = {"value": 12000 * len(CLASSES), "unit": "drawings", "source": "ak.data.quickdraw (TinyCNN split)"}
    Xu8 = (X * 255).round().to(torch.uint8).numpy().reshape(-1, 28, 28)
    presets = {"note": "Quick, Draw! test drawings (CC BY 4.0, Google Creative Lab); b64 = 784 uint8, white on black. "
                       "Per class: 6 typical drawings the model redraws well (stats.py picks); heldOut: classes it never saw.",
               "classes": [], "heldOut": []}
    with torch.no_grad():
        pmu = {}
        for k, name in enumerate(CLASSES):
            presets["classes"].append({"id": name, "en": name, "zh": ZH[name],
                                       "items": [{"b64": b64(Xu8[j]), "fileIndex": int(j)} for j in picks[k]]})
            pmu[name] = [round(float(v), 4) for v in model.encode(X[picks[k][0]].reshape(1, 1, 28, 28))[0][0]]
        Xh, yh, ih = ak.data.quickdraw(HELD_OUT, "test")
        for k, name in enumerate(HELD_OUT):
            j = int(np.where(yh == k)[0][0])
            presets["heldOut"].append({"id": name, "en": name, "zh": ZH[name], "items": [{"b64": b64(Xh[j]), "fileIndex": int(ih[j])}]})
            pmu[f"heldout-{name}"] = [round(float(v), 4) for v in model.encode(torch.tensor(Xh[j], dtype=torch.float32).reshape(1, 1, 28, 28) / 255)[0][0]]
    latent["presets"] = pmu
    cs = cases(X, picks)
    mod = Module(model)

    def module_input(c):
        mod.eps = EPS[c.id].reshape(1, 2)
        return c.input.unsqueeze(0)

    report = ak.export_model(
        describe, cs, slug="vae-2d", name=("VAE with a 2D latent space", "二维隐空间 VAE"), kinds=("trace",),
        module=mod, module_input=module_input,
        extras={"kinds": [{"id": c, "en": c, "zh": ZH[c]} for c in CLASSES], "heldOutKinds": HELD_OUT},
        facts=facts, presets=presets,
        training={"dataset": f"Quick, Draw! numpy bitmaps, {len(CLASSES)} classes ({', '.join(CLASSES)}), TinyCNN's split",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"Adam lr {tm['lr']}, cosine decay to 0; loss = BCE summed over 784 pixels + KL, mean over the batch",
                  "hardware": tm["hardware"],
                  # flat scalars: the page's Training section lists scalar metrics only
                  "metrics": {"test_bce": round(tm["test"]["rec_sampled"], 2), "test_bce_at_mu": round(tm["test"]["rec_mu"], 2),
                              "test_kl": round(tm["test"]["kl"], 3), "test_loss": round(tm["test"]["loss"], 2), "test_n": tm["test_n"]}},
    )
    ak.site_path("static", "models", "vae-2d", "latent.json").write_text(json.dumps(latent, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(report)
