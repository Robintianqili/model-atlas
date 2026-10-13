"""vq-vae: export the trained VQ-VAE to site/static/models/vq-vae/ (kind A).

    npm run atlas -- export vq-vae          # runs this file, then parity --write and the lesson draft

describe() names every step of VQVAE.forward(x) (model.py) with a builder call: the patch encoder (a 7x7 convolution
with stride 7, ReLU, a 1x1 convolution: z_e), the local op vq (ops_vq.py: the nearest codebook entry of each patch, with
the sub-steps dist, all squared distances, and idx, the code indices), the decoder (a 3x3 convolution, ReLU, two 1x1
convolutions) and the local op pixel_shuffle (ops_pixel_shuffle.py: the 49 scores of each cell -> its 7x7 block) before
the sigmoid. export_model() checks it against the module (AK103) and writes spec.json, weights.bin, reference.json
(2 test drawings per class, the blank and one-dot edge cases), facts.json (stats.py), presets.json (TinyCNN's format:
base64 uint8 28x28) and usage.json (how often each code is used on the test drawings).
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
import ops_pixel_shuffle  # noqa: E402
import ops_vq  # noqa: E402
from model import BETA, CLASSES, DECAY, D, G, H, K, P, VQVAE, ZH  # noqa: E402
from stats import SRC, load, measure  # noqa: E402

HELD_OUT = ["house", "fish"]   # presets of classes the model never saw
model = load("model_final.pt", K)
LINES = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/vq-vae/model.py", next(i + 1 for i, s in enumerate(LINES) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="drawing28", kind="drawing", axes="c,y,x",
                label=("Your drawing", "你的画"), desc=("28x28 numbers: 0 is empty, 1 is full ink.", "28x28 个数：0 是空白，1 是满墨。"))
    h = b.conv2d("enc1", x, model.enc1, label=(f"Encoder: each {P}×{P} patch → {H} numbers", f"编码器：每个 {P}×{P} 图块 → {H} 个数"), code=code("self.enc1 ="))
    h = b.relu("enc1.relu", h, label=("Encoder ReLU", "编码器 ReLU"), code=code("self.enc2(F.relu"))
    ze = b.conv2d("ze", h, model.enc2, label=(f"z_e: {D} numbers per patch", f"z_e：每个图块 {D} 个数"), code=code("self.enc2 ="),
                  desc=("The encoder's output for each of the 16 patches, before it snaps to a code.", "编码器对 16 个图块各自的输出，还没吸附到码字上。"))
    zq = b.op("vq", "zq", [ze], weights={"codebook": model.codebook}, ref=ops_vq.reference, axes="d,y,x",
              label=("z_q: the nearest code of each patch", "z_q：每个图块最近的码字"), code=code("dist = ((ze[:, None]"),
              desc=(f"Each patch's {D} numbers are replaced by the nearest of the {K} codebook entries.", f"每个图块的 {D} 个数被换成 {K} 个码本条目里最近的那个。"))
    h = b.conv2d("dec1", zq, model.dec1, label=(f"Decoder: each code with its neighbours → {H}", f"解码器：每个码字连同邻居 → {H}"), code=code("self.dec1 ="))
    h = b.relu("dec1.relu", h, label=("Decoder ReLU 1", "解码器 ReLU 1"), code=code("F.relu(self.dec1"))
    h = b.conv2d("dec2", h, model.dec2, label=(f"Decoder layer 2 ({H})", f"解码器第 2 层（{H}）"), code=code("self.dec2 ="))
    h = b.relu("dec2.relu", h, label=("Decoder ReLU 2", "解码器 ReLU 2"), code=code("F.relu(self.dec2"))
    h = b.conv2d("dec3", h, model.dec3, label=(f"{P * P} pixel scores per patch", f"每个图块 {P * P} 个像素分数"), code=code("self.dec3 ="))
    h = b.op("pixel_shuffle", "img", [h], params={"factor": P}, ref=ops_pixel_shuffle.reference, axes="c,y,x",
             label=(f"Scores laid out as {P}×{P} blocks", f"分数排成 {P}×{P} 的块"), code=code("F.pixel_shuffle"))
    b.output(b.sigmoid("out", h, axes="c,y,x", label=("The redrawing", "重画出的画"), code=code("torch.sigmoid(self.decode_logits")))


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


@torch.no_grad()
def gap(x: torch.Tensor) -> float:
    """the smallest gap between the nearest and second-nearest code over the 16 patches (a near tie could flip argmin)"""
    d = model.quantize(model.encode(x.reshape(1, 1, 28, 28)))[2][0].reshape(K, -1).sort(0).values
    return float((d[1] - d[0]).min())


def cases(X: torch.Tensor, picks: list[list[int]]) -> list[ak.Case]:
    out = []
    for k, name in enumerate(CLASSES):
        for j in [j for j in picks[k] if gap(X[j]) > 1e-3][:2]:
            out.append(ak.Case(f"{name}-{j}", name, X[j].reshape(1, 28, 28).clone()))
    dot = torch.zeros(1, 28, 28)
    dot[0, 14, 14] = 1.0
    return out + [ak.Case("blank", "blank", torch.zeros(1, 28, 28), edge=True), ak.Case("dot", "one dot", dot, edge=True)]


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    facts_raw, usage, picks, (X, y) = measure(model)
    facts = {k: {"value": v, "unit": u, "source": SRC} for k, (v, u) in facts_raw.items()}
    for k in facts:
        if k.startswith(("k128", "plainK128")):
            facts[k]["source"] = "pipeline/vq-vae/stats.py on " + ("model_plain_k128.pt (train.py --plain --codes 128" if k.startswith("plain") else "model_k128.pt (train.py --codes 128") + "; not shipped)"
    facts["params"] = {"value": sum(p.numel() for p in model.parameters()), "unit": "numbers", "source": "pipeline/vq-vae/model.py"}
    facts["epochs"] = {"value": tm["epochs"], "unit": "epochs", "source": "pipeline/vq-vae/train.py"}
    facts["trainN"] = {"value": 12000 * len(CLASSES), "unit": "drawings", "source": "ak.data.quickdraw (TinyCNN split)"}
    facts["beta"] = {"value": BETA, "unit": "weight", "source": "pipeline/vq-vae/model.py"}
    facts["emaStep"] = {"value": round(1 - DECAY, 4), "unit": "share", "source": "pipeline/vq-vae/model.py (EMA decay 0.99)"}
    Xu8 = (X * 255).round().to(torch.uint8).numpy().reshape(-1, 28, 28)
    presets = {"note": "Quick, Draw! test drawings (CC BY 4.0, Google Creative Lab); b64 = 784 uint8, white on black. "
                       "Per class: 6 drawings the model redraws well (stats.py picks); heldOut: classes it never saw.",
               "classes": [], "heldOut": []}
    for k, name in enumerate(CLASSES):
        presets["classes"].append({"id": name, "en": name, "zh": ZH[name], "items": [{"b64": b64(Xu8[j]), "fileIndex": int(j)} for j in picks[k]]})
    Xh, yh, ih = ak.data.quickdraw(HELD_OUT, "test")
    for k, name in enumerate(HELD_OUT):
        j = int(np.where(yh == k)[0][0])
        presets["heldOut"].append({"id": name, "en": name, "zh": ZH[name], "items": [{"b64": b64(Xh[j]), "fileIndex": int(ih[j])}]})
    report = ak.export_model(
        describe, cases(X, picks), slug="vq-vae", name=("VQ-VAE: a drawing as 16 codes", "VQ-VAE：一幅画变成 16 个码字"), kinds=("trace",),
        module=model, module_input=lambda c: c.input.unsqueeze(0),
        extras={"kinds": [{"id": c, "en": c, "zh": ZH[c]} for c in CLASSES], "heldOutKinds": HELD_OUT},
        facts=facts, presets=presets,
        training={"dataset": f"Quick, Draw! numpy bitmaps, {len(CLASSES)} classes ({', '.join(CLASSES)}), TinyCNN's split",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"Adam lr {tm['lr']}, cosine decay to 0 (encoder, decoder); codebook by moving average (decay {DECAY}); "
                               f"loss = BCE mean over 784 pixels + {BETA} x commitment, straight-through gradient",
                  "hardware": tm["hardware"],
                  "metrics": {"test_bce_px": round(tm["test"]["rec"], 4), "test_commit": round(tm["test"]["commit"], 4),
                              "codes_used": tm["test"]["used"], "test_n": tm["test_n"]}},
    )
    ak.site_path("static", "models", "vq-vae", "usage.json").write_text(json.dumps(usage, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(report)
