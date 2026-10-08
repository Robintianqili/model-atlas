"""tiny-unet-seg: export the trained tiny U-Net to site/static/models/tiny-unet-seg/ (kind A, design §4.11), and the same
network trained without skip connections to .../noskip/ (the lesson's "upsampling alone" model).

    npm run atlas -- export tiny-unet-seg          # runs this file, then parity --write and the lesson draft
    npm run atlas -- parity tiny-unet-seg --dir models/tiny-unet-seg/noskip --write

describe() names every step of UNet.logits (model.py) with a builder call: the down path (conv, ReLU, conv, ReLU, max
pool, twice), the bottleneck, the up path (the local op upsample = ops_upsample.py, the concat with the skip, conv, ReLU,
twice), the 1x1 head and the softmax over the 4 classes of every pixel (axis 0). Reference cases: two presets and two
edge fixtures, rendered by the page's own scenes.ts (runs/data/cases.bin, CRC-checked). facts.json: the measured numbers
of train_metrics.json and train_metrics_noskip.json (test set and the three sets outside the training range).
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

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ops_upsample  # noqa: E402
from data import case_ids, load, meta  # noqa: E402
from model import C1, C2, C3, CLASSES, ZH, UNet  # noqa: E402

LINES = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
M: list[UNet] = []


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/tiny-unet-seg/model.py", next(i + 1 for i, s in enumerate(LINES) if pattern in s))


def block(b: ak.SpecBuilder, name: str, x, c1, c2, en: str, zh: str, n: int, s: int):
    """conv 3x3, ReLU, conv 3x3, ReLU: one level of the U."""
    h = b.conv2d(f"{name}.conv1", x, c1, label=(f"{en}: conv 3×3 → {n} maps", f"{zh}：3×3 卷积 → {n} 张图"), code=code(f"F.relu(self.{c1name(name)}a"))
    h = b.relu(f"{name}.relu1", h, label=(f"{en}: ReLU", f"{zh}：ReLU"))
    h = b.conv2d(f"{name}.conv2", h, c2, label=(f"{en}: second conv 3×3", f"{zh}：第二个 3×3 卷积"), code=code(f"self.{c1name(name)}b("))
    return b.relu(name, h, label=(f"{en}: {n} maps of {s}×{s}", f"{zh}：{n} 张 {s}×{s} 的图"))


def c1name(name: str) -> str:
    return {"enc1": "enc1", "enc2": "enc2", "bott": "bot", "dec2": "up2", "dec1": "up1"}[name]


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    m = M[0]
    x = b.input("input", case.input, adapter="shapes32", kind="drawing", axes="c,y,x", label=("Your picture", "你的图"),
                desc=("32×32 grey numbers: dark noisy background, brighter shapes.", "32×32 个灰度数：带噪声的暗背景，更亮的形状。"))
    e1 = block(b, "enc1", x, m.enc1a, m.enc1b, "Down 1", "下行第 1 层", C1, 32)
    h = b.maxpool2d("pool1", e1, 2, label=("Max pool 2×2 → 16×16", "2×2 最大池化 → 16×16"), code=code("F.max_pool2d(e1"))
    e2 = block(b, "enc2", h, m.enc2a, m.enc2b, "Down 2", "下行第 2 层", C2, 16)
    h = b.maxpool2d("pool2", e2, 2, label=("Max pool 2×2 → 8×8", "2×2 最大池化 → 8×8"), code=code("F.max_pool2d(e2"))
    bt = block(b, "bott", h, m.bota, m.botb, "Bottleneck", "瓶颈", C3, 8)
    u = b.op("upsample", "up2", [bt], params={"scale": 2}, ref=ops_upsample.reference, axes="c,y,x",
             label=("Upsample ×2 → 16×16", "上采样 ×2 → 16×16"), code=code("u2 = self.up(b)"))
    if m.skip:
        u = b.concat("cat2", [e2, u], axis=0, axes="c,y,x", label=(f"Skip 2 + up: {C2} + {C3} maps", f"跳跃连接 2 + 上采样：{C2} + {C3} 张图"),
                     code=code("u2 = torch.cat"))
    d2 = block(b, "dec2", u, m.up2a, m.up2b, "Up 2", "上行第 2 层", C2, 16)
    u = b.op("upsample", "up1", [d2], params={"scale": 2}, ref=ops_upsample.reference, axes="c,y,x",
             label=("Upsample ×2 → 32×32", "上采样 ×2 → 32×32"), code=code("u1 = self.up(d2)"))
    if m.skip:
        u = b.concat("cat1", [e1, u], axis=0, axes="c,y,x", label=(f"Skip 1 + up: {C1} + {C2} maps", f"跳跃连接 1 + 上采样：{C1} + {C2} 张图"),
                     code=code("u1 = torch.cat"))
    d1 = block(b, "dec1", u, m.up1a, m.up1b, "Up 1", "上行第 1 层", C1, 32)
    lg = b.conv2d("logits", d1, m.head, axes="class,y,x", label=("Class scores: conv 1×1 → 4 per pixel", "类别分数：1×1 卷积 → 每个像素 4 个"),
                  code=code("return self.head(d1)"))
    b.output(b.softmax("probs", lg, axis=0, axes="class,y,x", label=("Per-pixel softmax: 4 shares per pixel", "逐像素 softmax：每个像素 4 个份额"),
                       code=code("torch.softmax(self.logits(x), dim=1)")))


def load_model(name: str, skip: bool) -> UNet:
    m = UNet(skip=skip)
    m.load_state_dict(torch.load(HERE / name, map_location="cpu"))
    return m.cpu().eval()


def cases(ids: list[str]) -> list[ak.Case]:
    X, _ = load("cases")
    all_ids = case_ids()
    edge = {"blank", "giant", "ring", "cut"}
    return [ak.Case(i, i, X[all_ids.index(i)].clone(), edge=i in edge) for i in ids]


def F(v, unit: str, src: str) -> dict:
    return {"value": v, "unit": unit, "source": src}


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    tn = json.loads((HERE / "train_metrics_noskip.json").read_text(encoding="utf-8"))
    src, srcn = "pipeline/tiny-unet-seg/train.py (train_metrics.json)", "pipeline/tiny-unet-seg/train.py --noskip (train_metrics_noskip.json)"
    t, n = tm["sets"]["test"], tn["sets"]["test"]
    px = meta()["sets"]["train"]["classPixels"]
    facts = {
        "trainN": F(tm["train_n"], "scenes", "pipeline/tiny-unet-seg/make_data.ts"), "testN": F(t["n"], "scenes", src),
        "epochs": F(tm["epochs"], "epochs", src), "params": F(tm["params"], "numbers", "pipeline/tiny-unet-seg/model.py"),
        "paramsNoskip": F(tn["params"], "numbers", "pipeline/tiny-unet-seg/model.py"),
        "bgShare": F(px[0] / sum(px), "share", "pipeline/tiny-unet-seg/make_data.ts (training pixels)"),
        "testMiou": F(t["miou"], "share", src), "testAcc": F(t["acc"], "share", src), "testEdgeAcc": F(t["edgeAcc"], "share", src),
        "testEdgeShare": F(t["edgeShare"], "share", src),
        "noskipMiou": F(n["miou"], "share", srcn), "noskipEdgeAcc": F(n["edgeAcc"], "share", srcn), "noskipAcc": F(n["acc"], "share", srcn),
        **{f"testIou{c.capitalize()}": F(t["iou"][k], "share", src) for k, c in enumerate(CLASSES)},
        "bigMiou": F(tm["sets"]["big"]["miou"], "share", src), "hollowMiou": F(tm["sets"]["hollow"]["miou"], "share", src),
        "darkMiou": F(tm["sets"]["dark"]["miou"], "share", src),
        "hollowShapeIou": F(sum(tm["sets"]["hollow"]["iou"][1:]) / 3, "share", src),
        "oodN": F(tm["sets"]["big"]["n"], "scenes", "pipeline/tiny-unet-seg/make_data.ts (big, hollow and dark sets)"),
    }
    training = {"dataset": f"{tm['train_n']:,} synthetic 32×32 scenes rendered by scenes.ts (circles, squares, triangles; sizes 3-8 px)",
                "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                "optimizer": f"Adam lr {tm['lr']}, cosine decay to 0; loss = cross-entropy of every pixel, averaged",
                "hardware": tm["hardware"],
                "metrics": {"test_miou": round(t["miou"], 4), "test_pixel_acc": round(t["acc"], 4), "test_edge_acc": round(t["edgeAcc"], 4),
                            "noskip_test_miou": round(n["miou"], 4), "test_n": t["n"]}}
    extras = {"classes": [{"id": c, "en": c, "zh": ZH[c]} for c in CLASSES]}
    M[:] = [load_model("model_final.pt", True)]
    print(ak.export_model(describe, cases(["three", "small", "blank", "giant"]), slug="tiny-unet-seg",
                          name=("Tiny U-Net segmentation", "小型 U-Net 分割"), kinds=("trace",),
                          module=M[0], module_input=lambda c: c.input.unsqueeze(0), extras=extras, facts=facts, training=training))
    M[:] = [load_model("model_noskip.pt", False)]
    print(ak.export_model(describe, cases(["three", "blank", "giant"]), slug="tiny-unet-seg",
                          name=("Tiny U-Net without skip connections", "去掉跳跃连接的小型 U-Net"), kinds=("trace",),
                          out=str(ak.site_path("static", "models", "tiny-unet-seg", "noskip")), module=M[0],
                          module_input=lambda c: c.input.unsqueeze(0), extras=extras,
                          note="the same network and training without the two concatenations (train.py --noskip)"))
