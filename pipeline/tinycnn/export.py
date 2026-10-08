"""Export the trained TinyCNN to site/static/models/tinycnn/ (contract format).
Writes spec.json, weights.bin, reference.json, presets.json, channels.json.
usage: python export.py --ckpt runs/c8e60/best.pt --run runs/c8e60"""
import argparse, base64, json, os, shutil
import numpy as np, torch
from common import TinyVGG, load_split, load_class, split_indices, n_params, N_TRAIN, N_VAL, N_TEST
from classes import BASE, HELD_OUT, ZH

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/c8e60/best.pt")
ap.add_argument("--run", default="runs/c8e60")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "site", "static", "models", "tinycnn")
os.makedirs(OUT, exist_ok=True)
TODAY = "2026-10-05"

run = json.load(open(os.path.join(a.run, "metrics.json")))
ch = tuple(run["ch"])
m = TinyVGG(ch, len(BASE))
m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
m.eval()
if os.path.abspath(a.ckpt) != os.path.abspath(os.path.join(HERE, "model_final.pt")):
    shutil.copy(a.ckpt, os.path.join(HERE, "model_final.pt"))
torch.set_grad_enabled(False)


def f9(x):
    return [float(f"{v:.9g}") for v in np.asarray(x, np.float32).ravel()]


# ---------------- metrics on CPU fp32 (the exported numbers)
Xte, yte, ite = load_split(BASE, "test")
Xva, yva, _ = load_split(BASE, "val")


def logits(X):
    return torch.cat([m(torch.tensor(X[i:i + 4096]).unsqueeze(1).float() / 255) for i in range(0, len(X), 4096)])


Lte = logits(Xte)
Pte = torch.softmax(Lte, 1).numpy()
pred = Pte.argmax(1)
test_acc = float((pred == yte).mean())
val_acc = float((logits(Xva).argmax(1).numpy() == yva).mean())
C = len(BASE)
cm = np.zeros((C, C), int)
for t, p in zip(yte, pred):
    cm[t, p] += 1
per_class = {BASE[i]: round(float(cm[i, i] / cm[i].sum()), 4) for i in range(C)}
print("test", test_acc, "val", val_acc)

# ---------------- weights.bin + tensors
tensors = []
chunks = []
off = 0
for name in ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "dense"]:
    mod = getattr(m, name)
    for part in ["weight", "bias"]:
        t = getattr(mod, part).detach().numpy().astype("<f4")
        tensors.append({"name": f"{name}.{part}", "shape": list(t.shape), "offset": off, "length": int(t.size)})
        chunks.append(t.ravel())
        off += t.size
w = np.concatenate(chunks).astype("<f4")
w.tofile(os.path.join(OUT, "weights.bin"))
assert off == n_params(m)

# ---------------- layers
c1, c2, c3, c4 = ch


def L(id, op, en, zh, inp, shape, params=None, rf=None, den="", dzh=""):
    d = {"id": id, "op": op, "label": {"en": en, "zh": zh}, "in": inp, "outShape": shape}
    if params:
        d["params"] = params
    if rf:
        d["rf"] = {"size": rf[0], "jump": rf[1]}
    if den:
        d["desc"] = {"en": den, "zh": dzh}
    return d


def conv(cin, cout):
    return {"kernel": 3, "stride": 1, "padding": 0, "inChannels": cin, "outChannels": cout}


pool = {"kernel": 2, "stride": 2}
layers = [
    L("input", "input", "Input", "输入", [], [1, 28, 28], rf=(1, 1),
      den="Your drawing as 28x28 numbers: 0 is empty, 1 is full ink.",
      dzh="你的画被缩成 28x28 的数字：0 表示空白，1 表示满墨。"),
    L("conv1_1", "conv2d", "Conv 1-1", "卷积 1-1", ["input"], [c1, 26, 26], conv(1, c1), (3, 1),
      "Each of the %d kernels (3x3) slides over the image; each output is a weighted sum of 9 pixels plus a bias." % c1,
      "%d 个 3x3 卷积核在图上滑动；每个输出是 9 个像素的加权和再加偏置。" % c1),
    L("relu1_1", "relu", "ReLU 1-1", "ReLU 1-1", ["conv1_1"], [c1, 26, 26], rf=(3, 1),
      den="Negative values become 0; positive values pass through.", dzh="负数变成 0，正数保持不变。"),
    L("conv1_2", "conv2d", "Conv 1-2", "卷积 1-2", ["relu1_1"], [c2, 24, 24], conv(c1, c2), (5, 1),
      "Each output looks at a 3x3 window in all %d maps of the previous layer." % c1,
      "每个输出查看上一层全部 %d 张特征图中的 3x3 窗口。" % c1),
    L("relu1_2", "relu", "ReLU 1-2", "ReLU 1-2", ["conv1_2"], [c2, 24, 24], rf=(5, 1),
      den="Negative values become 0.", dzh="负数变成 0。"),
    L("pool1", "maxpool2d", "Max pool 1", "最大池化 1", ["relu1_2"], [c2, 12, 12], pool, (6, 2),
      "Keeps the largest value in each 2x2 block, halving width and height.",
      "在每个 2x2 区块里保留最大值，宽高减半。"),
    L("conv2_1", "conv2d", "Conv 2-1", "卷积 2-1", ["pool1"], [c3, 10, 10], conv(c2, c3), (10, 2),
      "Second block: %d kernels, each 3x3 over %d maps. Cells now see a 10x10 patch of the drawing." % (c3, c2),
      "第二组：%d 个卷积核，每个覆盖 %d 张图的 3x3 窗口。此时每格能看到画中 10x10 的区域。" % (c3, c2)),
    L("relu2_1", "relu", "ReLU 2-1", "ReLU 2-1", ["conv2_1"], [c3, 10, 10], rf=(10, 2),
      den="Negative values become 0.", dzh="负数变成 0。"),
    L("conv2_2", "conv2d", "Conv 2-2", "卷积 2-2", ["relu2_1"], [c4, 8, 8], conv(c3, c4), (14, 2),
      "Cells see a 14x14 patch: parts of shapes such as wheels, roofs or petals.",
      "每格能看到 14x14 的区域：例如车轮、屋顶或花瓣这样的局部形状。"),
    L("relu2_2", "relu", "ReLU 2-2", "ReLU 2-2", ["conv2_2"], [c4, 8, 8], rf=(14, 2),
      den="Negative values become 0.", dzh="负数变成 0。"),
    L("pool2", "maxpool2d", "Max pool 2", "最大池化 2", ["relu2_2"], [c4, 4, 4], pool, (16, 4),
      "2x2 max pooling again: %d maps of 4x4." % c4, "再做一次 2x2 最大池化：得到 %d 张 4x4 的图。" % c4),
    L("flatten", "flatten", "Flatten", "展平", ["pool2"], [c4 * 16],
      den="The %d maps are laid end to end as one list of %d numbers (channel by channel, row by row)." % (c4, c4 * 16),
      dzh="把 %d 张图首尾相接排成一列 %d 个数（逐通道、逐行）。" % (c4, c4 * 16)),
    L("dense", "dense", "Dense", "全连接", ["flatten"], [len(BASE)], {"inFeatures": c4 * 16, "outFeatures": len(BASE)},
      den="One score (logit) per class: a weighted sum of all %d numbers plus a bias." % (c4 * 16),
      dzh="每个类别一个分数（logit）：全部 %d 个数的加权和再加偏置。" % (c4 * 16)),
    L("softmax", "softmax", "Softmax", "Softmax", ["dense"], [len(BASE)],
      den="Turns scores into probabilities that add up to 1: exp(score) / sum of exp(scores).",
      dzh="把分数变成总和为 1 的概率：exp(分数) / 所有 exp(分数) 之和。"),
]

classes = [{"id": c, "en": c, "zh": ZH[c]} for c in BASE]
held = [{"id": c, "en": c, "zh": ZH[c]} for c in HELD_OUT]
spe = (N_TRAIN * len(BASE)) // run["args"]["bs"]
spec = {
    "slug": "tinycnn",
    "name": {"en": "TinyCNN (drawing classifier)", "zh": "TinyCNN（涂鸦分类器）"},
    "version": TODAY,
    "params": int(n_params(m)),
    "precision": "fp32",
    "input": {
        "kind": "drawing", "shape": [1, 28, 28],
        "notes": "White strokes on black, 28x28, value = pixel/255. Framed like Quick, Draw!: the ink box's long side spans 26 px, centred, brightest pixel = 1. Use preprocessDrawing / preprocessStrokes / preprocessPhoto from tinycnn.ts.",
        "fit": 26, "margin": 1, "strokeWidth": 1.6, "suggestedBrushFraction": 0.055,
    },
    "tensors": tensors,
    "layers": layers,
    "classes": classes,
    "heldOutClasses": held,
    "training": {
        "dataset": "Quick, Draw! numpy bitmaps (first 15,000 drawings per class)",
        "datasetUrl": "https://github.com/googlecreativelab/quickdraw-dataset",
        "license": "CC BY 4.0",
        "attribution": "Quick, Draw! dataset by Google Creative Lab",
        "split": {"perClass": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}, "seed": 20261005},
        "epochs": run["epochs"], "bestEpoch": run["best_epoch"], "steps": run["epochs"] * spe,
        "batchSize": run["args"]["bs"],
        "optimizer": f"AdamW lr {run['args']['lr']} one-cycle, weight decay {run['args']['wd']} (weights only)",
        "augmentation": "random shift (up to 12% of half-width), scale 0.8-1.1, rotation +-12 deg, stroke thickness jitter (blend toward 3x3 dilation/erosion), max renormalised to 1",
        "hardware": "RTX 5090 Laptop GPU, PyTorch 2.11",
        "metrics": {
            "test_acc": round(test_acc, 4), "val_acc": round(val_acc, 4),
            "test_acc_augmented": round(run["test_acc_augmented"], 4),
            "test_n": int(len(yte)),
            "per_class_test_acc": per_class,
            "confusion": cm.tolist(),
            "confusion_note": "rows = true class, columns = predicted class, order = classes",
        },
        "history": [{"epoch": h["epoch"], "train_loss": round(h["train_loss"], 4), "val_acc": round(h["val_acc"], 4)}
                    for h in run["history"]],
    },
    "parity": {"against": "pytorch-fp32", "max_abs_diff": None, "checked": None},
}
old = os.path.join(OUT, "spec.json")
if os.path.exists(old):  # keep the drawing-path check written by the node script (re-run it after re-training)
    prev = json.load(open(old, encoding="utf-8"))
    pm = prev.get("training", {}).get("metrics", {})
    if "drawingPath" in pm and prev.get("params") == spec["params"]:
        spec["training"]["metrics"]["drawingPath"] = pm["drawingPath"]
json.dump(spec, open(old, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- reference.json: 3 test inputs, every layer
rng = np.random.default_rng(7)
ref_idx = [int(np.where(yte == k)[0][rng.integers(0, N_TEST)]) for k in [0, 5, 7]]  # apple, house, sun
refs = []
for i in ref_idx:
    x = torch.tensor(Xte[i]).float().reshape(1, 1, 28, 28) / 255
    tr = m.trace(x)
    refs.append({"label": BASE[int(yte[i])], "fileIndex": int(ite[i]), "input": f9(x),
                 "layers": {k: {"shape": list(v.shape[1:]), "data": f9(v[0])} for k, v in tr.items()}})
json.dump({"note": "PyTorch fp32 CPU outputs for every layer id, used by pipeline/parity/tinycnn.parity.ts. Inputs are Quick, Draw! test drawings (pixel/255).",
           "model": "tinycnn", "version": TODAY, "inputs": refs}, open(os.path.join(OUT, "reference.json"), "w"))

# ---------------- presets.json: 6 test drawings for each of the 12 classes
tr_, va_, te_ = split_indices()
presets = {"note": "Quick, Draw! test-split drawings (CC BY 4.0, Quick, Draw! dataset by Google Creative Lab). Each 'b64' is base64 of 784 uint8 (28x28, white on black); model input = value/255. fileIndex = row in the class's numpy_bitmap file. Base-class presets are drawings the model gets right with p > 0.9.",
           "classes": [], "heldOut": []}
prng = np.random.default_rng(11)
for group, names in [("classes", BASE), ("heldOut", HELD_OUT)]:
    for c in names:
        arr = load_class(c)[te_].reshape(-1, 28, 28)
        if c in BASE:
            k = BASE.index(c)
            P = torch.softmax(logits(arr), 1).numpy()
            ok = np.where((P.argmax(1) == k) & (P[:, k] > 0.9))[0]
        else:
            ok = np.arange(len(arr))
        pick = prng.choice(ok, 6, replace=False)
        presets[group].append({"id": c, "en": c, "zh": ZH[c], "items": [
            {"b64": base64.b64encode(arr[j].tobytes()).decode(), "fileIndex": int(te_[j])} for j in pick]})
json.dump(presets, open(os.path.join(OUT, "presets.json"), "w", encoding="utf-8"), ensure_ascii=False)

# ---------------- channels.json: top-6 test images per conv channel (by max pre-ReLU activation)
PAD = 2
X = torch.tensor(Xte).unsqueeze(1).float() / 255
acts = {}
for i in range(0, len(X), 3000):
    tr = m.trace(X[i:i + 3000])
    for k in ["conv1_1", "conv1_2", "conv2_1", "conv2_2"]:
        acts.setdefault(k, []).append(tr[k])
rfs = {"conv1_1": (3, 1), "conv1_2": (5, 1), "conv2_1": (10, 2), "conv2_2": (14, 2)}
chan = {"note": "For each conv channel: the 6 test drawings (10 base classes, 15,000 images) with the highest maximum pre-ReLU activation. 'patch' = base64 uint8 crop of the input around the receptive field 'rf' [y0,x0,y1,x1], padded by 'pad' pixels (zeros outside the image), patchSize x patchSize; the receptive field is the inner rfSize x rfSize square. (y, x) = the cell in the layer's own map. classMeanMax = mean over each class's test drawings of the channel's maximum activation (order = classes).",
        "pad": PAD, "classes": BASE, "layers": {}}
for k, (rs, jump) in rfs.items():
    A = torch.cat(acts[k])  # [N, C, H, W]
    N, Cc, H, W = A.shape
    mx, arg = A.reshape(N, Cc, -1).max(2)  # [N, C]
    P = rs + 2 * PAD
    chs = []
    for c in range(Cc):
        top = torch.argsort(mx[:, c], descending=True)[:6].tolist()
        items = []
        for n in top:
            pos = int(arg[n, c])
            y, x = divmod(pos, W)
            y0, x0 = y * jump, x * jump
            img = np.zeros((28 + 2 * PAD, 28 + 2 * PAD), np.uint8)
            img[PAD:PAD + 28, PAD:PAD + 28] = Xte[n]
            patch = img[y0:y0 + P, x0:x0 + P]
            assert patch.shape == (P, P)
            items.append({"class": int(yte[n]), "fileIndex": int(ite[n]), "act": round(float(mx[n, c]), 4),
                          "y": y, "x": x, "rf": [y0, x0, y0 + rs - 1, x0 + rs - 1],
                          "patch": base64.b64encode(patch.tobytes()).decode()})
        cmm = [round(float(mx[yte == j, c].mean()), 4) for j in range(len(BASE))]
        chs.append({"channel": c, "top": items, "classMeanMax": cmm})
    chan["layers"][k] = {"rfSize": rs, "jump": jump, "patchSize": P, "channels": chs}
json.dump(chan, open(os.path.join(OUT, "channels.json"), "w"))
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)))
