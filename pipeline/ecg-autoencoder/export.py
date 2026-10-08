"""ecg-autoencoder: export the trained autoencoder to site/static/models/ecg-autoencoder/ (kind A, design §4.11).

    npm run atlas -- export ecg-autoencoder          # runs this file, then parity --write and the lesson draft

describe() names every step of AE.forward(x) (model.py) with a builder call: the 140 heights, the encoder (140 -> 32,
ReLU, -> the 4-number code), the decoder (4 -> 32, ReLU, -> 140). export_model() checks it against the module (AK103)
and writes spec.json, weights.bin, reference.json (test beats of every kind, a flat line and a beat pinned at the top),
facts.json (measured below) and testset.json (the anomaly score, the root of the mean squared error, of every test beat, computed here by PyTorch in float32 on the CPU, with its kind: the
histogram, the ROC sweep and the AUC on the page). The size screen's AUCs come from train_metrics_c<k>_h32.json.
The input widget's beats (kind + seed) are the adapter's PRESETS in TS (manifest input.presetsFromAdapter): no presets file.
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
from model import AE, CODE, H, KINDS, N, n_params  # noqa: E402
from train import load_sets, scores_of  # noqa: E402

SRC = "pipeline/ecg-autoencoder/export.py (PyTorch float32 on the CPU)"
model = AE()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
LINES = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/ecg-autoencoder/model.py", next(i + 1 for i, s in enumerate(LINES) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="series", kind="series", axes="t",
                label=("Your beat: 140 heights", "你的心跳：140 个高度"), desc=("The height of the trace at 140 moments of one heartbeat.", "一次心跳中 140 个时刻的曲线高度。"))
    h = b.linear("enc1", x, model.enc1, label=(f"Encoder layer ({H})", f"编码器层（{H}）"), code=code("self.enc1 ="))
    h = b.relu("enc1.relu", h, label=("Encoder ReLU", "编码器 ReLU"), code=code("F.relu(self.enc1"))
    z = b.linear("code", h, model.code, label=(f"The code ({CODE} numbers)", f"编码（{CODE} 个数）"), code=code("self.code ="),
                 desc=("The bottleneck: all the decoder gets to know about the beat.", "瓶颈：解码器能知道的关于这次心跳的全部信息。"))
    h = b.linear("dec1", z, model.dec1, label=(f"Decoder layer ({H})", f"解码器层（{H}）"), code=code("self.dec1 ="))
    h = b.relu("dec1.relu", h, label=("Decoder ReLU", "解码器 ReLU"), code=code("F.relu(self.dec1"))
    b.output(b.linear("out", h, model.out, axes="t", label=("The rebuilt beat: 140 heights", "重建的心跳：140 个高度"), code=code("self.out =")))


def cases(D) -> list[ak.Case]:
    X, k = D["test"][0], D["test"][1].numpy()
    out = []
    for j, name in enumerate(KINDS):
        for i in np.where(k == j)[0][: (2 if j == 0 else 1)]:
            out.append(ak.Case(f"test-{int(i)}-{name}", f"test beat {int(i)} ({name})", X[int(i)].clone()))
    out.append(ak.Case("flat", "a flat line", torch.zeros(N), edge=True))
    out.append(ak.Case("top", "pinned at the top of the box", torch.full((N,), 1.4), edge=True))
    return out


def fact(v, unit, source=SRC):
    return {"value": v, "unit": unit, "source": source}


if __name__ == "__main__":
    D = load_sets()
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    st, sk = scores_of(model, D["test"][0]), D["test"][1].numpy()
    sv = scores_of(model, D["val"][0])[D["val"][1].numpy() == 0]
    normal, odd = st[sk == 0], st[sk > 0]
    thr = float(np.quantile(sv, 0.95))
    facts = {
        "params": fact(n_params(model), "numbers", "pipeline/ecg-autoencoder/model.py"),
        "code": fact(CODE, "numbers", "pipeline/ecg-autoencoder/model.py"),
        "trainN": fact(len(D["train"][0]), "beats", "beats.ts via the TS process data (train.py read them)"),
        "valNormalN": fact(int(len(sv)), "beats", "beats.ts via the TS process data"),
        "epochs": fact(tm["epochs"], "epochs", "pipeline/ecg-autoencoder/train.py"),
        "trainScore": fact(float(np.sqrt(tm["train_mse"])), "root of the mean squared error over the training beats", "pipeline/ecg-autoencoder/train.py"),
        "medianNormal": fact(float(np.median(normal)), "score", SRC),
        "medianOdd": fact(float(np.median(odd)), "score", SRC),
        "thr95": fact(thr, "score: 95th percentile of the normal validation beats' scores", SRC),
        "testAuc": fact(tm["test_auc"], "area under the ROC curve", "pipeline/ecg-autoencoder/train.py"),
        **{f"auc_{k}": fact(v, "area under the ROC curve, this kind against the normal test beats", "pipeline/ecg-autoencoder/train.py")
           for k, v in tm["test_auc_kind"].items()},
    }
    for c in (2, 8, 16):
        m = json.loads((HERE / f"train_metrics_c{c}_h{H}.json").read_text(encoding="utf-8"))
        src = f"pipeline/ecg-autoencoder/train.py --code {c} (not shipped)"
        facts[f"c{c}Auc"] = fact(m["test_auc"], "area under the ROC curve", src)
        facts[f"c{c}TrainScore"] = fact(float(np.sqrt(m["train_mse"])), "root of the mean squared error over the training beats", src)
        facts[f"c{c}AucNoP"] = fact(m["test_auc_kind"]["noP"], "area under the ROC curve", src)
    testset = {"note": "Anomaly score (root of the mean of the 140 squared errors) of every test beat, PyTorch float32 on the CPU; kinds index "
                       + ", ".join(KINDS) + "; beat i is beats.ts beat(kind, createRng(2026).fork('test<i>')).",
               "kinds": [int(v) for v in sk], "scores": [float(f"{v:.9g}") for v in st]}
    report = ak.export_model(
        describe, cases(D), slug="ecg-autoencoder", name=("ECG anomaly autoencoder", "心电图异常检测自编码器"), kinds=("trace", "process"),
        module=model, module_input=lambda c: c.input.unsqueeze(0),
        facts=facts,
        training={"dataset": f"synthetic beats made in TS (beats.ts): {len(D['train'][0]):,} normal training beats",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"Adam lr {tm['lr']}, cosine decay to 0; loss = mean squared error over the 140 points",
                  "hardware": tm["hardware"],
                  "metrics": {"train_mse": round(tm["train_mse"], 7), "val_mse": round(tm["val_mse"], 7), "test_auc": round(tm["test_auc"], 4)}},
    )
    ak.site_path("static", "models", "ecg-autoencoder", "testset.json").write_text(json.dumps(testset, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(report)
    print("thr95", thr, "median normal", np.median(normal), "median odd", np.median(odd))
