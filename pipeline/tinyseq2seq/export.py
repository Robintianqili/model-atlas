"""tinyseq2seq: export the date converter to site/static/models/tinyseq2seq/ (kind A, design §4.11).

    npm run atlas -- export tinyseq2seq

The input is the date's character ids, of symbolic length T (1..32); the cases include T = 1 and T = 32, so parity
covers both extremes. Two local ops (site/src/lib/explainers/tinyseq2seq/ops/): `gru` (the encoder, sub-steps r z n)
and `attn_decoder` (greedy decoding with Bahdanau attention, 11 steps; sub-steps keys query scores weights context
state next). Their PyTorch mirrors are ops_gru.py and ops_attn_decoder.py; export_model checks the recorded output
against model.GreedyProbs (AK103). Facts come from train_metrics.json (train.py) and runs/data (make_data.ts).
"""
from __future__ import annotations

import json
import re
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
from model import BOS, STEPS, GreedyProbs, Seq2Seq  # noqa: E402
from ops_attn_decoder import reference as dec_ref  # noqa: E402
from ops_gru import reference as gru_ref  # noqa: E402

T_MAX = 32
DATA = json.loads((HERE / "runs" / "data" / "train.json").read_text(encoding="utf-8"))
SRC, TGT = DATA["src_vocab"], DATA["tgt_vocab"]
METRICS = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
model = Seq2Seq(len(SRC), len(TGT))
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
H = model.enc.hidden_size


def encode(text: str) -> torch.Tensor:
    """As the page's adapter: one id per character (unknown: '?' = 1), the last T_MAX characters."""
    return torch.tensor([SRC.index(c) if c in SRC[2:] else 1 for c in text][-T_MAX:], dtype=torch.int64)


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    ids = b.input("input", case.input, adapter="date-text", kind="text", axes="t", shape=["T"],
                  dims={"T": (1, T_MAX, ("characters", "字符"))}, label=("Your date", "你的日期"))
    e = b.embedding("emb", ids, model.src_emb, label=("Character vectors", "字符向量"))
    g = model.enc
    h = b.op("gru", "enc", [e], params={"hidden": H}, ref=gru_ref, axes="t,d", label=("Encoder states", "编码器状态"),
             weights={"weight_ih": g.weight_ih_l0, "weight_hh": g.weight_hh_l0, "bias_ih": g.bias_ih_l0, "bias_hh": g.bias_hh_l0})
    d = model.dec
    z = b.op("attn_decoder", "dec", [h], params={"steps": STEPS, "bos": BOS}, ref=dec_ref, axes="t,d",
             label=("Decoder with attention", "带注意力的解码器"),
             weights={"emb": model.tgt_emb.weight, "wq": model.att_q.weight, "wk": model.att_k.weight, "bk": model.att_k.bias,
                      "v": model.att_v.weight, "weight_ih": d.weight_ih, "weight_hh": d.weight_hh, "bias_ih": d.bias_ih,
                      "bias_hh": d.bias_hh, "wo": model.out.weight, "bo": model.out.bias})
    # axes t,d (not t,vocab): the answer is the whole row of symbols, not one item of the last row (see notes, Kit issues)
    b.output(b.softmax("probs", z, axes="t,d", label=("Chances of each symbol, per step", "每一步各个符号的概率")))


TEXTS = [("long", "March 5, 2021"), ("dmy", "5 March 2021"), ("us", "3/5/2021"), ("eu", "05.03.2021"),
         ("weekday", "Friday, March 5, 2021"), ("impossible", "March 32, 2021"), ("yearFirst", "2021/03/05")]
CASES = [ak.Case("one", "one character (M)", encode("M"), edge=True)]
CASES += [ak.Case(i, t, encode(t)) for i, t in TEXTS]
CASES += [ak.Case("max", "32 characters", encode("Wednesday, September 22nd, 2021!"), edge=True)]
assert CASES[-1].input.shape[0] == T_MAX

PRESETS = {"items": [
    {"id": "long", "en": "March 5, 2021", "zh": "March 5, 2021（月 日, 年）", "text": "March 5, 2021"},
    {"id": "dmy", "en": "20 July 1969", "zh": "20 July 1969（日 月 年）", "text": "20 July 1969"},
    {"id": "us", "en": "12/25/2023 (US)", "zh": "12/25/2023（美式）", "text": "12/25/2023"},
    {"id": "eu", "en": "01.10.1949 (European)", "zh": "01.10.1949（欧式）", "text": "01.10.1949"},
    {"id": "weekday", "en": "Sunday, February 29, 2032", "zh": "Sunday, February 29, 2032（带星期）", "text": "Sunday, February 29, 2032"},
]}


def fact(v, source: str, unit: str | None = None) -> dict:
    return {"value": v, "source": source, **({"unit": unit} if unit else {})}


@torch.no_grad()
def year_look() -> tuple[float, float, int, int]:
    """On the test dates: how often decoder step 0 (the year's first digit) puts its largest weight on a digit of the year,
    and on the year's second digit (the predict question of step 5)."""
    pairs = json.loads((HERE / "runs" / "data" / "test.json").read_text(encoding="utf-8"))["pairs"]
    T = max(len(p[0]) for p in pairs)
    src = torch.zeros(len(pairs), T, dtype=torch.long)
    for i, p in enumerate(pairs):
        src[i, :len(p[0])] = encode(p[0])
    _, att = model.greedy(src, torch.tensor([len(p[0]) for p in pairs]), 1)
    top = att[:, 0].argmax(-1).tolist()
    pos = [re.search(r"\d{4}", p[0]).start() for p in pairs]
    inside = sum(y <= j < y + 4 for j, y in zip(top, pos)) / len(pairs)
    second = sum(j == y + 1 for j, y in zip(top, pos))
    return inside, second / len(pairs), len(pairs), second


def facts() -> dict:
    a, n = METRICS["attn"], METRICS["noattn"]
    a8, n8 = METRICS["attn_h8"], METRICS["noattn_h8"]
    tm = "pipeline/tinyseq2seq/train.py (train_metrics.json)"
    YL = year_look()
    print(f"year look on {YL[2]} test dates: inside the year {YL[0]:.4f}, its second digit {YL[1]:.4f}")
    off = a["off_formats"]
    return {
        "testExact": fact(a["test_exact"], tm, "share"), "testN": fact(a["test_n"], tm, "dates"),
        "testUnseenExact": fact(a["test_unseen_exact"], tm, "share"), "testUnseenN": fact(a["test_unseen_n"], tm, "dates"),
        "trainPairs": fact(DATA["n"], "pipeline/tinyseq2seq/make_data.ts", "pairs"), "trainSteps": fact(a["steps"], tm, "steps"),
        "batch": fact(a["batch"], tm, "pairs"), "hidden": fact(H, "model.py"),
        "noAttnExact": fact(n["test_exact"], tm, "share"),
        "h8": fact(a8["hidden"], tm), "h8AttnExact": fact(a8["test_exact"], tm, "share"), "h8NoAttnExact": fact(n8["test_exact"], tm, "share"),
        "offYearFirst": fact(off["yearFirst"], tm, "share"), "offDashes": fact(off["dashes"], tm, "share"),
        "offNoComma": fact(off["noComma"], tm, "share"), "offSpelled": fact(off["spelledDay"], tm, "share"),
        "offImpossible": fact(off["impossible"], tm, "share"), "offN": fact(1000, "pipeline/tinyseq2seq/make_data.ts", "dates"),
        "yearInside": fact(YL[0], "export.py year_look()", "share"), "yearSecond": fact(YL[1], "export.py year_look()", "share"),
        "yearSecondN": fact(YL[3], "export.py year_look()", "dates"),
        "dataSha": fact(DATA["sha256"], "pipeline/tinyseq2seq/make_data.ts (sha256 of the training pairs)"),
    }


if __name__ == "__main__":
    a = METRICS["attn"]
    report = ak.export_model(describe, CASES, slug="tinyseq2seq", name=("Tiny seq2seq with attention", "带注意力的小型 seq2seq"),
                             kinds=("trace", "process"), module=GreedyProbs(model), module_input=lambda c: c.input,
                             facts=facts(), presets=PRESETS,
                             training={"dataset": "synthetic dates (make_data.ts)", "steps": a["steps"],
                                       "metrics": {"Training pairs": f"{DATA['n']:,}", "Exact dates, test (5,000)": f"{a['test_exact']:.1%}",
                                                   "Exact dates, a format it never saw (year first)": f"{a['off_formats']['yearFirst']:.1%}"}},
                             extras={"alphabet": SRC, "unk": 1, "symbols": TGT, "steps": STEPS, "bos": BOS, "eos": 1})
    print(report)
