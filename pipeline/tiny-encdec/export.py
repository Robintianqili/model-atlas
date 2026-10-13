"""tiny-encdec: export the encoder-decoder Transformer to site/static/models/tiny-encdec/ (kind A, design §4.11).

    npm run atlas -- export tiny-encdec

The input is the date's character ids, of symbolic length T (1..32); the cases include T = 1 and T = 32. The graph:
the encoder block in kit ops (embedding, position embedding, LayerNorm, q k v, attention scores, softmax, mix, ...),
then the package's local op `greedy` (site/src/lib/explainers/tiny-encdec/ops/greedy.ts; mirror ops_greedy.py), which
decodes greedily and returns the decoder input [<s>, y_0 .. y_9], then the decoder block in kit ops on that input
(causal self-attention, cross-attention to the encoder's memory, feed-forward), the head and the softmax. Row t of
that one decoder pass equals step t of the greedy decoding (the causal mask); export_model checks the whole graph
against model.GreedyProbs (AK103). Facts come from train_metrics.json (train.py), runs/data (make_data.ts) and, for the
comparison with the GRU, pipeline/tinyseq2seq/train_metrics.json (read only).
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
from model import BOS, EOS, STEPS, T_MAX, EncDec, GreedyProbs  # noqa: E402
from ops_greedy import reference as greedy_ref  # noqa: E402
from train import tensors  # noqa: E402

DATA = json.loads((HERE / "runs" / "data" / "train.json").read_text(encoding="utf-8"))
SRC, TGT = DATA["src_vocab"], DATA["tgt_vocab"]
METRICS = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
GRU = json.loads((HERE.parent / "tinyseq2seq" / "train_metrics.json").read_text(encoding="utf-8"))
M = METRICS["model"]
model = EncDec(len(SRC), len(TGT), d=M["d"], heads=M["heads"], ff=M["ff"], n_enc=M["n_enc"], n_dec=M["n_dec"])
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
H = M["heads"]
SRC_FILE = "pipeline/tiny-encdec/model.py"


def encode(text: str) -> torch.Tensor:
    """As the page's adapter: one id per character (unknown: '?' = 1), the last T_MAX characters."""
    return torch.tensor([SRC.index(c) if c in SRC[2:] else 1 for c in text][-T_MAX:], dtype=torch.int64)


def block_attn(b, p, x_q, x_kv, att, causal, en, zh, cross=False):
    """q, k, v, scores, softmax, mix, o of one attention (kit ops); p = the id prefix. Cross-attention's queries are
    decoder positions and its keys input characters, but the kit names both from one token list (the input's): its
    scores get the axes h,t,tk and its mix k,d, so the zoom views say "position 3" / "index 3" instead of naming a
    decoder position with an input character (notes, Kit issues)."""
    q = b.linear(f"{p}q", x_q, att.q, n_head=H, label=(en + " · queries Q", zh + " · 查询 Q"), axes="t,d")
    k = b.linear(f"{p}k", x_kv, att.k, n_head=H, label=(en + " · keys K", zh + " · 键 K"), axes="t,d")
    v = b.linear(f"{p}v", x_kv, att.v, n_head=H, label=(en + " · values V", zh + " · 值 V"), axes="t,d")
    s = b.attention_scores(f"{p}scores", q, k, H, causal=causal, label=(en + " · scores", zh + " · 分数"), axes="h,t,tk" if cross else "h,tq,tk")
    w = b.softmax(f"{p}attn", s, label=(en + " · weights", zh + " · 权重"), axes="h,tq,tk")
    m = b.attention_mix(f"{p}mix", w, v, H, label=(en + " · mix of values", zh + " · 值的混合"), axes="k,d" if cross else "t,d")
    return b.linear(f"{p}o", m, att.o, label=(en + " · output", zh + " · 输出"), axes="t,d")


def ff(b, p, x, f, en, zh):
    u = b.linear(f"{p}up", x, f.up, label=(en + " · feed-forward up", zh + " · 前馈放大"), axes="t,d")
    u = b.relu(f"{p}relu", u, label=(en + " · ReLU", zh + " · ReLU"), axes="t,d")
    return b.linear(f"{p}down", u, f.down, label=(en + " · feed-forward down", zh + " · 前馈缩回"), axes="t,d")


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    ids = b.input("input", case.input, adapter="date-text", kind="text", axes="t", shape=["T"], dtype="i32",
                  dims={"T": (1, T_MAX, ("characters", "字符"))}, label=("Your date", "你的日期"))
    m, E, D = model, model.enc[0], model.dec[0]
    e = b.embedding("src.emb", ids, m.src_emb, label=("Character vectors", "字符向量"), axes="t,d")
    pe = b.position_embedding("src.pos", ids, m.enc_pos, label=("Position vectors", "位置向量"), axes="t,d")
    x = b.add("enc.x0", e, pe, label=("Character + position", "字符 + 位置"), axes="t,d")
    h = b.layernorm("enc.ln1", x, E.ln1, label=("Encoder · LayerNorm 1", "编码器 · 层归一化 1"), axes="t,d")
    x = b.add("enc.add1", x, block_attn(b, "enc.", h, h, E.sa, False, "Encoder self-attention", "编码器自注意力"),
              label=("Encoder · add attention", "编码器 · 加上注意力"), axes="t,d")
    h = b.layernorm("enc.ln2", x, E.ln2, label=("Encoder · LayerNorm 2", "编码器 · 层归一化 2"), axes="t,d")
    x = b.add("enc.add2", x, ff(b, "enc.", h, E.ff, "Encoder", "编码器"), label=("Encoder · add feed-forward", "编码器 · 加上前馈"), axes="t,d")
    mem = b.layernorm("mem", x, m.enc_ln, label=("Encoder output (the memory)", "编码器输出（记忆）"), axes="t,d")
    W = {"emb": m.tgt_emb.weight, "pos": m.dec_pos.weight, "ln1_w": D.ln1.weight, "ln1_b": D.ln1.bias,
         "sq": D.sa.q.weight, "sk": D.sa.k.weight, "sv": D.sa.v.weight, "so": D.sa.o.weight, "ln2_w": D.ln2.weight,
         "ln2_b": D.ln2.bias, "cq": D.ca.q.weight, "ck": D.ca.k.weight, "cv": D.ca.v.weight, "co": D.ca.o.weight,
         "ln3_w": D.ln3.weight, "ln3_b": D.ln3.bias, "up": D.ff.up.weight, "down": D.ff.down.weight,
         "ln_w": m.dec_ln.weight, "ln_b": m.dec_ln.bias, "head": m.head.weight}
    y_in = b.op("greedy", "greedy", [mem], params={"steps": STEPS, "bos": BOS, "nHead": H}, weights=W, ref=greedy_ref,
                axes="t", label=("Greedy decoding: ▶ and the symbols written", "贪心解码：▶ 和写下的符号"))
    y = b.add("dec.x0", b.embedding("tgt.emb", y_in, m.tgt_emb, label=("Symbol vectors", "符号向量"), axes="t,d"),
              b.position_embedding("tgt.pos", y_in, m.dec_pos, label=("Decoder position vectors", "解码器位置向量"), axes="t,d"),
              label=("Symbol + position", "符号 + 位置"), axes="t,d")
    h = b.layernorm("dec.ln1", y, D.ln1, label=("Decoder · LayerNorm 1", "解码器 · 层归一化 1"), axes="t,d")
    y = b.add("dec.add1", y, block_attn(b, "dec.s", h, h, D.sa, True, "Decoder self-attention (causal)", "解码器自注意力（因果）"),
              label=("Decoder · add self-attention", "解码器 · 加上自注意力"), axes="t,d")
    h = b.layernorm("dec.ln2", y, D.ln2, label=("Decoder · LayerNorm 2", "解码器 · 层归一化 2"), axes="t,d")
    y = b.add("dec.add2", y, block_attn(b, "dec.c", h, mem, D.ca, False, "Cross-attention", "交叉注意力", cross=True),
              label=("Decoder · add cross-attention", "解码器 · 加上交叉注意力"), axes="t,d")
    h = b.layernorm("dec.ln3", y, D.ln3, label=("Decoder · LayerNorm 3", "解码器 · 层归一化 3"), axes="t,d")
    y = b.add("dec.add3", y, ff(b, "dec.", h, D.ff, "Decoder", "解码器"), label=("Decoder · add feed-forward", "解码器 · 加上前馈"), axes="t,d")
    h = b.layernorm("dec.ln", y, m.dec_ln, label=("Decoder · final LayerNorm", "解码器 · 最后的层归一化"), axes="t,d")
    z = b.linear("logits", h, m.head, label=("A score for every symbol", "每个符号的分数"), axes="t,d")
    # axes t,d (not t,vocab): the answer is the whole row of symbols, not one item of the last row (as tinyseq2seq)
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


NAMED = {"long", "short", "dmy", "dmyShort", "weekday", "weekdayShort"}


@torch.no_grad()
def looks() -> dict:
    """Where cross-attention looks on the 5,000 test dates (the decoder input is the model's own greedy output; both
    heads averaged, the largest weight): the row of the month's last digit (row 6) on dates that spell the month out,
    and the rows of the year's third and fourth digits (rows 2 and 3)."""
    pairs = json.loads((HERE / "runs" / "data" / "test.json").read_text(encoding="utf-8"))["pairs"]
    src, key_ok, _, _ = tensors(pairs, SRC, TGT, "cpu")
    mem, _ = model.encode(src, key_ok)
    _, y_in = model.greedy(src, key_ok)
    top = model.decode(y_in, mem, key_ok)[2][-1].mean(1).argmax(-1)          # [B, L]
    named = [(b, p) for b, p in enumerate(pairs) if p[2] in NAMED]
    gap = sum(not p[0][top[b, 6]].isalnum() for b, p in named)
    letter = sum(p[0][top[b, 6]].isalpha() for b, p in named)
    yp = [re.search(r"\d{4}", p[0]).start() for p in pairs]
    copy = [sum(int(top[b, k]) == yp[b] + k for b in range(len(pairs))) for k in (2, 3)]
    out = {"namedN": len(named), "monthGapN": gap, "monthLetterN": letter, "year2CopyN": copy[0], "year3CopyN": copy[1]}
    print("looks on the test dates:", out)
    return out


def facts() -> dict:
    tm, gm = "pipeline/tiny-encdec/train.py (train_metrics.json)", "pipeline/tinyseq2seq/train.py (its train_metrics.json)"
    off, goff, g, yl = M["off_formats"], GRU["attn"]["off_formats"], GRU["attn"], M["year_look"]["sum"]
    return {
        "testExact": fact(M["test_exact"], tm, "share"), "testN": fact(M["test_n"], tm, "dates"),
        "testUnseenExact": fact(M["test_unseen_exact"], tm, "share"), "testUnseenN": fact(M["test_unseen_n"], tm, "dates"),
        "trainPairs": fact(DATA["n"], "pipeline/tiny-encdec/make_data.ts", "pairs"), "trainSteps": fact(M["steps"], tm, "steps"),
        "params": fact(M["params"], tm), "d": fact(M["d"], "model.py"), "heads": fact(H, "model.py"), "ff": fact(M["ff"], "model.py"),
        "val500": fact(dict(M["val_curve"])[500], tm, "share"), "gruVal500": fact(dict(g["val_curve"])[500], gm, "share"),
        "offYearFirst": fact(off["yearFirst"], tm, "share"), "offDashes": fact(off["dashes"], tm, "share"),
        "offNoComma": fact(off["noComma"], tm, "share"), "offSpelled": fact(off["spelledDay"], tm, "share"),
        "offImpossible": fact(off["impossible"], tm, "share"), "offN": fact(1000, "pipeline/tiny-encdec/make_data.ts", "dates"),
        "gruParams": fact(g["params"], gm), "gruTestExact": fact(g["test_exact"], gm, "share"),
        "gruOffYearFirst": fact(goff["yearFirst"], gm, "share"), "gruOffDashes": fact(goff["dashes"], gm, "share"),
        "gruOffNoComma": fact(goff["noComma"], gm, "share"), "gruOffSpelled": fact(goff["spelledDay"], gm, "share"),
        "gruOffImpossible": fact(goff["impossible"], gm, "share"),
        "yearInside": fact(yl["year"], tm + " year_look (both heads added)", "share"),
        **{k: fact(v, "export.py looks() on the test dates (both heads averaged)", "dates") for k, v in looks().items()},
        "dataSha": fact(DATA["sha256"], "pipeline/tiny-encdec/make_data.ts (sha256 of the training pairs)"),
    }


if __name__ == "__main__":
    report = ak.export_model(describe, CASES, slug="tiny-encdec", name=("Tiny encoder–decoder Transformer", "小型编码器–解码器 Transformer"),
                             kinds=("trace", "process"), module=GreedyProbs(model), module_input=lambda c: c.input,
                             facts=facts(), presets=PRESETS,
                             training={"dataset": "synthetic dates (make_data.ts, the same pairs as tinyseq2seq)", "steps": M["steps"],
                                       "metrics": {"Training pairs": f"{DATA['n']:,}", "Exact dates, test (5,000)": f"{M['test_exact']:.1%}",
                                                   "Exact dates, a format it never saw (year first)": f"{M['off_formats']['yearFirst']:.1%}"}},
                             extras={"alphabet": SRC, "unk": 1, "symbols": TGT, "steps": STEPS, "bos": BOS, "eos": EOS})
    print(report)
