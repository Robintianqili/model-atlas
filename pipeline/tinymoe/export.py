"""tinymoe: export the MoE model to site/static/models/tinymoe/ (kind A).

    npm run atlas -- py pipeline/tinymoe/stats.py      # first: measured numbers -> stats.json
    npm run atlas -- export tinymoe                    # this file, then parity --write and the lesson draft

describe() names every step of TinyMoE.forward (model.py): token and position rows added up, two pre-norm causal
blocks (attention, then the MoE layer: the router's 4 scores, their softmax, and the local op 'moe' that keeps the top 2
experts, runs only those and adds their outputs weighted by the gates; pipeline/tinymoe/ops_moe.py), the last
LayerNorm, the next-character scores and their softmax. The input is a sequence of character ids of symbolic length T
(1..64): the cases include T = 1 and T = 64. Writes spec.json, weights.bin, reference.json, facts.json (train_metrics.json
and stats.json) and presets.json (texts for the input box).
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
from model import ALPHABET, CTX, D, DENSE_H, E, H, HEADS, K, LAYERS, UNK, TinyMoE, encode  # noqa: E402
from ops_moe import reference as moe_ref  # noqa: E402

LABELS = [{" ": chr(0x2423), chr(10): chr(0x21B5)}.get(c, c) for c in ALPHABET]
model = TinyMoE()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
# the four experts of each block stacked into the op's weight slots, built once (describe() runs once per case)
STACKED = [{"fc": torch.stack([e.fc.weight for e in blk.mlp.experts]).detach().clone(),
            "fc_b": torch.stack([e.fc.bias for e in blk.mlp.experts]).detach().clone(),
            "out": torch.stack([e.out.weight for e in blk.mlp.experts]).detach().clone(),
            "out_b": torch.stack([e.out.bias for e in blk.mlp.experts]).detach().clone()} for blk in model.blocks]


def code(pattern: str) -> tuple[str, int]:
    """("pipeline/tinymoe/model.py", line) of the first line of model.py that contains `pattern` (the Code tab)."""
    return ("pipeline/tinymoe/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def ids_of(text: str) -> torch.Tensor:
    return torch.tensor(encode(text)[-CTX:], dtype=torch.int64)


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    ids = b.input("input", case.input, adapter="text-presets", kind="text", axes="t", shape=["T"], dtype="i32",
                  dims={"T": (1, CTX, ("characters", "字符"))}, label=("Your text", "你的文字"))
    t = b.embedding("tok", ids, model.tok, label=("Character rows", "字符行"), code=code("self.tok ="), axes="t,d")
    p = b.position_embedding("pos", ids, model.pos, label=("Position rows", "位置行"), code=code("self.pos ="), axes="t,d")
    x = b.add("embed", t, p, label=("Character + position", "字符 + 位置"), code=code("x = self.tok(ids)"), axes="t,d")
    for i, blk in enumerate(model.blocks):
        n, en, zh = f"B{i + 1}", f"Block {i + 1} · ", f"第 {i + 1} 块 · "
        a = b.layernorm(f"{n}.ln1", x, blk.ln1, label=(en + "LayerNorm 1", zh + "层归一化 1"), code=code("h = self.ln1(x)"), axes="t,d")
        q = b.linear(f"{n}.q", a, blk.q, n_head=HEADS, label=(en + "Queries Q", zh + "查询 Q"), code=code("self.q ="), axes="t,d")
        k = b.linear(f"{n}.k", a, blk.k, n_head=HEADS, label=(en + "Keys K", zh + "键 K"), code=code("self.k ="), axes="t,d")
        v = b.linear(f"{n}.v", a, blk.v, n_head=HEADS, label=(en + "Values V", zh + "值 V"), code=code("self.v ="), axes="t,d")
        sc = b.attention_scores(f"{n}.scores", q, k, HEADS, label=(en + "Attention scores", zh + "注意力分数"), code=code("s = (q @ k"), axes="h,tq,tk")
        sc = b.causal_mask(f"{n}.mask", sc, label=(en + "Causal mask", zh + "因果掩码"), code=code("s = s.masked_fill"), axes="h,tq,tk")
        w = b.softmax(f"{n}.attn", sc, label=(en + "Attention weights", zh + "注意力权重"), code=code("a = (s.softmax"), axes="h,tq,tk")
        m = b.attention_mix(f"{n}.mix", w, v, HEADS, label=(en + "Mix of values", zh + "值的混合"), code=code("a = (s.softmax"), axes="t,d")
        o = b.linear(f"{n}.proj", m, blk.proj, label=(en + "Attention output", zh + "注意力输出"), code=code("self.proj ="), axes="t,d")
        x = b.add(f"{n}.add1", x, o, label=(en + "Add attention", zh + "加上注意力"), code=code("x = x + self.proj(a)"), axes="t,d")
        c = b.layernorm(f"{n}.ln2", x, blk.ln2, label=(en + "LayerNorm 2", zh + "层归一化 2"), code=code("self.ln2 ="), axes="t,d")
        r = b.linear(f"{n}.router", c, blk.mlp.router, label=(en + "Router scores", zh + "路由分数"), code=code("self.router ="), axes="t,k",
                     desc=("One score per expert for each character: a linear layer with no bias.", "每个字符对每个专家打一个分：一个没有偏置的线性层。"))
        pr = b.softmax(f"{n}.route", r, label=(en + "Router chances", zh + "路由概率"), code=code("p = scores.softmax"), axes="t,k")
        y = b.op("moe", f"{n}.moe", [c, pr], params={"k": K}, weights=STACKED[i], ref=moe_ref,
                 label=(en + "Top-2 experts", zh + "前 2 个专家"), code=code("class MoE"), axes="t,d",
                 desc=("The two experts with the largest chances run; their outputs are added, weighted by the gates.", "概率最大的两个专家运行；它们的输出按门控加权相加。"))
        x = b.add(f"{n}.add2", x, y, label=(en + "Add experts", zh + "加上专家输出"), code=code("return x + y, p, gates"), axes="t,d")
    h = b.layernorm("ln", x, model.ln, label=("Final LayerNorm", "最后的层归一化"), code=code("self.ln ="), axes="t,d")
    z = b.linear("head", h, model.head, label=("Next-character scores", "下一个字符的分数"), code=code("self.head ="), axes="t,vocab")
    b.output(b.softmax("probs", z, label=("Next-character chances", "下一个字符的概率"), axes="t,vocab"))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return torch.softmax(self.m(ids), dim=-1)


TEXTS = {
    "lily": "Once upon a time, there was a little girl named Lily.",
    "dialog": 'Tom said, "Can we go to the park?" Mom smiled.',
    "unknown": "Zoë ate 3 crêpes (yum)",
}
STORY = ("One day, a big dog ran to the park. He saw a red ball and was very happy. "
         "The dog played all day with his friends.")
CASES = [ak.Case("one", "one character", ids_of("O"), edge=True)]
CASES += [ak.Case(k, v[:24], ids_of(v)) for k, v in TEXTS.items()]
CASES += [ak.Case("max", f"{CTX} characters", ids_of(STORY), edge=True)]
assert len(CASES[-1].input) == CTX

PRESETS = {"texts": [
    {"id": "lily", "en": "A story opening", "zh": "故事开头", "text": "Once upon a time, there was a little girl named Lily."},
    {"id": "dialog", "en": "Someone speaks", "zh": "有人说话", "text": 'Tom said, "Can we go to the park?" Mom smiled.'},
    {"id": "spaces", "en": "Many short words", "zh": "很多短词", "text": "He is a big cat. It is on the mat. We go to it."},
    {"id": "same", "en": "One letter, over and over", "zh": "同一个字母，一遍又一遍", "text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
]}

if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    st = json.loads((HERE / "stats.json").read_text(encoding="utf-8"))
    src_t, src_s = "pipeline/tinymoe/train.py", "pipeline/tinymoe/stats.py (256,000 held-out characters, CPU float32)"
    facts = {
        "valLoss": {"value": tm["moe"]["val_loss"], "unit": "nats per character", "source": src_t + ": 256,000 held-out characters (TinyStoriesV2-GPT4-valid.txt), windows of 64, CPU fp32"},
        "valLossDense": {"value": tm["dense"]["val_loss"], "unit": "nats per character", "source": src_t + " --variant dense"},
        "valLossNoaux": {"value": tm["noaux"]["val_loss"], "unit": "nats per character", "source": src_t + " --variant noaux"},
        "params": {"value": tm["moe"]["params"], "unit": "parameters", "source": "model.py"},
        "active": {"value": tm["moe"]["active"], "unit": "parameters one character uses", "source": "model.py active_params"},
        "denseParams": {"value": tm["dense"]["params"], "unit": "parameters", "source": "model.py (dense twin)"},
        "trainSteps": {"value": tm["moe"]["steps"], "unit": "steps of 256 windows of 64 characters", "source": src_t},
        "trainChars": {"value": tm["moe"]["train_chars"], "unit": "characters", "source": src_t},
        "alpha": {"value": tm["moe"]["alpha"], "unit": "weight of the balancing loss", "source": src_t},
        "vocab": {"value": len(ALPHABET), "unit": "characters in the alphabet", "source": "model.py"},
    }
    for v, tag in (("moe", ""), ("noaux", "Noaux")):
        for bi, s in enumerate(st[v]):
            p = f"b{bi + 1}{tag}"
            facts[p + "Busiest"] = {"value": s["busiest"], "unit": "share of slots", "source": src_s}
            facts[p + "Quietest"] = {"value": s["quietest"], "unit": "share of slots", "source": src_s}
            facts[p + "Balance"] = {"value": s["balance"], "unit": "balancing loss (1 = balanced)", "source": src_s}
            for c in ("1.0", "1.25", "2.0"):
                facts[p + "Drop" + c.replace(".", "_")] = {"value": s["dropped"][c]["slots"], "unit": "share of slots dropped", "source": src_s + f", capacity factor {c}"}
    report = ak.export_model(describe, CASES, slug="tinymoe", name=("Tiny mixture of experts", "小型混合专家模型"), kinds=("trace", "process"),
                             module=WithSoftmax(model), module_input=lambda c: c.input[None],
                             extras={"alphabet": ALPHABET, "unk": UNK, "context": CTX, "labels": LABELS,
                                     "config": {"d": D, "heads": HEADS, "layers": LAYERS, "experts": E, "k": K, "hidden": H, "denseHidden": DENSE_H, "vocab": len(ALPHABET)},
                                     "routing": {v: [{"share": s["share"], "top": s["top"]} for s in st[v]] for v in ("moe", "noaux")}},
                             training={"dataset": f"TinyStories (TinyStoriesV2-GPT4): {tm['moe']['train_chars']:,} characters from the start of the train split", "chars": tm["moe"]["train_chars"],
                                       "steps": tm["moe"]["steps"], "metrics": {"val_loss": tm["moe"]["val_loss"], "val_loss_dense": tm["dense"]["val_loss"], "val_loss_noaux": tm["noaux"]["val_loss"]},
                                       "recipe": tm["moe"]["optimizer"] + f", lr {tm['moe']['lr']}, batch {tm['moe']['batch']} x {tm['moe']['window']} characters, balancing loss x {tm['moe']['alpha']}"},
                             facts=facts, presets=PRESETS)
    print(report)
