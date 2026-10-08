"""tinybert: export the trained TinyBERT to site/static/models/tinybert/ (kind A, design §4.11).

    npm run atlas -- py pipeline/tinybert/stats.py     # first: measured numbers -> stats.json (GPU, ~1 min)
    npm run atlas -- export tinybert                   # this file, then parity --write and the lesson draft

describe() names every step of TinyBERT.forward (model.py): three table lookups (token, position, segment) added up,
four pre-norm blocks without a causal mask, a last LayerNorm and the output head tied to the token table, then softmax.
export_model() checks it against the module (AK103) and writes spec.json, weights.bin, reference.json (two sentences,
T = 1 and T = 64), facts.json and presets.json. This file also copies TinyGPT's tokenizer.json (the same tokenizer; the
input widget encodes typed text with it) and writes the per-token flags and display texts the page needs into spec.json
(tokenFlags, vocab), so the page splits sentences exactly as data.py does.
"""
from __future__ import annotations

import json
import shutil
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
import data  # noqa: E402
from model import CTX, EOT, HEADS, MASK, SPECIAL, VOCAB, TinyBERT, n_params  # noqa: E402

model = TinyBERT()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
TOK = data.tokenizer()


def code(pattern: str) -> tuple[str, int]:
    """("pipeline/tinybert/model.py", line) of the first line of model.py that contains `pattern` (the Code tab)."""
    return ("pipeline/tinybert/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    ids = b.input("ids", case.input[0], adapter="masked-text", kind="text", axes="t", shape=["T"], dtype="i32",
                  dims={"T": (1, CTX, ("tokens", "词元数"))}, label=("Token ids", "词元 id"))
    seg = b.input("segment", case.input[1], adapter="masked-text", kind="text", axes="t", shape=["T"], dtype="i32",
                  label=("Segment ids", "段 id"))
    t = b.embedding("tok", ids, model.tok, label=("Token rows", "词元行"), code=code("self.tok ="), axes="t,d")
    p = b.position_embedding("pos", ids, model.pos, label=("Position rows", "位置行"), code=code("self.pos ="), axes="t,d")
    s = b.embedding("seg", seg, model.seg, label=("Segment rows", "段行"), code=code("self.seg ="), axes="t,d")
    x = b.add("tokpos", t, p, label=("Token + position", "词元 + 位置"), code=code("x = self.tok(ids)"), axes="t,d")
    x = b.add("embed", x, s, label=("+ segment", "+ 段"), code=code("x = self.tok(ids)"), axes="t,d")
    for i, blk in enumerate(model.blocks):
        n, en, zh = f"B{i + 1}", f"Block {i + 1} · ", f"第 {i + 1} 块 · "
        a = b.layernorm(f"{n}.ln1", x, blk.ln1, label=(en + "LayerNorm 1", zh + "层归一化 1"), code=code("h = self.ln1(x)"), axes="t,d")
        q = b.linear(f"{n}.q", a, blk.q, n_head=HEADS, label=(en + "Queries Q", zh + "查询 Q"), code=code("q = self.q(h)"), axes="t,d")
        k = b.linear(f"{n}.k", a, blk.k, n_head=HEADS, label=(en + "Keys K", zh + "键 K"), code=code("k = self.k(h)"), axes="t,d")
        v = b.linear(f"{n}.v", a, blk.v, n_head=HEADS, label=(en + "Values V", zh + "值 V"), code=code("v = self.v(h)"), axes="t,d")
        sc = b.attention_scores(f"{n}.scores", q, k, HEADS, label=(en + "Attention scores (no mask)", zh + "注意力分数（无掩码）"),
                                code=code("s = (q @ k"), axes="h,tq,tk")
        w = b.softmax(f"{n}.attn", sc, label=(en + "Attention weights", zh + "注意力权重"), code=code("w = s.softmax(-1)"), axes="h,tq,tk")
        m = b.attention_mix(f"{n}.mix", w, v, HEADS, label=(en + "Mix of values", zh + "值的混合"), code=code("a = (w @ v)"), axes="t,d")
        o = b.linear(f"{n}.proj", m, blk.proj, label=(en + "Attention output", zh + "注意力输出"), code=code("self.proj(a)"), axes="t,d")
        x = b.add(f"{n}.add1", x, o, label=(en + "Add attention", zh + "加上注意力"), code=code("x = x + self.proj(a)"), axes="t,d")
        c = b.layernorm(f"{n}.ln2", x, blk.ln2, label=(en + "LayerNorm 2", zh + "层归一化 2"), code=code("self.ln2(x)"), axes="t,d")
        f = b.linear(f"{n}.fc", c, blk.fc, label=(en + "MLP up (304)", zh + "MLP 放大（304）"), code=code("self.fc ="), axes="t,d")
        f = b.gelu(f"{n}.gelu", f, approx="tanh", label=(en + "GELU", zh + "GELU"), code=code("def gelu_tanh"), axes="t,d")
        f = b.linear(f"{n}.out", f, blk.out, label=(en + "MLP down (76)", zh + "MLP 缩回（76）"), code=code("self.out ="), axes="t,d")
        x = b.add(f"{n}.add2", x, f, label=(en + "Add MLP", zh + "加上 MLP"), code=code("return x + self.out"), axes="t,d")
    h = b.layernorm("ln", x, model.ln, label=("Final LayerNorm", "最后的层归一化"), code=code("self.ln ="), axes="t,d")
    y = b.linear("logits", h, tied_to="tok.weight", label=("Score of every token", "每个词元的分数"), code=code("@ self.tok.weight.t()"), axes="t,d")
    b.output(b.softmax("probs", y, label=("Probabilities", "概率"), axes="t,d"))


class WithSoftmax(torch.nn.Module):
    """module(x) for AK103: x = [1, 2, T] (ids and segments stacked) -> probabilities [1, T, 1027]."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x[:, 0], x[:, 1]), dim=-1)


def masked_case(text: str, word: str) -> dict:
    """{text, ids, mask}: the text's ids and the index of the first token whose text is `word`."""
    ids = TOK.encode(text)
    return {"text": text, "ids": ids, "mask": next(i for i, t in enumerate(ids) if TOK.token_to_display(t) == word)}


PRESETS = [
    ("park", "At the park", "在公园", "Lily went to the park to play with her friends. She had a lot of fun.", " park"),
    ("ball", "The lost ball", "丢了的球", "Tom was sad because he lost his ball.", " sad"),
    ("dog", "A dog named Max", "一只叫 Max 的狗", "Once upon a time, there was a little dog named Max.", " dog"),
    ("cake", "A birthday cake", "生日蛋糕", "Mom made a big cake for my birthday.", " cake"),
    ("box", "A box", "一个盒子", "She opened the box and found a red ball inside.", " box"),
]


def with_layout(p: dict) -> dict:
    """A preset plus its BERT layout as data.py makes it (the claim test checks the page makes the same)."""
    i, s, _ = data.layout(p["ids"], {p["mask"]})
    return {**p, "bert": i, "seg": s}


def tensors_of(content: list[int], mask: int) -> tuple[torch.Tensor, torch.Tensor]:
    i, s, _ = data.layout(content, {mask})
    return torch.tensor(i), torch.tensor(s)


def cases() -> list:
    out = []
    for pid, text, word in (("ball", PRESETS[1][3], PRESETS[1][4]), ("cat", "The cat sat on the mat. It was happy.", " cat")):
        p = masked_case(text, word)
        out.append(ak.Case(pid, text, tensors_of(p["ids"], p["mask"])))
    out.append(ak.Case("one", "[MASK] alone (T = 1)", (torch.tensor([MASK]), torch.tensor([0])), edge=True))
    val = data.load("val")   # the first 61-token stretch of a validation story with two sentences: T = 64
    starts = data.sentence_starts(val)
    for s in starts:
        content = [int(x) for x in val[s:s + data.N_MAX]]
        if EOT not in content and data.split_at(content) >= 0:
            break
    out.append(ak.Case("full", "61 tokens of a validation story (T = 64)", tensors_of(content, 30), edge=True))
    return out


def token_flags() -> str:
    return "".join(str(int(data.ENDS[i]) + 2 * int(data.STARTS_WS[i]) + 4 * int(data.ONLY_WS[i])) for i in range(EOT + 1))


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    st = json.loads((HERE / "stats.json").read_text(encoding="utf-8"))
    gpt = json.loads(ak.site_path("static", "models", "tinygpt", "spec.json").read_text(encoding="utf-8"))
    src = "pipeline/tinybert/stats.py (TinyStories validation split, GPU float32, TF32 off)"
    units = {"valAcc": "share", "valTop5": "share", "valLoss": "nats", "valN": "tokens", "valAccMasked": "share", "valMaskedN": "tokens",
             "accBoth": "share", "accLeft": "share", "accGpt": "share", "cmpN": "tokens"}
    facts = {k: {"value": st[k], "unit": units[k], "source": src} for k in units}
    tsrc = "pipeline/tinybert/train.py + data.py"
    facts.update({
        "maskRate": {"value": data.RATE, "unit": "share", "source": tsrc},
        "maskShare": {"value": data.P_MASK, "unit": "share", "source": tsrc},
        "randomShare": {"value": data.P_RANDOM, "unit": "share", "source": tsrc},
        "keepShare": {"value": round(1 - data.P_MASK - data.P_RANDOM, 10), "unit": "share", "source": tsrc},
        "steps": {"value": tm["steps"], "unit": "steps", "source": tsrc},
        "batch": {"value": tm["batch"], "unit": "examples", "source": tsrc},
        "examples": {"value": tm["examples_seen"], "unit": "examples", "source": tsrc},
        "gptParams": {"value": gpt["params"], "unit": "parameters", "source": "static/models/tinygpt/spec.json"},
    })
    vocab = [TOK.token_to_display(i) for i in range(EOT + 1)] + [SPECIAL[i] for i in sorted(SPECIAL)]
    assert len(vocab) == VOCAB
    presets = {"note": "Sentences in the style of TinyStories (written for this page); ids from TinyGPT's tokenizer; mask = the hidden token's index.",
               "items": [{"id": pid, "en": en, "zh": zh, **with_layout(masked_case(text, word))} for pid, en, zh, text, word in PRESETS]}
    report = ak.export_model(
        describe, cases(), slug="tinybert", name=("TinyBERT (masked LM)", "TinyBERT（掩码语言模型）"), kinds=("trace", "process"),
        module=WithSoftmax(model), module_input=lambda c: torch.stack(c.input)[None],
        extras={"vocabSize": VOCAB, "context": CTX, "special": {"mask": MASK, "cls": 1025, "sep": 1026, "eot": EOT},
                "tokenizerFile": "tokenizer.json", "vocab": vocab, "tokenFlags": token_flags()},
        facts=facts, presets=presets,
        training={"dataset": "TinyStories (TinyGPT's train.bin: the first 100 MiB of TinyStoriesV2-GPT4-train, TinyGPT's tokenizer)",
                  "task": "masked-language modelling: 15% of text tokens chosen (at least one per example); 80% [MASK], 10% random, 10% unchanged",
                  "steps": tm["steps"], "batchSize": tm["batch"], "examples": tm["examples_seen"], "optimizer": tm["optimizer"] + f", lr {tm['lr']}, weight decay {tm['weight_decay']}",
                  "hardware": "RTX 5090 Laptop GPU (training); CPU float32 (export)",
                  "metrics": {"hidden tokens right (unseen stories)": round(st["valAcc"], 4), "top 5 (unseen stories)": round(st["valTop5"], 4),
                              "[MASK] tokens right": round(st["valAccMasked"], 4), "one gap, both sides": round(st["accBoth"], 4),
                              "one gap, left side only": round(st["accLeft"], 4), "TinyGPT on the same gaps": round(st["accGpt"], 4)}},
    )
    shutil.copyfile(ak.site_path("static", "models", "tinygpt", "tokenizer.json"), ak.site_path("static", "models", "tinybert", "tokenizer.json"))
    print(report)
    print("params", n_params(model))
