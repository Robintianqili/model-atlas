"""llava-toy: export the connector toy to site/static/models/llava-toy/ (kind A, design §4.11).

    npm run atlas -- py pipeline/llava-toy/train.py --run v1      # model_final.pt (projector + LM; TinyViT stays frozen)
    npm run atlas -- export llava-toy                             # this file, then parity --write and the lesson draft

One graph with two inputs: `image` [1, 28, 28] (the learner's drawing, TinyCNN's pixel/255) and `text` [T] ids: 16 <image>
placeholders (id 0), then the question (1 to 8 known words), so T is 17 to 24. describe() names every step: the frozen TinyViT's patch convolution, [CLS] row,
positions and blocks 1 to 3 (pipeline/tinyvit/model.py, weights read from pipeline/tinyvit/model_final.pt), drop_cls
(the 16 patch rows are the image features), the projector, the word rows, the joined sequence of 16 + L rows with its
positions, the LM's two causal blocks, and the next-word shares of every row (the page reads the last row).
The reference cases are TinyCNN's preset drawings (the same bytes the page decodes) with questions of every length
from 1 to 8 words, plus the blank pad. facts.json: stats.py's test numbers and the training run.
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
import ops_tokens as ops  # noqa: E402
from model import (BLOCKS, HEADS, ID, MAX_WORDS, N_IMG, VIT_BLOCKS, VOCAB, ZH, Full, LlavaToy, load_vit,  # noqa: E402
                   n_params)
from stats import measure  # noqa: E402

vit = load_vit()
model = LlavaToy()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
VIT_SRC = ak.root_path("pipeline", "tinyvit", "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str, vit_file: bool = False) -> tuple[str, int]:
    """(file, line) of the first line that contains `pattern` (the Code tab)."""
    src, name = (VIT_SRC, "pipeline/tinyvit/model.py") if vit_file else (SRC, "pipeline/llava-toy/model.py")
    return (name, next(i + 1 for i, s in enumerate(src) if pattern in s))


def block(b, h, blk, n: str, en: str, zh: str, causal: bool, vit_file: bool, axes_attn: str):
    """One pre-norm block, recorded as tinyvit's export does (ids n.ln1 ... n.add2)."""
    c = lambda p: code(p, vit_file)  # noqa: E731
    a = b.layernorm(f"{n}.ln1", h, blk.ln1, label=(en + "LayerNorm 1", zh + "层归一化 1"), code=c("h = self.ln1(x)"), axes="t,d")
    q = b.linear(f"{n}.q", a, blk.q, n_head=HEADS, label=(en + "Queries Q", zh + "查询 Q"), code=c("q = self.q(h)"), axes="t,d")
    k = b.linear(f"{n}.k", a, blk.k, n_head=HEADS, label=(en + "Keys K", zh + "键 K"), code=c("k = self.k(h)"), axes="t,d")
    v = b.linear(f"{n}.v", a, blk.v, n_head=HEADS, label=(en + "Values V", zh + "值 V"), code=c("v = self.v(h)"), axes="t,d")
    s = b.attention_scores(f"{n}.scores", q, k, HEADS, causal=causal, label=(en + "Attention scores", zh + "注意力分数"),
                           code=c("s = (q @ k"), axes=axes_attn)
    w = b.softmax(f"{n}.attn", s, label=(en + "Attention weights", zh + "注意力权重"), code=c("softmax(-1)"), axes=axes_attn)
    m = b.attention_mix(f"{n}.mix", w, v, HEADS, label=(en + "Mix of values", zh + "值的混合"), code=c("a = (w @ v)"), axes="t,d")
    o = b.linear(f"{n}.proj", m, blk.proj, label=(en + "Attention output", zh + "注意力输出"), code=c("self.proj(a)"), axes="t,d")
    h = b.add(f"{n}.add1", h, o, label=(en + "Add attention", zh + "加上注意力"), code=c("x = x + self.proj(a)"), axes="t,d")
    g = b.layernorm(f"{n}.ln2", h, blk.ln2, label=(en + "LayerNorm 2", zh + "层归一化 2"), code=c("self.ln2(x)"), axes="t,d")
    f = b.linear(f"{n}.fc1", g, blk.fc1, label=(en + "MLP up", zh + "MLP 放大"), code=c("self.fc1 =" if vit_file else "self.fc1 = nn.Linear(D, MLP)"), axes="t,d")
    f = b.gelu(f"{n}.gelu", f, label=(en + "GELU", zh + "GELU"), code=c("F.gelu" if vit_file else "F.gelu(self.fc1(self.ln2"), axes="t,d")
    f = b.linear(f"{n}.fc2", f, blk.fc2, label=(en + "MLP down", zh + "MLP 缩回"), code=c("self.fc2 =" if vit_file else "self.fc2 = nn.Linear(MLP, D)"), axes="t,d")
    return b.add(f"{n}.add2", h, f, label=(en + "Add MLP", zh + "加上 MLP"), code=c("return x + self.fc2"), axes="t,d")


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    img = b.input("image", case.input["image"], adapter="drawing-question", kind="drawing", axes="c,y,x",
                  label=("Your drawing", "你的画"), desc=("28×28 numbers: 0 is empty, 1 is full ink.", "28×28 个数：0 是空白，1 是满墨。"))
    ids = b.input("text", case.input["text"], adapter="drawing-question", kind="text", axes="t", shape=["T"], dtype="i32",
                  dims={"T": (N_IMG + 1, N_IMG + MAX_WORDS, ("rows", "行"))}, label=("16 image placeholders, then your question", "16 个图像占位符，然后是你的问题"))
    # 1. the frozen TinyViT, blocks 1 to 3
    v = lambda p: code(p, True)  # noqa: E731
    fold = {"lesson": "fold"}
    h = b.conv2d("vit.patch", img, vit.patch, label=("TinyViT · Patch embedding", "TinyViT · 图块嵌入"), code=v("self.patch ="), axes="c,y,x", teach=fold)
    h = b.op("to_tokens", "vit.tokens", [h], ref=ops.to_tokens, label=("TinyViT · Patches become rows", "TinyViT · 图块变成行"), axes="t,d",
             code=v("flatten(2).transpose"), teach=fold)
    h = b.op("cls_token", "vit.cls", [h], weights={"weight": vit.cls}, ref=ops.cls_token, axes="t,d",
             label=("TinyViT · [CLS] row in front", "TinyViT · 前面加上 [CLS] 行"), code=v("self.cls ="), teach=fold)
    p = b.position_embedding("vit.pos", h, weight=vit.pos, label=("TinyViT · Position rows", "TinyViT · 位置行"), code=v("self.pos ="), axes="t,d", teach=fold)
    h = b.add("vit.embed", h, p, label=("TinyViT · Rows + positions", "TinyViT · 行 + 位置"), code=v("+ self.pos"), axes="t,d", teach=fold)
    for i in range(VIT_BLOCKS):
        h = block(b, h, vit.blocks[i], f"vit.B{i + 1}", f"TinyViT block {i + 1} · ", f"TinyViT 第 {i + 1} 块 · ", False, True, "h,tq,tk")
    f = b.op("drop_cls", "feat", [h], ref=ops.drop_cls, axes="t,d", label=("Image features (16 patch rows)", "图像特征（16 个图块行）"),
             code=code("h[:, 1:]"))
    # 2. the projector
    u = b.linear("proj.fc1", f, model.proj.fc1, label=("Projector · Linear 48 → 64", "投影层 · 线性 48 → 64"), code=code("self.fc1 = nn.Linear(D_VIT"), axes="t,d")
    u = b.gelu("proj.gelu", u, label=("Projector · GELU", "投影层 · GELU"), code=code("self.fc2(F.gelu(self.fc1(f)))"), axes="t,d")
    t = b.linear("proj.fc2", u, model.proj.fc2, label=("Image tokens (64 each)", "图像词元（每个 64 维）"), code=code("self.fc2 = nn.Linear(D, D)"), axes="t,d")
    # 3. the language model
    lm = model.lm
    w = b.embedding("lm.tok", ids, lm.tok, label=("Word rows (16 placeholders, then the question)", "词行（16 个占位符，然后是问题）"), code=code("self.tok ="), axes="t,d")
    s = b.op("merge_image", "lm.seq", [t, w], ref=ops.merge_image, label=("Image tokens, then the words", "先是图像词元，再是词"),
             code=code("torch.cat([img, self.tok(ids)]"), axes="t,d")
    pe = b.position_embedding("lm.pos", s, weight=lm.pos, label=("LM position rows", "语言模型的位置行"), code=code("self.pos = nn.Parameter(torch.zeros(MAX_T"), axes="t,d")
    h = b.add("lm.embed", s, pe, label=("Sequence + positions", "序列 + 位置"), code=code("h = h + self.pos"), axes="t,d")
    for i in range(BLOCKS):
        h = block(b, h, lm.blocks[i], f"lm.L{i + 1}", f"LM block {i + 1} · ", f"语言模型第 {i + 1} 块 · ", True, False, "h,tq,tk")
    h = b.layernorm("lm.ln", h, lm.ln, label=("LM final LayerNorm", "语言模型最后的层归一化"), code=code("self.ln = nn.LayerNorm(D)"), axes="t,d")
    y = b.linear("lm.head", h, lm.head, label=("Next-word scores", "下一个词的分数"), code=code("self.head ="), axes="t,vocab")
    b.output(b.softmax("probs", y, label=("Next-word shares", "下一个词的份额"), axes="t,vocab"))


PRESETS = json.loads(ak.site_path("static", "models", "tinycnn", "presets.json").read_text(encoding="utf-8"))


def drawing(cls: str) -> torch.Tensor:
    """The first preset drawing of a class, as the page decodes it (b64 -> uint8 / 255)."""
    c = next(c for c in PRESETS["classes"] + PRESETS["heldOut"] if c["id"] == cls)
    px = np.frombuffer(base64.b64decode(c["items"][0]["b64"]), np.uint8).reshape(1, 28, 28)
    return torch.tensor(px, dtype=torch.float32) / 255


def case(id_: str, label: str, img: torch.Tensor, q: str, edge: bool = False) -> ak.Case:
    return ak.Case(id_, label, {"image": img, "text": torch.tensor([0] * N_IMG + [ID[w] for w in q.split()], dtype=torch.int32)}, edge=edge)


def cases() -> list:
    dot = torch.zeros(1, 28, 28)
    dot[0, 14, 14] = 1.0
    return [
        case("house-what", "house: what is this ?", drawing("house"), "what is this ?"),
        case("apple-yes", "apple: is this an apple ?", drawing("apple"), "is this an apple ?"),
        case("car-no", "car: is it a house ?", drawing("car"), "is it a house ?"),
        case("fish-what", "fish: what is it ?", drawing("fish"), "what is it ?"),
        case("ladder-what", "ladder: what is this ?", drawing("ladder"), "what is this ?"),
        case("blank", "blank: what is this ?", torch.zeros(1, 28, 28), "what is this ?", edge=True),
        case("dot-one-word", "one dot: ?", dot, "?", edge=True),
        case("star-eight", "star: 8 words", drawing("star"), "is this a star ? what is it", edge=True),
    ]


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    src = "pipeline/llava-toy/stats.py (CPU float32, 15,000 test drawings)"
    facts = {k: {"value": v, "unit": u, "source": src} for k, (v, u) in measure(vit, model).items()}
    # the limits step quotes the ladder preset's answer (precomputed here, so the page runs no second forward pass)
    lad = Full(vit, model)((drawing("ladder").unsqueeze(0), case("l", "l", drawing("ladder"), "what is this ?").input["text"].long().unsqueeze(0)))[0, -1]
    lsrc = "pipeline/llava-toy/export.py (the ladder preset, \"what is this ?\", CPU float32)"
    facts["ladderAnswer"] = {"value": int(lad.argmax()), "unit": "vocab id", "source": lsrc}
    facts["ladderP"] = {"value": round(float(lad.max()), 4), "unit": "share", "source": lsrc}
    for k, key in (("bestEpoch", "best_epoch"), ("epochs", "epochs"), ("steps", "steps")):
        facts[k] = {"value": tm[key], "unit": "epoch" if k != "steps" else "count", "source": "pipeline/llava-toy/train.py"}
    presets = dict(PRESETS)
    presets["note"] = "TinyCNN's presets (static/models/tinycnn/presets.json): " + PRESETS.get("note", "")
    report = ak.export_model(
        describe, cases(), slug="llava-toy", name=("LLaVA-style connector (toy)", "LLaVA 式连接器（玩具）"), kinds=("trace",),
        module=Full(vit, model), module_input=lambda c: (c.input["image"].unsqueeze(0), c.input["text"].long().unsqueeze(0)),
        extras={"vocab": VOCAB, "vocabZh": [ZH.get(w, w) for w in VOCAB], "imageTokens": 16, "vitBlocks": VIT_BLOCKS},
        facts=facts, presets=presets,
        training={"dataset": "Quick, Draw! numpy bitmaps (TinyCNN's 10 classes and split), one generated question per drawing per epoch",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"AdamW lr {tm['lr']} (1 warm-up epoch, cosine), weight decay {tm['weight_decay']} on matrices",
                  "trained": "the projector and the language model only; TinyViT frozen",
                  "augmentation": "TinyCNN's augment(): shift, scale 0.8-1.1, rotation +-12 deg, stroke thickness jitter",
                  "hardware": "RTX 5090 Laptop GPU (training); CPU float32 (evaluation, export)",
                  "metrics": {"test_what": tm["test_what"], "test_yesno": tm["test_yesno"], "test_n": tm["test_n"],
                              "trainable_params": n_params(model)}},
    )
    print(report)
