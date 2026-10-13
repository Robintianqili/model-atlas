"""llava-toy: the measured numbers behind the lesson (facts.json), on the CPU in float32 (TF32 off) over TinyCNN's 15,000
test drawings.

    npm run atlas -- py pipeline/llava-toy/stats.py          # prints them; export.py writes them into facts.json

- whatAcc: "what is this ?" answered with the drawing's class word; vitAcc: TinyViT's own class head on the same
  drawings (the frozen encoder's full model); agreeVit: the share where both give the same class.
- yesTrue: "is this a(n) <its class> ?" answered "yes"; noOther: the same question with each of the 9 other classes
  answered "no" (averaged over the 9); yesnoAcc: the mean of the two.
- answerWords: the share of all those answers that are one of the words the question type allows (a class word, or
  yes / no).
- Where the answer row looks ("?", the last row, which writes the answer), per block of the LM, averaged over heads and
  drawings for "what is this ?": imgShare, the share of its attention on the 16 image tokens (the rest goes to the
  words); inkLean, the weight per image token whose patch has ink (brightest pixel >= 0.5, as pipeline/tinyvit/stats.py)
  divided by the weight per image token whose patch is empty.
- heldOutClassWord: on TinyCNN's held-out ladders and envelopes (1,500 test drawings each), the share of "what is this ?"
  answers that are one of the 10 class words (it has no word for "I do not know").
- trainable: the projector and the LM; vitRead: TinyViT's weights the graph reads (patches, [CLS], positions, blocks 1-3),
  frozen; vitParams, vitBlocksAll: the whole TinyViT.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import (BLOCKS, CLASSES, ID, N_IMG, VOCAB, TV, LlavaToy, article, encode, image_features,  # noqa: E402
                   load_vit, n_params)

INK = 0.5
CLS_IDS = torch.tensor([ID[c] for c in CLASSES])


def inked_patches(X: torch.Tensor) -> torch.Tensor:
    p = X.unfold(2, TV.PATCH, TV.PATCH).unfold(3, TV.PATCH, TV.PATCH).reshape(len(X), N_IMG, TV.PATCH * TV.PATCH)
    return (p.amax(-1) >= INK).float()


@torch.no_grad()
def answers(m, feats, words: list[str], keep=None):
    ids = torch.tensor([encode(words)]).expand(len(feats), -1)
    return m(feats, ids, keep)[:, -1].softmax(-1)


@torch.no_grad()
def measure(vit, m) -> dict:
    """{fact key: (value, unit)}"""
    X, y, _ = ak.data.quickdraw(CLASSES, "test")
    X = torch.tensor(X).float().unsqueeze(1) / 255
    y = torch.tensor(y)
    feats = torch.cat([image_features(vit, X[i:i + 1500]) for i in range(0, len(X), 1500)])
    keep: list = []
    p = answers(m, feats, "what is this ?".split(), keep)
    pick = p.argmax(1)
    pv = torch.cat([vit(X[i:i + 1500]).argmax(1) for i in range(0, len(X), 1500)])
    out = {
        "whatAcc": (round(float((pick == CLS_IDS[y]).float().mean()), 4), "share"),
        "vitAcc": (round(float((pv == y).float().mean()), 4), "share"),
        "agreeVit": (round(float((pick == CLS_IDS[pv]).float().mean()), 4), "share"),
        "testN": (len(y), "drawings"),
    }
    in_words = [float(torch.isin(pick, CLS_IDS).float().mean())]
    yes, no = [], []
    for k, c in enumerate(CLASSES):
        a = answers(m, feats, ["is", "this", article(c), c, "?"]).argmax(1)
        mine = y == k
        yes.append(float((a[mine] == ID["yes"]).float().sum()))
        no.append(float((a[~mine] == ID["no"]).float().sum()))
        in_words.append(float(((a == ID["yes"]) | (a == ID["no"])).float().mean()))
    yes_true = sum(yes) / len(y)
    no_other = sum(no) / (len(y) * (len(CLASSES) - 1))
    out["yesTrue"] = (round(yes_true, 4), "share")
    out["noOther"] = (round(no_other, 4), "share")
    out["yesnoAcc"] = (round((yes_true + no_other) / 2, 4), "share")
    out["answerWords"] = (round(sum(in_words) / len(in_words), 4), "share")
    ink = inked_patches(X)                                            # [N, 16]
    for b in range(BLOCKS):
        w = keep[b][:, :, -1, :]                                      # [N, H, T]: the answer row
        img = w[..., :N_IMG]
        share = img.sum(-1)                                           # [N, H]
        out[f"imgShare{b + 1}"] = (round(float(share.mean()), 4), "share")
        per_ink = (img * ink[:, None]).sum(-1) / ink.sum(-1, keepdim=True).clamp(min=1)
        per_empty = (img * (1 - ink[:, None])).sum(-1) / (1 - ink).sum(-1, keepdim=True).clamp(min=1)
        has_both = (ink.sum(-1) > 0) & (ink.sum(-1) < N_IMG)
        out[f"inkLean{b + 1}"] = (round(float(per_ink[has_both].mean() / per_empty[has_both].mean()), 3), "ratio")
        top = img.mean(1).argmax(-1)                                  # the image token with the most weight (mean of heads)
        out[f"topInked{b + 1}"] = (round(float(ink[torch.arange(len(X)), top].mean()), 4), "share")
    out["inkedShare"] = (round(float(ink.mean()), 4), "share")
    Xh, yh, _ = ak.data.quickdraw(TV.HELD_OUT, "test")
    Xh = torch.tensor(Xh).float().unsqueeze(1) / 255
    fh = torch.cat([image_features(vit, Xh[i:i + 1500]) for i in range(0, len(Xh), 1500)])
    ph = answers(m, fh, "what is this ?".split()).argmax(1)
    out["heldOutN"] = (len(Xh), "drawings")
    out["heldOutClassWord"] = (round(float(torch.isin(ph, CLS_IDS).float().mean()), 4), "share")
    out["trainable"] = (n_params(m), "count")
    out["vitRead"] = (sum(p.numel() for n, p in vit.named_parameters() if not n.startswith(("blocks.3", "ln.", "head."))), "count")
    out["vitParams"] = (n_params(vit), "count")
    out["vitBlocksAll"] = (len(vit.blocks), "count")
    return out


if __name__ == "__main__":
    m = LlavaToy()
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    for k, (v, u) in measure(load_vit(), m.eval()).items():
        print(f"{k:24s} {v} {u}")
