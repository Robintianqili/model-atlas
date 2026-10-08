"""char-rnn: export the character LSTM to site/static/models/char-rnn/ (kind A, design §4.11, §7, §8.3 M4).

    npm run atlas -- export char-rnn

The input is a sequence of character ids of symbolic length T (1..128): the cases include T = 1 and T = 128, so
parity covers both extremes (M4). b.lstm records the gate sub-steps (i f g o c h) and proves them against nn.LSTM
(AK104). The alphabet goes into spec.json as `alphabet` (the text-char adapter maps characters to ids with it; id 0 is
the unknown character). Also writes facts.json (training and held-out measurements from train_metrics.json and
units.py), presets.json (texts for the input box) and unitTop.json (the top contexts of every unit, units.py).
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
from atlas_kit._io import json_bytes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import ALPHABET, UNK, CharLSTM, encode  # noqa: E402
from units import analyse  # noqa: E402

T_MAX = 128
# what the views print for each output (the side panel and the probability bars): spaces and newlines made visible
LABELS = [{" ": chr(0x2423), chr(10): chr(0x21B5)}.get(c, c) for c in ALPHABET]
model = CharLSTM()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = ("pipeline/char-rnn/model.py", 34)


def ids_of(text: str) -> torch.Tensor:
    return torch.tensor(encode(text)[-T_MAX:], dtype=torch.int64)


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    ids = b.input("input", case.input, adapter="text-presets", kind="text", axes="t", shape=["T"], dtype="i32",
                  dims={"T": (1, T_MAX, ("characters", "字符"))}, label=("Your text", "你的文字"))
    e = b.embedding("emb", ids, model.emb, label=("Character vectors", "字符向量"), axes="t,d", code=SRC,
                    desc=("Each character id picks its own row of 24 learned numbers.", "每个字符 id 取出自己那一行 24 个学来的数。"))
    h = b.lstm("lstm", e, model.lstm, label=("LSTM memory", "LSTM 记忆"), code=SRC)
    z = b.linear("head", h, model.head, label=("Next-character scores", "下一个字符的分数"), axes="t,vocab", code=SRC)
    b.output(b.softmax("probs", z, label=("Next-character chances", "下一个字符的概率"), axes="t,vocab"))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return torch.softmax(self.m(ids), dim=-1)


STORY = ("Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a "
         "big red ball in the park. She ran to get it and laughed.")
TEXTS = {
    "lily": "Once upon a time, there was a little girl named Lily.",
    "dialog": 'Tom said, "Can we go to the park?" Mom smiled and said, "Yes!"',
    "lines": "The end.\n\nOne day, a dog and a cat found a wonderful treasure.",
    "unknown": "Zoë ate 3 crêpes (yum)",
}
CASES = [ak.Case("one", "one character", ids_of("O"), edge=True)]
CASES += [ak.Case(k, v[:24], ids_of(v)) for k, v in TEXTS.items()]
CASES += [ak.Case("max", f"{T_MAX} characters", ids_of(STORY[:T_MAX]), edge=True)]
assert len(CASES[-1].input) == T_MAX

PRESETS = {"texts": [
    {"id": "lily", "en": "A story opening", "zh": "故事开头", "text": "Once upon a time, there was a little girl named Lily."},
    {"id": "dialog", "en": "Someone speaks", "zh": "有人说话", "text": 'Tom said, "Can we go to the park?" Mom smiled.'},
    {"id": "long-words", "en": "Long words", "zh": "长单词", "text": "The grandmother told a wonderful story about friendship."},
    {"id": "two-lines", "en": "A new paragraph", "zh": "新的一段", "text": "They were happy.\n\nOne day, a dog found a ball."},
]}

if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    unit_facts, unit_top = analyse(model)
    facts = {
        "valLoss": {"value": tm["val_loss"], "unit": "nats per character", "source": "train.py: cross-entropy on 1,966,956 held-out characters (TinyStoriesV2-GPT4-valid.txt), windows of 128, CPU fp32"},
        "valBpc": {"value": tm["val_bpc"], "unit": "bits per character", "source": "train.py"},
        "trainChars": {"value": tm["train_chars"], "unit": "characters", "source": "train.py: ak.data.tinystories_chars(n_chars=40,000,000) on the TinyStories train head (story markers become blank lines), normalized"},
        "trainSteps": {"value": tm["steps"], "unit": "steps of 256 windows of 128 characters", "source": "train.py"},
        "params": {"value": tm["params"], "unit": "parameters", "source": "model.py"},
        "vocab": {"value": len(ALPHABET), "unit": "characters in the alphabet", "source": "model.py"},
        **unit_facts,
    }
    report = ak.export_model(describe, CASES, slug="char-rnn", name=("Character LSTM", "字符级 LSTM"), kinds=("trace", "process"),
                             module=WithSoftmax(model), module_input=lambda c: c.input,
                             extras={"alphabet": ALPHABET, "unk": UNK, "context": T_MAX, "labels": LABELS},
                             training={"dataset": f"TinyStories (TinyStoriesV2-GPT4): {tm['train_chars']:,} characters from the start of the train split", "chars": tm["train_chars"],
                                       "steps": tm["steps"], "metrics": {"val_loss": tm["val_loss"], "val_bpc": tm["val_bpc"]},
                                       "recipe": tm["optimizer"] + f", lr {tm['lr']}, batch {tm['batch']} x {tm['window']} characters"},
                             facts=facts, presets=PRESETS)
    out = report.out / "unitTop.json"
    out.write_bytes(json_bytes(unit_top, where="unitTop.json"))
    print(report)
    print("unitTop.json", out.stat().st_size, "bytes")
