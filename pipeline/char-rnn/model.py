"""char-rnn: the PyTorch model (kind A) and the text normalization shared by train, export, units and reference.

A character-level LSTM: each character id -> an embedding (24 numbers) -> one LSTM layer (64 units) -> a score for
every character of the alphabet. Submodule names become layer ids and tensor names (emb.weight, lstm.weight_ih_l0,
head.weight, ...).

    npm run atlas -- py pipeline/char-rnn/model.py        (smoke test: shapes and the parameter count)
"""
from __future__ import annotations

import torch
import torch.nn as nn

UNK = "�"  # the "unknown character" symbol; id 0
# The alphabet: the unknown symbol, space and newline, the 52 letters, 9 punctuation marks and the 10 digits (74).
# TinyStories' curly quotes and dashes are mapped to their plain forms first (normalize), so the learner's phone
# keyboard (which types curly quotes) reads the same characters the model was trained on.
ALPHABET = [UNK, " ", "\n"] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] \
    + list(".,\"'!?-:;") + [str(d) for d in range(10)]
INDEX = {c: i for i, c in enumerate(ALPHABET)}
NORMALIZE = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", " ": " "}


def normalize(text: str) -> str:
    for a, b in NORMALIZE.items():
        text = text.replace(a, b)
    return text


def encode(text: str) -> list[int]:
    return [INDEX.get(c, 0) for c in normalize(text)]


class CharLSTM(nn.Module):
    def __init__(self, vocab: int = len(ALPHABET), emb: int = 24, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden)          # one layer, batch_first=False (values carry no batch dim)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [T] or [T, B] (int) -> logits [T, vocab] or [T, B, vocab]."""
        h, _ = self.lstm(self.emb(ids))
        return self.head(h)


if __name__ == "__main__":
    m = CharLSTM()
    n = sum(p.numel() for p in m.parameters())
    print({"vocab": len(ALPHABET), "params": n, "names": [k for k in m.state_dict()]})
    print(m(torch.tensor(encode("Once upon a time"))).shape)
