"""tinybert: training and evaluation examples from TinyStories, with BERT's input format and BERT's masking.

Text: the token files TinyGPT's prepare.py wrote (pipeline/data/tinystories/train.bin and val.bin: uint16 ids of
TinyGPT's tokenizer, stories joined by <|endoftext|>). No new download.

One example = a window of n content tokens (1..61) that starts at the start of a sentence and stays inside one story,
laid out as BERT lays out its input:

    [CLS] sentence A [SEP]                      when the window holds one sentence
    [CLS] sentence A [SEP] the rest B [SEP]     when it holds more: A = the first sentence, B = everything after it

The split rule is the page's (adapters/masked-text/adapter.ts splitAt): A ends at the first token that ends in . ! or ?
(closing quotes allowed) and is followed by a token that starts with white space, provided B has a token that is not
only white space. Segment ids: 0 for [CLS], A and its [SEP]; 1 for B and the last [SEP].

Masking (BERT, Devlin et al. 2019): each content token is chosen with probability 15% (at least one per example, so a
short window still teaches something); a chosen token becomes [MASK] 80% of the time, a random text token 10% of the
time, and stays unchanged 10% of the time. The loss is cross-entropy on the chosen positions only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("tinygpt_bpe", ROOT / "pipeline" / "tinygpt" / "bpe.py")
bpe = importlib.util.module_from_spec(_spec)  # TinyGPT's tokenizer: the same merges, the same ids
_spec.loader.exec_module(bpe)
sys.path.insert(0, str(HERE))
from model import CLS, CTX, EOT, MASK, SEP  # noqa: E402

DATA = ROOT / "pipeline" / "data" / "tinystories"
TOKENIZER = ROOT / "pipeline" / "tinygpt" / "out" / "tokenizer.json"
N_MAX = CTX - 3                 # content tokens: 64 positions minus [CLS] and two [SEP]
RATE, P_MASK, P_RANDOM = 0.15, 0.8, 0.1   # 15% chosen; of those 80% [MASK], 10% random, 10% unchanged
PAD = 0                         # padding id (padded keys get no attention weight and no loss)

_tok = None


def tokenizer() -> "bpe.BPETokenizer":
    global _tok
    if _tok is None:
        _tok = bpe.BPETokenizer.load(str(TOKENIZER))
    return _tok


def tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per text id: ends a sentence; starts with white space; is only white space."""
    tok = tokenizer()
    ends = np.zeros(EOT + 1, bool)
    starts_ws = np.zeros(EOT + 1, bool)
    only_ws = np.zeros(EOT + 1, bool)
    for i in range(EOT):
        b = tok.token_bytes[i]
        s = b.decode("utf-8", errors="replace")
        core = s.rstrip("\"'”’)")
        ends[i] = len(core) > 0 and core[-1] in ".!?"
        starts_ws[i] = len(b) > 0 and b[:1] in (b" ", b"\n", b"\t", b"\r")
        only_ws[i] = len(s) > 0 and s.strip() == ""
    return ends, starts_ws, only_ws


ENDS, STARTS_WS, ONLY_WS = tables()


def split_at(content: list[int]) -> int:
    """Index of the last token of sentence A, or -1 when the content stays one segment (the page's splitAt)."""
    n = len(content)
    for j in range(n - 1):
        if ENDS[content[j]] and STARTS_WS[content[j + 1]] and any(not ONLY_WS[c] for c in content[j + 1:]):
            return j
    return -1


def layout(content: list[int], masked: set[int] | None = None) -> tuple[list[int], list[int], list[int]]:
    """content ids -> (BERT ids, segment ids, position of each content token). `masked`: content indices shown as [MASK].
    The split is decided on the original content (before masking), as in training."""
    j = split_at(content)
    ids, seg, where = [CLS], [0], []
    for c, t in enumerate(content):
        if j >= 0 and c == j + 1:
            ids.append(SEP)
            seg.append(0)
        where.append(len(ids))
        ids.append(MASK if masked and c in masked else t)
        seg.append(1 if j >= 0 and c > j else 0)
    ids.append(SEP)
    seg.append(1 if j >= 0 else 0)
    return ids, seg, where


def load(split: str) -> np.ndarray:
    return np.fromfile(DATA / f"{split}.bin", dtype=np.uint16).astype(np.int64)


def sentence_starts(stream: np.ndarray) -> np.ndarray:
    """Positions where a sentence starts (after <|endoftext|>, or after a sentence end + a white-space token),
    excluding <|endoftext|> and white-space-only tokens, with room for a full window."""
    prev = np.concatenate([[EOT], stream[:-1]])
    nxt_ok = (stream != EOT)
    text = np.where(stream == EOT, 0, stream)
    after_end = (prev == EOT) | (ENDS[np.where(prev == EOT, 0, prev)] & STARTS_WS[text])
    ok = after_end & nxt_ok & ~ONLY_WS[text]
    ok[len(stream) - N_MAX - 1:] = False
    return np.nonzero(ok)[0]


class Batches:
    """Training or evaluation batches. Every array is numpy; train.py moves them to the GPU."""

    def __init__(self, split: str):
        self.stream = load(split)
        self.starts = sentence_starts(self.stream)

    def windows(self, rng: np.random.Generator, B: int, n_lo: int, n_hi: int, p_full: float):
        s = self.starts[rng.integers(len(self.starts), size=B)]
        n = np.where(rng.random(B) < p_full, N_MAX, rng.integers(n_lo, n_hi + 1, size=B))
        ar = np.arange(N_MAX)
        win = self.stream[s[:, None] + ar[None]]
        first_eot = np.where(win == EOT, ar[None], N_MAX).min(1)
        n = np.minimum(n, first_eot)
        return win, n

    def example_arrays(self, win: np.ndarray, n: np.ndarray):
        """Lay out a batch of windows: ids, seg, content mask, pad mask [B, T] and the split index j (-1: none)."""
        B = len(n)
        ar = np.arange(N_MAX)
        valid = ar[None] < n[:, None]
        w = np.where(valid, win, 0)
        nonws = valid & ~ONLY_WS[np.where(w == EOT, 0, w)]
        later_nonws = np.flip(np.cumsum(np.flip(nonws, 1), 1), 1) - nonws  # count of non-white-space tokens after k
        cand = np.zeros((B, N_MAX), bool)
        cand[:, :-1] = ENDS[w[:, :-1]] & STARTS_WS[w[:, 1:]] & valid[:, 1:] & (later_nonws[:, :-1] > 0)
        has = cand.any(1)
        j = np.where(has, cand.argmax(1), -1)
        L = n + 2 + has
        T = int(L.max())
        inB = has[:, None] & (ar[None] > j[:, None])
        pos = ar[None] + 1 + inB
        rows = np.broadcast_to(np.arange(B)[:, None], (B, N_MAX))
        ids = np.full((B, T), PAD, np.int64)
        seg = np.zeros((B, T), np.int64)
        content = np.zeros((B, T), bool)
        ids[rows[valid], pos[valid]] = w[valid]
        seg[rows[valid], pos[valid]] = inB[valid]
        content[rows[valid], pos[valid]] = True
        ids[:, 0] = CLS
        ids[np.nonzero(has)[0], j[has] + 2] = SEP
        ids[np.arange(B), L - 1] = SEP
        seg[np.arange(B), L - 1] = has
        pad = np.arange(T)[None] >= L[:, None]
        return ids, seg, content, pad, j

    def mlm(self, rng: np.random.Generator, B: int, n_lo: int = 1, n_hi: int = N_MAX, p_full: float = 0.5):
        """A masked batch: (inputs, seg, pad, chosen [B, T] bool, targets [B, T])."""
        win, n = self.windows(rng, B, n_lo, n_hi, p_full)
        keep = n > 0
        win, n = win[keep], n[keep]
        ids, seg, content, pad, _ = self.example_arrays(win, n)
        u = rng.random(ids.shape)
        chosen = content & (u < RATE)
        none = ~chosen.any(1)
        if none.any():  # at least one chosen token per example
            r = np.where(content, rng.random(ids.shape), -1.0)
            pick = r.argmax(1)
            chosen[np.nonzero(none)[0], pick[none]] = True
        v = rng.random(ids.shape)
        inputs = ids.copy()
        to_mask = chosen & (v < P_MASK)
        to_rand = chosen & (v >= P_MASK) & (v < P_MASK + P_RANDOM)
        inputs[to_mask] = MASK
        inputs[to_rand] = rng.integers(0, EOT, size=int(to_rand.sum()))
        return inputs, seg, pad, chosen, ids
