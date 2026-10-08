"""Prepare TinyStories for TinyGPT: learn the BPE tokenizer and encode train/val to uint16 token files.

Inputs (pipeline/data/tinystories/):
  train_head_100MB.raw.txt     first 100 MiB of TinyStoriesV2-GPT4-train.txt (HTTP Range request)
  TinyStoriesV2-GPT4-valid.txt full validation split (optional; if missing, hold out 2% of train)
Outputs:
  pipeline/data/tinystories/train.bin, val.bin   uint16 token ids, stories joined by <|endoftext|>
  pipeline/tinygpt/out/tokenizer.json             the learned tokenizer
  pipeline/tinygpt/out/data_stats.json

Run: python prepare.py [--vocab 1024]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

import bpe

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "tinystories")
OUT = os.path.join(HERE, "out")
EOT = "<|endoftext|>"


def read_stories(path: str, cut_partial: bool) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if cut_partial:  # the range request ends mid-story: keep only complete stories
        text = text[: text.rfind(EOT)]
    stories = []
    for s in text.split(EOT):
        # light clean-up: strip trailing spaces on each line and blank lines at the ends
        s = "\n".join(line.rstrip() for line in s.strip().split("\n"))
        if s:
            stories.append(s)
    return stories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    t0 = time.time()
    train = read_stories(os.path.join(DATA, "train_head_100MB.raw.txt"), cut_partial=True)
    vpath = os.path.join(DATA, "TinyStoriesV2-GPT4-valid.txt")
    if os.path.exists(vpath):
        val = read_stories(vpath, cut_partial=False)
        val_source = "TinyStoriesV2-GPT4-valid.txt (official validation split)"
    else:
        n = len(train) * 2 // 100
        val, train = train[-n:], train[:-n]
        val_source = "last 2% of the downloaded train head (official valid split could not be downloaded)"
    print(f"stories: train {len(train)}, val {len(val)}  ({time.time() - t0:.1f}s)")

    # 1) learn the tokenizer from pre-token counts over the training stories
    t0 = time.time()
    counts: Counter = Counter()
    for s in train:
        counts.update(bpe.pretokenize(s))
    print(f"pre-tokens: {sum(counts.values())} total, {len(counts)} distinct ({time.time() - t0:.1f}s)")
    t0 = time.time()
    num_merges = args.vocab - 256 - len(bpe.SPECIAL_TOKENS)
    merges = bpe.train_bpe(counts, num_merges, verbose=True)
    tok = bpe.BPETokenizer(merges)
    assert tok.vocab_size == args.vocab, tok.vocab_size
    print(f"learned {len(merges)} merges ({time.time() - t0:.1f}s)")

    # 2) encode
    t0 = time.time()
    eot = tok.special[EOT]

    def encode_all(stories):
        out = []
        for s in stories:
            out.extend(tok.encode(s, allow_special=False))
            out.append(eot)
        return np.array(out, dtype=np.uint16)

    tr = encode_all(train)
    va = encode_all(val)
    tr.tofile(os.path.join(DATA, "train.bin"))
    va.tofile(os.path.join(DATA, "val.bin"))
    train_bytes = sum(len(s.encode("utf-8")) for s in train)
    val_bytes = sum(len(s.encode("utf-8")) for s in val)
    print(f"tokens: train {len(tr)}, val {len(va)} ({time.time() - t0:.1f}s)")

    stats = {
        "dataset": "TinyStories (Eldan & Li 2023), TinyStoriesV2-GPT4",
        "train_source": "first 100 MiB of TinyStoriesV2-GPT4-train.txt, cut at the last complete story",
        "val_source": val_source,
        "train_stories": len(train), "val_stories": len(val),
        "train_tokens": int(len(tr)), "val_tokens": int(len(va)),
        "train_bytes": train_bytes, "val_bytes": val_bytes,
        # bytes per token, counting the <|endoftext|> token as one token with 0 bytes
        "train_bytes_per_token": train_bytes / len(tr), "val_bytes_per_token": val_bytes / len(va),
        "vocab_size": tok.vocab_size,
    }
    with open(os.path.join(OUT, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    tj = tok.to_json({
        "trainedOn": "TinyStoriesV2-GPT4 train head (first 100 MiB), Eldan & Li 2023, CDLA-Sharing-1.0",
        "mergeCounts": [int(m[2]) for m in merges],
    })
    with open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tj, f)
    print(json.dumps(stats, indent=2))
    print("first 40 merges:", [tok.token_to_display(256 + i) for i in range(40)])
    print("last 20 merges:", [tok.token_to_display(256 + i) for i in range(len(merges) - 20, len(merges))])


if __name__ == "__main__":
    main()
