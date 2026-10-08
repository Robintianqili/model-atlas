"""Verification: fresh tokenizer test strings (not the build agent's 300), encoded by the Python
tokenizer. fresh_bpe.check.ts compares the TypeScript tokenizer against them.

Run: python pipeline/parity/verify_tinygpt/make_fresh_bpe.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "pipeline", "tinygpt"))
import bpe  # noqa: E402

tok = bpe.BPETokenizer.load(os.path.join(ROOT, "site", "static", "models", "tinygpt", "tokenizer.json"))
C = chr

story = ("Pip the small brown mouse lived under the old red barn. Every morning he ran to the big field to "
         "look for seeds, and every evening he ran back home before the owl woke up. One windy day the barn "
         "door blew shut and Pip could not get in! He sat on a stone and thought hard. \"I can dig,\" he said. "
         "So he dug and dug until his paws were sore, and at last he was home, warm and happy.")
fixed = [
    "我爱自然语言处理",
    "我爱自然语言处理，因为它让机器读懂人话。",
    "TinyGPT 说：“从前有一只小兔子。” 然后呢？",
    "日本語のテキストも試します。カタカナとひらがな。",
    "한국어 문장도 넣어 봅니다.",
    "Привет, мир! Это проверка.",
    "مرحبا بالعالم ١٢٣",
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    "Emoji: " + C(0x1F600) + C(0x1F602) + C(0x1F923) + " " + C(0x1F431) + C(0x1F436) + "!",
    "family " + C(0x1F469) + C(0x200D) + C(0x1F469) + C(0x200D) + C(0x1F467) + C(0x200D) + C(0x1F466),
    "flags " + C(0x1F1E8) + C(0x1F1F3) + C(0x1F1EF) + C(0x1F1F5) + " end",
    "skin " + C(0x1F44B) + C(0x1F3FF) + C(0x1F44D) + C(0x1F3FD),
    "heart on fire " + C(0x2764) + C(0xFE0F) + C(0x200D) + C(0x1F525),
    "astronaut " + C(0x1F9D1) + C(0x1F3FF) + C(0x200D) + C(0x1F680),
    "newer emoji " + C(0x1FAE8) + C(0x1FAF7) + C(0x1FABF) + C(0x1FAE9) + C(0x1FAC6),
    " " * 200,
    "a" + " " * 37 + "b",
    "\t\t\t   \n\n\n    x",
    "\r\n\r\n  \r\n",
    "line one\n\n\n\n\n\nline two" + "\n" * 30,
    C(0x3000) * 5 + "ideographic spaces" + C(0xA0) * 3 + "nbsp" + C(0x2028) + "ls" + C(0x2029) + "ps",
    "   leading and trailing   ",
    "tabs\tin\tthe\tmiddle\t\t\t",
    C(0xFEFF) + "BOM at start",
    "zero" + C(0x200B) + "width" + C(0x2060) + "joiner" + C(0x200C) + "x",
    "NEL" + C(0x85) + "char and VT" + C(0x0B) + "FF" + C(0x0C) + "end",
    "e" + C(0x301) + " vs " + C(0xE9) + " and Z" + C(0x336) + C(0x334) + "algo",
    "I'm sure you'll see we've done what they're asking; it's 'quoted' and it" + C(0x2019) + "s curly.",
    "DON'T SHOUT, I'LL LISTEN. It'S odd.",
    "3.14159 2026-10-05 1,000,000 0x1F600 " + C(0xBD) + " " + C(0x2167) + " " + C(0x0663) + C(0x0664),
    "def f(x):\n    return x**2  # square\n\nprint(f(12))",
    "<|endoftext|>",
    "<|endoftext|><|endoftext|>",
    "The end.<|endoftext|>Once upon a time",
    "almost special <|endoftext| and <|endof",
    "<|endoftext|>" * 3 + " tail",
    story,
    story + " " + story + " " + story,
    "Once upon a time, " * 40,
    "a" * 300,
    "ab" * 150,
    "!!!???...,,," * 20,
    C(0x10D40) + C(0x10D41) + " Garay, " + C(0x16D43) + C(0x16D44) + " Kirat Rai, " + C(0x1E6C0) + C(0x1E6C1) + " (Unicode 16/17 letters)",
    C(0x31350) + C(0x2EBF0) + " CJK ext H/I, " + C(0x1CC00) + C(0x1CC01) + " legacy computing",
    C(0x1D7CE) + C(0x1D7CF) + " math digits " + C(0x1D400) + C(0x1D401),
    "",
    " ",
    "x",
]

rnd = random.Random(424242)
pools = [
    list(range(0x20, 0x7F)) * 4,
    list(range(0x4E00, 0x4E00 + 400)),
    [0x20, 0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C, 0xA0, 0x3000, 0x2028, 0x2029, 0x202F, 0x205F, 0x1680, 0x2003, 0xFEFF, 0x85, 0x200B],
    list(range(0x1F300, 0x1F650)) + [0x200D, 0xFE0F, 0x1F3FB, 0x1F3FF],
    list(range(0x0400, 0x0460)) + list(range(0x0600, 0x0660)) + list(range(0x0900, 0x0970)) + list(range(0x3040, 0x30FF)),
    list(range(0x300, 0x370)),
    [ord(c) for c in "'sdmtlvre"] * 5,
]


def rand_cp():
    if rnd.random() < 0.08:  # anywhere in Unicode except surrogates
        while True:
            cp = rnd.randrange(1, 0x110000)
            if not 0xD800 <= cp <= 0xDFFF:
                return cp
    return rnd.choice(rnd.choice(pools))


fuzz = []
for i in range(400):
    n = rnd.choice([1, 2, 3, 5, 8, 13, 30, 80, 200])
    fuzz.append("".join(C(rand_cp()) for _ in range(n)))
# random mixes of English words, Chinese, emoji, whitespace runs
words = ["the", " cat", "Lily", " ran", "!", "?", "我", "爱", "你", C(0x1F600), " ", "  ", "\n", "\n\n", "\t", "'s", "'ll",
         "123", " 45", "...", "<|endoftext|>", C(0x3000), "naïve", "Ω", "x"]
for i in range(200):
    fuzz.append("".join(rnd.choice(words) for _ in range(rnd.randrange(1, 60))))

out = []
for s in fixed + fuzz:
    ids = tok.encode(s)
    out.append({"text": s, "ids": ids, "decoded": tok.decode(ids), "nTokens": len(ids)})
steps_cases = [fixed[0], fixed[1], fixed[9], fixed[17], fixed[32], story[:120]]
steps = [{"text": s, "result": tok.encode_with_steps(s)} for s in steps_cases]
# trainBPE on fresh texts (same algorithm and tie-break as bpe.py train_bpe)
from collections import Counter  # noqa: E402
train_texts = [
    (story + " " + story, 80),
    ("我爱自然语言处理。自然语言处理很有趣。我爱机器学习。" * 3, 40),
    (" ".join(fixed[9:16]) + "  " + chr(10) * 2 + "  " + " ".join(fixed[16:24]), 60),
    ("aaaaaaa bbbb aaaa abab " * 5, 12),  # runs and ties
]
train = []
for text, k in train_texts:
    merges = bpe.train_bpe(Counter(bpe.pretokenize(text)), k)
    train.append({"text": text, "numMerges": k, "merges": [[a, b] for a, b, _ in merges], "counts": [n for _, _, n in merges]})
# the real tokenizer cut to k merges
cut = []
for k in (0, 5, 50, 300, 700):
    t2 = bpe.BPETokenizer(tok.merges[:k])
    for s in fixed[:12] + [story] + fuzz[:40]:
        cut.append({"k": k, "text": s, "ids": t2.encode(s)})
# decode fuzz: random id lists heavy in byte tokens 0x80-0xFF and merged tokens
drnd = random.Random(777)
dec = []
for i in range(600):
    ids = [drnd.choice([drnd.randrange(0x80, 0x100), drnd.randrange(0, 1024), drnd.randrange(0xC0, 0x100)]) for _ in range(drnd.randrange(1, 12))]
    dec.append({"ids": ids, "text": tok.decode(ids)})
with open(os.path.join(HERE, "fresh_bpe_cases.json"), "w", encoding="utf-8") as f:
    json.dump({"cases": out, "encodeWithSteps": steps, "train": train, "cut": cut, "decode": dec}, f, ensure_ascii=False)
print(f"{len(out)} strings ({len(fixed)} hand-written, {len(fuzz)} fuzz), {sum(c['nTokens'] for c in out)} tokens")
for c in out[:3]:
    print(repr(c["text"][:40]), c["nTokens"], c["ids"][:30])
print("story x3 tokens:", out[fixed.index(story + ' ' + story + ' ' + story)]["nTokens"], "| story tokens:", out[fixed.index(story)]["nTokens"])
print("round trip failures (py):", sum(1 for c in out if c["decoded"] != c["text"]))
