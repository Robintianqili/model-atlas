"""Append TinyGPT's third-party sources to site/static/licenses.json (re-reads it, appends only
entries whose name is not there yet)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "..", "site", "static", "licenses.json")
ENTRIES = [
    {"name": "TinyStories (TinyStoriesV2-GPT4)",
     "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
     "license": "CDLA-Sharing-1.0",
     "usedFor": "TinyGPT training data (first 100 MiB of TinyStoriesV2-GPT4-train.txt) and validation "
                "(TinyStoriesV2-GPT4-valid.txt); TinyGPT's BPE tokenizer was learned from the same text. "
                "Only the trained weights, tokenizer and short generated samples ship with the site, not the dataset.",
     "attribution": "Ronen Eldan and Yuanzhi Li, 'TinyStories: How Small Can Language Models Be and Still Speak "
                    "Coherent English?' (2023), arXiv:2305.07759"},
    {"name": "nanoGPT / minGPT (ideas)",
     "url": "https://github.com/karpathy/nanoGPT",
     "license": "MIT",
     "usedFor": "Ideas for TinyGPT's model and training loop (pre-LN GPT-2 block, tied head, init scaling, "
                "AdamW + warm-up + cosine). Code written fresh for this project.",
     "attribution": "Andrej Karpathy, nanoGPT and minGPT (MIT License)"},
    {"name": "minbpe and GPT-2 encoder (ideas)",
     "url": "https://github.com/karpathy/minbpe",
     "license": "MIT",
     "usedFor": "Ideas for TinyGPT's byte-level BPE tokenizer (byte fallback, GPT-2-style pre-tokenize regex, "
                "rank-ordered merges). Code written fresh for this project.",
     "attribution": "Andrej Karpathy, minbpe (MIT); OpenAI, GPT-2 encoder.py (MIT)"},
    {"name": "Transformer Explainer (ideas)",
     "url": "https://github.com/poloclub/transformer-explainer",
     "license": "MIT",
     "usedFor": "Ideas for what the TinyGPT page traces and shows (Q/K/V, scores, causal mask, softmax, "
                "sampling with temperature / top-k / top-p). No code copied.",
     "attribution": "Cho et al., Georgia Tech Polo Club, Transformer Explainer (MIT License)"},
]
cur = []
if os.path.exists(P):
    with open(P, encoding="utf-8") as f:
        cur = json.load(f)
have = {e.get("name") for e in cur}
added = [e for e in ENTRIES if e["name"] not in have]
cur.extend(added)
with open(P, "w", encoding="utf-8") as f:
    json.dump(cur, f, ensure_ascii=False, indent=2)
print(f"appended {len(added)} entries; licenses.json now has {len(cur)}")
