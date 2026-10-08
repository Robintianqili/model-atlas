"""word2vec: the preset mini-corpora, chosen from TinyStories by fixed rules (no hand-picking of sentences).

    npm run atlas -- py pipeline/word2vec/presets.py        # writes $ATLAS_SITE/static/models/word2vec/presets.json

Source: pipeline/data/tinystories/TinyStoriesV2-GPT4-valid.txt (TinyStories, CDLA-Sharing-1.0; already on disk).
A "short sentence" is 2 to 7 words of ASCII letters and spaces ending in a full stop. Rules:
  pets   for each of "The cat", "The dog", "The bird": its 3 most frequent short sentences that begin "<X> was",
         then its 2 most frequent that begin "<X> wanted" (ties: alphabetical). 15 sentences.
  kids   the same rule for "Tim", "Lily" and "Sue".
  story  the first story of the file with 50 to 75 words and no quotation marks, as it is (line breaks become spaces).
  pets-zh  a Chinese translation of `pets`, sentence by sentence (written here; every character is one token).
"""
from __future__ import annotations

import collections
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("ATLAS_ROOT") or Path(__file__).resolve().parents[2])
SITE = Path(os.environ.get("ATLAS_SITE") or ROOT / "site")
SRC = ROOT / "pipeline" / "data" / "tinystories" / "TinyStoriesV2-GPT4-valid.txt"
OUT = SITE / "static" / "models" / "word2vec" / "presets.json"

ZH = {
    "The cat was very happy.": "猫非常开心。", "The cat was sad.": "猫很难过。", "The cat was happy.": "猫很开心。",
    "The cat wanted to catch the bird.": "猫想抓鸟。", "The cat wanted to climb the tree.": "猫想爬树。",
    "The dog was sad.": "狗很难过。", "The dog was very happy.": "狗非常开心。", "The dog was not nice.": "狗不友好。",
    "The dog wanted to play too.": "狗也想玩。", "The dog wanted the ball.": "狗想要球。",
    "The bird was very happy.": "鸟非常开心。", "The bird was sad.": "鸟很难过。", "The bird was happy.": "鸟很开心。",
    "The bird wanted to help.": "鸟想帮忙。", "The bird wanted to play too.": "鸟也想玩。",
}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    short = collections.Counter(s for s in sentences(text) if 2 <= len(s.split()) <= 7 and re.fullmatch(r"[A-Za-z ]+\.", s))

    def top(prefix: str, k: int) -> list[str]:
        hits = sorted(((c, s) for s, c in short.items() if s.startswith(prefix + " ")), key=lambda x: (-x[0], x[1]))
        return [s for _, s in hits[:k]]

    def rule(subjects: list[str]) -> list[str]:
        return [s for x in subjects for s in top(x + " was", 3) + top(x + " wanted", 2)]

    pets = rule(["The cat", "The dog", "The bird"])
    kids = rule(["Tim", "Lily", "Sue"])
    story = next(st for st in (p.strip() for p in text.split("<|endoftext|>")) if 50 <= len(st.split()) <= 75 and '"' not in st)
    assert all(s in ZH for s in pets), [s for s in pets if s not in ZH]
    presets = [
        {"id": "pets", "en": "Cats, dogs and birds", "zh": "猫、狗和鸟", "text": " ".join(pets)},
        {"id": "kids", "en": "Tim, Lily and Sue", "zh": "Tim、Lily 和 Sue", "text": " ".join(kids)},
        {"id": "story", "en": "A real story", "zh": "一个真实的故事", "text": " ".join(story.split())},
        {"id": "pets-zh", "en": "Cats, dogs and birds (Chinese)", "zh": "猫、狗和鸟（中文）", "text": "".join(ZH[s] for s in pets)},
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": "word2vec-presets/1", "source": "TinyStoriesV2-GPT4-valid.txt (CDLA-Sharing-1.0); rules in pipeline/word2vec/presets.py", "presets": presets}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for p in presets:
        print(f"{p['id']}: {len(p['text'].split())} words | {p['text'][:110]}")
    print(f"wrote {OUT.as_posix()}")


if __name__ == "__main__":
    main()
