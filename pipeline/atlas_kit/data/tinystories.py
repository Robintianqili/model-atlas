"""ak.data.tinystories_chars(path, n_chars): raw TinyStories text for character-level models.

Files in <ATLAS_ROOT>/pipeline/data/tinystories/ (already downloaded for TinyGPT; CDLA-Sharing-1.0, TinyStories by
Eldan and Li, 2023): train_head_100MB.raw.txt (the first 100 MiB of the train split) and
TinyStoriesV2-GPT4-valid.txt. Stories are separated by <|endoftext|>.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..errors import fail
from ..paths import root_path

EOT = "<|endoftext|>"
DEFAULT = "train_head_100MB.raw.txt"


def tinystories_chars(path: Optional[str] = None, n_chars: int = 1_000_000, *, keep_eot: bool = False,
                      whole_stories: bool = True) -> str:
    """The first `n_chars` characters (UTF-8 decoded) of a TinyStories file.

    path: a file name inside pipeline/data/tinystories/ (default: the train head) or a full path.
    keep_eot=False replaces each <|endoftext|> marker with a blank line, so the text is plain prose.
    whole_stories=True cuts the text after the last complete story that fits."""
    p = Path(path) if path and Path(path).is_absolute() else root_path("pipeline", "data", "tinystories", path or DEFAULT)
    if not p.is_file():
        fail("AK002", f"TinyStories file not found: {p.as_posix()}",
             fix="use train_head_100MB.raw.txt or TinyStoriesV2-GPT4-valid.txt (downloaded for TinyGPT); no new downloads")
    if n_chars <= 0:
        fail("AK301", "n_chars must be positive")
    # read a little more than needed (a story marker may straddle the cut)
    with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
        text = f.read(n_chars + len(EOT) * 4 + 4096)
    text = text.replace("\r\n", "\n")
    if whole_stories:
        cut = text.rfind(EOT, 0, n_chars + len(EOT))
        text = text[:cut] if cut > 0 else text[:n_chars]
    if not keep_eot:
        text = "\n\n".join(s.strip("\n") for s in text.split(EOT) if s.strip())
    return text[:n_chars]
