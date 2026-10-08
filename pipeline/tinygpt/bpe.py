"""A small, readable byte-level BPE tokenizer (GPT-2 style).

How it works
------------
1. Split the text on special tokens such as "<|endoftext|>" (they become one id each).
2. Pre-tokenize every other piece with a regex: words, numbers, punctuation runs and whitespace runs.
   A word keeps its leading space (" cat"), so merges never cross a word boundary.
3. Turn every pre-token into UTF-8 bytes. Token ids 0..255 are the 256 raw byte values.
4. Apply the learned merges. Merge i (0-based) joins the pair (a, b) into the new token 256 + i.
   Encoding repeatedly picks the adjacent pair with the lowest merge rank and merges every
   occurrence of it, left to right, until no learned pair is left (same as GPT-2).

Training learns the merges: count every adjacent pair over all pre-tokens (weighted by how often
each pre-token occurs), merge the most frequent pair, repeat. Ties are broken by the smaller pair
(a, b) so training is deterministic and the TypeScript trainer gives the same merges.

The pre-tokenize regex is written so Python's `regex` module and JavaScript (`new RegExp(p, "gu")`)
read it the same way: whitespace is spelled out as an explicit class instead of `\\s`, because
`\\s` means slightly different characters in the two languages.

Ideas from GPT-2's encoder.py (OpenAI, MIT) and Andrej Karpathy's minbpe (MIT); code written fresh.
"""
from __future__ import annotations

import json
from collections import Counter

import regex

SPECIAL_TOKENS = ["<|endoftext|>"]

# Whitespace characters, spelled out (tab, LF, VT, FF, CR, space, NBSP, Ogham space, en/em spaces,
# line/paragraph separators, narrow NBSP, math space, ideographic space U+3000, BOM).
WS = "\\x09-\\x0d \\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"

PRETOKENIZE_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"            # English contractions: 's 'd 'm 't 'll 've 're
    r"| ?\p{L}+"                        # a word (letters of any script), with one leading space
    r"| ?\p{N}+"                        # a number, with one leading space
    + r"| ?[^" + WS + r"\p{L}\p{N}]+"   # punctuation, symbols, emoji
    + r"|[" + WS + r"]+(?![^" + WS + r"])"  # whitespace run (leaves 1 space for the next word)
    + r"|[" + WS + r"]+"                # whitespace run at the end
)

_PRETOKENIZE_RE = regex.compile(PRETOKENIZE_PATTERN)
_SPECIAL_RE = regex.compile("(" + "|".join(regex.escape(s) for s in SPECIAL_TOKENS) + ")")


def pretokenize(text: str) -> list[str]:
    """Split text (without special tokens) into pre-tokens."""
    return _PRETOKENIZE_RE.findall(text)


def split_special(text: str) -> list[str]:
    """Split text into pieces; special tokens become their own pieces."""
    return [p for p in _SPECIAL_RE.split(text) if p != ""]


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def _pairs(word: list[int]) -> list[tuple[int, int]]:
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def _merge_word(word: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe(pretoken_counts: Counter, num_merges: int, verbose: bool = False):
    """Learn `num_merges` merges from a Counter {pretoken string: count}.

    Returns a list of (a, b, count) in merge order.
    """
    words = [list(p.encode("utf-8")) for p in pretoken_counts]
    freqs = list(pretoken_counts.values())

    pair_counts: Counter = Counter()
    where: dict[tuple[int, int], set[int]] = {}  # pair -> indices of words that contain it
    for wi, w in enumerate(words):
        for p in _pairs(w):
            pair_counts[p] += freqs[wi]
            where.setdefault(p, set()).add(wi)

    merges = []
    for m in range(num_merges):
        if not pair_counts:
            break
        # most frequent pair; ties -> smaller (a, b)
        best, best_n = None, 0
        for p, n in pair_counts.items():
            if n > best_n or (n == best_n and p < best):
                best, best_n = p, n
        if best_n <= 0:
            break
        new_id = 256 + m
        merges.append((best[0], best[1], best_n))
        for wi in sorted(where.get(best, ())):
            w = words[wi]
            for p in _pairs(w):  # remove this word's old pairs
                pair_counts[p] -= freqs[wi]
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                s = where.get(p)
                if s is not None:
                    s.discard(wi)
            w = _merge_word(w, best, new_id)
            words[wi] = w
            for p in _pairs(w):  # add the new pairs
                pair_counts[p] += freqs[wi]
                where.setdefault(p, set()).add(wi)
        where.pop(best, None)
        if verbose and (m < 20 or m % 100 == 0):
            print(f"merge {m}: {best} -> {new_id} (count {best_n})")
    return merges


# ----------------------------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self, merges, special_tokens: list[str] = SPECIAL_TOKENS):
        self.merges = [(int(m[0]), int(m[1])) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self.special = {}
        for s in special_tokens:
            self.special[s] = len(self.token_bytes)
            self.token_bytes.append(s.encode("utf-8"))
        self.special_ids = {v: k for k, v in self.special.items()}
        self.vocab_size = len(self.token_bytes)
        self._cache: dict[str, list[int]] = {}

    # -- encoding --
    def bpe_steps(self, ids: list[int]):
        """Apply merges to one pre-token's byte ids. Returns (final ids, steps)."""
        steps = []
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            pair = (ids[best_i], ids[best_i + 1])
            ids = _merge_word(ids, pair, 256 + best_rank)
            steps.append({"pair": list(pair), "rank": best_rank, "newId": 256 + best_rank, "tokens": list(ids)})
        return ids, steps

    def encode_pretoken(self, piece: str) -> list[int]:
        ids = self._cache.get(piece)
        if ids is None:
            ids, _ = self.bpe_steps(list(piece.encode("utf-8")))
            if len(self._cache) < 500_000:
                self._cache[piece] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        pieces = split_special(text) if allow_special else [text]
        for piece in pieces:
            if allow_special and piece in self.special:
                out.append(self.special[piece])
                continue
            for pt in pretokenize(piece):
                out.extend(self.encode_pretoken(pt))
        return out

    def encode_with_steps(self, text: str):
        pretokens, ids = [], []
        for piece in split_special(text):
            if piece in self.special:
                sid = self.special[piece]
                pretokens.append({"text": piece, "bytes": [], "steps": [], "ids": [sid], "special": True})
                ids.append(sid)
                continue
            for pt in pretokenize(piece):
                b = list(pt.encode("utf-8"))
                final, steps = self.bpe_steps(b)
                pretokens.append({"text": pt, "bytes": b, "steps": steps, "ids": final, "special": False})
                ids.extend(final)
        return {"pretokens": pretokens, "ids": ids}

    # -- decoding --
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def token_to_display(self, tid: int) -> str:
        """Readable form of one token. Bytes that are not a complete UTF-8 character on their own
        are shown as <0xE7>."""
        if tid in self.special_ids:
            return self.special_ids[tid]
        return bytes_to_display(self.token_bytes[tid])

    # -- export --
    def to_json(self, extra: dict | None = None) -> dict:
        d = {
            "type": "byte-level-bpe",
            "vocabSize": self.vocab_size,
            "pretokenize": {"pattern": PRETOKENIZE_PATTERN, "flags": "gu"},
            "byteToToken": list(range(256)),
            "merges": [list(m) for m in self.merges],
            "tokenBytes": [b.hex() for b in self.token_bytes],
            "specialTokens": dict(self.special),
            "notes": (
                "Token ids 0-255 are raw bytes (byteToToken[b] = b). Merge i joins merges[i] = [a, b] "
                "into token 256 + i. Special tokens follow the merges. tokenBytes[id] is the hex of the "
                "UTF-8 bytes each token stands for. Encoding: split on special tokens, pre-tokenize with "
                "the regex (flags 'gu' in JavaScript, Python `regex` module), UTF-8 encode each pre-token, "
                "then repeatedly merge the adjacent pair with the lowest merge index (all occurrences, "
                "left to right)."
            ),
        }
        if extra:
            d.update(extra)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "BPETokenizer":
        tok = cls([tuple(m) for m in d["merges"]], list(d["specialTokens"].keys()))
        assert tok.vocab_size == d["vocabSize"]
        return tok

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))


def bytes_to_display(b: bytes) -> str:
    """Decode bytes as UTF-8; any byte that is not part of a complete, valid character is shown
    as <0xXX>."""
    out, i, n = [], 0, len(b)
    while i < n:
        c = b[i]
        if c < 0x80:
            out.append(chr(c))
            i += 1
            continue
        need = 1 if 0xC2 <= c <= 0xDF else 2 if 0xE0 <= c <= 0xEF else 3 if 0xF0 <= c <= 0xF4 else -1
        ch = None
        if need > 0 and i + need <= n - 1 and all(0x80 <= b[i + k] <= 0xBF for k in range(1, need + 1)):
            try:
                ch = b[i:i + need + 1].decode("utf-8")  # rejects overlong forms and surrogates
            except UnicodeDecodeError:
                ch = None
        if ch is not None:
            out.append(ch)
            i += need + 1
        else:
            out.append(f"<0x{c:02X}>")
            i += 1
    return "".join(out)
