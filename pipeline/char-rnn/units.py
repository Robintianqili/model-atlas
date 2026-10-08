"""char-rnn: what the 64 LSTM units do on held-out text (the measured facts and unitTop.json that export.py writes).

    npm run atlas -- py pipeline/char-rnn/units.py        (prints the facts and the counter unit's top contexts)

Held-out text: the first 400,000 characters of TinyStoriesV2-GPT4-valid.txt (not used for training), normalized as in
training and cut into windows of 128 characters that each start from an empty memory (h = c = 0), as the page does.
CPU, float32, TF32 off. Positions 16 and later of a window are used for correlations (earlier ones have little context).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import ALPHABET, CharLSTM, encode, normalize  # noqa: E402

T = 128
EVAL_CHARS = 400_000


def heldout() -> str:
    text = normalize(ak.data.tinystories_chars("TinyStoriesV2-GPT4-valid.txt", n_chars=EVAL_CHARS))
    return text[: (len(text) // T) * T]


@torch.no_grad()
def activations(m: CharLSTM, text: str) -> dict[str, np.ndarray]:
    """c, h and the forget gate f of every unit at every character ([N, 64] each, text order), windows of T."""
    ids = torch.tensor(encode(text))
    n = len(ids) // T
    x = m.emb(ids[: n * T].view(n, T).T)                       # [T, n, 24]
    W_ih, W_hh = m.lstm.weight_ih_l0, m.lstm.weight_hh_l0
    b = m.lstm.bias_ih_l0 + m.lstm.bias_hh_l0
    H = m.lstm.hidden_size
    h, c = torch.zeros(n, H), torch.zeros(n, H)
    hs, cs, fs = [], [], []
    for t in range(T):
        z = x[t] @ W_ih.T + h @ W_hh.T + b
        i, f, g, o = z.sigmoid()[:, :H], z[:, H:2 * H].sigmoid(), z[:, 2 * H:3 * H].tanh(), z[:, 3 * H:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()
        hs.append(h); cs.append(c); fs.append(f)
    flat = lambda a: torch.stack(a).permute(1, 0, 2).reshape(-1, H).numpy()
    return {"h": flat(hs), "c": flat(cs), "f": flat(fs)}


def features(text: str) -> dict[str, np.ndarray]:
    """letters so far in the current word; inside a quote (counted from the window start); window position."""
    N = len(text)
    letters = np.zeros(N)
    run = 0
    for k, ch in enumerate(text):
        run = run + 1 if ch.isalpha() else 0
        letters[k] = run
    inq = np.zeros(N)
    seen = np.zeros(N, bool)
    for w in range(N // T):
        q = s = False
        for k in range(w * T, (w + 1) * T):
            if text[k] == '"':
                q, s = not q, True
            inq[k], seen[k] = q, s
    return {"letters": letters, "inquote": inq, "quoteSeen": seen, "pos": np.tile(np.arange(T), N // T)}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def analyse(m: CharLSTM) -> tuple[dict, dict]:
    text = heldout()
    A = activations(m, text)
    F = features(text)
    ok = F["pos"] >= 16
    H = A["c"].shape[1]
    # the counter: the unit whose cell value rises most clearly with the letters of the current word
    rs = [corr(A["c"][ok, j], F["letters"][ok]) for j in range(H)]
    u = int(np.argmax(rs))
    prev_alpha = np.array([False] + [ch.isalpha() for ch in text[:-1]])
    is_alpha = np.array([ch.isalpha() for ch in text])
    is_space = np.array([ch == " " for ch in text])
    reset = ok & is_space & prev_alpha                       # a space right after a letter
    inword = ok & is_alpha & prev_alpha                      # a letter right after a letter
    cprev = np.concatenate([[0.0], A["c"][:-1, u]])
    rise = A["c"][inword, u] - cprev[inword]
    # quotes: how well any unit (c or h) follows "a quote is open" (only where the window has seen a quote)
    qk = ok & F["quoteSeen"]
    qr = max(abs(corr(A[a][qk, j], F["inquote"][qk])) for a in ("c", "h") for j in range(H))
    facts = {
        "evalChars": {"value": len(text), "unit": "characters", "source": "units.py: first 400,000 characters of TinyStoriesV2-GPT4-valid.txt, windows of 128"},
        "counterUnit": {"value": u, "unit": "unit index (from 0)", "source": "units.py: largest correlation of c with the letters so far in the word"},
        "counterR": {"value": round(rs[u], 4), "unit": "Pearson r", "source": "units.py"},
        "counterResetF": {"value": round(float(np.median(A["f"][reset, u])), 4), "unit": "median forget gate at a space after a letter", "source": "units.py"},
        "counterLetterF": {"value": round(float(np.median(A["f"][inword, u])), 4), "unit": "median forget gate at a letter after a letter", "source": "units.py"},
        "counterRise": {"value": round(float(np.median(rise)), 4), "unit": "median change of c per letter inside a word", "source": "units.py"},
        "quoteR": {"value": round(qr, 4), "unit": "largest |Pearson r| of any unit's c or h with 'a quote is open'", "source": "units.py"},
        "quoteChars": {"value": int(text.count('"')), "unit": "quote marks in the held-out text", "source": "units.py"},
        "medianF": {"value": round(float(np.median(A["f"][ok])), 4), "unit": "median forget gate, all units and characters", "source": "units.py"},
        "meanF": {"value": round(float(np.mean(A["f"][ok])), 4), "unit": "mean forget gate, all units and characters", "source": "units.py"},
    }
    tops = []
    for j in range(H):
        order = np.argsort(-A["c"][:, j], kind="stable")
        picked: list[int] = []
        for k in order:
            if F["pos"][k] < 16 or any(abs(int(k) - q) < 12 for q in picked):
                continue
            picked.append(int(k))
            if len(picked) == 6:
                break
        tops.append({"unit": j, "r": round(rs[j], 4), "top": [{"before": text[k - 15:k], "at": text[k], "after": text[k + 1:k + 4], "c": round(float(A["c"][k, j]), 3)} for k in picked]})
    unit_top = {"format": "char-rnn-units/1", "by": "cell value c", "window": T, "chars": len(text),
                "source": "pipeline/char-rnn/units.py on the first 400,000 characters of TinyStoriesV2-GPT4-valid.txt (held out from training)",
                "units": tops}
    return facts, unit_top


if __name__ == "__main__":
    m = CharLSTM()
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    facts, top = analyse(m.eval())
    for k, v in facts.items():
        print(k, v["value"])
    u = facts["counterUnit"]["value"]
    for e in top["units"][u]["top"]:
        print(repr(e["before"] + "[" + e["at"] + "]" + e["after"]), e["c"])
