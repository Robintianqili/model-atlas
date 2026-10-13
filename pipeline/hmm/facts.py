"""hmm: copy the char-RNN page's measured numbers that the comparison step quotes into static/models/hmm/facts.json.

The hmm page compares a hidden Markov model's memory (one of two states) with the char-RNN's LSTM. The numbers come
from the char-RNN package's own measured files, read here, never typed: its spec.json (the LSTM layer's width) and its
facts.json (parameters, and the "letter counter" cell found by pipeline/char-rnn/units.py). Run it again whenever the
char-RNN package is re-exported.

    npm run atlas -- py pipeline/hmm/facts.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak


def main() -> None:
    src = ak.site_path("static", "models", "char-rnn")
    spec = json.loads((src / "spec.json").read_text(encoding="utf-8"))
    cf = json.loads((src / "facts.json").read_text(encoding="utf-8"))["facts"]
    lstm = next(l for l in spec["layers"] if l["op"] == "lstm")
    hidden = int(lstm["outShape"][-1])
    today = _dt.date.today().isoformat()

    def fact(value, unit, source):
        return {"value": value, "unit": unit, "source": source, "measured": today}

    facts = {
        "charRnnHidden": fact(hidden, "numbers in the LSTM's h (and as many in its cell c), per character",
                              "static/models/char-rnn/spec.json: the outShape of layer 'lstm'"),
        "charRnnParams": fact(cf["params"]["value"], "parameters, learned by gradient descent", "static/models/char-rnn/facts.json: params (" + cf["params"]["source"] + ")"),
        "counterUnit": fact(cf["counterUnit"]["value"], "cell index (counting from 0)", "static/models/char-rnn/facts.json: counterUnit (" + cf["counterUnit"]["source"] + ")"),
        "counterR": fact(cf["counterR"]["value"], "Pearson r with the number of letters so far in the word", "static/models/char-rnn/facts.json: counterR (" + cf["counterR"]["source"] + ")"),
    }
    out = ak.site_path("static", "models", "hmm", "facts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ak.write_json(out, {"format": "atlas-facts/1", "facts": facts}, indent=1)
    print(f"wrote {out}: " + ", ".join(f"{k} = {v['value']}" for k, v in facts.items()))


if __name__ == "__main__":
    main()
