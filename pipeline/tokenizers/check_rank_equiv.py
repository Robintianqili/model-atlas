"""Does tiktoken-style BPE (merge the adjacent pair whose concatenation has the lowest rank)
reproduce HF merges-based BPE for GPT-2 and Qwen2.5? Checks the merge table structure and
compares both algorithms on many pre-tokens."""
import os, json, random
import regex
from tokenizers import Tokenizer
R = os.path.join(os.path.dirname(__file__), 'raw')

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
B2U = bytes_to_unicode(); U2B = {v: k for k, v in B2U.items()}
def tok_bytes(s): return bytes(U2B[c] for c in s)

for name in ['gpt2', 'qwen2.5']:
    j = json.load(open(f'{R}/{name}/tokenizer.json', encoding='utf8'))
    vocab = j['model']['vocab']; merges = j['model']['merges']
    merges = [tuple(m.split(' ')) if isinstance(m, str) else tuple(m) for m in merges]
    ranks = {tok_bytes(t): i for t, i in vocab.items()}
    ok = sum(1 for i, (a, b) in enumerate(merges) if vocab.get(a + b) == 256 + i)
    prod = {}
    for a, b in merges: prod.setdefault(a + b, []).append((a, b))
    multi = sum(1 for v in prod.values() if len(v) > 1)
    print(name, 'vocab', len(vocab), 'merges', len(merges), 'merge i -> id 256+i:', ok, 'tokens with >1 merge:', multi)
    # ids 0..255 single bytes?
    single = sorted(i for t, i in vocab.items() if len(tok_bytes(t)) == 1)
    print('  single-byte ids', single[:3], '...', single[-3:], len(single))
    mranks = {m: i for i, m in enumerate(merges)}
    def hf_bpe(bs):
        parts = [B2U[b] for b in bs]
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = mranks.get((parts[i], parts[i+1]))
                if r is not None and (best is None or r < best[0]): best = (r, i)
            if best is None: break
            r, _ = best; a, b = merges[r]
            out = []; i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i+1] == b: out.append(a + b); i += 2
                else: out.append(parts[i]); i += 1
            parts = out
        return [vocab[p] for p in parts]
    def tt_bpe(bs):
        parts = [bytes([b]) for b in bs]
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = ranks.get(parts[i] + parts[i+1])
                if r is not None and (best is None or r < best[0]): best = (r, i)
            if best is None: break
            _, i = best
            parts = parts[:i] + [parts[i] + parts[i+1]] + parts[i+2:]
        return [ranks[p] for p in parts]
    random.seed(0)
    toks = list(vocab.keys())
    words = []
    for _ in range(20000):
        k = random.randint(1, 4)
        words.append(b''.join(tok_bytes(random.choice(toks)) for _ in range(k)))
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'tinystories', 'TinyStoriesV2-GPT4-valid.txt'), encoding='utf8').read()[:300000]
    pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    words += [w.encode() for w in set(pat.findall(txt))]
    diff = 0; ex = []
    for w in words:
        a = hf_bpe(w); b = tt_bpe(w)
        if a != b:
            diff += 1
            if len(ex) < 5: ex.append((w, a, b))
    print('  words', len(words), 'disagreements', diff, ex)
