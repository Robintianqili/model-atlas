"""Export real tokenizers for the /train/tokenization/ page into compact self-hosted files.

Byte-level BPE (GPT-2, cl100k_base, Qwen2.5) -> <name>.tokb
    'TOKB' | u32 LE count N | N records: u8 len | len bytes      (record i = bytes of token id i)
  Encoding in JS is tiktoken-style: merge the adjacent pair whose concatenation has the lowest id.
  check_rank_equiv.py showed this equals HF's merges-based BPE for GPT-2 and Qwen2.5
  (merge i always creates id 256 + i, no token has two merges, 0 disagreements on 22,927 words).
WordPiece (bert-base-uncased) -> bert-base-uncased.txt (the original vocab.txt, one token per line).

Usage: python export.py <out_dir>
"""
import os, sys, json, base64, struct, shutil
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
U2B = {v: k for k, v in bytes_to_unicode().items()}

def write_tokb(path, toks):
    with open(path, 'wb') as f:
        f.write(b'TOKB'); f.write(struct.pack('<I', len(toks)))
        for t in toks:
            assert 1 <= len(t) <= 255, (len(t), t[:20])
            f.write(bytes([len(t)])); f.write(t)
    return os.path.getsize(path)

info = {}
# GPT-2 and Qwen2.5 from HF tokenizer.json (vocab ids are BPE ranks)
for name, src in [('gpt2', 'gpt2'), ('qwen2.5', 'qwen2.5')]:
    j = json.load(open(f'{R}/{src}/tokenizer.json', encoding='utf8'))
    vocab = j['model']['vocab']
    n = len(vocab)
    toks = [None] * n
    for s, i in vocab.items():
        toks[i] = bytes(U2B[c] for c in s)
    assert all(t is not None for t in toks)
    specials = {a['content']: a['id'] for a in j['added_tokens']}
    size = write_tokb(f'{out}/{name}.tokb', toks)
    info[name] = dict(n=n, bytes=size, maxlen=max(map(len, toks)), specials=specials)

# cl100k_base from the tiktoken file (base64 token, rank)
ranks = {}
for line in open(f'{R}/cl100k_base.tiktoken', 'rb'):
    if line.strip():
        t, r = line.split()
        ranks[int(r)] = base64.b64decode(t)
n = len(ranks)
assert sorted(ranks) == list(range(n)), 'cl100k ranks have gaps'
toks = [ranks[i] for i in range(n)]
size = write_tokb(f'{out}/cl100k_base.tokb', toks)
info['cl100k_base'] = dict(n=n, bytes=size, maxlen=max(map(len, toks)),
    specials={'<|endoftext|>': 100257, '<|fim_prefix|>': 100258, '<|fim_middle|>': 100259, '<|fim_suffix|>': 100260, '<|endofprompt|>': 100276})

shutil.copyfile(f'{R}/bert-base-uncased-vocab.txt', f'{out}/bert-base-uncased.txt')
lines = open(f'{out}/bert-base-uncased.txt', encoding='utf8').read().split('\n')
info['bert-base-uncased'] = dict(n=len([l for l in lines if l != '']) , bytes=os.path.getsize(f'{out}/bert-base-uncased.txt'))
print(json.dumps(info, ensure_ascii=False, indent=1))
