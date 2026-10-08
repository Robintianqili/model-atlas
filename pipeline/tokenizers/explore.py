import os, json
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(os.path.dirname(__file__), 'cache')
import tiktoken
from tokenizers import Tokenizer
R = os.path.join(os.path.dirname(__file__), 'raw')
enc_g = tiktoken.get_encoding('gpt2')
enc_c = tiktoken.get_encoding('cl100k_base')
hf_g = Tokenizer.from_file(f'{R}/gpt2/tokenizer.json')
hf_b = Tokenizer.from_file(f'{R}/bert-base-uncased-tokenizer.json')
hf_q = Tokenizer.from_file(f'{R}/qwen2.5/tokenizer.json')
texts = ['我爱自然语言处理 unbelievable!', '我爱自然语言处理', 'The weather is nice today', '今天天气很好', 'Hello world 🙂', ' the the The THE', '12345678 + 3.14159 = ?', 'unbelievable', 'naïve café']
for s in texts:
    print(repr(s))
    for name, f in [('gpt2', lambda s: [enc_g.decode_single_token_bytes(i) for i in enc_g.encode_ordinary(s)]),
                    ('cl100k', lambda s: [enc_c.decode_single_token_bytes(i) for i in enc_c.encode_ordinary(s)])]:
        t = f(s); print('  ', name, len(t), t)
    for name, tk in [('hf-gpt2', hf_g), ('bert', hf_b), ('qwen', hf_q)]:
        e = tk.encode(s); print('  ', name, len(e.tokens), e.tokens, e.ids[:12])
print(json.dumps(hf_q.to_str()[:0]))
q = json.loads(open(f'{R}/qwen2.5/tokenizer.json', encoding='utf8').read())
print(q['normalizer']); print(json.dumps(q['pre_tokenizer'])[:600]); print(q['post_processor']); print(q['decoder']); print(q['model']['type'], len(q['model']['vocab']), len(q['model']['merges']), [a['content'] for a in q['added_tokens']][:30], len(q['added_tokens']))
b = json.loads(open(f'{R}/bert-base-uncased-tokenizer.json', encoding='utf8').read())
print(b['normalizer'], b['pre_tokenizer'], json.dumps(b['post_processor'])[:300], {k:v for k,v in b['model'].items() if k!='vocab'})
g = json.loads(open(f'{R}/gpt2/tokenizer.json', encoding='utf8').read())
print(g['normalizer'], g['pre_tokenizer'], g['post_processor'], {k:v for k,v in g['model'].items() if k not in ('vocab','merges')})
print(enc_c._pat_str); print(enc_g._pat_str); print(enc_c._special_tokens, enc_g._special_tokens)
